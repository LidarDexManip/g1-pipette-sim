"""Collect independently replayed motor-target pipette demonstrations.

Dataset actions are 29 motor targets, NOT the real dataset's 35-D mimic goals.
All geometry/action variants are fixed before replay. Nothing is teleported during
execution. Simulation contact/attachment criteria are inherited unchanged.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,copy,fcntl,hashlib,json,math,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.simulation import set_robot_pose
from g1_pipette_sim.scene import rack_target,build_xml
from g1_pipette_sim.pickup_layouts import ROOT,MANIFEST,model_fingerprints,sha256
from g1_pipette_sim.camera_views import apply_camera_views, assert_camera_views
from tools.fit_pickup_layouts import write_json
from tools.search_replay_layout import on_table

SEEDS=[14,64,68,144,6,13,24,30,35,40,41,45]
SIZES={'rgb':(512,288),'wrist_right':(384,384),'wrist_left':(384,384)}
FPS=30


def placed_config(base,cell,shift):
    cfg=copy.deepcopy(base);old=rack_target(cfg);cfg['rack']['active_cell']=list(cell)
    cfg['rack']['center_xy']=(np.asarray(cfg['rack']['center_xy'])+old[:2]-rack_target(cfg)[:2]+np.asarray(shift)).tolist()
    cfg.pop('pickup_layout',None);cfg.pop('layout_search',None)
    cfg['episode_scene']={'enabled':False}
    return cfg


def smoothstep(x):
    x=np.clip(x,0.,1.);return x*x*(3-2*x)


def make_actions(ep,cfg,shift,noise_seed,kind):
    original=ep.arrays['action.low_level'].copy()
    if kind=='well_only':return original,{'max_joint_delta_rad':0.}
    # Jacobian maps a sampled Cartesian translation into small coordinated arm
    # corrections. A ramp preserves the recorded initial pose/grasp transient.
    model=mujoco.MjModel.from_xml_string(build_xml(cfg));data=mujoco.MjData(model)
    names=ep.names[22:29];joints=np.array([model.joint(n).id for n in names]);dofs=model.jnt_dofadr[joints]
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);rng=np.random.default_rng(noise_seed)
    knot_times=np.arange(0,ep.duration+.2,.2);offsets=[]
    null_coeff=float(rng.uniform(-.0003,.0003))
    for t in knot_times:
        body,left,right=ep.sample(float(t),'low_level');set_robot_pose(model,data,ep.names,body,left,right,cfg);mujoco.mj_forward(model,data)
        mujoco.mj_jacSite(model,data,jp,jr,model.site('nozzle').id)
        J=np.vstack([jp[:,dofs],.1*jr[:,dofs]])
        inv=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-7,np.eye(6))
        delta=inv@np.r_[shift,0.,0.,0.,0.]
        # Smooth, low-amplitude null-space variation; fixed random direction.
        direction=np.array([.2,-.3,.1,.2,-.4,.3,.1]);direction/=np.linalg.norm(direction)
        delta+=(np.eye(7)-inv@J)@direction*null_coeff*np.sin(np.pi*min(t/ep.duration,1.))
        delta*=smoothstep(t/2.)
        offsets.append(delta)
    offsets=np.asarray(offsets)
    correction=np.stack([np.interp(ep.action_times,knot_times,offsets[:,i]) for i in range(7)],axis=1)
    original[:,22:29]+=correction
    limits=model.jnt_range[joints]
    if np.any(original[:,22:29]<limits[:,0]) or np.any(original[:,22:29]>limits[:,1]):raise ValueError('Augmented arm target exceeds joint range')
    if np.max(np.abs(correction))>(.08 if kind=='retarget_well' else .03):raise ValueError('IK correction too large')
    if np.max(abs(np.diff(correction,axis=0)/np.diff(ep.action_times)[:,None]))>.08:raise ValueError('Augmentation velocity too large')
    return original,{'max_joint_delta_rad':float(np.max(abs(correction))),'max_added_joint_velocity_rad_s':float(np.max(abs(np.diff(correction,axis=0)/np.diff(ep.action_times)[:,None]))),'nullspace_coefficient_rad':null_coeff,'ramp_s':2.}


from g1_pipette_sim.augmentation.sources import load_episode


def validate(candidate,record=False):
    if (Path(candidate['folder'])/'accepted.json').exists():
        raise ValueError('Accepted captures are immutable; use the shared runner to verify reuse')
    import tools.fit_pickup_layouts as fit
    cfg=json.loads(Path(candidate['config']).read_text());ep=load_episode(candidate)
    if record and candidate.get('run_id'):assert_camera_views(cfg,ep)
    capture=[];factory=None;diagnostic=[]
    if record:
        def factory(c,e,full=False):
            sim=RecordedSimulation(c,e,candidate);capture.append(sim);return sim
    else:
        from g1_pipette_sim.augmentation.failure_data import make_diagnostic_simulation
        def factory(c,e,full=False):
            sim=make_diagnostic_simulation(c,e);diagnostic.append(sim);return sim
    try:
        result,trace,report=fit.rollout(cfg,ep,full=True,lift_frame='world',simulation_factory=factory)
    finally:
        for sim in capture:sim.close_recording()
        for sim in diagnostic:
            sim.save_diagnostics(Path(candidate['folder'])/'screen-state.npz')
    from g1_pipette_sim.augmentation.failure_data import save_trace, atomic_npz
    save_trace(candidate['folder'],'record' if record else 'screen',trace)
    events=result.get('events',[]);attached=next((e['time'] for e in events if e['state']=='attached'),None)
    # Prevent noisy variants from acquiring an unintended early attachment.
    timing_ok=attached is not None and abs(attached-candidate['seed_attachment_s'])<=.75 and attached>=1.
    result['seed_phase_timing_passed']=timing_ok
    result['pickup']=bool(result['pickup'] and result['complete'] and timing_ok)
    if record and capture:
        # Failed recording attempts retain numeric evidence even without acceptance.
        sim=capture[0]
        if not result.get('pickup') or not result.get('complete'):
            atomic_npz(Path(candidate['folder'])/'failed-recording-state.npz',
                       **{k:np.asarray(v) for k,v in sim.rows.items()},qpos=np.asarray(sim.full_q),qvel=np.asarray(sim.full_v))
    folder=Path(candidate['folder']);write_json(folder/('verification.json' if record else 'screen.json'),{'result':result,'report':report})
    if record and result['pickup']:
        sim=capture[0];sim.save_rows()
        result['frame_count']=len(sim.rows['timestamp'])
        if not all(np.isfinite(np.asarray(values)).all() for values in sim.rows.values()):
            raise ValueError('Nonfinite captured numeric data')
        audit_videos(folder,result['frame_count'])
        from g1_pipette_sim.augmentation.contracts import atomic_json, file_hash
        artifact_hashes={name:file_hash(folder/name) for name in ['config.json','commands.npz','trajectory.npz','camera-poses.json','verification.json',*[v+'.mp4' for v in SIZES]]}
        atomic_json(folder/'accepted.json',{'artifact_hashes':artifact_hashes,'candidate':candidate,'validation':result,'video_sizes':SIZES,'fps':FPS,'action_semantics':'29 motor joint targets in radians; separate from real high-level mimic actions','training_split_group':f"hil194:{candidate['parent_episode']}",'camera_calibration':'approximate','independently_replayed':True})
    return result


class RecordedSimulation(DynamicSimulation):
    def __init__(self,cfg,ep,candidate):
        assert_camera_views(cfg,ep)
        super().__init__(cfg,ep,source='low_level')
        from g1_pipette_sim.camera_streams import CameraStreams
        import cv2
        self.folder=Path(candidate['folder']);self.cameras={};self.writers={};self.rows={k:[] for k in ['timestamp','simulation_time','observation.state','action','action.hand_left','action.hand_right','observation.hand_joint_positions','tip_position','nozzle_position','tip_attached']}
        self.full_q=[];self.full_v=[];self.camera_poses=[];self.next_frame=0
        self.body_adr=np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in ep.names]);self.hand_adr=np.array([self.model.jnt_qposadr[int(j)] for j in self.joints if self.model.joint(int(j)).name.startswith(('L_','R_'))])
        try:
            for name,size in SIZES.items():
                self.cameras[name]=CameraStreams(self.model,[name],*size)
                writer=cv2.VideoWriter(str(self.folder/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),FPS,size)
                if not writer.isOpened():raise RuntimeError('Cannot open video encoder')
                self.writers[name]=writer
            self.record_frame()
        except BaseException:self.close_recording();raise
    def record_frame(self):
        import cv2
        t=self.next_frame/FPS
        body,left,right=self.episode.sample(self.time,'low_level')
        values=[t,self.time,self.data.qpos[self.body_adr].copy(),body,left,right,self.data.qpos[self.hand_adr].copy(),self.data.body('tip').xpos.copy(),self.data.site('nozzle').xpos.copy(),self.state=='attached']
        for key,value in zip(self.rows,values):self.rows[key].append(value)
        self.full_q.append(self.data.qpos.copy());self.full_v.append(self.data.qvel.copy())
        poses={}
        for name,camera in self.cameras.items():
            pixels=camera.render(self.data)[name];self.writers[name].write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR));poses[name]=camera.poses(self.data)[name]
            if self.next_frame==0:cv2.imwrite(str(self.folder/(name+'-first.jpg')),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
        self.camera_poses.append(poses);self.next_frame+=1
    def step(self):
        super().step()
        if self.time+1e-10>=self.next_frame/FPS and self.next_frame/FPS<=self.episode.duration:self.record_frame()
    def close_recording(self):
        for w in getattr(self,'writers',{}).values():w.release()
        for c in getattr(self,'cameras',{}).values():c.close()
        self.writers={};self.cameras={}
    def save_rows(self):
        np.savez_compressed(self.folder/'trajectory.npz',**{k:np.asarray(v) for k,v in self.rows.items()},qpos=np.asarray(self.full_q),qvel=np.asarray(self.full_v))
        write_json(self.folder/'camera-poses.json',self.camera_poses)


def audit_videos(folder,expected):
    import cv2
    for name,size in SIZES.items():
        cap=cv2.VideoCapture(str(folder/(name+'.mp4')));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame.shape[:2]!=(size[1],size[0]):raise ValueError('Wrong camera resolution')
            count+=1
        cap.release()
        if count!=expected:raise ValueError(f'{name}: {count} decoded frames, expected {expected}')


def prepare(job):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    number,out,manifest,seed_value=job
    index=SEEDS[number%len(SEEDS)];round_id=number//len(SEEDS);entry=manifest['episodes'][str(index)]
    base=json.loads((ROOT/entry['config']).read_text());ep=Episode(index=index);base=apply_camera_views(base,ep)
    rng=np.random.default_rng(np.random.SeedSequence([seed_value,number]))
    rows,cols=base['rack']['rows'],base['rack']['cols'];old=base['rack']['active_cell']
    # First round checks nearby alternate wells. Later draws include broad wells
    # and local cells, with continuous, reproducible spatial/action variation.
    if round_id<8:
        neighbors=[(r,c) for r in range(rows) for c in range(cols) if [r,c]!=old]
        neighbors.sort(key=lambda cell:(abs(cell[0]-old[0])+abs(cell[1]-old[1]),cell))
        cell=neighbors[round_id];kind='well_only';shift=np.zeros(2)
    else:
        if rng.random()<.65:
            cell=(int(np.clip(old[0]+rng.integers(-2,3),0,rows-1)),int(np.clip(old[1]+rng.integers(-2,3),0,cols-1)))
        else:cell=(int(rng.integers(rows)),int(rng.integers(cols)))
        kind='smooth_action_and_placement';amplitude=float(rng.choice([.0005,.0015]));shift=rng.uniform(-amplitude,amplitude,2)
        if np.linalg.norm(shift)<.0002:shift[0]+=.00025
        if round_id%5==0:
            neighbors=[(r,c) for r,c in [(old[0]-1,old[1]),(old[0]+1,old[1]),(old[0],old[1]-1),(old[0],old[1]+1)] if 0<=r<rows and 0<=c<cols]
            cell=neighbors[int(rng.integers(len(neighbors)))];kind='retarget_well'
            changed=copy.deepcopy(base);changed['rack']['active_cell']=list(cell)
            shift=rack_target(changed)[:2]-rack_target(base)[:2]
    cfg=placed_config(base,cell,shift)
    if not on_table(cfg):return {'number':number,'rejected':'off_table'}
    folder=Path(out)/'candidates'/f'{number:06}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'candidate.json').exists():return json.loads((folder/'candidate.json').read_text())
    try:actions,noise=make_actions(ep,base,shift,seed_value+number,kind)
    except ValueError as error:return {'number':number,'rejected':str(error)}
    commands=folder/'commands.npz';np.savez_compressed(commands,times=ep.action_times,motor_targets=actions,hand_left=ep.arrays['action.hand_left'],hand_right=ep.arrays['action.hand_right'])
    config=folder/'config.json';write_json(config,cfg)
    digest=hashlib.sha256(json.dumps({'rack':cfg['rack'],'table':cfg['table']},sort_keys=True).encode()+actions.tobytes()).hexdigest()
    candidate={'number':number,'parent_episode':index,'kind':kind,'cell':list(cell),'target_shift_m':shift.tolist(),'noise':noise,'seed_attachment_s':next(e['time'] for e in entry['validation']['events'] if e['state']=='attached'),'config':str(config),'commands':str(commands),'folder':str(folder),'content_hash':digest,'seed_value':seed_value+number,'config_sha256':sha256(config),'commands_sha256':sha256(commands),'seed_config_sha256':entry['sha256']}
    write_json(folder/'candidate.json',candidate)
    return candidate


def screen_job(job):
    candidate=prepare(job)
    if 'rejected' in candidate:return candidate,None
    path=Path(candidate['folder'])/'screen.json'
    result=json.loads(path.read_text())['result'] if path.exists() else validate(candidate)
    return candidate,result


def main():
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',type=int,default=600);p.add_argument('--workers',type=int,default=4);p.add_argument('--render-workers',type=int,default=2);p.add_argument('--seed',type=int,default=20260912);p.add_argument('--max-candidates',type=int,default=20000);p.add_argument('--output',type=Path,default=ROOT/'outputs/hil194/augmentation600');a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'.collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(MANIFEST.read_text());fp=model_fingerprints()
    if fp!=manifest['model_fingerprints']:raise ValueError('Stale seed model')
    provenance={'model_fingerprints':fp,'seed_manifest_sha256':sha256(MANIFEST),'collector_sha256':sha256(Path(__file__)),'seed':a.seed,'dataset_snapshot':manifest['dataset_snapshot'],'action_semantics':'motor_targets_29_rad','fps':FPS,'sizes':SIZES}
    prov=a.output/'provenance.json'
    if prov.exists() and json.loads(prov.read_text())!=json.loads(json.dumps(provenance)):raise ValueError('Collector/model/seed provenance changed; use a new output')
    write_json(prov,provenance)
    statepath=a.output/'progress.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'accepted':[],'screened':0,'rejected':0,'errors':[],'next_candidate':0,'pending_render':[],'processed':[]}
    state['accepted']=list(dict.fromkeys(state['accepted']));accepted=set(state['accepted']);seen={json.loads((a.output/'candidates'/f'{i:06}'/'candidate.json').read_text())['content_hash'] for i in accepted}
    processed=set(state.get('processed',[]));todo=[]
    # Resume in-flight work from disk; only completed accepted.json counts.
    for folder in sorted((a.output/'candidates').glob('*')) if (a.output/'candidates').exists() else []:
        if (folder/'accepted.json').exists():
            i=int(folder.name)
            if i not in accepted:accepted.add(i);state['accepted'].append(i)
            seen.add(json.loads((folder/'candidate.json').read_text())['content_hash'])
        elif (folder/'screen.json').exists() and json.loads((folder/'screen.json').read_text())['result']['pickup']:
            if not (folder/'verification.json').exists() or json.loads((folder/'verification.json').read_text())['result']['pickup']:
                candidate=json.loads((folder/'candidate.json').read_text());todo.append(candidate);seen.add(candidate['content_hash'])
    state['target']=a.target;state['status']='running';state['started_at']=state.get('started_at',time.time());write_json(statepath,state)
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as screens, ProcessPoolExecutor(max_workers=a.render_workers,max_tasks_per_child=1) as renders:
        sf={};rf={}
        while len(accepted)<a.target:
            while todo and len(rf)<a.render_workers and len(accepted)+len(rf)<a.target:
                candidate=todo.pop(0);rf[renders.submit(validate,candidate,True)]=candidate
            while len(sf)<a.workers and len(todo)<a.render_workers*2 and state['next_candidate']<a.max_candidates and len(accepted)+len(rf)+len(todo)<a.target:
                n=state['next_candidate'];state['next_candidate']+=1
                if n in accepted:continue
                sf[screens.submit(screen_job,(n,str(a.output),manifest,a.seed))]=n
            if not sf and not rf:
                if not todo:break
                continue
            done,_=wait([*sf,*rf],timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    if future in sf:
                        number=sf.pop(future);candidate,result=future.result();state['screened']+=1
                        if result and result['pickup'] and candidate['content_hash'] not in seen:
                            seen.add(candidate['content_hash']);todo.append(candidate)
                        else:state['rejected']+=1
                        processed.add(number)
                    else:
                        candidate=rf.pop(future);result=future.result()
                        if result['pickup']:
                            accepted.add(candidate['number']);state['accepted'].append(candidate['number']);print(json.dumps({'ACCEPTED':len(accepted),'candidate':candidate['number'],'parent':candidate['parent_episode'],'kind':candidate['kind'],'cell':candidate['cell'],'lift_mm':result['lift_m']*1000,'target_mm':result['lift_target_m']*1000}),flush=True)
                        else:state['rejected']+=1
                except Exception as error:
                    state['errors'].append({'error':repr(error),'time':time.time()});print('ERROR',repr(error),flush=True)
            state['processed']=sorted(processed);state['pending_render']=[c['number'] for c in todo]+[c['number'] for c in rf.values()];state['updated_at']=time.time();write_json(statepath,state)
        for future in sf:future.cancel()
        # Screening jobs already running finish but are not counted as exports.
    state['status']='collected' if len(accepted)>=a.target else 'candidate_budget_exhausted';state['updated_at']=time.time();write_json(statepath,state)
    print(json.dumps({'status':state['status'],'accepted':len(accepted),'target':a.target}),flush=True)
if __name__=='__main__':main()
