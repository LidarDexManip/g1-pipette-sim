"""Search larger arm translations with paired rack/tip placements and full validation."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,copy,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.scene import build_xml
from g1_pipette_sim.simulation import set_robot_pose
from g1_pipette_sim.pickup_layouts import sha256,model_fingerprints
from tools import collect_pickup_augmentations as collection
from tools.fit_pickup_layouts import write_json
from tools.search_replay_layout import on_table
OUT=ROOT/'outputs/hil194/larger-augmentation-study'
BASE=ROOT/'outputs/hil194/augmentation600/candidates/000000'


from g1_pipette_sim.augmentation.sources import load_episode as exact_episode


def actions(ep,cfg,shift,lift_trim=0.):
    m=mujoco.MjModel.from_xml_string(build_xml(cfg));d=mujoco.MjData(m)
    names=ep.names[22:29];j=np.array([m.joint(n).id for n in names]);dofs=m.jnt_dofadr[j];qa=m.jnt_qposadr[j]
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);knots=np.arange(0,ep.duration+.1,.1);corrections=[]
    for t in knots:
        body,left,right=ep.sample(float(t),'low_level');set_robot_pose(m,d,ep.names,body,left,right,cfg);mujoco.mj_forward(m,d)
        initial=d.qpos[qa].copy();site=m.site('nozzle').id
        target=d.site(site).xpos.copy()+np.r_[shift,0.]*collection.smoothstep(t/2.)
        target[2]-=lift_trim*collection.smoothstep((t-14.)/4.)
        orientation=d.site(site).xmat.reshape(3,3).copy()
        for _ in range(12):
            poserr=target-d.site(site).xpos
            roterr=Rotation.from_matrix(orientation@d.site(site).xmat.reshape(3,3).T).as_rotvec()
            if np.linalg.norm(poserr)<1e-7 and np.linalg.norm(roterr)<1e-6:break
            mujoco.mj_jacSite(m,d,jp,jr,site);J=np.vstack([jp[:,dofs],.1*jr[:,dofs]])
            delta=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-7,np.r_[poserr,.1*roterr])
            d.qpos[qa]+=delta;mujoco.mj_forward(m,d)
        if np.linalg.norm(target-d.site(site).xpos)>1e-5:raise ValueError('IK did not converge')
        corrections.append(d.qpos[qa]-initial)
    correction=np.stack([np.interp(ep.action_times,knots,np.asarray(corrections)[:,i]) for i in range(7)],axis=1)
    result=ep.arrays['action.low_level'].copy();result[:,22:29]+=correction
    velocity=np.max(abs(np.diff(correction,axis=0)/np.diff(ep.action_times)[:,None]))
    if np.any(result[:,22:29]<m.jnt_range[j,0]) or np.any(result[:,22:29]>m.jnt_range[j,1]):raise ValueError('Joint limit')
    if np.max(abs(correction))>.35 or velocity>.35:raise ValueError('Perturbation bound exceeded')
    return result,{'max_joint_delta_rad':float(np.max(abs(correction))),'max_added_joint_velocity_rad_s':float(velocity),'ramp_s':2.,'method':'iterative nozzle translation IK with orientation preserved'}


def prepare(number,shift,placement_offset=(0.,0.)):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    folder=OUT/'candidates'/f'{number:06}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'candidate.json'
    if path.exists():return json.loads(path.read_text())
    record=json.loads((BASE/'accepted.json').read_text());base=json.loads((BASE/'config.json').read_text())
    provenance=json.loads((BASE.parents[1]/'provenance.json').read_text())
    if provenance['model_fingerprints']!=model_fingerprints():raise ValueError('Baseline model changed')
    ep=Episode(provenance['dataset_snapshot'],14)
    with np.load(BASE/'commands.npz') as a:
        ep.arrays['action.low_level']=a['motor_targets'].copy()
        for side in ['left','right']:ep.arrays['action.hand_'+side]=a['hand_'+side].copy()
    motor,noise=actions(ep,base,np.asarray(shift))
    cfg=collection.placed_config(base,base['rack']['active_cell'],np.asarray(shift)+np.asarray(placement_offset)[:2])
    if len(placement_offset)==3:cfg['table']['top_z']+=placement_offset[2]
    if not on_table(cfg):raise ValueError('Off table')
    config=folder/'config.json';write_json(config,cfg)
    commands=folder/'commands.npz';np.savez_compressed(commands,times=ep.action_times,motor_targets=motor,hand_left=ep.arrays['action.hand_left'],hand_right=ep.arrays['action.hand_right'])
    c={'number':number,'parent_episode':14,'kind':'larger_arm_and_placement','cell':cfg['rack']['active_cell'],'target_shift_m':list(shift),'placement_refinement_m':list(placement_offset),'noise':noise,'seed_attachment_s':next(e['time'] for e in record['validation']['events'] if e['state']=='attached'),'config':str(config),'commands':str(commands),'folder':str(folder),'dataset_snapshot':str(ep.root),'config_sha256':sha256(config),'commands_sha256':sha256(commands),'source_capture':str(BASE)}
    write_json(path,c);return c


def job(spec):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    number,shift,offset,record=spec
    c=prepare(number,shift,offset);folder=Path(c['folder']);path=folder/('accepted.json' if record else 'screen.json')
    if path.exists():return {'candidate':c,'result':json.loads(path.read_text())['validation' if record else 'result']}
    result=collection.validate(c,record=record)
    return {'candidate':c,'result':result}


def main():
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');p.add_argument('--refine',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    shifts=[(x,y) for r in [.005,.01,.02] for x,y in [(r,0),(-r,0),(0,r),(0,-r)]]
    specs=[(i,list(s),[0.,0.],a.record) for i,s in enumerate(shifts)]
    if a.refine:
        specs=[]
        reference=json.loads((BASE/'verification.json').read_text())['report']['attachment_diagnostics']['closest_approach']
        bias=np.array(reference['nozzle_world_m'])[:2]-np.array(reference['tip_mouth_world_m'])[:2]
        for i,shift in enumerate(shifts):
            path=OUT/'candidates'/f'{i:06}'/'screen.json'
            if not path.exists():continue
            screen=json.loads(path.read_text());closest=screen['report']['attachment_diagnostics']['closest_approach']
            if closest is None:continue
            offset=np.array(closest['nozzle_world_m'])[:2]-np.array(closest['tip_mouth_world_m'])[:2]-bias
            if screen['result'].get('attached'):
                offsets=[[0.,0.,.0005],[0.,0.,.001],[0.,0.,-.0005]]
            else:
                offsets=[[*offset,0.],[*(offset*.5),0.],[*offset,.0005]]
            for j,delta in enumerate(offsets):specs.append((100+i*3+j,list(shift),delta,False))
    if a.record:
        specs=[]
        for path in sorted((OUT/'candidates').glob('*/screen.json')):
            if json.loads(path.read_text())['result']['pickup']:
                c=json.loads((path.parent/'candidate.json').read_text())
                specs.append((c['number'],c['target_shift_m'],c['placement_refinement_m'],True))
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(job,s):s for s in specs}
        for f in as_completed(futures):
            try:
                item=f.result();r=item['result'];results.append(item)
                print(json.dumps({'candidate':item['candidate']['number'],'shift_mm':(np.asarray(item['candidate']['target_shift_m'])*1000).tolist(),'pickup':r['pickup'],'lift_mm':r.get('lift_m',0)*1000,'reason':r.get('invalid_reason'),'events':r.get('events'),'max_joint_delta_deg':np.degrees(item['candidate']['noise']['max_joint_delta_rad'])}),flush=True)
            except Exception as e:results.append({'spec':futures[f],'error':repr(e)});print(repr(e),flush=True)
            write_json(OUT/('recording-summary.json' if a.record else ('refinement-summary.json' if a.refine else 'screening-summary.json')),results)
    print('DONE',flush=True)
if __name__=='__main__':main()
