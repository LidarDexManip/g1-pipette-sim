"""Resume bounded, unchanged-command pickup placement searches over HIL194.

Only rack XY/yaw and (optional) common table/object height are fitted. Grasp,
commands, contacts and attachment gates are never changed. Each installed result
is independently replayed in full; failures remain explicitly unverified.
"""
import argparse,copy,hashlib,json,math,sys,time,gc
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode,DEFAULT_DATASET
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.scene import rack_target
from g1_pipette_sim.pickup_layouts import model_fingerprints,sha256,asset_fingerprints_from_xml
from tools.search_replay_layout import shifted_config,on_table,rack_penetrations

CRITERIA={'lift_reference':'recorded_joint_fk_v1','lift_dwell_s':.1,'max_tool_penetration_mm':.5,'max_tool_force_N':100.,'max_lid_penetration_mm':.1}

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def candidate_config(base,delta):
    cfg=shifted_config(base,np.asarray(delta[:2]))
    cfg['table']['top_z']+=float(delta[2])
    cfg['rack']['yaw_deg']+=float(delta[3])
    return cfg

def candidate_simulation(cfg,ep,full=False):
    if full:return DynamicSimulation(cfg,ep,source='low_level')
    # Candidate proposal proxy only. The saved config and final validation retain
    # the full 48-sector tip collision shell. Each worker runs one build at a time.
    import g1_pipette_sim.object_meshes as meshes
    coarse=copy.deepcopy(cfg);coarse['pipette']['search_collision_segments']=16
    original=meshes.tip_shell_parts
    try:
        meshes.tip_shell_parts=lambda p:original(p,segments=16)
        return DynamicSimulation(coarse,ep,source='low_level')
    finally:meshes.tip_shell_parts=original

def rollout(cfg,ep,full=False,lift_frame="rack",simulation_factory=None):
    if lift_frame not in {"rack","world"}:raise ValueError(lift_frame)
    gc.collect()
    criteria={**CRITERIA,**cfg['pickup_fit_criteria']}
    target=criteria['target_m'];required=criteria['required_lift_m']
    sim=(simulation_factory or candidate_simulation)(cfg,ep,full);overlaps=rack_penetrations(sim)
    result={'valid':False,'pickup':False,'attached':False,'lift_m':0.,'lift_dwell_s':0.,'score':1e6,'initial_overlaps':overlaps,'complete':False,'lift_target_m':target,'lift_tolerance_m':criteria['tolerance_m'],'required_lift_m':required,'evaluation_model':'full_48_sector_tip' if full else 'proposal_16_sector_tip'}
    if overlaps:
        report=sim.report()
        if full:report['asset_fingerprints']=asset_fingerprints_from_xml(sim.xml)
        result['lift_frame']=lift_frame
        return result,np.empty((0,8)),report
    m,d=sim.model,sim.data;rid=m.body('rack').id;lid=m.body('waste').id
    max_lid=0.;lift=0.;rack_lift=0.;world_lift=0.;world_anchor=None;rack_anchor=None;rack_motion=0.;run=0.;held=0.;trace=[];violations=[];a=cfg['attachment'];stride=max(1,round(.005/m.opt.timestep));steps=0;invalid=None
    while sim.time<ep.duration-1e-9:
        sim.step();steps+=1
        if d.ncon:
            b1=m.geom_bodyid[d.contact.geom1];b2=m.geom_bodyid[d.contact.geom2]
            mask=((b1==rid)&(b2==lid))|((b1==lid)&(b2==rid))
            if np.any(mask):max_lid=max(max_lid,float(np.max(-d.contact.dist[mask]))*1000)
        for stats in sim.contact_diagnostics.values():
            if stats['maximum_penetration_mm']>CRITERIA['max_tool_penetration_mm'] or stats['maximum_normal_force_N']>CRITERIA['max_tool_force_N']:invalid='tool contact limit exceeded'
        if max_lid>CRITERIA['max_lid_penetration_mm']:invalid='rack/lid penetration'
        if invalid:break
        tip=d.body('tip');R=tip.xmat.reshape(3,3);mouth=tip.xpos+R[:,2]*cfg['pipette']['tip_length']/2
        if sim.state=='attached':
            rack_z=float(d.site('rack_mouth').xpos[2])
            if world_anchor is None:world_anchor=float(mouth[2]);rack_anchor=rack_z
            rack_height=float(mouth[2]-rack_z);world_height=float(mouth[2]-world_anchor)
            rack_lift=max(rack_lift,rack_height);world_lift=max(world_lift,world_height);rack_motion=max(rack_motion,abs(rack_z-rack_anchor))
            height=world_height if lift_frame=='world' else rack_height;lift=max(lift,height)
            lifted=height>=required and (lift_frame!='world' or rack_height>=.002)
            run=run+m.opt.timestep if lifted else 0.;held=max(held,run)
            if held>=CRITERIA['lift_dwell_s'] and not full:break
        if steps%stride or sim.state!='in_rack':continue
        nozzle=d.site('nozzle');err=R.T@(nozzle.xpos-mouth);radial=float(np.linalg.norm(err[:2]));depth=-err[2]
        angle=float(np.degrees(np.arccos(np.clip(R[:,2]@nozzle.xmat.reshape(3,3)[:,2],-1,1))))
        v=max(radial/a['radial_tolerance'],angle/a['angle_tolerance_deg'],1+max(a['min_insertion_depth']-depth,depth-a['max_insertion_depth'])/(a['max_insertion_depth']-a['min_insertion_depth']))
        trace.append([sim.time,*nozzle.xpos,*mouth,angle]);violations.append(v)
    report=sim.report()
    if full:report['asset_fingerprints']=asset_fingerprints_from_xml(sim.xml)
    valid=not invalid and report['finite'] and not report['warnings'];attached=any(e['state']=='attached' for e in sim.events)
    window=max(1,math.ceil(a['dwell_seconds']/(stride*m.opt.timestep)))
    score=float(np.min(np.max(np.lib.stride_tricks.sliding_window_view(violations,window),axis=1))) if len(violations)>=window else 1e6
    if attached:score=-1-max(0.,1-abs(lift-target)/max(target,.002))
    if not valid:score=1e6
    result.update(valid=valid,pickup=bool(valid and attached and held>=CRITERIA['lift_dwell_s'] and sim.state=='attached' and lift<=target+criteria['tolerance_m']),attached=attached,lift_m=lift,lift_dwell_s=held,score=score,complete=sim.time>=ep.duration-1e-9,invalid_reason=invalid,maximum_lid_penetration_mm=max_lid,events=sim.events,contacts=sim.contact_diagnostics)
    result.update(lift_frame=lift_frame,rack_clearance_m=rack_lift,world_lift_since_attachment_m=world_lift,maximum_rack_vertical_motion_after_attachment_m=rack_motion)
    report['lift_measurement']={k:result[k] for k in ['lift_frame','lift_m','rack_clearance_m','world_lift_since_attachment_m','maximum_rack_vertical_motion_after_attachment_m']}
    return result,np.asarray(trace).reshape(-1,8),report

def seeds_from_trace(trace,base,allow_height=False):
    if len(trace)==0:return []
    depth=trace[:,6]-trace[:,3];desired=.011
    future=np.maximum.accumulate(trace[::-1,3])[::-1]-trace[:,3]
    reference=base.get('pickup_fit_criteria',{})
    target=reference.get('target_m',.025);tolerance=reference.get('tolerance_m',.002)
    lift_error=np.maximum(abs(future-target)-tolerance,0.)
    rank=np.argsort(abs(depth-desired)+.001*np.maximum(trace[:,7]-5,0)+lift_error)
    seeds=[]
    for i in rank:
        delta=np.r_[trace[i,1:3]-trace[i,4:6],0.,0.]
        if all(np.linalg.norm(delta[:2]-s[:2])>.003 for s in seeds):seeds.append(delta)
        if len(seeds)>=4:break
    if allow_height:
        rank=np.argsort(lift_error+.0005*np.maximum(trace[:,7]-5,0))
        extra=[]
        for i in rank:
            dz=desired-depth[i]
            if abs(dz)>.02:continue
            delta=np.r_[trace[i,1:3]-trace[i,4:6],dz,0.]
            if all(np.linalg.norm(delta[:3]-s[:3])>.004 for s in extra):extra.append(delta)
            if len(extra)>=4:break
        seeds=extra+seeds
    return seeds

def _rollout_child(connection,cfg,root,index,full,lift_frame="rack"):
    try:connection.send(('ok',rollout(cfg,Episode(root,index),full,lift_frame=lift_frame)))
    except BaseException as error:connection.send(('error',repr(error)))
    finally:connection.close()

def isolated_rollout(cfg,ep,full=False,lift_frame="rack"):
    # MuJoCo/native allocator memory otherwise accumulates across many model builds.
    # A fresh process per candidate bounds resident memory without changing physics.
    import multiprocessing
    context=multiprocessing.get_context('spawn');receiver,sender=context.Pipe(duplex=False)
    process=context.Process(target=_rollout_child,args=(sender,cfg,str(ep.root),ep.index,full,lift_frame));process.start();sender.close()
    try:
        kind,value=receiver.recv()
    except EOFError as error:
        raise RuntimeError('Candidate worker exited without a result') from error
    finally:
        receiver.close();process.join()
    if kind!='ok':raise RuntimeError(value)
    return value

def fit_episode(job):
    index,root,out,baseline,max_evals,phase,screening_evals=job;ep=Episode(root,index);out=Path(out)/f'episode-{index:03}';out.mkdir(parents=True,exist_ok=True)
    baseline=Path(baseline)/f'episode-{index:03}';base=json.loads((baseline/'workspace.json').read_text())
    original_report=json.loads((baseline/'report.json').read_text());base.pop('pickup_layout',None)
    references=json.loads((out.parent/'lift-targets.json').read_text())
    base['pickup_fit_criteria']=references[str(index)]
    path=out/'search.json'
    state=json.loads(path.read_text()) if path.exists() else {'episode':index,'dataset':str(ep.root),'criteria':CRITERIA,'lift_reference':base['pickup_fit_criteria'],'trials':[],'best':None,'phase':phase,'verified_pickup':False,'baseline_scene_source':original_report['validation']['scene_source']}
    if state['criteria']!=CRITERIA or state['dataset']!=str(ep.root):raise ValueError('Search provenance mismatch')
    fingerprint=model_fingerprints()
    if state.get('model_fingerprints',fingerprint)!=fingerprint:raise ValueError('Model changed since search began')
    state['model_fingerprints']=fingerprint
    baseline_hash=sha256(baseline/'workspace.json')
    if state.get('baseline_sha256',baseline_hash)!=baseline_hash:raise ValueError('Baseline placement changed')
    state['baseline_sha256']=baseline_hash
    if state['verified_pickup']:return state
    # Older interrupted trials can contain a provisional pickup whose full replay failed.
    for trial in state['trials']:
        verification=trial.get('verification')
        if verification is not None and not verification['pickup']:
            trial['provisional_score']=trial['score'];trial.update(verification)
    state['best']=min(state['trials'],key=lambda r:r['score']) if state['trials'] else None
    write_json(out/'baseline.json',base);state['phase']=phase
    archive=np.load(baseline/'trace.npz');nozzle=archive['nozzle'];mouth=np.broadcast_to(rack_target(base),nozzle.shape)
    trace=np.c_[archive['time'],nozzle,mouth,np.zeros(len(nozzle))]
    # Baseline archive has positions but not angles. These are proposal seeds only.
    pending=seeds_from_trace(trace,base,phase=='height');pending.insert(0,np.zeros(4))
    if state.get('seed_deltas'):pending[0:0]=[np.array(d) for d in state['seed_deltas'][:2]]
    if index==46:
        prior=Path('outputs/hil194/episode46-layout-search/attached-12mm.json')
        if prior.exists():
            old=json.loads(prior.read_text());pending.insert(0,np.r_[np.array(old['rack']['center_xy'])-base['rack']['center_xy'],0.,0.])
    seen={tuple(np.round(t['delta'],6)) for t in state['trials']};used_before=len(state['trials']);best=state['best'];start=time.monotonic()
    refinements=[.004,.002,.001,.0005];refine_stage=0;origin_key=None;feedback_count=0
    while len(state['trials'])-used_before<max_evals:
        if screening_evals and len(state['trials'])>=screening_evals and best is not None and best['score']>3:
            state['stopping_reason']='Screening budget reached without a near-feasible insertion; further placement refinement deferred'
            break
        if not pending:
            if best is None:break
            origin=np.array(best['delta']);key=tuple(origin)
            if key!=origin_key:refine_stage=0;origin_key=key
            if refine_stage>=len(refinements):break
            step=refinements[refine_stage];refine_stage+=1
            for axis in range(3 if phase=='height' else 2):
                for sign in [1,-1]:
                    delta=origin.copy();delta[axis]+=sign*step;pending.append(delta)
            if phase=='height' and refine_stage==1:
                for yaw in [-5,5]:
                    delta=origin.copy();delta[3]+=yaw;pending.append(delta)
        delta=np.asarray(pending.pop(0),float);key=tuple(np.round(delta,6))
        if key in seen or np.linalg.norm(delta[:2])>.12 or abs(delta[2])>.02 or abs(delta[3])>10:continue
        seen.add(key);cfg=candidate_config(base,delta)
        if not on_table(cfg):continue
        try:result,newtrace,report=isolated_rollout(cfg,ep)
        except Exception as error:
            result={'valid':False,'pickup':False,'attached':False,'lift_m':0.,'score':1e6,'error':repr(error)};newtrace=np.empty((0,8));report={}
        row={**result,'delta':delta.tolist(),'trial':len(state['trials']),'phase':phase,'proposal_reference':'episode_lift_v2'};state['trials'].append(row)
        if best is None or row['score']<best['score']:
            best=row;state['best']=row;write_json(out/'best-config.json',cfg);write_json(out/'best-report.json',report)
            if len(newtrace):np.save(out/'best-trace.npy',newtrace)
            if feedback_count<3 and not row.get('attached') and row.get('valid'):
                # Recompute the needed correction after the moved scene changes contacts.
                feedback=seeds_from_trace(newtrace,cfg,phase=='height')[:2]
                pending[0:0]=[delta+s for s in feedback];feedback_count+=1
        if row.get('pickup'):
            verified,_,validation=isolated_rollout(cfg,ep,full=True);row['verification']=verified
            write_json(out/f"verification-{row['trial']:03}.json",validation)
            if not verified['pickup']:
                row['provisional_score']=row['score'];row.update(verified)
                best=min(state['trials'],key=lambda r:r['score']);state['best']=best
                write_json(out/'best-config.json',candidate_config(base,best['delta']))
            if verified['pickup'] and verified['complete']:
                state['verified_pickup']=True;state['best']=row;best=row
                write_json(out/'best-config.json',cfg);write_json(out/'best-report.json',validation)
        state['elapsed_last_phase_s']=time.monotonic()-start;write_json(path,state)
        print(f"ep {index:03} trial {row['trial']:02} score={row['score']:.3f} attached={row.get('attached')} lift={row.get('lift_m',0)*1000:.1f}mm verified={state['verified_pickup']}",flush=True)
        if state['verified_pickup']:break
    state['finished_phase']=phase;write_json(path,state);return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--output',type=Path,default=Path('outputs/hil194/pickup-layouts'));p.add_argument('--baseline',type=Path,default=Path('outputs/hil194/final-replays'));p.add_argument('--episodes',nargs='+',default=['all']);p.add_argument('--workers',type=int,default=12);p.add_argument('--max-evaluations',type=int,default=24);p.add_argument('--phase',choices=['xy','height'],default='xy');p.add_argument('--screening-evaluations',type=int,default=0,help='Optional early screening budget: defer refinements when best insertion score remains above 3; zero disables');p.add_argument('--summary-name',help='Separate progress filename for a disjoint episode retry; never run concurrent searches for the same episode');p.add_argument('--total-budget',action='store_true',help='Count already completed trials in this phase against the budget');p.add_argument('--prioritize',action='store_true',help='Try short episodes whose baseline lift is already close to the reference first');a=p.parse_args()
    ep=Episode(a.dataset,0);indices=list(range(ep.info['total_episodes'])) if a.episodes==['all'] else list(map(int,a.episodes));a.output.mkdir(parents=True,exist_ok=True)
    if a.prioritize:
        refs=json.loads((a.output/'lift-targets.json').read_text())
        def priority(i):
            data=np.load(a.baseline/f'episode-{i:03}'/'trace.npz');z=data['nozzle'][:,2];rise=float(np.max(np.maximum.accumulate(z[::-1])[::-1]-z));duration=float(data['time'][-1])
            return abs(math.log((rise+.001)/(refs[str(i)]['target_m']+.001)))+duration/100
        indices.sort(key=priority)
    jobs=[]
    for i in indices:
        budget=a.max_evaluations;existing=a.output/f'episode-{i:03}'/'search.json'
        if a.total_budget and existing.exists():budget=max(0,budget-sum(r['phase']==a.phase for r in json.loads(existing.read_text())['trials']))
        jobs.append((i,str(ep.root),str(a.output),str(a.baseline),budget,a.phase,a.screening_evaluations))
    errors=[];finished=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fit_episode,j):j[0] for j in jobs}
        for f in as_completed(futures):
            i=futures[f]
            try:s=f.result();finished.append({'episode':i,'verified_pickup':s['verified_pickup'],'trials':len(s['trials']),'best':s['best']})
            except Exception as error:errors.append({'episode':i,'error':repr(error)})
            write_json(a.output/f'summary-{a.summary_name or a.phase}.json',{'completed':finished,'errors':errors,'requested':indices})
            print(f"FINISHED {len(finished)+len(errors)}/{len(indices)} verified={sum(r['verified_pickup'] for r in finished)} errors={len(errors)}",flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
