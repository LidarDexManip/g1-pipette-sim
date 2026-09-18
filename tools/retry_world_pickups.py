"""Retry failed HIL194 pickups using full-model world lift and rack-well placement.

Robot commands, grasp, meshes, insertion gates and physics stay unchanged.
A contact-free proposal trace is only a heuristic; every trial uses full physics.
"""
import argparse,copy,fcntl,hashlib,json,multiprocessing,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.pickup_layouts import ROOT,MANIFEST,model_fingerprints,sha256
from g1_pipette_sim.scene import rack_target
from tools.fit_pickup_layouts import isolated_rollout,write_json
from tools.publish_world_pickup_layouts import config_for_row,WORLD_CRITERIA
from tools.search_replay_layout import on_table

FOLDER=ROOT/'outputs/hil194/pickup-retry-world'

def placement_key(row):return tuple(np.round(row['delta'],7))+tuple(row['active_cell'])

def allowed(base,row):
    d=row['delta'];r,c=row['active_cell']
    return (np.linalg.norm(d[:2])<=.12 and abs(d[2])<=.02 and abs(d[3])<=10 and
            0<=r<base['rack']['rows'] and 0<=c<base['rack']['cols'] and on_table(config_for_row(base,row)))

def move_well(base,row,cell):
    old=config_for_row(base,row);new=copy.deepcopy(row);new['active_cell']=list(cell);shift=rack_target(old)-rack_target(config_for_row(base,new));new['delta']=(np.asarray(row['delta'])+np.r_[shift[:2],0.,0.]).tolist();return new

def window_proposals(trace,base,origin,count=8):
    if len(trace)<20:return []
    a=base['attachment'];ref=base['pickup_fit_criteria'];dt=float(np.median(np.diff(trace[:,0])));window=max(2,int(np.ceil(a['dwell_seconds']/dt))+1);future=np.maximum.accumulate(trace[::-1,3])[::-1];candidates=[]
    for i in range(0,len(trace)-window,5):
        end=i+window-1
        if trace[i,0]<.5:continue
        segment=trace[i:i+window];xy=segment[:,1:3].mean(axis=0);z=(segment[:,3].min()+segment[:,3].max())/2;mouth=segment[:,4:7].mean(axis=0);rise=future[end]-segment[-1,3]
        correction=np.r_[xy-mouth[:2],z+(a['min_insertion_depth']+a['max_insertion_depth'])/2-mouth[2],0.]
        row={'delta':(np.asarray(origin['delta'])+correction).tolist(),'active_cell':origin['active_cell'],'proposal_time_s':float(segment[-1,0]),'proposal':'dwell_window'}
        if not allowed(base,row):continue
        radius=float(np.linalg.norm(segment[:,1:3]-xy,axis=1).max());span=float(np.ptp(segment[:,3]));angle=float(segment[:,7].max())
        score=2*max(abs(rise-ref['target_m'])-ref['tolerance_m'],0)+5*max(radius-a['radial_tolerance'],0)+2*max(span-(a['max_insertion_depth']-a['min_insertion_depth']),0)+.002*max(angle-a['angle_tolerance_deg'],0)+.0001*abs(segment[-1,0]-ref['reference_low_time_s'])
        candidates.append((score,row))
    selected=[]
    for _,row in sorted(candidates,key=lambda item:item[0]):
        if all(np.linalg.norm(np.array(row['delta'])[:3]-np.array(other['delta'])[:3])>.001 for other in selected):selected.append(row)
        if len(selected)>=count:break
    return selected

def _free_worker(connection,cfg,root,index,folder):
    try:
        from tools.free_motion_proposals import free_trace
        from g1_pipette_sim.dynamics import DynamicSimulation
        trace=free_trace(cfg,Episode(root,index),folder,lambda c,e:DynamicSimulation(c,e,source='low_level'))
        connection.send(('ok',trace))
    except BaseException as error:connection.send(('error',repr(error)))
    finally:connection.close()

def proposal_trace(cfg,ep,folder):
    signature=hashlib.sha256(json.dumps({'cfg':cfg,'root':str(ep.root),'episode':ep.index,'model':model_fingerprints()},sort_keys=True).encode()).hexdigest();stamp=folder/'free-provenance.json';trace_path=folder/'free-motion-proposal.npy'
    if stamp.exists() and json.loads(stamp.read_text()).get('signature')==signature and trace_path.exists():return np.load(trace_path)
    if trace_path.exists():raise ValueError('Proposal trace provenance changed; use a new retry output directory')
    context=multiprocessing.get_context('spawn');reader,writer=context.Pipe(duplex=False);process=context.Process(target=_free_worker,args=(writer,cfg,str(ep.root),ep.index,str(folder)));process.start();writer.close()
    try:kind,value=reader.recv()
    finally:reader.close();process.join()
    if kind!='ok':raise RuntimeError(value)
    if not np.isfinite(value).all():raise ValueError('Nonfinite proposal trajectory')
    write_json(stamp,{'signature':signature,'scope':'Proposal only; object contacts disabled. All candidate and validation replays restore full contact physics.'});return value

def archive_row(index,row):
    folder=ROOT/f'outputs/hil194/pickup-layouts/episode-{index:03}';path=folder/'world-search.json'
    with (folder/'.world-validation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);world=json.loads(path.read_text()) if path.exists() else {'episode':index,'lift_frame':'world','model_fingerprints':model_fingerprints(),'trials':[]}
        match=next((j for j,r in enumerate(world['trials']) if r.get('retry_run')==row['retry_run'] and r.get('retry_trial')==row['retry_trial']),None)
        if match is None:world['trials'].append({**row,'trial':len(world['trials'])})
        else:world['trials'][match]={**row,'trial':world['trials'][match]['trial']}
        write_json(path,world)

def make_entry(base,row,report,ep,folder,previous_trials):
    cfg=config_for_row(base,row);v={k:v for k,v in row.items() if k not in {'delta','active_cell','proposal','proposal_time_s','retry_run','retry_trial','trial'}}
    cfg['layout_search']={'method':'world_lift_and_rack_well_retry','episode':ep.index,'dataset':str(ep.root),'source':'low_level','clock':'recorded','result':{**v,'success':v['pickup']},'criteria':WORLD_CRITERIA,'delta_xy_z_yaw':row['delta'],'active_cell':row['active_cell'],'trials':previous_trials+row['retry_trial']+1,'meaning':'Trajectory-conditioned rack/tip placement with unchanged robot actions and physics; not measured real object placement.'}
    path=folder/f"validated-{row['retry_trial']:03}.json";rp=folder/f"validated-{row['retry_trial']:03}-report.json";write_json(path,cfg);write_json(rp,report)
    return {'episode':ep.index,'status':'verified_pickup' if v['pickup'] else 'attachment_only' if v['attached'] else 'no_pickup','config':str(path.relative_to(ROOT)),'report':str(rp.relative_to(ROOT)),'sha256':sha256(path),'trials':cfg['layout_search']['trials'],'validation':v,'asset_fingerprints':report['asset_fingerprints']}

def retry(job):
    index,budget,output,initial=job;folder=Path(output)/f'episode-{index:03}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return retry_locked(index,budget,folder,initial)

def retry_locked(index,budget,folder,initial):
    p=ROOT/f'outputs/hil194/pickup-layouts/episode-{index:03}';base=json.loads((p/'baseline.json').read_text());ep=Episode(index=index);fp=model_fingerprints();path=folder/'search.json'
    if path.exists():state=json.loads(path.read_text())
    else:
        if initial.get('config'):
            cfg=json.loads((ROOT/initial['config']).read_text());origin={'delta':cfg['layout_search']['delta_xy_z_yaw'],'active_cell':cfg['rack']['active_cell']}
        else:
            old=json.loads((p/'search.json').read_text());origin={'delta':old['best']['delta'],'active_cell':base['rack']['active_cell']}
        state={'episode':index,'model_fingerprints':fp,'origin':origin,'initial_entry':initial,'trials':[],'verified':False}
        write_json(path,state)
    if state['model_fingerprints']!=fp:raise ValueError('Model changed during retry')
    if state.get('verified') or len(state['trials'])>=budget:return state
    origin=state['origin'];trace=proposal_trace(config_for_row(base,origin),ep,folder);seeds=window_proposals(trace,base,origin)
    pending=[]
    for row in seeds[:4]:pending.append(row)
    corners=[(0,0),(0,base['rack']['cols']-1),(base['rack']['rows']-1,0),(base['rack']['rows']-1,base['rack']['cols']-1)]
    for cell in corners:pending.append({**move_well(base,seeds[0] if seeds else origin,cell),'proposal':'different_well'})
    pending.extend(seeds[4:])
    best=state.get('best');best_score=best['score'] if best else state['initial_entry'].get('validation',{}).get('score',1e6);center=best or origin
    for scale in [.00025,.0005,.001,.002]:
        for axis in [2,0,1]:
            for sign in [1,-1]:
                row={'delta':list(center['delta']),'active_cell':center['active_cell'],'proposal':'local_refinement'};row['delta'][axis]+=sign*scale;pending.append(row)
    seen={placement_key(r) for r in state['trials']};seen.add(placement_key(origin));feedback=0
    while pending and len(state['trials'])<budget:
        candidate=pending.pop(0)
        if placement_key(candidate) in seen or not allowed(base,candidate):continue
        seen.add(placement_key(candidate));cfg=config_for_row(base,candidate)
        result,newtrace,report=isolated_rollout(cfg,ep,full=True,lift_frame='world');row={**candidate,**result,'retry_run':str(folder.parent.relative_to(ROOT)),'retry_trial':len(state['trials'])}
        if result['pickup'] and result['complete']:
            verification,_,report=isolated_rollout(cfg,ep,full=True,lift_frame='world');row.update(verification);row['independently_replayed']=True
        state['trials'].append(row);write_json(folder/f"trial-{row['retry_trial']:03}-report.json",report);archive_row(index,row)
        if row['valid'] and row['complete'] and (row['pickup'] or row['score']<best_score):
            best_score=row['score'];state['best']=row;state['entry']=make_entry(base,row,report,ep,folder,state['initial_entry'].get('trials',0))
            if not row['attached'] and feedback<2:
                pending[0:0]=window_proposals(newtrace,base,candidate,2);feedback+=1
        state['verified']=bool(row.get('pickup') and row.get('independently_replayed'));write_json(path,state)
        print(json.dumps({'episode':index,'trial':row['retry_trial'],'attached':row['attached'],'world_lift_mm':1000*row['lift_m'],'target_mm':1000*row['lift_target_m'],'valid':row['valid'],'verified':state['verified'],'active_cell':row['active_cell']}),flush=True)
        if state['verified']:break
    state['finished']=True;state['budget']=budget;write_json(path,state);return state

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--episodes',nargs='+',type=int);parser.add_argument('--budget',type=int,default=12);parser.add_argument('--workers',type=int,default=10);parser.add_argument('--output',type=Path,default=FOLDER);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    snapshot=a.output/'starting-manifest.json'
    if not snapshot.exists():write_json(snapshot,json.loads(MANIFEST.read_text()))
    initial=json.loads(snapshot.read_text())
    if initial['model_fingerprints']!=model_fingerprints():raise ValueError('Initial model changed')
    indices=a.episodes if a.episodes is not None else [int(k) for k,e in initial['episodes'].items() if e['status']!='verified_pickup']
    indices.sort(key=lambda i:(i!=46,initial['episodes'][str(i)]['status']!='attachment_only',abs(initial['episodes'][str(i)].get('validation',{}).get('lift_m',0)-initial['episodes'][str(i)].get('validation',{}).get('lift_target_m',0)),i))
    progress={'requested':indices,'completed':[],'errors':[],'newly_verified':[]}
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(retry,(i,a.budget,str(a.output.resolve()),initial['episodes'][str(i)])):i for i in indices}
        for f in as_completed(futures):
            i=futures[f]
            try:
                state=f.result();entry=state.get('entry');progress['completed'].append(i)
                if entry:
                    with (MANIFEST.parent/'.pickup-manifest.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX);manifest=json.loads(MANIFEST.read_text())
                        if manifest['model_fingerprints']!=model_fingerprints():raise ValueError('Published model changed')
                        old=manifest['episodes'][str(i)]
                        if old['status']!='verified_pickup' and (entry['status']=='verified_pickup' or entry['validation']['score']<old.get('validation',{}).get('score',1e6)):
                            manifest['episodes'][str(i)]=entry;write_json(MANIFEST,manifest)
                if state['verified']:
                    progress['newly_verified'].append(i);v=entry['validation'];print(json.dumps({'READY':i,'world_lift_mm':v['lift_m']*1000,'recorded_target_mm':v['lift_target_m']*1000,'attachment_s':next(e['time'] for e in v['events'] if e['state']=='attached'),'command':f'./run.sh --episode {i}'}),flush=True)
            except Exception as error:progress['errors'].append({'episode':i,'error':repr(error)});print('ERROR',i,repr(error),flush=True)
            write_json(a.output/'progress.json',progress);print(f"COMPLETED {len(progress['completed'])}/{len(indices)} new_matches={len(progress['newly_verified'])} errors={len(progress['errors'])}",flush=True)
    if progress['errors']:raise SystemExit(1)
if __name__=='__main__':main()
