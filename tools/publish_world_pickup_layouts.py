"""Independently validate world-frame lift and publish episode-specific replay configs."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fit_pickup_layouts import rollout,candidate_config,CRITERIA,write_json
from g1_pipette_sim.dataset import Episode,DEFAULT_DATASET
from g1_pipette_sim.pickup_layouts import ROOT,MANIFEST,model_fingerprints,sha256

WORLD_CRITERIA={**CRITERIA,'lift_frame':'world_tip_rise_since_attachment','minimum_rack_clearance_m':.002}

def config_for_row(base,row):
    cfg=candidate_config(base,row["delta"])
    if "active_cell" in row:cfg["rack"]["active_cell"]=list(row["active_cell"])
    return cfg

def proposals(state,world):
    valid=[r for r in state['trials'] if r.get('valid') and (not r.get('verification') or r['verification'].get('valid'))]
    extra=[r for r in world.get('trials',[]) if r.get('valid') and r.get('complete')]
    chosen=[]
    if extra:chosen.append(min(extra,key=lambda r:(not r.get('pickup',False),r['score'])))
    full=[r for r in valid if (r.get('verification') or r).get('evaluation_model')=='full_48_sector_tip' and (r.get('verification') or r).get('complete')]
    if full:chosen.append(min(full,key=lambda r:(r.get('verification') or r)['score']))
    if valid:chosen.append(min(valid,key=lambda r:r['score']))
    unique=[]
    for row in chosen:
        if not any((np.allclose(row['delta'],r['delta'],rtol=0,atol=1e-9) and row.get('active_cell')==r.get('active_cell')) for r in unique):unique.append(row)
    return unique

def reuse_unattached_checkpoint(folder,cfg,fingerprint):
    checkpoint=folder/'validated.json'
    if not checkpoint.exists():return None
    saved=json.loads(checkpoint.read_text());entry=saved['entry'];v=entry.get('validation',{})
    if saved.get('model_fingerprints')!=fingerprint or entry.get('status')!='no_pickup' or not v.get('valid') or not v.get('complete') or v.get('attached') or v.get('events'):return None
    path=ROOT/entry['config'];assets=entry.get('asset_fingerprints')
    if not assets or not path.exists() or sha256(path)!=entry['sha256']:return None
    old=json.loads(path.read_text());old.pop('layout_search',None);candidate=dict(cfg);candidate.pop('layout_search',None)
    if old!=candidate or any(not (ROOT/name).exists() or sha256(ROOT/name)!=digest for name,digest in assets.items()):return None
    report_path=folder/'default-report.json'
    if not report_path.exists():return None
    report=json.loads(report_path.read_text())
    if report.get('events') or not report.get('finite') or report.get('warnings') or report.get('asset_fingerprints')!=assets:return None
    result={**v,'lift_frame':'world','lift_m':0.,'lift_dwell_s':0.,'rack_clearance_m':0.,'world_lift_since_attachment_m':0.,'maximum_rack_vertical_motion_after_attachment_m':0.,'validation_reuse':'Unchanged full replay with no attachment; attached-tip world lift is identically zero'}
    report['lift_measurement']={k:result[k] for k in ['lift_frame','lift_m','rack_clearance_m','world_lift_since_attachment_m','maximum_rack_vertical_motion_after_attachment_m']}
    trace_path=folder/'default-approach-trace.npy';trace=np.load(trace_path) if trace_path.exists() else np.empty((0,8))
    return result,trace,report

def finalize_world(job):
    i,folder,_=job
    with (Path(folder)/f"episode-{i:03}"/".world-validation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _finalize_world_locked(job)

def _finalize_world_locked(job):
    i,folder,root=job;p=Path(folder)/f'episode-{i:03}';state=json.loads((p/'search.json').read_text());base=json.loads((p/'baseline.json').read_text());fp=model_fingerprints()
    if state.get('model_fingerprints')!=fp:raise ValueError('Search model changed')
    world_path=p/'world-search.json';world=json.loads(world_path.read_text()) if world_path.exists() else {}
    if world and world.get('model_fingerprints')!=fp:raise ValueError('World search model changed')
    rows=proposals(state,world);signature=hashlib.sha256(json.dumps({'placements':[{'delta':r['delta'],'active_cell':r.get('active_cell')} for r in rows],'base':base,'criteria':WORLD_CRITERIA},sort_keys=True).encode()).hexdigest();checkpoint=p/'validated-world.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text());e=saved['entry']
        if saved['input_signature']==signature and saved['model_fingerprints']==fp:
            assets=e.get('asset_fingerprints')
            assets_valid=assets is not None and all((ROOT/name).exists() and sha256(ROOT/name)==digest for name,digest in assets.items())
            if not e.get('config'):
                if assets_valid:return e
            else:
                path=ROOT/e['config']
                if path.exists() and sha256(path)==e['sha256'] and assets_valid:return e
    ep=Episode(root,i);evaluated=[]
    for row in rows:
        cfg=config_for_row(base,row);reused=reuse_unattached_checkpoint(p,cfg,fp)
        result,trace,report=reused if reused is not None else rollout(cfg,ep,full=True,lift_frame='world');evaluated.append((row,cfg,result,trace,report))
        if result['valid'] and result['complete'] and result['pickup']:break
    valid=[r for r in evaluated if r[2]['valid'] and r[2]['complete']]
    if valid:
        row,cfg,v,trace,report=min(valid,key=lambda r:r[2]['score']);status='verified_pickup' if v['pickup'] else 'attachment_only' if v['attached'] else 'no_pickup'
        cfg['layout_search']={'method':'bounded_dynamic_replay_placement_search','episode':i,'dataset':str(ep.root),'source':'low_level','clock':'recorded','result':{**v,'success':v['pickup']},'criteria':WORLD_CRITERIA,'delta_xy_z_yaw':row['delta'],'baseline_scene_source':state['baseline_scene_source'],'trials':len(state['trials'])+len(world.get('trials',[])),'meaning':'Trajectory-conditioned placement; world-frame tip rise from first attached sample; commands and physics unchanged.'}
        config=p/'validated-world-default.json';report_path=p/'world-default-report.json';write_json(config,cfg);write_json(report_path,report);np.save(p/'world-default-approach-trace.npy',trace)
        entry={'episode':i,'status':status,'config':str(config.relative_to(ROOT)),'report':str(report_path.relative_to(ROOT)),'sha256':sha256(config),'trials':cfg['layout_search']['trials'],'validation':v,'asset_fingerprints':report['asset_fingerprints']}
    else:
        entry={'episode':i,'status':'no_valid_layout','config':None,'trials':len(state['trials'])+len(world.get('trials',[])),'reason':'No candidate passed independent full collision-screened validation'}
        if evaluated:
            row,cfg,v,trace,report=evaluated[0];report_path=p/'world-rejected-report.json';write_json(report_path,report);entry.update(validation=v,report=str(report_path.relative_to(ROOT)),asset_fingerprints=report.get('asset_fingerprints'))
    write_json(checkpoint,{'input_signature':signature,'model_fingerprints':fp,'entry':entry});return entry

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);parser.add_argument('--input',type=Path,default=ROOT/'outputs/hil194/pickup-layouts');parser.add_argument('--episodes',nargs='+',type=int);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--install',action='store_true');a=parser.parse_args();ep=Episode(a.dataset,0);indices=a.episodes if a.episodes is not None else list(range(ep.info['total_episodes']));entries={};errors=[]
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(finalize_world,(i,str(a.input.resolve()),str(ep.root))):i for i in indices}
        for f in as_completed(futures):
            try:
                e=f.result();entries[str(e['episode'])]=e;v=e.get('validation',{});print(json.dumps({'completed':len(entries),'requested':len(indices),'episode':e['episode'],'status':e['status'],'world_lift_mm':1000*v.get('lift_m',0),'recorded_target_mm':1000*v.get('lift_target_m',0)}),flush=True)
            except Exception as error:errors.append({'episode':futures[f],'error':repr(error)});print('ERROR',errors[-1],flush=True)
            write_json(a.input/'world-publication-progress.json',{'episodes':entries,'errors':errors})
    if errors:raise SystemExit('Incomplete world validation; manifest unchanged')
    if a.episodes is not None and MANIFEST.exists():
        old=json.loads(MANIFEST.read_text())
        if old.get('dataset_snapshot')==str(ep.root) and old.get('model_fingerprints')==model_fingerprints():entries={**{i:e for i,e in old['episodes'].items() if e.get('validation',{}).get('lift_frame')=='world'},**entries}
    manifest={'complete':len(entries)==ep.info['total_episodes'],'dataset_snapshot':str(ep.root),'model_fingerprints':model_fingerprints(),'source':'low_level','clock':'recorded','physics':'dynamic','criteria':WORLD_CRITERIA,'episodes':entries};write_json(a.input/'world-manifest.json',manifest)
    if a.install:write_json(MANIFEST,manifest)
    print({s:sum(e['status']==s for e in entries.values()) for s in ['verified_pickup','attachment_only','no_pickup','no_valid_layout']},flush=True)
if __name__=='__main__':main()
