"""Independently validate best episode layouts and atomically install default lookup."""
import argparse,json,sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fit_pickup_layouts import rollout,candidate_config,CRITERIA,write_json
from g1_pipette_sim.dataset import Episode,DEFAULT_DATASET
from g1_pipette_sim.scene import ROOT
from g1_pipette_sim.pickup_layouts import model_fingerprints,sha256,MANIFEST

def finalize(job):
    index,folder,root=job;p=Path(folder)/f'episode-{index:03}';state=json.loads((p/'search.json').read_text());fingerprint=model_fingerprints()
    if state.get('model_fingerprints',fingerprint)!=fingerprint:raise ValueError('Search model fingerprint changed')
    base=json.loads((p/'baseline.json').read_text());ep=Episode(root,index)
    candidates=[r for r in state['trials'] if r.get('valid') and (not r.get('verification') or r['verification']['valid'])]
    if not candidates:return {'episode':index,'status':'no_valid_layout','config':None,'trials':len(state['trials']),'reason':'Every evaluated layout exceeded collision/force limits or failed initialization'}
    verified=[r for r in candidates if r.get('verification',{}).get('pickup') and r['verification'].get('complete')]
    row=min(verified or candidates,key=lambda r:r['score']);cfg=candidate_config(base,row['delta'])
    full_candidates=[]
    for trial in candidates:
        full=trial.get('verification') or trial
        if full.get('evaluation_model')=='full_48_sector_tip' and full.get('valid') and full.get('complete'):
            full_candidates.append((trial,full))
    known_best=min(full_candidates,key=lambda pair:pair[1]['score']) if full_candidates else None
    checkpoint=p/'validated.json';cfg_hash=__import__('hashlib').sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if (saved['config_content_sha256']==cfg_hash and saved['model_fingerprints']==fingerprint
                and (known_best is None or saved['entry'].get('validation',{}).get('score',1e6)<=known_best[1]['score'])):
            entry=saved['entry']
            if not entry.get('config'):return entry
            saved_path=ROOT/entry['config'];assets=entry.get('asset_fingerprints')
            if (saved_path.exists() and sha256(saved_path)==entry['sha256'] and assets is not None
                    and all((ROOT/name).exists() and sha256(ROOT/name)==digest for name,digest in assets.items())):
                return entry

    result,trace,report=rollout(cfg,ep,full=True)
    if known_best is not None and known_best[1]['score']<result['score']:
        row=known_best[0];cfg=candidate_config(base,row['delta'])
        result,trace,report=rollout(cfg,ep,full=True)
    if not result['complete'] or not result['valid']:
        write_json(p/'default-rejected-report.json',report)
        entry={'episode':index,'status':'no_valid_layout','config':None,'trials':len(state['trials']),'reason':'Best candidate failed independent full collision-screened validation','validation':result}
        write_json(checkpoint,{'config_content_sha256':cfg_hash,'model_fingerprints':fingerprint,'entry':entry})
        return entry
    status='verified_pickup' if result['pickup'] else 'attachment_only' if result['attached'] else 'no_pickup'
    if result['pickup'] and not state.get('verified_pickup') and json.loads((p/'search.json').read_text())==state:
        row['verification']=result;row.update(result);state['best']=row;state['verified_pickup']=True
        write_json(p/f"verification-{row['trial']:03}.json",report)
        write_json(p/'search.json',state)
    cfg['layout_search']={'method':'bounded_dynamic_replay_placement_search','episode':index,'dataset':str(ep.root),'source':'low_level','clock':'recorded','result':{**result,'success':result['pickup']},'criteria':CRITERIA,'delta_xy_z_yaw':row['delta'],'baseline_scene_source':state['baseline_scene_source'],'trials':len(state['trials']),'meaning':'Trajectory-conditioned placement, not recovered real-world ground truth; commands and physics parameters unchanged.'}
    config_path=p/'validated-default.json';write_json(config_path,cfg);write_json(p/'default-report.json',report);np.save(p/'default-approach-trace.npy',trace)
    entry={'episode':index,'status':status,'config':str(config_path.resolve().relative_to(ROOT)),'sha256':sha256(config_path),'trials':len(state['trials']),'validation':result,'asset_fingerprints':report['asset_fingerprints']}
    write_json(checkpoint,{'config_content_sha256':cfg_hash,'model_fingerprints':fingerprint,'entry':entry})
    return entry

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--input',type=Path,default=Path('outputs/hil194/pickup-layouts'));p.add_argument('--workers',type=int,default=12);p.add_argument('--install',action='store_true');a=p.parse_args();ep=Episode(a.dataset,0)
    entries={};errors=[]
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(finalize,(i,str(a.input),str(ep.root))):i for i in range(ep.info['total_episodes'])}
        for f in as_completed(futures):
            try:r=f.result();entries[str(r['episode'])]=r;print(f"{len(entries)}/194 ep {r['episode']}: {r['status']}",flush=True)
            except Exception as error:errors.append({'episode':futures[f],'error':repr(error)});print('ERROR',errors[-1],flush=True)
            write_json(a.input/'publication-progress.json',{'episodes':entries,'errors':errors})
    if errors or len(entries)!=ep.info['total_episodes']:raise SystemExit('Incomplete validation; default manifest was not changed')
    manifest={'complete':True,'dataset_snapshot':str(ep.root),'model_fingerprints':model_fingerprints(),'source':'low_level','clock':'recorded','physics':'dynamic','criteria':CRITERIA,'episodes':entries}
    write_json(a.input/'manifest.json',manifest)
    if a.install:write_json(MANIFEST,manifest)
    print({status:sum(e['status']==status for e in entries.values()) for status in ['verified_pickup','attachment_only','no_pickup','no_valid_layout']})
if __name__=='__main__':main()
