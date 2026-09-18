"""Publish independently verified pickups incrementally while the batch runs."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fit_pickup_layouts import write_json,candidate_config,CRITERIA
from g1_pipette_sim.pickup_layouts import MANIFEST,ROOT,model_fingerprints,sha256
from g1_pipette_sim.dataset import Episode

def publish(folder):
    folder=Path(folder);fingerprints=model_fingerprints();root=str(Episode(index=0).root)
    manifest=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'complete':False,'dataset_snapshot':root,'model_fingerprints':fingerprints,'source':'low_level','clock':'recorded','physics':'dynamic','criteria':CRITERIA,'episodes':{}}
    if manifest.get('criteria',{}).get('lift_frame')=='world_tip_rise_since_attachment':raise ValueError('Use tools/watch_world_pickup_layouts.py for the current world-lift manifest')
    if manifest.get('complete'):return
    if manifest['model_fingerprints']!=fingerprints:raise ValueError('Model changed during incremental publication')
    for path in sorted(folder.glob('episode-*/search.json')):
        state=json.loads(path.read_text());index=state['episode']
        existing=manifest['episodes'].get(str(index),{})
        if existing.get('status')=='verified_pickup':
            checkpoint=path.parent/'validated.json'
            if not existing.get('asset_fingerprints') and checkpoint.exists():
                saved=json.loads(checkpoint.read_text());entry=saved['entry']
                if saved['model_fingerprints']==fingerprints and entry.get('status')=='verified_pickup' and entry.get('asset_fingerprints'):
                    manifest['episodes'][str(index)]=entry;write_json(MANIFEST,manifest)
                    print(json.dumps({'refreshed_mesh_checks':index}),flush=True)
            continue
        if not state['verified_pickup']:continue
        if state.get('model_fingerprints',fingerprints)!=fingerprints:raise ValueError('Search model mismatch')
        row=state['best'];result=row.get('verification')
        if not result or not result['pickup'] or not result['valid'] or not result['complete']:continue
        p=path.parent;base=json.loads((p/'baseline.json').read_text());cfg=candidate_config(base,row['delta'])
        cfg['layout_search']={'method':'bounded_dynamic_replay_placement_search','episode':index,'dataset':root,'source':'low_level','clock':'recorded','result':{**result,'success':True},'criteria':CRITERIA,'delta_xy_z_yaw':row['delta'],'trials':len(state['trials']),'meaning':'Trajectory-conditioned placement; unchanged commands, physics and attachment gates; independently verified full replay.'}
        report_path=p/f"verification-{row['trial']:03}.json"
        if not report_path.exists():report_path=p/'exact-refinement-validation.json'
        validation_report=json.loads(report_path.read_text()) if report_path.exists() else {}
        out=p/'default.json';write_json(out,cfg)
        entry={'episode':index,'status':'verified_pickup','config':str(out.relative_to(ROOT)),'sha256':sha256(out),'validation':result,'trials':len(state['trials'])}
        if 'asset_fingerprints' in validation_report:entry['asset_fingerprints']=validation_report['asset_fingerprints']
        manifest['episodes'][str(index)]=entry;write_json(MANIFEST,manifest)
        print(json.dumps({'ready':index,'attachment_s':result['events'][0]['time'],'lift_mm':result['lift_m']*1000,'command':f'./run.sh --episode {index}'}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'outputs/hil194/pickup-layouts');p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        publish(a.input.resolve())
        if a.once:return
        time.sleep(10)
if __name__=='__main__':main()
