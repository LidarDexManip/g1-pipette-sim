"""Publish finished world-lift validation checkpoints during an incomplete batch."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.pickup_layouts import ROOT,MANIFEST,model_fingerprints,sha256
from tools.fit_pickup_layouts import write_json
from tools.publish_world_pickup_layouts import WORLD_CRITERIA

def publish():
    if not MANIFEST.exists():return
    manifest=json.loads(MANIFEST.read_text())
    if manifest.get('complete') or manifest.get('criteria',{}).get('lift_frame')!='world_tip_rise_since_attachment':return
    fp=model_fingerprints()
    if manifest.get('model_fingerprints')!=fp:raise ValueError('Published model changed')
    changed=False
    for path in sorted((ROOT/'outputs/hil194/pickup-layouts').glob('episode-*/validated-world.json')):
        saved=json.loads(path.read_text());entry=saved['entry'];key=str(entry['episode'])
        if saved.get('model_fingerprints')!=fp or manifest['episodes'].get(key)==entry:continue
        if entry.get('config') and sha256(ROOT/entry['config'])!=entry['sha256']:continue
        manifest['episodes'][key]=entry;changed=True
        if entry['status']=='verified_pickup':
            v=entry['validation'];print(json.dumps({'ready':entry['episode'],'world_lift_mm':1000*v['lift_m'],'recorded_target_mm':1000*v['lift_target_m'],'attachment_s':next(e['time'] for e in v['events'] if e['state']=='attached'),'command':f'./run.sh --episode {key}'}),flush=True)
    if changed:write_json(MANIFEST,manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        publish()
        if a.once:return
        time.sleep(10)
if __name__=='__main__':main()
