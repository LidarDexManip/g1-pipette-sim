"""List pickup configs available now and progress of the resumable search."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.pickup_layouts import MANIFEST,ROOT,recorded_reviews
folder=ROOT/'outputs/hil194/pickup-layouts'
manifest=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'episodes':{}}
reviews=recorded_reviews(manifest.get('dataset_snapshot',''))
for key,entry in sorted(manifest['episodes'].items(),key=lambda pair:int(pair[0])):
    if not entry.get('config'):continue
    v=entry.get('validation',{})
    target=v.get('lift_target_m',entry.get('recorded_lift_reference',{}).get('target_m',0))
    attached=next((e['time'] for e in v.get('events',[]) if e['state']=='attached'),None)
    attachment=f"{attached:.3f}s" if attached is not None else 'none'
    print(f"Episode {key:>3}: attach {attachment:>8}, lift {v.get('lift_m',0)*1000:6.2f} mm / target {target*1000:6.2f} mm [{entry['status']}] | ./run.sh --episode {key}")

    if key in reviews:print('  Recorded video: '+reviews[key]['note'])

states=[json.loads(p.read_text()) for p in folder.glob('episode-*/search.json')]
started=sum(bool(s['trials']) for s in states)
print(f"\nEpisodes with current-target trials: {started}/194; completed trials: {sum(len(s['trials']) for s in states)}; published pickups: {sum(e['status']=='verified_pickup' for e in manifest['episodes'].values())}")
