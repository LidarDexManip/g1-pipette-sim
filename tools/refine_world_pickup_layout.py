"""Refine existing placements against world-frame tip rise, with unchanged physics."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fit_pickup_layouts import rollout,candidate_config,write_json
from tools.search_replay_layout import on_table
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.pickup_layouts import ROOT,model_fingerprints

def evaluate(job):
 base,delta,root,i=job
 return rollout(candidate_config(base,delta),Episode(root,i),full=True,lift_frame='world')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,required=True);a=p.parse_args()
 folder=ROOT/f'outputs/hil194/pickup-layouts/episode-{a.episode:03}';original=json.loads((folder/'search.json').read_text());base=json.loads((folder/'baseline.json').read_text());origin=np.array(original['best']['delta']);path=folder/'world-search.json';state=json.loads(path.read_text()) if path.exists() else {'episode':a.episode,'lift_frame':'world','model_fingerprints':model_fingerprints(),'trials':[]}
 if state['model_fingerprints']!=model_fingerprints():raise ValueError('Model changed')
 deltas=[origin]
 for dz in [.0005,.001,.0015,.002,-.0005,-.001]:d=origin.copy();d[2]+=dz;deltas.append(d)
 for axis in [0,1]:
  for offset in [-.0005,.0005,-.001,.001]:d=origin.copy();d[axis]+=offset;d[2]+=.001;deltas.append(d)
 seen={tuple(np.round(r['delta'],7)) for r in state['trials']}
 with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
  for d in deltas:
   if tuple(np.round(d,7)) in seen or np.linalg.norm(d[:2])>.12 or abs(d[2])>.02 or not on_table(candidate_config(base,d)):continue
   result,trace,report=pool.submit(evaluate,(base,d.tolist(),original['dataset'],a.episode)).result();row={**result,'delta':d.tolist(),'trial':len(state['trials'])};state['trials'].append(row)
   write_json(folder/f"world-trial-{row['trial']:03}.json",report);write_json(path,state)
   print(json.dumps({'episode':a.episode,'trial':row['trial'],'world_lift_mm':1000*result['lift_m'],'target_mm':1000*result['lift_target_m'],'pickup':result['pickup'],'valid':result['valid']}),flush=True)
   if result['pickup'] and result['complete']:break
if __name__=='__main__':main()
