"""Validate completed episode searches using world lift; write checkpoints only."""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.publish_world_pickup_layouts import finalize_world
from tools.fit_pickup_layouts import write_json
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.pickup_layouts import ROOT

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);a=p.parse_args();folder=ROOT/'outputs/hil194/pickup-layouts';root=str(Episode(index=0).root);done={};errors={};pending={}
 with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
  while len(done)+len(errors)<192:
   first=json.loads((folder/'summary-height.json').read_text());retry=json.loads((folder/'summary-target-aware-retries.json').read_text());retry_done={r['episode'] for r in retry['completed']};retry_all=set(retry['requested'])
   for row in first['completed']:
    i=row['episode']
    if i in {14,52} or i in done or i in errors or i in pending.values() or (i in retry_all and i not in retry_done):continue
    if len(pending)>=a.workers:break
    pending[pool.submit(finalize_world,(i,str(folder),root))]=i
   if not pending:time.sleep(10);continue
   finished,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
   for f in finished:
    i=pending.pop(f)
    try:
     e=f.result();done[i]=e;v=e.get('validation',{});print(json.dumps({'episode':i,'status':e['status'],'world_lift_mm':v.get('lift_m',0)*1000,'target_mm':v.get('lift_target_m',0)*1000}),flush=True)
    except Exception as error:errors[i]=repr(error);print('ERROR',i,repr(error),flush=True)
    write_json(folder/'world-incremental-validation.json',{'episodes':done,'errors':errors})
if __name__=='__main__':main()
