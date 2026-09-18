"""Validate completed search episodes while the remaining searches continue.

Writes validation checkpoints only; the publisher installs the final manifest.
Each worker handles one episode to release native MuJoCo allocations.
"""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.publish_pickup_layouts import finalize
from tools.fit_pickup_layouts import write_json
from g1_pipette_sim.dataset import Episode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--input',type=Path,default=Path('outputs/hil194/pickup-layouts'))
    a=p.parse_args();root=str(Episode(index=0).root);done={};errors={};pending={}
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        while len(done)+len(errors)<194:
            summary=a.input/'summary-height.json'
            completed=json.loads(summary.read_text()).get('completed',[]) if summary.exists() else []
            for row in completed:
                i=row['episode']
                if i in done or i in errors or i in pending.values():continue
                # Episode 10 has a separate full-model refinement still in progress.
                if i==10:continue
                if len(pending)>=a.workers:break
                pending[pool.submit(finalize,(i,str(a.input),root))]=i
            if not pending:
                if len(done)+len(errors)>=193:return
                time.sleep(10);continue
            finished,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for f in finished:
                i=pending.pop(f)
                try:
                    result=f.result();done[i]=result
                    v=result.get('validation',{})
                    print(json.dumps({'episode':i,'status':result['status'],'lift_mm':v.get('lift_m',0)*1000,'target_mm':v.get('lift_target_m',0)*1000}),flush=True)
                except Exception as error:
                    errors[i]=repr(error);print(f'ERROR episode {i}: {error!r}',flush=True)
                write_json(a.input/'incremental-validation.json',{'episodes':done,'errors':errors})
if __name__=='__main__':main()
