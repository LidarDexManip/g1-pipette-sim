"""Refine a near-feasible layout with the full collision model, then recheck it."""
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fit_pickup_layouts import candidate_config,rollout,write_json,isolated_rollout
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.pickup_layouts import model_fingerprints

def probe(job):
    cfg,root,index,delta=job;result,trace,report=rollout(cfg,Episode(root,index),full=True)
    return delta,result,trace,report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    folder=Path('outputs/hil194/pickup-layouts')/f'episode-{a.episode:03}';path=folder/'search.json';state=json.loads(path.read_text());base=json.loads((folder/'baseline.json').read_text());root=state['dataset'];origin=np.array(state['best']['delta']);fingerprint=model_fingerprints()
    if state['model_fingerprints']!=fingerprint:raise ValueError('Model changed since search')
    deltas=[]
    for spacing in [.0005,.001]:
        for axis in range(3):
            for sign in [-1,1]:
                d=origin.copy();d[axis]+=sign*spacing;deltas.append(d.tolist())
    for dx,dy in [(-.0005,-.0005),(-.0005,.0005),(.0005,-.0005),(.0005,.0005)]:
        d=origin.copy();d[:3]+=[dx,dy,.0005];deltas.append(d.tolist())
    seen={tuple(np.round(r['delta'],6)) for r in state['trials']};deltas=[d for d in deltas if tuple(np.round(d,6)) not in seen]
    from tools.search_replay_layout import on_table
    deltas=[d for d in deltas if np.linalg.norm(d[:2])<=.12 and abs(d[2])<=.02 and abs(d[3])<=10 and on_table(candidate_config(base,d))]
    iterator=iter(deltas)
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures=set()
        def submit_next():
            d=next(iterator,None)
            if d is not None:futures.add(pool.submit(probe,(candidate_config(base,d),root,a.episode,d)))
        for _ in range(a.workers):submit_next()
        while futures:
            finished,_=wait(futures,return_when=FIRST_COMPLETED)
            f=next(iter(finished));futures.remove(f)
            delta,result,trace,report=f.result();row={**result,'delta':delta,'trial':len(state['trials']),'phase':'exact_refine'};state['trials'].append(row)
            write_json(folder/f"refine-{row['trial']:03}-report.json",report)
            if len(trace):np.save(folder/f"refine-{row['trial']:03}-trace.npy",trace)
            if result['score']<state['best']['score']:state['best']=row
            print(f"ep {a.episode} exact trial {row['trial']} score={result['score']:.3f} attached={result['attached']} lift={result['lift_m']*1000:.2f}mm",flush=True)
            if result['pickup']:
                write_json(path,state)
                verification,_,validation=isolated_rollout(candidate_config(base,delta),Episode(root,a.episode),full=True);row['verification']=verification
                if verification['pickup'] and verification['complete']:
                    state['verified_pickup']=True;state['best']=row;write_json(folder/'exact-refinement-validation.json',validation);write_json(path,state)
                    for pending in futures:pending.cancel()
                    print('VERIFIED',a.episode,flush=True);return
            write_json(path,state);submit_next()
if __name__=='__main__':main()
