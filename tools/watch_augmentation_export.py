"""Wait for collection completion, then publish immutable v2.1 and v3 snapshots."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.augmentation.export import catalog,publish
from g1_pipette_sim.augmentation.contracts import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    while True:
        state=json.loads((a.collection/'progress.json').read_text())
        if state['status']=='collected':break
        if state['status'] in ['incomplete','candidate_budget_exhausted']:raise RuntimeError('Collection incomplete')
        time.sleep(30)
    value=catalog([a.collection])
    publish(value,a.output/'lerobot-v21')
    publish(value,a.output/'lerobot-v3',v3=True)
    atomic_json(a.output/'export-progress.json',{'complete':True,'episodes':len(value['entries'])})
if __name__=='__main__':main()
