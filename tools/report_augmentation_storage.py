"""Measure storage without deleting anything."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.augmentation.storage import write_storage
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--depth',type=int,default=1);a=p.parse_args()
    r=write_storage(a.root,a.output,a.depth)
    print(json.dumps({k:v for k,v in r.items() if k not in ['groups','notes']}))
