"""Run the separate, balanced placement/joint placement/joint campaign."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.augmentation.campaign import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260915);p.add_argument('--workers',type=int,default=4);p.add_argument('--draws',type=int,default=12);a=p.parse_args()
    print(json.dumps(run(a.output,a.seed,a.workers,a.draws)['totals']),flush=True)
