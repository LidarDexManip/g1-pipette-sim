"""Replay an accepted augmentation using its exact saved dense motor commands."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.collect_pickup_augmentations import load_episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.__main__ import configure_viewer_camera
from g1_pipette_sim.pickup_layouts import ROOT,model_fingerprints


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True,type=int);p.add_argument('--collection',type=Path,default=ROOT/'outputs/hil194/augmentation600');p.add_argument('--headless',action='store_true');p.add_argument('--camera',default='overview',choices=['overview','rgb','wrist_right','wrist_left']);a=p.parse_args()
 from g1_pipette_sim.augmentation.sources import import_seed
 candidate,ep,record=import_seed(a.collection/'candidates'/f'{a.candidate:06}')
 cfg=json.loads(Path(candidate['config']).read_text());sim=DynamicSimulation(cfg,ep,source='low_level')
 if a.headless:
  while sim.time<ep.duration-1e-9:sim.step()
 else:
  import mujoco.viewer
  with mujoco.viewer.launch_passive(sim.model,sim.data) as viewer:
   with viewer.lock():configure_viewer_camera(viewer.cam,sim.model,sim.data,cfg,a.camera);viewer.opt.geomgroup[3]=False
   start=time.monotonic();next_sync=0.
   while viewer.is_running() and sim.time<ep.duration-1e-9:
    sim.step()
    if sim.time>=next_sync:
     viewer.sync();next_sync+=1/60;time.sleep(max(0,start+sim.time-time.monotonic()))
 report=sim.report();print(json.dumps({'candidate':a.candidate,'parent_episode':candidate['parent_episode'],'augmentation':candidate['kind'],'state':report['state'],'events':report['events'],'finite':report['finite'],'warnings':report['warnings']},indent=2))
if __name__=='__main__':main()
