"""Compare published tip height above the rack with world-frame rise after attachment."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.pickup_layouts import ROOT,select_pickup_layout

def check(i):
 ep=Episode(index=i);cfg,entry=select_pickup_layout({},ep);sim=DynamicSimulation(cfg,ep,source='low_level');a=None;world=0.;relative=0.;rack_change=0.
 while sim.time<ep.duration-1e-9:
  sim.step()
  if sim.state!='attached':continue
  tip=sim.data.body('tip');z=float(tip.xpos[2]+tip.xmat.reshape(3,3)[2,2]*cfg['pipette']['tip_length']/2);r=float(sim.data.site('rack_mouth').xpos[2])
  if a is None:a=(z,r,sim.time)
  world=max(world,z-a[0]);relative=max(relative,z-r);rack_change=max(rack_change,abs(r-a[1]))
 v=entry['validation'];return {'episode':i,'attachment_s':a[2] if a else None,'world_rise_since_attachment_mm':1000*world,'maximum_tip_height_above_rack_mm':1000*relative,'maximum_rack_vertical_motion_after_attachment_mm':1000*rack_change,'recorded_fk_target_mm':1000*v['lift_target_m'],'world_rise_matches_target':abs(world-v['lift_target_m'])<=v['lift_tolerance_m'],'finite':sim.report()['finite'],'warnings':sim.report()['warnings']}
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--episodes',nargs='+',type=int,default=[5,14,52,104]);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
 out=ROOT/'outputs/hil194/pickup-layouts/lift-frame-check.json';rows=[]
 with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
  for r in pool.map(check,args.episodes):
   rows.append(r);out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(r),flush=True)
