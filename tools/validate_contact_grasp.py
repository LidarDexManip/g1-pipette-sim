from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco,json
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dynamics import DynamicSimulation
root='/home/ytung34/.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hgdagger-20260904';out=[]
for idx,lift in [(0,.25),(0,0),(15,0),(45,0)]:
 c=load_config('config/hgdagger-cameras.json');c['base_position'][2]+=lift;s=DynamicSimulation(c,Episode(root,idx),source='low_level')
 if lift:
  reference=s._reference;s._reference=lambda t:reference(0)
 samples=[]
 for t in [1,3,8]:
  while s.time<t:s.step()
  b=s.data.body('right_wrist_yaw_link');samples.append({'t':t,'local':(b.xmat.reshape(3,3).T@(s.data.body('pipette').xpos-b.xpos)).tolist(),'weld':bool(s.data.eq_active[s.grip_id])})
 result={'episode':idx,'lift':lift,'samples':samples,'contacts':[p for p in s.contact_pairs if 'pipette' in p],'warnings':s.report()['warnings']}
 if lift:
  z=float(s.data.body('pipette').xpos[2]);s.release_grip()
  for _ in range(round(.7/s.model.opt.timestep)):s.step()
  result['drop_after_open_m']=z-float(s.data.body('pipette').xpos[2])
 out.append(result);print(result,flush=True)
open('outputs/contact-grasp-validation.json','w').write(json.dumps(out,indent=2))
