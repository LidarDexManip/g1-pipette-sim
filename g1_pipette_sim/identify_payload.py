"""Identify an effective wrist payload from slow free-space motor effort.

This is an aggregate inertial correction, not a measured pipette mass.
"""
import json,numpy as np,pyarrow.parquet as pq,mujoco,xml.etree.ElementTree as ET
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
from g1_pipette_sim.scene import build_xml,load_config,element
from g1_pipette_sim.dataset import Episode,LEGACY_DATASET
from g1_pipette_sim.simulation import set_robot_pose
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--train',type=int,nargs='+',default=[53,54,55,56])
parser.add_argument('--controller-config',type=Path,default=Path('config/identified.json'))
parser.add_argument('--config-output',type=Path,default=Path('config/calibrated-physics.json'))
parser.add_argument('--output',type=Path,default=Path('outputs/calibration/payload_fit.json'))
args=parser.parse_args()
cfg=load_config('config/workspace.json');root=ET.fromstring(build_xml(cfg));w=root.find(".//body[@name='right_wrist_yaw_link']");b=element(w,'body',name='extra_payload');element(b,'inertial',pos=[0,0,0],mass=1,diaginertia=[.001,.001,.001]);m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);bid=m.body('extra_payload').id
points=np.array([[.05,0,0],[.22,0,0],[.12,.08,0],[.12,-.08,0],[.12,0,.08],[.12,0,-.08]])
X=[];Y=[]
for idx in args.train:
 e=Episode(LEGACY_DATASET,index=idx);path=e.root/e.info['data_path'].format(episode_chunk=0,episode_index=idx);eff=np.array(pq.read_table(path,columns=['observation.effort'])['observation.effort'].to_pylist());q=e.arrays['observation.state'][:,:29];v=savgol_filter(q,15,2,deriv=1,delta=np.median(np.diff(e.times)),axis=0)
 ids=np.array([m.jnt_dofadr[m.joint(n).id] for n in e.names[22:]])
 for i in range(15,len(q)-15,5):
  t=e.times[i]
  if 7<t<14 or np.linalg.norm(v[i,22:])>.2:continue
  set_robot_pose(m,d,e.names,*e.sample(t),cfg);m.body_mass[bid]=0;mujoco.mj_forward(m,d);base=d.qfrc_bias[ids].copy();cols=[]
  for point in points:
   m.body_mass[bid]=1;m.body_pos[bid]=point;mujoco.mj_forward(m,d);cols.append(d.qfrc_bias[ids]-base)
  X.append(np.array(cols).T);Y.append(eff[i,22:]-base)
X=np.concatenate(X);Y=np.concatenate(Y);res=lsq_linear(X,Y,bounds=(0,1.5));mass=sum(res.x);pos=res.x@points/mass
print('samples',len(Y)//7,'mass',mass,'pos',pos,'before',np.sqrt(np.mean(Y**2)),'after',np.sqrt(np.mean((X@res.x-Y)**2)),flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
json.dump(dict(mass=mass,position=pos.tolist(),point_masses=res.x.tolist(),static_effort_rmse_before=float(np.sqrt(np.mean(Y**2))),static_effort_rmse_after=float(np.sqrt(np.mean((X@res.x-Y)**2)))),args.output.open('w'),indent=2)

c=load_config(args.controller_config)
c['effective_wrist_payload']={'mass':float(mass),'position':pos.tolist()}
c['calibration']['payload_training_episodes']=args.train
c['calibration']['payload_note']='Effective inertial correction fitted to slow free-space effort; not a measured physical component or pipette mass.'
args.config_output.parent.mkdir(parents=True,exist_ok=True)
args.config_output.write_text(json.dumps(c,indent=2)+'\n')
