"""Approximate session-specific cameras from episode-46 rack corners.

Focal lengths, rack world pose and dimensions remain estimates. Four planar
corners do not establish calibrated camera or wrist-to-tool transforms.
"""
import sys,json,copy
from pathlib import Path
import numpy as np,cv2,mujoco
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.scene import load_config,build_xml
from g1_pipette_sim.episode_scene import corners
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.simulation import set_robot_pose

c=load_config('config/hil194.json');ep=Episode(index=46)
ann={'rgb':{'wh':[700,394],'uv':[[299,243],[359,247],[352,331],[290,328]],'fovy':60},
     'wrist_right':{'wh':[850,850],'uv':[[294,625],[340,468],[598,474],[640,637]],'fovy':85},
     'wrist_left':{'wh':[500,500],'uv':[[287,284],[216,237],[315,169],[374,230]],'fovy':105}}
xyz=corners(c,'rack');m=mujoco.MjModel.from_xml_string(build_xml(c));d=mujoco.MjData(m);set_robot_pose(m,d,ep.names,*ep.sample(0),c);mujoco.mj_forward(m,d)
cameras=copy.deepcopy(c['cameras'])
for name,a in ann.items():
 w,h=a['wh'];f=h/2/np.tan(np.deg2rad(a['fovy']/2));K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
 ok,rv,tv=cv2.solvePnP(xyz,np.array(a['uv'],float),K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
 if not ok:raise ValueError(name)
 R=cv2.Rodrigues(rv)[0];pos=-R.T@tv.ravel();rot=R.T@np.diag([1,-1,-1]);projected=cv2.projectPoints(xyz,rv,tv,K,None)[0].reshape(-1,2);a['rms']=float(np.sqrt(np.mean((projected-a['uv'])**2)))
 camera=copy.deepcopy(cameras[name]);camera['position']=pos.tolist();camera['quaternion']=np.roll(Rotation.from_matrix(rot).as_quat(),1).tolist()
 if name=='wrist_left':
  b=d.body('right_wrist_yaw_link');Br=b.xmat.reshape(3,3);camera['position']=(Br.T@(pos-b.xpos)).tolist();camera['quaternion']=np.roll(Rotation.from_matrix(Br.T@rot).as_quat(),1).tolist()
 cameras[name]=camera;print(name,a['rms'])
newfit={**copy.deepcopy(c['camera_fit']),'episode':46,'annotations':ann,'provenance':'Manual rack corners from HIL194 episode 46; assumed rack pose and focal lengths. Approximate session-specific PnP, not calibrated extrinsics.'}
previous_new=next((s['overrides'] for s in c['dataset_profile'].get('sessions',[]) if s['name']=='20260909'),{})
c['dataset_profile']['sessions']=[{'name':'20260904','start':0,'end':46,'overrides':{'cameras':copy.deepcopy(c['cameras']),'camera_fit':copy.deepcopy(c['camera_fit'])}}, {'name':'20260909','start':46,'end':194,'overrides':{**previous_new,'cameras':cameras,'camera_fit':newfit}}]
Path('config/hil194.json').write_text(json.dumps(c,indent=2)+'\n')
