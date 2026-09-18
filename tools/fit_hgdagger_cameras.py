"""Reproduce an approximate camera initialization from manually annotated rack corners."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2,json,mujoco
from scipy.spatial.transform import Rotation
from g1_pipette_sim.scene import load_config,build_xml,rack_target
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.simulation import set_robot_pose
c=load_config('config/hgdagger-cameras.json');anchor=rack_target(c);c['rack']['active_cell']=[2,4];c['rack']['center_xy']=(anchor[:2]+[.0135,.0135]).tolist();c['waste']['offset_from_rack']=[0,-.094]
x,y=c['rack']['center_xy'];z=c['table']['top_z']+c['rack']['height'];rx,ry=np.array(c['rack']['size_xy'])/2
# Coordinates manually read from resized previews, with corner identity tied to world axes.
xyz=np.array([[x+rx,y+ry,z],[x+rx,y-ry,z],[x-rx,y-ry,z],[x-rx,y+ry,z]])
ann={'rgb':{'wh':[700,394],'uv':[[312,255],[375,259],[368,350],[301,343]],'fovy':60},'wrist_right':{'wh':[850,850],'uv':[[257,573],[319,400],[595,405],[658,575]],'fovy':85},'wrist_left':{'wh':[700,700],'uv':[[402,403],[314,334],[511,211],[554,325]],'fovy':105}}
# Wrist image coordinates are relative to its own panel in the displayed pair.
root=str(Path.home()/'.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hgdagger-20260904/snapshots/e74439f3eb2257661145571571b2dfaa5592f734');ep=Episode(root,index=0);m=mujoco.MjModel.from_xml_string(build_xml(c));d=mujoco.MjData(m);set_robot_pose(m,d,ep.names,*ep.sample(0),c);mujoco.mj_forward(m,d)
for name,a in ann.items():
 w,h=a['wh'];f=h/2/np.tan(np.deg2rad(a['fovy']/2));K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
 ok,rv,tv=cv2.solvePnP(xyz,np.array(a['uv'],float),K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE);R=cv2.Rodrigues(rv)[0];pos=-R.T@tv.ravel();rot=R.T@np.diag([1,-1,-1]);proj=cv2.projectPoints(xyz,rv,tv,K,None)[0].reshape(-1,2);a['rms']=float(np.sqrt(np.mean((proj-a['uv'])**2)))
 cam={'position':pos.tolist(),'quaternion':np.roll(Rotation.from_matrix(rot).as_quat(),1).tolist(),'fovy':a['fovy'],'resolution_wh':([1280,720] if name=='rgb' else [1080,1080]),'role':{'rgb':'fixed_ego','wrist_right':'fixed_side','wrist_left':'right_wrist'}[name]}
 if name=='wrist_left':
  b=d.body('right_wrist_yaw_link');Br=b.xmat.reshape(3,3);cam['position']=(Br.T@(pos-b.xpos)).tolist();cam['quaternion']=np.roll(Rotation.from_matrix(Br.T@rot).as_quat(),1).tolist();cam['parent_body']='right_wrist_yaw_link'
 c['cameras'][name]=cam;print(name,'rms',a['rms'],'worldpos',pos,'local',cam['position'])
c['camera_fit']={'method':'Approximate planar rack PnP with assumed focal lengths; not intrinsic calibration','dataset_snapshot':str(root),'episode':0,'annotations':ann,'camera_order':['rgb','wrist_right','wrist_left']}
Path('config/hgdagger-cameras.json').write_text(json.dumps(c,indent=2)+'\n')
