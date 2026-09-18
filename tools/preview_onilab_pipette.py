from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ['MUJOCO_GL']='egl'
import xml.etree.ElementTree as ET,copy
import mujoco
from PIL import Image
from g1_pipette_sim.scene import build_xml,load_config,element,camera_axes
c=load_config('config/hgdagger-cameras.json');full=ET.fromstring(build_xml(c));root=ET.Element('mujoco');root.append(copy.deepcopy(full.find('compiler')));root.append(copy.deepcopy(full.find('asset')));visual=element(root,'visual');element(visual,'global',offwidth=1200,offheight=900);world=element(root,'worldbody');tool=copy.deepcopy(full.find(".//body[@name='pipette']"));tool.set('pos','0 0 0');tool.set('quat','1 0 0 0');world.append(tool)
element(world,'light',pos=[.2,-.3,.5],diffuse=[.8,.8,.8]);element(world,'light',pos=[-.3,.2,.1],diffuse=[.5,.5,.5]);pos=[.20,-.35,.06];element(world,'camera',name='product',pos=pos,xyaxes=camera_axes(pos,[0,0,-.025]),fovy=40)
m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
opt=mujoco.MjvOption();opt.geomgroup[3]=False
with mujoco.Renderer(m,height=900,width=650) as r:
 r.update_scene(d,camera='product',scene_option=opt);Image.fromarray(r.render()).save('outputs/onilab-pipette.png')
