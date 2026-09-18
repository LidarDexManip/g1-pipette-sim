"""Render the generated assets at close range (no dataset needed)."""
import os,sys
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import xml.etree.ElementTree as ET
import mujoco
from PIL import Image
from g1_pipette_sim.scene import load_config,build_xml

cfg=load_config();cfg['table']['top_z']=.0;cfg['table']['thickness']=.001
cfg['rack']['center_xy']=[0,0];cfg['waste']['offset_from_rack']=[0,-.098]
root=ET.fromstring(build_xml(cfg));world=root.find('worldbody');pelvis=world.find("body[@name='pelvis']")
wrist=pelvis.find(".//body[@name='right_wrist_yaw_link']");pipe=wrist.find("body[@name='pipette']");wrist.remove(pipe);world.append(pipe);pipe.set('pos','.11 0 .15');pipe.set('quat','1 0 0 0');world.remove(pelvis)
for geom in list(world.findall('geom')):
    if geom.get('name')!='floor':world.remove(geom)
tip=world.find("body[@name='tip']");tip.set('pos','.085 -.06 .025')
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
cam=mujoco.MjvCamera();cam.lookat[:]=[.035,-.035,.09];cam.distance=.52;cam.azimuth=130;cam.elevation=-30
opt=mujoco.MjvOption();opt.geomgroup[3]=0
out=Path('outputs/photo-assets');out.mkdir(parents=True,exist_ok=True)
with mujoco.Renderer(model,height=1000,width=1400) as renderer:
    renderer.update_scene(data,camera=cam,scene_option=opt);Image.fromarray(renderer.render()).save(out/'objects.png')
print(out/'objects.png')
