"""Render closed, intermediate, flat-open and table-resting lid poses."""
from pathlib import Path
import os,sys
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image,ImageDraw
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET
from g1_pipette_sim.dynamics import DynamicSimulation

sim=DynamicSimulation(load_config('config/hgdagger-cameras.json'),Episode(HGDAGGER_DATASET,15),source='low_level')
cam=mujoco.MjvCamera();cam.lookat[:]=sim.data.body('rack').xpos+[0,-.03,.035];cam.distance=.43;cam.azimuth=65;cam.elevation=-30
opt=mujoco.MjvOption();opt.geomgroup[3]=False;opt.geomgroup[1]=False
out=Path('outputs/box-hinge');out.mkdir(exist_ok=True)
sheet=Image.new('RGB',(1200,420),'white');draw=ImageDraw.Draw(sheet)
with mujoco.Renderer(sim.model,width=300,height=390) as renderer:
    for i,angle in enumerate([0,90,180,None]):
        sim.reset()
        if angle is None:
            for _ in range(round(.8/sim.model.opt.timestep)):sim.step()
            label=f"Table rest: {np.rad2deg(sim.data.joint('lid_hinge').qpos[0]):.1f} deg"
        else:
            sim.data.joint('lid_hinge').qpos[0]=np.deg2rad(angle);mujoco.mj_forward(sim.model,sim.data);label=f'{angle} deg'
        renderer.update_scene(sim.data,camera=cam,scene_option=opt)
        sheet.paste(Image.fromarray(renderer.render()),(i*300,30));draw.text((i*300+8,8),label,fill='black')
sheet.save(out/'hinge-preview.jpg',quality=90)
print(out/'hinge-preview.jpg')
