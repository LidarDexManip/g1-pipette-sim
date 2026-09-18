"""Render prescribed seated mesh poses, not a successful episode replay."""
from pathlib import Path
import os,sys
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image,ImageDraw
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET
from g1_pipette_sim.simulation import Simulation

out=Path('outputs/tip-fit');out.mkdir(exist_ok=True)
sheet=Image.new('RGB',(1200,590),'white');draw=ImageDraw.Draw(sheet)
old=load_config(out/'before.json');new=load_config('config/hgdagger-cameras.json')
for i,(cfg,depth,transparent,label) in enumerate([(old,.003,False,'Previous: 3 mm insertion'),(new,new['pipette']['seated_insertion_depth'],False,'Revised: 11 mm insertion'),(new,new['pipette']['seated_insertion_depth'],True,'Cutaway: nozzle highlighted')]):
    sim=Simulation(cfg,Episode(HGDAGGER_DATASET,15));sim.state='attached';sim.seated_depth=depth;sim._hold_tip();mujoco.mj_forward(sim.model,sim.data)
    for g in range(sim.model.ngeom):
        name=sim.model.body(sim.model.geom_bodyid[g]).name
        if name not in ('pipette','ejector','tip'):sim.model.geom_rgba[g,3]=0
        elif name=='tip' and transparent:sim.model.geom_rgba[g,3]=.3
        elif transparent and sim.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and sim.model.mesh(sim.model.geom_dataid[g]).name=='object_pipette_nozzle':sim.model.geom_rgba[g,:3]=[.7,.72,.75]
    cam=mujoco.MjvCamera();site=sim.data.site('nozzle');cam.lookat[:]=site.xpos+site.xmat.reshape(3,3)@np.array([0,0,-.009]);cam.distance=.10;cam.azimuth=70;cam.elevation=-5
    opt=mujoco.MjvOption();opt.geomgroup[3]=False;opt.sitegroup[:]=False
    with mujoco.Renderer(sim.model,width=400,height=550) as renderer:
        renderer.update_scene(sim.data,camera=cam,scene_option=opt);sheet.paste(Image.fromarray(renderer.render()),(i*400,40))
    draw.text((i*400+10,10),label,fill='black')
sheet.save(out/'seated-comparison.jpg',quality=92)
print(out/'seated-comparison.jpg')
