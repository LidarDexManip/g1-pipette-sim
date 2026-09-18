"""Compare the HG-DAgger camera fit against recorded frames, without dynamics fitting."""
import os,sys,json
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,mujoco
from PIL import Image,ImageDraw
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.simulation import Simulation
from g1_pipette_sim.camera_streams import CameraStreams
cfg=load_config('config/hgdagger-cameras.json');root=cfg['camera_fit']['dataset_snapshot'];out=Path('outputs/hgdagger-cameras');out.mkdir(exist_ok=True)
for idx in [0,15,30,45]:
 ep=Episode(root,index=idx);sim=Simulation(cfg,ep);times=[0,ep.duration*.5,ep.duration*.9]
 sheet=Image.new('RGB',(1200,3*680),(22,22,22));draw=ImageDraw.Draw(sheet)
 streams={name:CameraStreams(sim.model,[name],width=400,height=225 if name=='rgb' else 400) for name in ['rgb','wrist_left','wrist_right']}
 try:
  for row,t in enumerate(times):
   sim._pose(t);sim._hold_tip();mujoco.mj_forward(sim.model,sim.data)
   for col,name in enumerate(['rgb','wrist_right','wrist_left']):
    cap=cv2.VideoCapture(str(ep.video_path(name)));cap.set(cv2.CAP_PROP_POS_FRAMES,ep.frame_at(t));ok,real=cap.read();cap.release()
    if not ok:raise RuntimeError((idx,t,name))
    pixels=streams[name].render(sim.data)[name];im=Image.fromarray(pixels);im.thumbnail((395,310));actual=Image.fromarray(cv2.cvtColor(real,cv2.COLOR_BGR2RGB));actual.thumbnail((395,310));y=row*680;sheet.paste(actual,(col*400,y+25));sheet.paste(im,(col*400,y+360));draw.text((col*400+5,y+5),f'REAL {name} ep{idx} t{t:.2f}',fill='white');draw.text((col*400+5,y+340),'SIM '+cfg['cameras'][name]['role'],fill='white')
    if idx==0 and row==0:Image.fromarray(pixels).save(out/(name+'-preview.png'))
 finally:
  for s in streams.values():s.close()
 sheet.save(out/f'comparison-episode-{idx:03}.jpg')
print(out)
