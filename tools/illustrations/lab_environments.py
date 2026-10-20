"""Cycle procedural lab surroundings over a verified continuous pickup capture.

Presentation-only state playback: decorative geometry exists in the renderer,
not the physics model. This does not generate validated training demonstrations.
"""
from contextlib import ExitStack
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from g1_pipette_sim.dynamics import build_dynamic_xml
from g1_pipette_sim.augmentation.sources import import_seed
from g1_pipette_sim.augmentation.contracts import file_hash

from g1_pipette_sim.environments.procedural import THEMES, LIGHTING, lab_layout


def add_layout(scene, layout):
    for obj in layout['objects']:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Renderer geometry capacity exceeded')
        geom=scene.geoms[scene.ngeom]
        kind=mujoco.mjtGeom.mjGEOM_BOX if obj['kind']=='box' else mujoco.mjtGeom.mjGEOM_CYLINDER
        mujoco.mjv_initGeom(geom,kind,np.asarray(obj['size']),np.asarray(obj['pos']),np.asarray(obj.get('mat',np.eye(3).ravel())),np.asarray(obj['rgba'],dtype=np.float32))
        geom.category=mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom+=1


def illuminate(scene, lighting):
    tint=np.asarray(lighting['tint']);strength=lighting['intensity']
    for i in range(scene.nlight):
        light=scene.lights[i]
        light.diffuse[:]=tint*strength*(.85 if i==0 else .4)
        light.ambient[:]=tint*.2
        light.specular[:]=tint*.12
        light.pos[:]=lighting['position'] if i==0 else [-.6,.7,2.5]
        direction=np.array([.35,0,.65])-light.pos
        light.dir[:]=direction/np.linalg.norm(direction)
        light.type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,default=ROOT/'outputs/hil194/augmentation600')
    p.add_argument('--candidate',type=int,default=25,help='Accepted augmentation candidate; 25 is the episode-64 illustration baseline.')
    p.add_argument('--swap-seconds',type=float,default=.3,help='Seconds between lab changes; motion never resets.')
    p.add_argument('--seed',type=int,default=64)
    p.add_argument('--resolution',choices=['4k','1080p'],default='4k',help='Composite output resolution (default: 3840x2160).')
    p.add_argument('--name',default='episode64-lab-environments')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/augmentation-illustration')
    p.add_argument('--duration',type=float,help='Optional short preview duration, in seconds.')
    args=p.parse_args()
    if not np.isfinite(args.swap_seconds) or args.swap_seconds<=0:p.error('swap-seconds must be positive and finite')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration<=0):p.error('duration must be positive and finite')
    if Path(args.name).name!=args.name:p.error('name must be a filename stem')
    folder=args.collection.resolve()/f'candidates/{args.candidate:06d}'
    _,episode,record=import_seed(folder)
    catalog_path=ROOT/'outputs/hil194/augmentation-catalog-v1.json'
    catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else {'entries':[]}
    receipt=next((e['files'] for e in catalog['entries'] if e['folder']==str(folder.resolve())), record.get('artifact_hashes',{}))
    if not receipt:raise ValueError('Capture has no integrity receipt')
    for name,digest in receipt.items():
        if file_hash(folder/name)!=digest:raise ValueError('Changed source capture: '+name)
    cfg=json.loads((folder/'config.json').read_text())
    from g1_pipette_sim.camera_views import apply_camera_views
    cfg=apply_camera_views(cfg,episode)
    model=mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg));data=mujoco.MjData(model)
    with np.load(folder/'trajectory.npz') as archive:
        qpos=archive['qpos'].copy();qvel=archive['qvel'].copy();times=archive['timestamp'].copy();simtimes=archive['simulation_time'].copy()
    if qpos.shape!=(len(times),model.nq) or qvel.shape!=(len(times),model.nv):raise ValueError('Capture/model state dimensions differ')
    fps=30
    if not np.allclose(times,np.arange(len(times))/fps):raise ValueError('Unexpected capture clock')
    frames=len(times) if args.duration is None else min(len(times),max(1,round(args.duration*fps)))
    swap=max(1,round(args.swap_seconds*fps));layouts=[lab_layout(i,args.seed,cfg) for i in range(len(THEMES))]
    order=np.random.default_rng(args.seed).permutation(len(layouts)).tolist()
    width,height=(3840,2160) if args.resolution=='4k' else (1920,1080)
    scale=width/1280
    px=lambda n:round(n*scale)
    main_height=px(400);header=px(40);label_height=px(26);view_height=px(240)
    policy_y=header+main_height+label_height
    camera=mujoco.MjvCamera();camera.lookat[:]=[.22,0,1.02];camera.distance=1.65;camera.azimuth=135;camera.elevation=-20
    option=mujoco.MjvOption();option.geomgroup[3]=False
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',px(16))
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',px(23))
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',px(11))
    views=[('rgb','Ego camera'),('wrist_right','Side camera'),('wrist_left','Right wrist camera')]
    sizes={name:(round(view_height*cfg['cameras'][name]['resolution_wh'][0]/cfg['cameras'][name]['resolution_wh'][1]),view_height) for name,_ in views}
    model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(main_height,view_height,model.vis.global_.offheight)
    args.output.mkdir(parents=True,exist_ok=True);output=args.output/(args.name+'.mp4');temp=args.output/(args.name+'.tmp.mp4')
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    segments=[];previews=[];camera_frames=[]
    try:
        with ExitStack() as stack:
            renderer=stack.enter_context(mujoco.Renderer(model,height=main_height,width=width))
            cameras={name:stack.enter_context(mujoco.Renderer(model,width=size[0],height=size[1])) for name,size in sizes.items()}
            for frame in range(frames):
                idx=order[(frame//swap)%len(order)];layout=layouts[idx]
                if frame%swap==0:segments.append({'start_frame':frame,'end_frame_exclusive':min(frame+swap,frames),'layout':idx,'name':layout['name']})
                data.qpos[:]=qpos[frame];data.qvel[:]=qvel[frame];data.time=simtimes[frame];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=option);add_layout(renderer.scene,layout);illuminate(renderer.scene,layout['lighting'])
                canvas=Image.new('RGB',(width,height),'#111b29');canvas.paste(Image.fromarray(renderer.render()),(0,header))
                draw=ImageDraw.Draw(canvas)
                draw.text((width/2,px(6)),'Lab Environment Augmentation',anchor='mt',font=bold,fill='white')
                poses={}
                for column,(name,label) in enumerate(views):
                    view=cameras[name];view.update_scene(data,camera=name,scene_option=option);add_layout(view.scene,layout);illuminate(view.scene,layout['lighting'])
                    x=column*(width//3)+(width//3-sizes[name][0])//2
                    canvas.paste(Image.fromarray(view.render()),(x,policy_y))
                    draw.text((column*(width//3)+px(14),header+main_height+px(3)),label,font=font,fill='#73d8bf')
                    cam_id=model.camera(name).id
                    poses[name]={'position_world_m':data.cam_xpos[cam_id].tolist(),'rotation_camera_to_world':data.cam_xmat[cam_id].reshape(3,3).tolist()}
                camera_frames.append({'frame':frame,'simulation_time_s':float(data.time),'cameras':poses})
                draw.text((px(16),height-px(14)),f't = {frame/fps:05.2f} s   |   {layout["name"]}   |   Synchronized policy camera views',font=small,fill='#b8c7d8')
                process.stdin.write(canvas.tobytes())
                if frame%swap==0 and len(previews)<len(layouts):previews.append(canvas.resize((640,360)))
                if frame==frames//2:canvas.save(args.output/(args.name+'.png'))
                if (frame+1)%30==0:print(f'Rendered {frame+1}/{frames}',flush=True)
    except BaseException:
        process.stdin.close();process.wait();raise
    process.stdin.close()
    if process.wait():raise RuntimeError('Video encoder failed')
    import cv2
    cap=cv2.VideoCapture(str(temp));count=0
    if abs(cap.get(cv2.CAP_PROP_FPS)-fps)>.01:raise ValueError('Unexpected video FPS')
    while True:
        ok,pixels=cap.read()
        if not ok:break
        if pixels.shape!=(height,width,3):raise ValueError('Unexpected video size')
        count+=1
    cap.release()
    if count!=frames:raise ValueError(f'Incomplete video: {count}/{frames}')
    temp.replace(output)
    sheet=Image.new('RGB',(1280,360*((len(previews)+1)//2)))
    for i,preview in enumerate(previews):sheet.paste(preview.resize((640,360)),((i%2)*640,(i//2)*360))
    sheet.save(args.output/(args.name+'-layouts.jpg'))
    metadata={'title':'Lab Environment Augmentation','type':'visual-only lab surroundings over saved successful state trajectory','training_demonstration':False,'physics_rerun':False,'task_state_modified':False,'robot_trajectory_cuts':False,'baseline_candidate':args.candidate,'parent_episode':record['candidate']['parent_episode'],'source_capture':str(folder),'source_integrity':receipt,'seed':args.seed,'resolution_wh':[width,height],'policy_view_order':[name for name,_ in views],'policy_view_render_sizes':sizes,'camera_configuration':cfg['cameras'],'camera_view_correction':cfg.get('camera_view_correction'),'camera_views_note':'Same configured simulator observation cameras before policy preprocessing; aspect ratios preserved, no simulation step between views. Decorative environment rendered in every view.','presentation_camera':{'lookat':camera.lookat.tolist(),'distance':camera.distance,'azimuth':camera.azimuth,'elevation':camera.elevation},'fps':fps,'frames_verified':count,'duration_s':count/fps,'swap_seconds':swap/fps,'layouts':layouts,'segments':segments,'script_sha256':file_hash(Path(__file__))}
    (args.output/(args.name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/(args.name+'-camera-poses.json')).write_text(json.dumps(camera_frames,indent=2)+'\n')
    print(output,flush=True)


if __name__=='__main__':main()
