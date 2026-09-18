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

THEMES = ['Open steel shelving', 'Wire storage racks', 'Mobile lab carts',
          'Storage cabinets', 'Mixed wire and cabinet storage', 'Compact benchtop storage',
          'Tall utility shelving', 'Wide low storage']
# Fixed material palette: scene-to-scene color shifts come from illumination.
STEEL = [.58,.62,.64]
SHELF = [.72,.75,.76]
BOX = [.72,.65,.53]
LID = [.82,.8,.73]
LIGHTING = [
    {'name':'Neutral overhead','tint':[1.,.97,.92],'intensity':.82,'position':[.4,-.4,2.8]},
    {'name':'Cool diffuse','tint':[.87,.94,1.],'intensity':.85,'position':[.9,.4,2.6]},
    {'name':'Warm overhead','tint':[1.,.88,.73],'intensity':.79,'position':[-.4,-.6,2.5]},
    {'name':'Daylight from left','tint':[.92,.97,1.],'intensity':.93,'position':[-1.2,.3,2.]},
    {'name':'Soft neutral','tint':[1.,.96,.9],'intensity':.7,'position':[.2,.9,2.8]},
    {'name':'Warm task lighting','tint':[1.,.9,.78],'intensity':.88,'position':[.8,-.8,2.]},
    {'name':'Cool overhead','tint':[.88,.95,1.],'intensity':.75,'position':[0.,0.,3.]},
    {'name':'Bright neutral','tint':[1.,.98,.94],'intensity':.95,'position':[1.2,.3,2.7]},
]


def lab_layout(index, seed, cfg):
    """Renderer primitives for room furniture and task-adjacent benchtop equipment."""
    name=THEMES[index]
    wall=[.84,.85,.84];floor=[.55,.57,.57];accent=[.44,.49,.5]
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    objects = []
    def box(pos, size, color):
        objects.append(dict(kind='box', pos=list(pos), size=list(size), rgba=list(color)+[1]))
    def cylinder(pos, radius, halfheight, color):
        objects.append(dict(kind='cylinder', pos=list(pos), size=[radius,halfheight,0], rgba=list(color)+[1]))
    box([.4,0,-.02],[2.6,2.5,.025],floor)
    # Back wall and a side wall leave the foreground camera unobstructed.
    box([1.95,0,1.15],[.045,2.5,1.15],wall)
    box([.3,-1.55,1.15],[1.65,.045,1.15],wall)
    # Panel seams and an accent strip give scale to the room.
    box([1.897,0,.92],[.006,2.45,.055],accent)
    for y in [-1.1,-.2,.7,1.6]:
        box([1.895,y,1.45],[.007,.009,.8],[.65,.69,.7])
    # Move and resize the rear cabinetry between environments.
    y0 = float(rng.uniform(-.45,.1))
    for j in range(3):
        y = y0+(j-1)*.56
        box([1.55,y,.38],[.24,.26,.38],accent)
        box([1.53,y,.78],[.29,.28,.025],[.82,.85,.84])
        for z in [.2,.44,.64]:
            box([1.301,y,z],[.01,.235,.075],[min(1,c+.12) for c in accent])
            box([1.285,y,z+.015],[.012,.08,.012],[.75,.78,.8])
        # Vary equipment between bottles, a benchtop instrument and tube racks.
        equipment=(j+index)%3
        if equipment==0:
            for k in range(4):
                cylinder([1.5,y-.17+k*.11,.9],.035,.09,[.5,.7,.78])
                cylinder([1.5,y-.17+k*.11,1.0],.022,.015,accent)
        elif equipment==1:
            box([1.51,y,.94],[.16,.2,.13],[.91,.92,.9])
            box([1.342,y,.98],[.008,.13,.055],[.05,.19,.22])
            cylinder([1.51,y,1.085],.11,.015,[.36,.4,.45])
        else:
            box([1.49,y,.84],[.12,.19,.025],accent)
            for k in range(5):
                cylinder([1.48,y-.14+k*.07,.9],.017,.045,[.94,.76,.22])
    # Side shelving alternates with a large instrument enclosure.
    if index%2==0:
        for x in [.65,1.25]:
            box([x,-1.18,.88],[.025,.025,.88],[.48,.53,.56])
        for z in [.25,.7,1.15,1.6]:
            box([.95,-1.18,z],[.34,.2,.025],[.78,.81,.82])
            for k in range(3):
                box([.72+.21*k,-1.18,z+.11],[.07,.12,.085],BOX)
    else:
        box([.9,-1.19,.72],[.34,.25,.72],[.86,.88,.9])
        box([.9,-.93,1.0],[.28,.012,.3],[.19,.32,.37])
        box([.9,-.91,.61],[.28,.025,.025],accent)
    # Wall cabinet / fume hood and a differently located utility cart.
    box([1.69,.8,1.55],[.2,.38,.3],[.87,.9,.91])
    box([1.48,.8,1.53],[.012,.32,.22],[.3,.47,.51])
    cart_y = .9+float(rng.uniform(0,.18))
    for z in [.2,.65]:
        box([.9,cart_y,z],[.23,.18,.018],[.68,.73,.76])
    for x in [.69,1.11]:
        for y in [cart_y-.16,cart_y+.16]:
            cylinder([x,y,.34],.015,.32,[.5,.56,.6])
    box([.9,cart_y,.76],[.14,.12,.09],accent)
    # Camera is at +x/-y; put furniture behind the task, at -x/+y.
    for obj in objects:
        obj['pos'][0] = .4-obj['pos'][0]
        obj['pos'][1] = -obj['pos'][1]
    room_count=len(objects)
    top=cfg['table']['top_z']
    tx,ty=cfg['table']['center_xy']
    # Three separated slots along the far side of the 60 x 65 cm task table.
    # Keep the original task rack, open lid and insertion corridor clear.
    slots=[(tx+.15,ty-.19),(tx+.15,ty+.18),(tx+.16,ty-.005)]
    equipment=[]
    for slot,(x,y) in enumerate(slots):
        x+=float(rng.uniform(-.009,.009));y+=float(rng.uniform(-.01,.01))
        kind='centrifuge' if slot==index%2 else 'tube_rack'
        color=[.34,.45,.49]
        start=len(objects)
        if kind=='centrifuge':
            box([x,y,top+.035],[.065,.06,.035],[.9,.92,.94])
            cylinder([x,y,top+.073],.054,.009,color)
            cylinder([x,y,top+.085],.043,.003,[.18,.26,.3])
            # Visible lid ring and front display/controls.
            cylinder([x,y,top+.09],.012,.003,[.68,.73,.77])
            box([x-.0655,y,top+.047],[.001,.028,.011],[.04,.19,.24])
            box([x-.067,y+.042,top+.045],[.002,.006,.006],[.3,.8,.53])
        else:
            box([x,y,top+.015],[.055,.052,.015],color)
            box([x,y,top+.035],[.055,.052,.004],color)
            for row in range(3):
                for col in range(4):
                    xx=x+(col-1.5)*.024;yy=y+(row-1)*.032
                    cylinder([xx,yy,top+.041],.007,.018,[.76,.87,.9])
                    cylinder([xx,yy,top+.061],.008,.004,[.68,.71,.69])
        for obj in objects[start:]:obj['benchtop']=True
        equipment.append({'kind':kind,'center_xy':[x,y],'top_z':top,'first_geom':start,'end_geom':len(objects)})
    storage_start=len(objects)
    # Low storage shelves immediately beyond the table edges fall inside the
    # downward-looking policy views; the distant room shelves remain in place.
    storage=[]
    style_pairs=[('open','open'),('wire','wire'),('cart','cart'),('cabinet','cabinet'),
                 ('wire','cabinet'),('cart','open'),('open','wire'),('cabinet','cart')]
    for shelf,style in enumerate(style_pairs[index]):
        tall=[.85,1.03,.66,.94,1.08,.7,1.16,.62][index]
        levels=2 if style=='cart' or index in [5,7] else (4 if index==6 else 3)
        depth=[.10,.13,.11,.15,.12,.09,.11,.15][index]
        span=[.25,.31,.22,.26,.28,.19,.23,.34][index]
        hx,hy=(depth,span) if shelf==0 else (span,depth)
        yaw=np.deg2rad([-7,5,0,-4,7,0,-6,4][index])
        R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        extent=np.abs(R[:2,:2])@np.array([hx,hy])
        if shelf==0:x=tx+.3+extent[0]+.045+.025*(index%3);y=ty+[-.05,.07,-.12,.04,-.08,.13,.0,-.04][index]
        else:x=tx+[-.09,.1,-.02,.07,-.06,.04,.13,-.03][index];y=ty-.325-extent[1]-.055-.025*(index%3)
        first=len(objects)
        for xx in [-hx+.012,hx-.012]:
            for yy in [-hy+.012,hy-.012]:
                box([xx,yy,tall/2],[.009,.009,tall/2],STEEL)
                if style=='cart':cylinder([xx,yy,.035],.025,.026,[.17,.18,.19])
        if style=='cabinet':
            # Closed side/back panels, open front cubbies with boxed supplies.
            if shelf==0:
                box([hx,0,tall/2],[.01,hy,tall/2],SHELF)
                for yy in [-hy,hy]:box([0,yy,tall/2],[hx,.01,tall/2],SHELF)
            else:
                box([0,-hy,tall/2],[hx,.01,tall/2],SHELF)
                for xx in [-hx,hx]:box([xx,0,tall/2],[.01,hy,tall/2],SHELF)
        for level,z in enumerate(np.linspace(.15,tall-.06,levels)):
            if style=='wire':
                for yy in np.linspace(-hy,hy,max(3,int(hy/.025))):box([0,yy,z],[hx,.003,.003],STEEL)
                for xx in np.linspace(-hx,hx,max(3,int(hx/.025))):box([xx,0,z],[.003,hy,.003],STEEL)
            else:box([0,0,z],[hx,hy,.01],SHELF)
            count=2 if index in [2,5] else 3
            for k in range(count):
                offset=(k-(count-1)/2)*(2*span/count)
                bx,by=(0,offset) if shelf==0 else (offset,0)
                half=[depth*.72,span/count*.78,.033+.007*((k+level)%3)] if shelf==0 else [span/count*.78,depth*.72,.033+.007*((k+level)%3)]
                box([bx,by,z+.012+half[2]],half,BOX)
                box([bx,by,z+.018+2*half[2]],[half[0]+.002,half[1]+.002,.005],LID)
                box([bx-half[0]-.001,by,z+.04],[.001,.023,.011],[.94,.93,.87])
        for obj in objects[first:]:
            obj['pos']=(R@np.asarray(obj['pos'])+[x,y,0]).tolist()
            obj['mat']=R.ravel().tolist()
        storage.append({'kind':style,'center_xy':[x,y],'size_xyz':[2*hx,2*hy,tall],'yaw_deg':float(np.rad2deg(yaw)),'levels':levels})
    # Small labeled consumables boxes on the near table corners.
    for x,y in [(tx-.19,ty-.20),(tx-.18,ty+.22)]:
        z=top
        for level in range(1+index%2):
            box([x,y,z+.019],[.045,.035,.019],list(BOX))
            box([x,y,z+.041],[.047,.037,.003],accent)
            box([x,y-.0355,z+.02],[.025,.001,.01],[.96,.96,.92])
            z+=.044
        storage.append({'kind':'consumables_boxes','center_xy':[x,y]})
    for obj in objects[storage_start:]:obj['policy_storage']=True
    return {'name':name,'objects':objects,'benchtop_equipment':equipment,'room_geom_count':room_count,'policy_storage':storage,'storage_geom_start':storage_start,'lighting':LIGHTING[index],'material_colors_fixed':True}


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
