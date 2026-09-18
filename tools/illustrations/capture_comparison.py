"""Render an accepted capture over the original, optionally zooming into insertion."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,subprocess,sys
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from g1_pipette_sim.dynamics import build_dynamic_xml


def main():
    p=argparse.ArgumentParser();p.add_argument('--resolution',choices=['4k','1080p'],default='4k');p.add_argument('--capture',type=Path,required=True);p.add_argument('--base',type=Path,default=ROOT/'outputs/hil194/augmentation600/candidates/000000');p.add_argument('--azimuth',type=float);p.add_argument('--elevation',type=float);p.add_argument('--distance',type=float);p.add_argument('--name',required=True);p.add_argument('--title',required=True);p.add_argument('--zoom',action='store_true');p.add_argument('--start',type=float,default=0.);p.add_argument('--end',type=float);p.add_argument('--repeat',type=int,default=1);a=p.parse_args()
    if a.repeat<1 or Path(a.name).name!=a.name:raise ValueError('Invalid render arguments')
    base=a.base.resolve();out=ROOT/'outputs/augmentation-illustration';out.mkdir(parents=True,exist_ok=True)
    sources=[base,a.capture.resolve()];captures=[]
    for folder in sources:
        from g1_pipette_sim.augmentation.sources import import_seed
        from g1_pipette_sim.augmentation.contracts import file_hash
        _,_,record=import_seed(folder)
        for name,expected in record.get('artifact_hashes',{}).items():
            if file_hash(folder/name)!=expected:raise ValueError('Changed accepted artifact: '+name)
        cfg=json.loads((folder/'config.json').read_text());model=mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg));data=mujoco.MjData(model)
        with np.load(folder/'trajectory.npz') as archive:
            captures.append({'model':model,'data':data,'qpos':archive['qpos'].copy(),'time':archive['timestamp'].copy(),'nozzle':archive['nozzle_position'].copy(),'attached':archive['tip_attached'].copy(),'record':record,'config':cfg})
    if captures[0]['record']['candidate']['parent_episode']!=captures[1]['record']['candidate']['parent_episode']:raise ValueError('Cannot compare different source episodes')
    assert np.array_equal(captures[0]['time'],captures[1]['time'])
    times=captures[0]['time'];end=float(times[-1]) if a.end is None else a.end
    indices=np.flatnonzero((times>=a.start-1e-9)&(times<=end+1e-9))
    if not len(indices):raise ValueError('Empty time interval')
    m,d=captures[0]['model'],captures[0]['data'];d.qpos[:]=captures[0]['qpos'][indices[0]];mujoco.mj_forward(m,d)
    cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-40;cam.distance=.60;cam.lookat[:]=d.body('rack').xpos+[0,0,.10]
    if a.zoom:
        cam.lookat[:]=d.body('tip').xpos+[0,0,.021]
        cam.distance=.115;cam.azimuth=45;cam.elevation=-23
    if a.azimuth is not None:cam.azimuth=a.azimuth
    if a.elevation is not None:cam.elevation=a.elevation
    if a.distance is not None:cam.distance=a.distance
    pulse_start=captures[1]['record']['candidate'].get('pulse_ramp_out_s',[11.4])[0]
    for transform in captures[1]['record']['candidate'].get('lineage',[]):
        if 'pulse_s' in transform:pulse_start=transform['pulse_s'][2]
    option=mujoco.MjvOption();option.geomgroup[3]=False
    width,height=(3840,2160) if a.resolution=='4k' else (1920,1080)
    px=lambda n:round(n*width/1280)
    for c in captures:
        c['model'].vis.global_.offwidth=width
        c['model'].vis.global_.offheight=px(620)
    fonts='/usr/share/fonts/truetype/dejavu/';font=ImageFont.truetype(fonts+'DejaVuSans.ttf',px(19));bold=ImageFont.truetype(fonts+'DejaVuSans-Bold.ttf',px(30))
    if a.zoom:
        deviation=np.linalg.norm(captures[1]['nozzle']-captures[0]['nozzle'],axis=1);preview=int(indices[np.argmax(deviation[indices])])
    else:
        deviation=np.linalg.norm(captures[1]['nozzle']-captures[0]['nozzle'],axis=1);preview=int(indices[np.argmax(deviation[indices])])
    temp=out/(a.name+'.tmp.mp4');dest=out/(a.name+'.mp4')
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    renderers=[]
    try:
        renderers=[mujoco.Renderer(c['model'],height=px(620),width=width) for c in captures]
        for f in indices:
            combined=np.zeros((px(620),width,3),dtype=np.float32)
            for c,r in zip(captures,renderers):
                c['data'].qpos[:]=c['qpos'][f];mujoco.mj_forward(c['model'],c['data']);r.update_scene(c['data'],camera=cam,scene_option=option);combined+=r.render()*.5
            im=Image.new('RGB',(width,height),'#111b29');im.paste(Image.fromarray(np.rint(combined).clip(0,255).astype(np.uint8)),(0,px(60)));draw=ImageDraw.Draw(im)
            draw.text((width/2,px(14)),a.title,anchor='mt',font=bold,fill='white')
            caption='Original: 100%  |  Augmentation: 50%'
            if a.zoom:
                state='Tip attached' if captures[1]['attached'][f] else ('Planned return to alignment' if times[f]>=pulse_start else '')
                caption+=(f'  |  {state}' if state else '')+f'  |  {1/a.repeat:g}x speed'
            draw.text((px(24),px(690)),caption,font=font,fill='#bbc9d7');draw.text((px(1090),px(690)),f't = {times[f]:05.2f} s',font=font,fill='#73d8bf')
            pixels=im.tobytes()
            for _ in range(a.repeat):encoder.stdin.write(pixels)
            if f==preview:im.save(out/(a.name+'.png'))
            if f%90==0:print('Rendered source frame',f,flush=True)
    finally:
        encoder.stdin.close()
        for r in renderers:r.close()
    if encoder.wait():raise RuntimeError('Encoding failed')
    import cv2
    v=cv2.VideoCapture(str(temp));count=0
    while True:
        ok,frame=v.read()
        if not ok:break
        assert frame.shape==(height,width,3);count+=1
    v.release();assert count==len(indices)*a.repeat;temp.replace(dest)
    (out/(a.name+'.json')).write_text(json.dumps({'title':a.title,'resolution_wh':[width,height],'source_captures':[str(s) for s in sources],'layer_opacities':[1.,.5],'source_interval_s':[float(times[indices[0]]),float(times[indices[-1]])],'playback_speed':1/a.repeat,'camera_zoom':a.zoom,'camera_distance':cam.distance,'camera_azimuth':cam.azimuth,'camera_elevation':cam.elevation,'parent_episode':captures[0]['record']['candidate']['parent_episode'],'frames_verified':count,'fps':30,'motion_scale':1},indent=2)+'\n')
    print(dest)
if __name__=='__main__':main()
