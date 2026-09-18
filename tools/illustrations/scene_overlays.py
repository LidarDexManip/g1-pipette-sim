"""Blend complete accepted scenes on a shared clock over the original capture."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import argparse
from pathlib import Path
import sys
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from g1_pipette_sim.dynamics import build_dynamic_xml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',action='store_true');parser.add_argument('--single-max',action='store_true');parser.add_argument('--joint',action='store_true');args=parser.parse_args()
    if args.joint:args.single_max=True
    root=ROOT/'outputs/hil194/augmentation600';out=ROOT/'outputs/augmentation-illustration'
    baseline=root/'candidates/000000'
    original_config=json.loads((baseline/'config.json').read_text())
    captures=[]
    sources=[root/f'candidates/{number:06}' for number in json.loads((root/'progress.json').read_text())['accepted']]
    if args.study:
        available=[]
        for marker in (ROOT/'outputs/hil194/larger-augmentation-study/candidates').glob('*/accepted.json'):
            r=json.loads(marker.read_text())
            if r['validation']['pickup']:available.append((marker.parent,r))
        available.sort(key=lambda item:-np.linalg.norm(item[1]['candidate']['target_shift_m']))
        chosen=[];seen=set()
        for folder,r in available:
            shift=tuple(r['candidate']['target_shift_m'])
            if shift in seen:continue
            seen.add(shift);chosen.append(folder)
            if len(chosen)==4:break
        if not chosen:raise ValueError('No validated larger augmentations')
        sources=[baseline,*chosen]
    selection=None
    if args.single_max:
        with np.load(baseline/'trajectory.npz') as a:
            reference=a['observation.state'][:,22:29].copy();reference_times=a['timestamp'].copy();reference_nozzle=a['nozzle_position'].copy()
        ranked=[]
        for collection in ([ROOT/'outputs/hil194/joint-perturbation-study'] if args.joint else [root,ROOT/'outputs/hil194/larger-augmentation-study']):
            for marker in (collection/'candidates').glob('*/accepted.json'):
                record=json.loads(marker.read_text())
                if record['candidate']['parent_episode']!=14 or not record['validation']['pickup'] or not record['independently_replayed']:continue
                with np.load(marker.parent/'trajectory.npz') as a:
                    if not np.array_equal(a['timestamp'],reference_times):raise ValueError('Capture clocks differ')
                    maximum=(float(1000*np.max(np.linalg.norm(a['nozzle_position']-reference_nozzle,axis=1))) if args.joint else float(np.degrees(np.max(np.abs(a['observation.state'][:,22:29]-reference)))))
                ranked.append((maximum,marker.parent))
        if not ranked:raise ValueError('No independently accepted captures for this comparison')
        maximum,selected=max(ranked,key=lambda item:item[0])
        sources=[baseline,selected]
        selection={'metric':('maximum nozzle path separation in millimeters' if args.joint else 'maximum absolute measured right-arm joint deviation over the synchronized trajectory, degrees'),'value':maximum,'selected_capture':str(selected)}
        if args.joint:
            with np.load(selected/'trajectory.npz') as a:preview_frame=int(np.argmax(np.linalg.norm(a['nozzle_position']-reference_nozzle,axis=1)))
        print(json.dumps(selection),flush=True)
    for folder in sources:
        record=json.loads((folder/'accepted.json').read_text());number=record['candidate']['number']
        if record['candidate']['parent_episode']!=14:continue
        assert record['validation']['pickup'] and record['independently_replayed']
        cfg=json.loads((folder/'config.json').read_text())
        # Each capture keeps its own model, including its rack/lid configuration.
        model=mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg));data=mujoco.MjData(model)
        with np.load(folder/'trajectory.npz') as a:
            q=a['qpos'].copy();times=a['timestamp'].copy()
        captures.append({'number':number,'model':model,'data':data,'qpos':q,'times':times})
    captures.sort(key=lambda c:(c['number']!=0,c['number']))
    base=captures[0];base['data'].qpos[:]=base['qpos'][0];mujoco.mj_forward(base['model'],base['data'])
    camera=mujoco.MjvCamera();camera.lookat[:]=base['data'].body('rack').xpos+[0,0,.10]
    camera.distance=.60;camera.azimuth=135;camera.elevation=-40
    option=mujoco.MjvOption();option.geomgroup[3]=False
    n=len(base['times'])
    if not args.joint:preview_frame=n//2
    for c in captures:assert np.array_equal(c['times'],base['times'])
    # Linear image blending preserves the common background and gives the
    # original a strong visible anchor; all other full scenes contribute equally.
    weights=([.5,.5] if args.single_max else [.10]+[.90/(len(captures)-1)]*(len(captures)-1))
    # A 50%-alpha full-frame foreground over a 100%-opaque background
    # produces 0.5*foreground + 0.5*background (not additive exposure).
    fonts='/usr/share/fonts/truetype/dejavu/'
    bold=ImageFont.truetype(fonts+'DejaVuSans-Bold.ttf',30)
    font=ImageFont.truetype(fonts+'DejaVuSans.ttf',19)
    stem='joint-perturbation-example' if args.joint else ('original-plus-max-augmentation' if args.single_max else 'arm-perturbations')
    temp=out/(stem+'.tmp.mp4');dest=out/(stem+'.mp4')
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    renderers=[]
    try:
        for c in captures:
            renderers.append(mujoco.Renderer(c['model'],height=620,width=1280))
        for f in range(n):
            combined=np.zeros((620,1280,3),dtype=np.float32)
            for c,r,weight in zip(captures,renderers,weights):
                c['data'].qpos[:]=c['qpos'][f];mujoco.mj_forward(c['model'],c['data'])
                r.update_scene(c['data'],camera=camera,scene_option=option)
                combined+=r.render().astype(np.float32)*weight
            canvas=Image.new('RGB',(1280,720),'#111b29')
            canvas.paste(Image.fromarray(np.clip(np.rint(combined),0,255).astype(np.uint8)),(0,60))
            d=ImageDraw.Draw(canvas);d.text((24,14),'Direct joint perturbation: original + successful variant' if args.joint else ('Original + largest validated augmentation' if args.single_max else ('Larger arm perturbations + matched placements' if args.study else 'Original + augmented demonstrations')),font=bold,fill='white')
            d.text((24,690),f'Original: 100% opacity  |  Augmentation: 50% opacity  |  Capture {captures[1]["number"]}' if args.single_max else f'Original + {len(captures)-1} full-scene overlays',font=font,fill='#bbc9d7')
            d.text((1090,690),f't = {f/30:05.2f} s',font=font,fill='#73d8bf')
            encoder.stdin.write(canvas.tobytes())
            if f in [0,preview_frame,n-1]:canvas.save(out/(stem+'.png' if f==preview_frame else f'{stem}-frame-{f:04}.png'))
            if f%30==0:print(f'Rendered {f}/{n} full-scene overlay frames',flush=True)
    finally:
        encoder.stdin.close()
        for r in renderers:r.close()
    if encoder.wait():raise RuntimeError('Video encoding failed')
    import cv2
    video=cv2.VideoCapture(str(temp));count=0
    while True:
        ok,frame=video.read()
        if not ok:break
        assert frame.shape==(720,1280,3);count+=1
    video.release();assert count==n
    temp.replace(dest)
    (out/(stem+'.json')).write_text(json.dumps({'type':'pixel blend of complete validated scenes, aligned by saved frame timestamp','selection':selection,'layer_opacities':{'original':1.0,'augmentation':0.5} if args.single_max else None,'original_candidate':0,'original_weight':weights[0],'augmented_weight_each':weights[1],'candidates':[c['number'] for c in captures],'source_folders':[str(p) for p in sources],'frames_verified':count,'fps':30,'motion_scale':1},indent=2)+'\n')
    print(dest)

if __name__=='__main__':main()
