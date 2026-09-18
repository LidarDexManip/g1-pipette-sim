"""Overlay saved validated arm perturbations; the baseline arm is invisible."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import colorsys
import json
from pathlib import Path
import subprocess
import sys
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from g1_pipette_sim.dynamics import build_dynamic_xml


def main():
    root = ROOT / 'outputs/hil194/augmentation600'
    out = ROOT / 'outputs/augmentation-illustration'
    base = root / 'candidates/000000'
    cfg = json.loads((base/'config.json').read_text())
    model = mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg))
    data = mujoco.MjData(model)
    with np.load(base/'trajectory.npz') as a:
        original = a['qpos'].copy(); times = a['timestamp'].copy()
    with np.load(base/'commands.npz') as a:
        base_times = a['times'].copy(); base_commands = a['motor_targets'].copy()
    variants = []
    for number in json.loads((root/'progress.json').read_text())['accepted']:
        path = root/f'candidates/{number:06}'
        record = json.loads((path/'accepted.json').read_text())
        c = record['candidate']
        if c['parent_episode'] != 14 or c['kind'] != 'smooth_action_and_placement':
            continue
        assert record['validation']['pickup'] and record['independently_replayed']
        with np.load(path/'trajectory.npz') as a:
            q = a['qpos'].copy()
            assert q.shape == original.shape and np.allclose(a['timestamp'], times)
        with np.load(path/'commands.npz') as a:
            assert np.array_equal(a['times'], base_times)
            deviation = np.max(np.abs(a['motor_targets'][:,22:29]-base_commands[:,22:29]), axis=1)*180/np.pi
            delta = np.interp(times, base_times, deviation)
        variants.append({'number': number, 'qpos': q, 'delta': delta, 'source': str(path)})
    # Hide the original right arm, held tool, and tip. All visible arm copies
    # below come from actual perturbed saved trajectories at this same time.
    shoulder = model.body('right_shoulder_pitch_link').id
    descendants = {shoulder, model.body('pipette').id, model.body('tip').id}
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) in descendants:
            descendants.add(b)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in descendants and model.geom_group[g] != 3:
            model.geom_group[g] = 4
    background = mujoco.MjvOption(); background.geomgroup[3:] = False
    ghosts = mujoco.MjvOption(); ghosts.geomgroup[:] = False; ghosts.geomgroup[4] = True
    ghosts.sitegroup[:] = False
    data.qpos[:] = original[0]; mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera(); camera.lookat[:] = data.body('rack').xpos+[0,0,.12]
    camera.distance = .66; camera.azimuth = 135; camera.elevation = -35
    perturb = mujoco.MjvPerturb()
    for i, v in enumerate(variants):
        v['rgb'] = colorsys.hsv_to_rgb(.46+.40*i/max(1,len(variants)-1), .65, 1.)
        v['opacity'] = .05+.15*i/max(1,len(variants)-1)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    limit = max(float(v['delta'].max()) for v in variants)*1.15
    plot = Image.new('RGB', (370, 205), '#111b29'); d = ImageDraw.Draw(plot)
    d.text((14, 10), 'Added arm target', font=font, fill='white')
    d.text((14, 37), 'Largest joint offset (degrees)', font=small, fill='#bbc9d7')
    d.line((44,68,44,173,353,173), fill='#66768a', width=1)
    d.text((4,62), f'{limit:.2f}', font=small, fill='#bbc9d7');d.text((22,162), '0', font=small, fill='#bbc9d7')
    d.text((44,182), '0 s', font=small, fill='#bbc9d7');d.text((305,182), f'{times[-1]:.1f} s', font=small, fill='#bbc9d7')
    for v in variants:
        layer = Image.new('RGBA', plot.size)
        pen = ImageDraw.Draw(layer)
        points = [(44+309*j/(len(times)-1),173-105*float(y)/limit) for j,y in enumerate(v['delta'])]
        pen.line(points, fill=tuple(int(x*255) for x in v['rgb'])+(int(255*(.3+v['opacity']*3)),), width=2)
        plot = Image.alpha_composite(plot.convert('RGBA'), layer).convert('RGB')
    dest = out/'arm-perturbations.mp4'; temp = out/'arm-perturbations.tmp.mp4'
    encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=620,width=1280,max_geom=30000) as renderer:
            for f in range(len(times)):
                data.qpos[:] = original[f]; mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=background)
                for v in variants:
                    data.qpos[:] = v['qpos'][f]; mujoco.mj_forward(model,data)
                    begin = renderer.scene.ngeom
                    mujoco.mjv_addGeoms(model,data,ghosts,perturb,mujoco.mjtCatBit.mjCAT_ALL,renderer.scene)
                    for g in range(begin, renderer.scene.ngeom):
                        renderer.scene.geoms[g].rgba[:] = [*v['rgb'],v['opacity']]
                canvas = Image.new('RGB',(1280,720),'#111b29')
                canvas.paste(Image.fromarray(renderer.render()),(0, sixty := 60))
                draw = ImageDraw.Draw(canvas)
                draw.text((24,14),'Arm perturbations → new demonstrations',font=bold,fill='white')
                inset = plot.copy(); pen=ImageDraw.Draw(inset)
                x=44+309*f/(len(times)-1);pen.line((x,68,x,173),fill='white',width=2)
                canvas.paste(inset,(890,80))
                draw.text((24,690),f'{len(variants)} validated arm variants  |  Original: 0% opacity  |  Actual motion scale',font=font,fill='#bbc9d7')
                draw.text((1090,690),f't = {times[f]:05.2f} s',font=font,fill='#73d8bf')
                encoder.stdin.write(canvas.tobytes())
                if f == len(times)//2:canvas.save(out/'arm-perturbations.png')
                if f%90 == 0:print(f'Rendered {f}/{len(times)} overlay frames',flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait():raise RuntimeError('Encoding failed')
    import cv2
    video=cv2.VideoCapture(str(temp));count=0
    while True:
        ok, frame=video.read()
        if not ok:break
        assert frame.shape==(720,1280,3);count+=1
    video.release()
    assert count==len(times)
    temp.replace(dest)
    (out/'arm-perturbations.json').write_text(json.dumps({'original_opacity':0,'source_parent_episode':14,'frames_verified':count,'fps':30,'motion_scale':1,'new_data_generated_in_this_render':False,'description':'Overlay of existing validated augmented arm states, with baseline workspace for context. Each variant was validated in its own rack placement.','variants':[{k:v for k,v in item.items() if k not in ['qpos','delta']} for item in variants]},indent=2)+'\n')
    print(dest)

if __name__=='__main__':main()
