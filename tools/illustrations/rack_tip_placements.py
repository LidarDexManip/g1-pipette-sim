"""Illustrate validated placements over an uninterrupted baseline trajectory.

Object layouts come from accepted captures; robot and pipette motion stay on the
baseline clock. This presentation composite is not a new validated demonstration.
"""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', choices=['4k','1080p'], default='4k')
    parser.add_argument('--collection',type=Path,default=ROOT/'outputs/hil194/augmentation600')
    parser.add_argument('--candidate', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default='augmentation-illustration')
    parser.add_argument('--swap-seconds', type=float, default=.5)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/augmentation-illustration')
    args = parser.parse_args()
    if args.swap_seconds <= 0:
        parser.error('--swap-seconds must be positive')
    if Path(args.name).name != args.name:parser.error('name must be a filename stem')
    args.output.mkdir(parents=True, exist_ok=True)
    collection = args.collection.resolve()
    folder = collection / f'candidates/{args.candidate:06}'
    accepted = json.loads((folder / 'accepted.json').read_text())
    parent = accepted['candidate']['parent_episode']
    index_path = collection/'illustration-index.json'
    progress = json.loads((index_path if index_path.exists() else collection/'progress.json').read_text())
    from g1_pipette_sim.augmentation.sources import import_seed
    from g1_pipette_sim.augmentation.contracts import file_hash
    catalog_path = ROOT/'outputs/hil194/augmentation-catalog-v1.json'
    known = {entry['folder']:entry for entry in json.loads(catalog_path.read_text())['entries']} if catalog_path.exists() else {}
    captures = []
    for number in progress['accepted']:
        path = collection / f'candidates/{number:06}'
        record = json.loads((path / 'accepted.json').read_text())
        if record['candidate']['parent_episode'] != parent:
            continue
        if not record['validation']['pickup'] or not record['independently_replayed']:
            raise ValueError('Unvalidated capture')
        source,_,_=import_seed(path)
        expected = known.get(str(path.resolve()), {}).get('files', record.get('artifact_hashes', {}))
        if not expected:raise ValueError('Source has no capture integrity receipt; import into a catalog first')
        for name,digest in expected.items():
            if file_hash(path/name)!=digest:raise ValueError('Changed placement source: '+str(path/name))
        with np.load(path / 'trajectory.npz') as archive:
            captures.append({'number': number, 'path': path, 'record': record,
                             'qpos': archive['qpos'].copy(), 'times': archive['timestamp'].copy(),
                             'simulation_times': archive['simulation_time'].copy(),
                             'source_hashes': expected})
    baseline = next(c for c in captures if c['number'] == args.candidate)
    cfg = json.loads((baseline['path'] / 'config.json').read_text())
    model = mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg))
    data = mujoco.MjData(model)
    data.qpos[:] = baseline['qpos'][0]
    mujoco.mj_forward(model, data)
    baseline_schema=[(model.joint(j).name,int(model.jnt_type[j]),int(model.jnt_qposadr[j])) for j in range(model.njnt)]
    object_joints={'rack_free','lid_hinge','tip_free','table_free'}
    continuous=[]
    for name,kind,address in baseline_schema:
        if name not in object_joints:continuous.extend(range(address,address+(7 if kind==0 else 4 if kind==1 else 1)))
    continuous=np.asarray(continuous,dtype=int)
    for capture in captures:
        if capture['qpos'].shape!=baseline['qpos'].shape or not np.array_equal(capture['times'],baseline['times']):
            raise ValueError('Source clocks/state shapes differ')
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.body('rack').xpos + [0, 0, .10]
    camera.distance = .60
    camera.azimuth = 135
    camera.elevation = -40
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    fps = 30
    frame_count = min(len(c['times']) for c in captures)
    swap_frames = max(1, round(args.swap_seconds*fps))
    rng = np.random.default_rng(args.seed)
    order = [c for c in captures if c is not baseline]
    rng.shuffle(order)
    order.insert(0, baseline)
    width,height=(3840,2160) if args.resolution=='4k' else (1920,1080)
    scale=width/1280
    px=lambda n:round(n*scale)
    title='Visual Data Augmentation'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', px(20))
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', px(30))
    output = args.output / (args.name+'.mp4')
    temp = output.with_name(args.name+'.tmp.mp4')
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    segments = []
    try:
        for segment, start in enumerate(range(0, frame_count, swap_frames)):
            capture = order[segment % len(order)]
            config = json.loads((capture['path'] / 'config.json').read_text())
            model = mujoco.MjModel.from_xml_string(build_dynamic_xml(config))
            if [(model.joint(j).name,int(model.jnt_type[j]),int(model.jnt_qposadr[j])) for j in range(model.njnt)]!=baseline_schema:
                raise ValueError('Joint schema changed between layouts')
            data = mujoco.MjData(model)
            end = min(start+swap_frames, frame_count)
            segments.append({'candidate': capture['number'], 'start_frame': start, 'end_frame_exclusive': end,
                             'source_capture': str(capture['path']), 'source_frame_start': start})
            model.vis.global_.offwidth=width
            model.vis.global_.offheight=px(600)
            with mujoco.Renderer(model, height=px(600), width=width) as renderer:
                for frame in range(start, end):
                    if not np.isclose(capture['times'][frame], frame/fps):
                        raise ValueError('Capture clocks differ')
                    data.qpos[:] = capture['qpos'][frame]
                    data.qpos[continuous] = baseline['qpos'][frame,continuous]
                    data.time = capture['simulation_times'][frame]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas = Image.new('RGB', (width, height), '#111b29')
                    canvas.paste(Image.fromarray(renderer.render()), (0, px(76)))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((width/2, px(10)), title, anchor='mt', font=bold, fill='white')
                    row, col = capture['record']['candidate']['cell']
                    draw.text((px(24), px(688)), f"t = {frame/fps:05.2f} s", font=font, fill='#73d8bf')
                    process.stdin.write(canvas.tobytes())
                    if frame == frame_count//2:
                        canvas.save(args.output / (args.name+'.png'))
            print(f'Rendered {end}/{frame_count} frames; capture {capture["number"]}', flush=True)
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError('Video encoder failed')
    # Decode every frame before replacing the requested existing video.
    import cv2
    video = cv2.VideoCapture(str(temp))
    decoded = 0
    while True:
        ok, pixels = video.read()
        if not ok:
            break
        if pixels.shape != (height, width, 3):
            raise ValueError('Unexpected video dimensions')
        decoded += 1
    video.release()
    if decoded != frame_count:
        raise ValueError(f'Decoded {decoded} frames; expected {frame_count}')
    temp.replace(output)
    metadata = {'title':title,'resolution_wh':[width,height],'parent_episode': parent, 'seed': args.seed, 'fps': fps,
                'duration_s': frame_count/fps, 'frames_verified': decoded,
                'unique_captures': len({s['candidate'] for s in segments}),
                'swap_seconds': swap_frames/fps,
                'type': 'validated object placements composited over continuous baseline robot and pipette motion',
                'baseline_candidate':args.candidate,'robot_trajectory_cuts':False,
                'training_demonstration':False,
                'source_integrity':{str(c['path']):c['source_hashes'] for c in captures},
                'new_placements_generated': False, 'segments': segments}
    (args.output / (args.name+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
