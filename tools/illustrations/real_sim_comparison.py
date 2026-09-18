"""Rebuild a saved real/sim comparison at source-quality resolution, without labels."""
import argparse,json,os,sys,subprocess,hashlib
from pathlib import Path
from contextlib import ExitStack
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,mujoco,numpy as np
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.video import VideoFrames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=ROOT/'outputs/episode14-comparison-ego-fixed')
    p.add_argument('--physics',choices=['dynamic','kinematic'],default='dynamic')
    p.add_argument('--source',choices=['low_level','state'],help='Override saved source. Dynamic state replay uses measured positions as servo targets.')
    p.add_argument('--cell-width',type=int,default=1080,help='Pixels per camera column; default produces 3240x2160.')
    a=p.parse_args();root=a.folder.resolve();size=a.cell_width
    if size<2 or size%2:p.error('cell-width must be an even positive integer')
    previous=json.loads((root/'report.json').read_text());cfg=json.loads((root/'workspace.json').read_text())
    if previous.get('mode')!='replay' or previous.get('physics')!='dynamic':raise ValueError('Requires a saved dynamic replay')
    ep=Episode(previous['dataset'],previous['episode'],previous['timing']['clock'])
    source=a.source or previous['source']
    if a.physics=='kinematic':
        if source!='state':raise ValueError('Kinematic diagnostic requires measured state')
        from g1_pipette_sim.simulation import Simulation
        from tools.illustrations.kinematic_diagnostics import KinematicDiagnostics
        sim=Simulation(cfg,ep,'replay',source);diagnostic=KinematicDiagnostics(sim)
    else:sim=DynamicSimulation(cfg,ep,'replay',source,False);diagnostic=None
    names=['rgb','wrist_left','wrist_right'];caps={};sizes={}
    for name in names:
        cap=VideoFrames(ep.video_path(name));caps[name]=cap
        frame=cap.read(0);h,w=frame.shape[:2];sizes[name]=(size,round(size*h/w))
        if sizes[name][1]>size:raise ValueError('Portrait camera requires taller cells')
        if cfg['cameras'][name].get('rectification'):raise ValueError('Rectified comparisons require explicit matching image rectification')
    sim.model.vis.global_.offwidth=max(size,sim.model.vis.global_.offwidth)
    sim.model.vis.global_.offheight=max(size,sim.model.vis.global_.offheight)
    option=mujoco.MjvOption();option.geomgroup[3]=False
    width,height=size*3,size*2;fps=30;count=0;next_frame=0.;limit=previous['time'];source_sizes={name:[int(cap.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))] for name,cap in caps.items()}
    temp=root/'simulation.highres.tmp.mp4';output=root/'simulation.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    times=[]
    try:
        with ExitStack() as stack:
            renders={name:stack.enter_context(mujoco.Renderer(sim.model,width=wh[0],height=wh[1])) for name,wh in sizes.items()}
            while sim.time<limit-1e-9:
                sim.step()
                if sim.time<next_frame-1e-9:continue
                if diagnostic:diagnostic.sample(count)
                canvas=np.zeros((height,width,3),dtype=np.uint8);index=ep.frame_at(sim.time)
                for col,name in enumerate(names):
                    real=caps[name].read(index);real=cv2.resize(real,sizes[name],interpolation=cv2.INTER_AREA if sizes[name][0]<=real.shape[1] else cv2.INTER_CUBIC)
                    h=sizes[name][1];x=col*size
                    canvas[:h,x:x+size]=cv2.cvtColor(real,cv2.COLOR_BGR2RGB)
                    renderer=renders[name];renderer.update_scene(sim.data,camera=name,scene_option=option)
                    canvas[size:size+h,x:x+size]=renderer.render()
                process.stdin.write(canvas.tobytes());times.append({'frame':count,'simulation_time_s':sim.time,'real_frame':index})
                if count==previous['video_frames']//2:cv2.imwrite(str(root/'highres-preview.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                count+=1;next_frame=count/fps
                if count%90==0:print(f'Rendered {count}/{previous["video_frames"]}',flush=True)
    finally:
        process.stdin.close()
        for cap in caps.values():cap.release()
    if process.wait():raise RuntimeError('Encoding failed')
    report=sim.report();report['physics']=a.physics
    if diagnostic:
        report['kinematic_diagnostics']=diagnostic.report()
        (root/'kinematic-diagnostics.json').write_text(json.dumps(report['kinematic_diagnostics'],indent=2)+'\n')
    differences=[field for field in ['events','state','contact_diagnostics'] if report.get(field)!=previous.get(field)]
    if source==previous['source'] and a.physics==previous['physics'] and differences:raise ValueError('Replay drift: '+', '.join(differences)+'; original video retained')
    if not report['finite'] or count!=previous['video_frames']:raise ValueError('Incomplete or invalid replay')
    video=cv2.VideoCapture(str(temp));decoded=0
    if abs(video.get(cv2.CAP_PROP_FPS)-fps)>.01:raise ValueError('Unexpected FPS')
    while True:
        ok,pixels=video.read()
        if not ok:break
        if pixels.shape!=(height,width,3):raise ValueError('Unexpected frame size')
        decoded+=1
    video.release()
    if decoded!=count:raise ValueError('Incomplete video')
    old_hash=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None;temp.replace(output)
    audit={'video_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'resolution_wh':[width,height],'fps':fps,'frames_verified':count,'duration_s':count/fps,'source_video_resolutions':source_sizes,'render_sizes':sizes,'text_overlays':False,'label_bands':False,'source':source,'physics':a.physics,'reference_source':previous['source'],'differences_from_reference':differences,'reference_report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),'replay_events_state_contacts_match':not differences,'previous_video_sha256':old_hash,'workspace_sha256':hashlib.sha256((root/'workspace.json').read_bytes()).hexdigest(),'frame_alignment':times,'replay_report':report}
    (root/'highres-render.json').write_text(json.dumps(audit,indent=2)+'\n');print(output,flush=True)


if __name__=='__main__':main()
