"""Clean, synchronized RGB observations and per-camera video export.

Names retain their configured meaning: no automatic ego/side/wrist reassignment.
No policy inference or hardware/network connections are made.
"""
import argparse,json,os
from contextlib import ExitStack
from pathlib import Path
import numpy as np

class CameraStreams:
    """Call render on the same simulation thread, without stepping between views.

    Returns {camera_name: uint8 RGB[height,width,3]}; no HUD or policy resizing.
    Set MUJOCO_GL before importing mujoco when running headless.
    """
    def __init__(self,model,names=('rgb','wrist_left','wrist_right'),width=640,height=480):
        import mujoco
        if width<=0 or height<=0:raise ValueError('Positive image dimensions required')
        self.model=model;self.names=tuple(names);self.option=mujoco.MjvOption();self.option.geomgroup[3]=False
        if not self.names or len(set(self.names))!=len(self.names):raise ValueError('Camera names must be nonempty and unique')
        for name in self.names:
            if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,name)<0:raise ValueError(f'Unknown camera: {name}')
        self.renderer=mujoco.Renderer(model,width=width,height=height)
    def render(self,data):
        result={}
        for name in self.names:
            self.renderer.update_scene(data,camera=name,scene_option=self.option)
            result[name]=self.renderer.render().copy()
        return result
    def poses(self,data):
        return {name:{'position_world_m':data.cam_xpos[self.model.camera(name).id].tolist(),
                      'rotation_camera_to_world':data.cam_xmat[self.model.camera(name).id].reshape(3,3).tolist(),
                      'axes':'MuJoCo/OpenGL: +x right, +y up, -z forward'} for name in self.names}
    def close(self):self.renderer.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/hil194.json')
    p.add_argument('--dataset',type=Path,default=None)
    p.add_argument('--camera-pose',choices=['corrected','config'],default='corrected',help='Apply dataset/session view corrections (default), or retain selected configuration cameras.')
    p.add_argument('--physics',choices=['kinematic','dynamic'],default='dynamic')
    p.add_argument('--scene-layout',choices=['auto','fixed'],default='auto',help='Use the episode-specific visual scene or the unchanged config.')
    p.add_argument('--episode',type=int,default=0);p.add_argument('--duration',type=float,default=5)
    p.add_argument('--source',choices=['state','action','low_level'],default='low_level')
    p.add_argument('--views',nargs='+',default=['rgb','wrist_left','wrist_right'])
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480);p.add_argument('--fps',type=float,default=30)
    p.add_argument('--native-resolution',action='store_true',help='Use per-camera resolution_wh from the scene config')
    p.add_argument('--output',type=Path,default=Path('outputs/camera-streams'));a=p.parse_args()
    if a.fps<=0 or a.duration<=0:p.error('fps and duration must be positive')
    os.environ.setdefault('MUJOCO_GL','egl')
    import cv2
    from .scene import load_config
    from .dataset import Episode, DEFAULT_DATASET
    from .simulation import Simulation
    from .dynamics import DynamicSimulation
    cfg=load_config(a.config);episode=Episode(root=a.dataset or DEFAULT_DATASET,index=a.episode)
    from .episode_scene import resolve_scene
    cfg,_=resolve_scene(cfg,episode,a.scene_layout)
    from .profiles import for_episode
    from .camera_views import apply_camera_views
    cfg=for_episode(cfg,episode)
    if a.camera_pose=='corrected':cfg=apply_camera_views(cfg,episode)
    cls=DynamicSimulation if a.physics=='dynamic' else Simulation
    sim=cls(cfg,episode,source=a.source)
    a.output.mkdir(parents=True,exist_ok=True);writers={};frames=[]
    limit=min(a.duration,sim.duration)
    sizes={name:tuple(cfg['cameras'][name]['resolution_wh']) if a.native_resolution else (a.width,a.height) for name in a.views}
    try:
        with ExitStack() as stack:
            cameras={name:stack.enter_context(CameraStreams(sim.model,[name],*sizes[name])) for name in a.views}
            for name in a.views:
                if Path(name).name!=name:raise ValueError('Camera names must be plain filenames for export')
                writer=cv2.VideoWriter(str(a.output/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,sizes[name])
                if not writer.isOpened():raise RuntimeError(f'Cannot open video writer: {name}')
                writers[name]=writer
            for index,target in enumerate(np.arange(0,limit,1/a.fps)):
                while sim.time+1e-10<target:sim.step()
                rgb={name:camera.render(sim.data)[name] for name,camera in cameras.items()}
                for name,pixels in rgb.items():
                    bgr=cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR);writers[name].write(bgr)
                    if index==0:cv2.imwrite(str(a.output/(name+'.png')),bgr)
                frames.append({'frame':index,'simulation_time_s':sim.time,'cameras':{name:camera.poses(sim.data)[name] for name,camera in cameras.items()}})
    finally:
        for writer in writers.values():writer.release()
    metadata={'dataset':str(episode.root),'physics':a.physics,'episode':a.episode,'source':a.source,'views':a.views,'resolution_wh':sizes,'fps':a.fps,
              'projection':'pinhole; match real intrinsics/aspect/crops before policy use','camera_configuration':cfg['cameras'],'camera_view_correction':cfg.get('camera_view_correction'),'frames':frames}
    (a.output/'streams.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Wrote {len(frames)} synchronized frames per view to {a.output.resolve()}')

if __name__=='__main__':main()
