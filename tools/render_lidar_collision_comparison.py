"""Render unchanged visual assets beside the active collision geometry."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import mujoco
import numpy as np
from g1_pipette_sim.environments import imported_xml, environment_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/lidar-teleop/collision-repair/visual-vs-collision.jpg'))
    args=parser.parse_args()
    model=mujoco.MjModel.from_xml_string(imported_xml('lidar'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    groups=model.geom_group.copy();rows=[]
    renderer=mujoco.Renderer(model,360,600,max_geom=max(10000,model.ngeom+100))
    try:
        for obj in environment_manifest('lidar')['objects']:
            bounds=np.asarray(obj['bounds_m']);camera=mujoco.MjvCamera()
            camera.lookat[:]=bounds.mean(axis=1)
            camera.distance=float(np.ptp(bounds,axis=1).max()*2.7)
            camera.azimuth=110;camera.elevation=-35
            ids=np.where(model.geom_bodyid==model.body(obj['body']).id)[0]
            model.geom_group[:]=5;model.geom_group[ids]=groups[ids]
            panels=[]
            for group,title in [(1,'Visual mesh'),(3,'Collision pieces')]:
                option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[group]=1
                if group==3:model.geom_rgba[ids[groups[ids]==3]]=[.85,.3,.12,1]
                renderer.update_scene(data,camera=camera,scene_option=option)
                image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                cv2.putText(image,title,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                cv2.putText(image,obj['body'],(15,345),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                panels.append(image)
            rows.append(np.hstack(panels))
    finally:renderer.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(args.output),np.vstack(rows))
    print(args.output)


if __name__=='__main__':main()
