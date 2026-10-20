"""Compose the real-size G1 asset with an imported scene, without resizing either."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from . import ROOT, ENVIRONMENTS, imported_xml, environment_manifest, _vec

DEFAULT_REGISTRATION=ROOT/'config/environments/lidar-robot.json'


def load_registration(path=None):
    return json.loads(Path(path or DEFAULT_REGISTRATION).read_text())


def build_robot_scene(name='lidar', registration=None):
    cfg=registration or load_registration()
    root=ET.fromstring(imported_xml(name));robot_path=ROOT/'assets/robot/g1_inspire.xml'
    robot=ET.parse(robot_path).getroot()
    for asset in robot.findall('./asset/*'):
        asset=copy.deepcopy(asset)
        if asset.get('file'):asset.set('file',str((robot_path.parent/'meshes'/asset.get('file')).resolve()))
        root.find('asset').append(asset)
    for tag in ('default','actuator','equality','sensor'):
        node=robot.find(tag)
        if node is not None:root.append(copy.deepcopy(node))
    pelvis=copy.deepcopy(robot.find("./worldbody/body[@name='pelvis']"))
    for joint in list(pelvis):
        if joint.tag=='freejoint' or (joint.tag=='joint' and joint.get('type')=='free'):pelvis.remove(joint)
    manifest=environment_manifest(name)
    floor=manifest['floor']['z_m'];theta=np.deg2rad(cfg['base_yaw_deg'])/2
    pelvis.set('pos',_vec([*cfg['base_xy'],floor+cfg['pelvis_height_above_floor_m']]))
    pelvis.set('quat',_vec([np.cos(theta),0,0,np.sin(theta)]))
    for geom in pelvis.iter('geom'):
        if geom.get('group')!='1':geom.set('group','3');geom.set('contype','2');geom.set('conaffinity','1')
    root.find('worldbody').append(pelvis)
    ET.indent(root)
    return ET.tostring(root,encoding='unicode')


def apply_joint_pose(model,data,pose):
    for name,value in pose.items():
        joint=model.joint(name);data.qpos[joint.qposadr[0]]=float(value)
    data.qvel[:]=0


class MeasuredReplay:
    """Kinematic diagnostic only: static props never count as successful grasps."""
    def __init__(self,model,data,episode):
        import pyarrow.parquet as pq
        from ..scene import load_config
        self.model=model;self.data=data;self.episode=episode;self.mapping=load_config()['hand_mapping']
        path=episode.root/episode.info['data_path'].format(episode_chunk=episode.index//episode.info['chunks_size'],episode_index=episode.index)
        self.hand_sources={}
        for side in ('left','right'):
            key=f'observation.state.hand_{side}'
            if key in episode.info['features']:
                values=np.asarray(pq.read_table(path,columns=[key])[key].to_pylist(),dtype=float)
                if np.isfinite(values).all() and np.any(np.ptp(values,axis=0)>0):
                    episode.arrays[key]=values;self.hand_sources[side]=key;continue
            self.hand_sources[side]=f'action.hand_{side}'

    def pose(self,t):
        import mujoco
        e=self.episode;pose=dict(zip(e.names,e.interpolate('observation.state',t)[:29]));m=self.mapping
        for side,prefix in [('left','L'),('right','R')]:
            raw=e.interpolate(self.hand_sources[side],t);u=np.clip(raw/m['register_max'],0,1)
            for i,finger in enumerate(['pinky','ring','middle','index']):
                for segment in ['proximal','intermediate']:pose[f'{prefix}_{finger}_{segment}_joint']=(1-u[i])*m['finger_closed_rad']
            bend=(1-u[4])*m['thumb_bend_closed_rad']
            pose.update({f'{prefix}_thumb_proximal_yaw_joint':u[5]*m['thumb_rotation_max_rad'],
                         f'{prefix}_thumb_proximal_pitch_joint':bend,f'{prefix}_thumb_intermediate_joint':bend*1.6,
                         f'{prefix}_thumb_distal_joint':bend*2.4})
        apply_joint_pose(self.model,self.data,pose);self.data.time=t;mujoco.mj_forward(self.model,self.data)
