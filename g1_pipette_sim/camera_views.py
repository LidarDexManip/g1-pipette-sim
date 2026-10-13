"""Dataset-bound camera-view corrections applied after physics-layout selection."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
from .scene import ROOT

OVERRIDES=ROOT/'config/camera-view-overrides.json'

def _dataset_identity(root):
    path=Path(root).resolve()
    if path.parent.name=='snapshots' and path.parent.parent.name.startswith('datasets--'):
        return (path.parent.parent.name,path.name)
    return str(path)


def apply_camera_views(config,episode,path=OVERRIDES):
    path=Path(path)
    if not path.exists():return config
    spec=json.loads(path.read_text())
    root=str(episode.root.resolve())
    identity=_dataset_identity(root)
    if spec.get('dataset_snapshot') is None or _dataset_identity(spec['dataset_snapshot'])!=identity:
        profile=next((v for k,v in spec.get('dataset_profiles',{}).items() if _dataset_identity(k)==identity),None)
        if profile is None:return config
        spec={**spec,**profile}
    views=copy.deepcopy(spec.get('defaults',{}))
    scopes=['dataset_default'] if views else []
    sessions=[s for s in spec.get('sessions',[]) if s['start']<=episode.index<s['end']]
    if len(sessions)>1:raise ValueError('Overlapping camera correction sessions')
    for session in sessions:
        for name,pose in session.get('views',{}).items():views[name]={**views.get(name,{}),**pose}
        scopes.append('session:'+session['name'])
    specific=spec.get('episodes',{}).get(str(episode.index),{})
    for name,pose in specific.items():views[name]={**views.get(name,{}),**pose}
    if specific:scopes.append('episode:'+str(episode.index))
    if not views:return config
    result=copy.deepcopy(config)
    for name,pose in views.items():
        if name not in result['cameras']:raise ValueError(f'Unknown camera view: {name}')
        if set(pose)-{'position','quaternion','fovy'}:raise ValueError('Camera-view corrections may only change position, quaternion and fovy')
        position=np.asarray(pose['position'],float);quaternion=np.asarray(pose['quaternion'],float)
        if position.shape!=(3,) or quaternion.shape!=(4,) or not np.isfinite(np.r_[position,quaternion]).all() or not np.isclose(np.linalg.norm(quaternion),1.) or not 0<float(pose['fovy'])<180:raise ValueError('Invalid camera-view pose')
        result['cameras'][name].update(copy.deepcopy(pose))
    result['camera_view_correction']={'source':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scopes':scopes,'episode':episode.index,'views':sorted(views),'note':spec.get('note','')}
    return result


def assert_camera_views(config, episode, path=OVERRIDES):
    """Reject stale dataset-bound cameras before writing a new capture."""
    expected = apply_camera_views(config, episode, path)
    if expected['cameras'] != config['cameras']:
        raise ValueError('Stale camera settings: apply dataset/session camera overrides before recording')
