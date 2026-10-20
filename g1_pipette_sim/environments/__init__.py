"""Environment registry, independent of datasets, robot actions and task geometry.

Missing environment configuration always selects the historical pipette workspace.
Imported scenes own their dimensions and are viewed independently until a task is
explicitly registered; selecting one must never rescale a calibrated task.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import copy
import gzip
import hashlib
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from .procedural import THEMES, lab_layout

ROOT = Path(__file__).resolve().parents[2]
LAB_IDS = ('lab-open-shelves','lab-wire-racks','lab-mobile-carts','lab-cabinets',
           'lab-mixed-storage','lab-compact-storage','lab-tall-shelves','lab-wide-storage')
ENVIRONMENTS = {
    'pipette': {'kind':'workspace','description':'Original calibrated G1 pipette workspace'},
    'lidar': {'kind':'imported','description':'LIDAR lab + fixed-base G1, physical servo replay and movable pipette/tube',
              'asset':ROOT/'assets/environments/lidar/scene.xml',
              'calibration':ROOT/'config/environments/lidar-scene.json'},
    **{key:{'kind':'procedural','description':title+' (visual surroundings)', 'index':i}
       for i,(key,title) in enumerate(zip(LAB_IDS,THEMES))},
}


def specification(config):
    value=config.get('environment','pipette')
    result={'name':value} if isinstance(value,str) else dict(value)
    name=result.get('name','pipette')
    if name not in ENVIRONMENTS: raise ValueError(f'Unknown environment {name!r}; choose from {", ".join(ENVIRONMENTS)}')
    result['name']=name
    return result


def add_surroundings(root, config):
    """Attach decorations under a static body, never to the movable task table."""
    spec=specification(config);entry=ENVIRONMENTS[spec['name']]
    if entry['kind']=='workspace':return
    if entry['kind']=='imported':
        raise ValueError('LIDAR is a standalone scene. Use ./run.sh --environment lidar for physical replay, or add --physics kinematic for measured-position diagnostics.')
    layout=lab_layout(entry['index'],int(spec.get('seed',20260914)),config)
    world=root.find('worldbody')
    backdrop=world.find("geom[@name='backdrop']")
    if backdrop is not None: world.remove(backdrop)
    body=ET.SubElement(world,'body',name='environment_surroundings')
    for i,obj in enumerate(layout['objects']):
        size=obj['size'][:2] if obj['kind']=='cylinder' else obj['size']
        attrs={'name':f'environment_decor_{i}','type':obj['kind'],'pos':_vec(obj['pos']),
               'size':_vec(size),'rgba':_vec(obj['rgba']),'contype':'0','conaffinity':'0','group':'1'}
        if 'mat' in obj:
            q=Rotation.from_matrix(np.array(obj['mat']).reshape(3,3)).as_quat()
            attrs['quat']=_vec(q[[3,0,1,2]])
        ET.SubElement(body,'geom',**attrs)
    light=world.find('light');lighting=layout['lighting']
    if light is not None:
        light.set('diffuse',_vec(np.array(lighting['tint'])*lighting['intensity']))
        light.set('pos',_vec(lighting['position']))


def _vec(values):return ' '.join(f'{float(x):.9g}' for x in values)


def unpack_meshes(asset_path):
    """Extract bundled gzip meshes atomically, checking the export manifest."""
    directory=asset_path.parent
    manifest=json.loads(asset_path.with_name('manifest.json').read_text())
    for compressed in (directory/'meshes').glob('*.gz'):
        target=compressed.with_suffix('')
        if target.exists():continue
        expected=manifest['artifacts'][target.relative_to(directory).as_posix()]['sha256']
        temporary=None
        try:
            digest=hashlib.sha256()
            with gzip.open(compressed,'rb') as source, tempfile.NamedTemporaryFile(
                    dir=target.parent,prefix='.unpack-',delete=False) as output:
                temporary=Path(output.name)
                while chunk:=source.read(1024*1024):
                    digest.update(chunk);output.write(chunk)
            if digest.hexdigest()!=expected:
                raise ValueError(f'Compressed mesh does not match export manifest: {compressed}')
            temporary.replace(target)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)


def imported_xml(name):
    entry=ENVIRONMENTS[name]
    if entry['kind']!='imported':raise ValueError(f'{name} is not an imported scene')
    path=entry['asset']
    if not path.exists():raise FileNotFoundError(f'{path}: first run tools/export_blender_environment.py with Blender')
    unpack_meshes(path)
    root=ET.parse(path).getroot();compiler=root.find('compiler')
    for key in ('meshdir','texturedir'):
        if compiler.get(key):compiler.set(key,str((path.parent/compiler.get(key)).resolve()))
    calibration=ENVIRONMENTS[name].get('calibration')
    if calibration and calibration.exists():
        cfg=json.loads(calibration.read_text());scale=float(cfg['uniform_scale'])
        if not np.isfinite(scale) or scale<=0:raise ValueError('Environment scale must be positive and finite')
        raw=json.loads(path.with_name('manifest.json').read_text())
        offset=np.array([0.,0.,-raw['floor']['z_m']*scale if cfg.get('floor_to_zero') else 0.])
        for mesh in root.findall('./asset/mesh'):
            mesh.set('scale',_vec(np.fromstring(mesh.get('scale','1 1 1'),sep=' ')*scale))
        world=root.find('worldbody')
        for node in world.iter():
            if node is world:continue
            if node.get('pos'):node.set('pos',_vec(np.fromstring(node.get('pos'),sep=' ')*scale))
            if node.tag=='geom' and node.get('size'):node.set('size',_vec(np.fromstring(node.get('size'),sep=' ')*scale))
        for node in world:
            if node.tag in ('body','geom','light','camera'):
                node.set('pos',_vec(np.fromstring(node.get('pos','0 0 0'),sep=' ')+offset))
        stat=root.find('statistic')
        if stat is not None:
            stat.set('center',_vec(np.fromstring(stat.get('center','0 0 0'),sep=' ')*scale+offset))
            stat.set('extent',str(float(stat.get('extent','1'))*scale))
    return ET.tostring(root,encoding='unicode')


def environment_manifest(name):
    entry=ENVIRONMENTS[name];result=json.loads(entry['asset'].with_name('manifest.json').read_text())
    calibration=entry.get('calibration')
    if not calibration or not calibration.exists():return result
    cfg=json.loads(calibration.read_text());scale=cfg['uniform_scale']
    offset=np.array([0.,0.,-result['floor']['z_m']*scale if cfg.get('floor_to_zero') else 0.])
    result['source_bounds_m']=copy.deepcopy(result['bounds_m'])
    result['bounds_m']=(np.asarray(result['bounds_m'])*scale+offset[:,None]).tolist()
    for obj in result['objects']:
        obj['source_bounds_m']=copy.deepcopy(obj['bounds_m'])
        obj['bounds_m']=(np.asarray(obj['bounds_m'])*scale+offset[:,None]).tolist()
        obj['dimensions_world_m']=(np.asarray(obj['dimensions_world_m'])*scale).tolist()
    result['floor']['z_m']=result['floor']['z_m']*scale+offset[2]
    result['calibration']=cfg
    result['coordinates']='Calibrated meters, Z up, floor at zero; source mesh files retain original coordinates.'
    result['artifacts_note']='Artifact hashes refer to raw source exports before runtime calibration.'
    return result
