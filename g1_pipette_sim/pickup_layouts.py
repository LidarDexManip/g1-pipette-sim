"""Select a dataset-bound, hash-checked per-episode replay layout."""
import hashlib,json
from pathlib import Path
from .scene import ROOT

MANIFEST=ROOT/('.local/pickup-layouts.json' if (ROOT/'.local/pickup-layouts.json').exists() else 'config/hil194-pickup-layouts.json')
REVIEWS=ROOT/('.local/recorded-reviews.json' if (ROOT/'.local/recorded-reviews.json').exists() else 'config/hil194-recorded-reviews.json')
MODEL_FILES=['g1_pipette_sim/'+name+'.py' for name in ['dynamics','scene','simulation','attachment','grasp_initialization','object_meshes','pipette_mesh','box_lid','profiles','dataset']]+['assets/robot/g1_inspire.xml','config/hil194.json']

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def model_fingerprints():return {name:sha256(ROOT/name) for name in MODEL_FILES}

def select_pickup_layout(config,episode,manifest_path=MANIFEST):
    path=Path(manifest_path)
    if not path.exists():return config,None
    manifest=json.loads(path.read_text())
    if manifest['dataset_snapshot']!=str(episode.root.resolve()):return config,None
    if manifest['model_fingerprints']!=model_fingerprints():
        raise ValueError('Saved pickup layouts were validated with a different model/config. Revalidate them or use --scene-layout visual for an unfitted replay')
    entry=manifest['episodes'].get(str(episode.index))
    if entry is None:
        if not manifest.get('complete',True):return config,None
        raise ValueError(f'No pickup search result for episode {episode.index}; use --scene-layout visual')
    if not entry.get('config'):raise ValueError(f"Episode {episode.index} has no collision-screened layout ({entry['status']}); use --scene-layout visual for diagnostics")
    config_path=ROOT/entry['config']
    if sha256(config_path)!=entry['sha256']:raise ValueError(f'Pickup config was modified since validation: {config_path}')
    # Object meshes are a generated cache, excluded from the source repository.
    if any(not (ROOT/name).exists() for name in entry.get('asset_fingerprints',{})):
        from .dynamics import build_dynamic_xml
        build_dynamic_xml(json.loads(config_path.read_text()))
    for name,expected in entry.get('asset_fingerprints',{}).items():
        asset=ROOT/name
        if not asset.exists() or sha256(asset)!=expected:raise ValueError(f'Mesh asset changed or missing since validation: {asset}. Revalidate or use --scene-layout visual')
    fitted=json.loads(config_path.read_text());provenance=fitted['layout_search']
    if provenance['episode']!=episode.index or Path(provenance['dataset']).resolve()!=episode.root.resolve() or provenance['source']!='low_level' or provenance['clock']!='recorded':
        raise ValueError('Pickup config episode/dataset/source/clock provenance mismatch')
    return fitted,{**entry,'path':str(config_path),'recorded_review':recorded_reviews(episode.root).get(str(episode.index),{})}

def recommended_episode(dataset_root,manifest_path=MANIFEST):
    """Choose a verified example for an otherwise unspecified default replay."""
    path=Path(manifest_path)
    if not path.exists():return 0
    manifest=json.loads(path.read_text())
    if manifest.get('dataset_snapshot')!=str(Path(dataset_root).resolve()):return 0
    reviews=recorded_reviews(dataset_root)
    verified=[e for e in manifest['episodes'].values() if e.get('status')=='verified_pickup' and e.get('config') and not reviews.get(str(e['episode']),{}).get('exclude_from_default',False)]
    if not verified:return 0
    # Prefer attachment during the recorded movement over initialization settling.
    def preference(entry):
        t=next((event['time'] for event in entry.get('validation',{}).get('events',[]) if event['state']=='attached'),0)
        index=int(entry['episode'])
        # HIL194 episodes 46–193 are the newer recording session.
        return (t<1,reviews.get(str(index),{}).get('recorded_pickup_visible') is not True,index<46,index)
    return int(min(verified,key=preference)['episode'])

def asset_fingerprints_from_xml(xml):
    """Hash actual model dependencies without regenerating or modifying meshes."""
    import xml.etree.ElementTree as ET
    tree=ET.fromstring(xml);compiler=tree.find('compiler')
    meshdir=Path(compiler.get('meshdir','')) if compiler is not None else ROOT
    paths={ROOT/'g1_pipette_sim/photo_objects.py'}
    for node in tree.findall('./asset/mesh'):
        if node.get('file'):paths.add((meshdir/node.get('file')).resolve())
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):sha256(path) for path in sorted(paths)}

def recorded_reviews(dataset_root,path=REVIEWS):
    if not Path(path).exists():return {}
    data=json.loads(Path(path).read_text())
    if data.get('dataset_snapshot')!=str(Path(dataset_root).resolve()):return {}
    return data.get('episodes',{})
