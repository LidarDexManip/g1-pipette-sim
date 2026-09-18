"""Check the completed failure-review artifacts against their recording certificates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from g1_pipette_sim.camera_views import apply_camera_views, OVERRIDES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(output):
    progress=json.loads((output/'recordings.json').read_text())
    manifest=json.loads((output/'input-manifest.json').read_text())
    expected={int(i) for i,e in manifest['episodes'].items() if e['status']!='verified_pickup'}
    records=progress['recordings'];seen={r['episode'] for r in records}
    if not progress['complete'] or progress['errors'] or seen!=expected or len(records)!=len(expected):
        raise ValueError(f'Incomplete recording set: {len(seen)}/{len(expected)}; errors={progress["errors"]}')
    if progress['contract']['manifest_sha256']!=sha(output/'input-manifest.json'):
        raise ValueError('Input manifest changed')
    if progress['contract'].get('camera_overrides_sha256')!=sha(OVERRIDES):
        raise ValueError('Camera correction changed since recording')
    frames=0;duration=0.;bytes_total=0;discrepancies=[]
    for r in records:
        folder=output/'episodes'/f'episode-{r["episode"]:03d}'
        saved=json.loads((folder/'recording.json').read_text())
        if saved!=r or saved['contract']!=progress['contract']:
            raise ValueError(f'Recording certificate mismatch: episode {r["episode"]}')
        video=output/r['video'];report=json.loads((folder/'report.json').read_text())
        if sha(video)!=r['video_sha256'] or not report['finite']:
            raise ValueError(f'Corrupt video/nonfinite replay: episode {r["episode"]}')
        if report['video_frames']!=r['frames']:
            raise ValueError(f'Encoded and generated frame counts differ: episode {r["episode"]}')
        if abs(report['time']-r['duration_s'])>.1:
            raise ValueError('Video/physics duration mismatch')
        workspace=json.loads((folder/'workspace.json').read_text())
        correction=workspace.get('camera_view_correction',{})
        expected_view=apply_camera_views(workspace,SimpleNamespace(root=Path(manifest['dataset_snapshot']),index=r['episode']))
        if (correction!=r.get('camera_view_correction')
                or correction.get('source_sha256')!=sha(OVERRIDES)
                or workspace['cameras']['rgb']!=r.get('ego_camera')
                or workspace['cameras']['rgb']!=expected_view['cameras']['rgb']):
            raise ValueError(f'Incorrect ego camera metadata: episode {r["episode"]}')
        camera=ET.parse(folder/'scene.xml').find('.//camera[@name="rgb"]')
        if camera is None:
            raise ValueError('Rendered ego camera is missing')
        for attribute,key in [('pos','position'),('quat','quaternion'),('fovy','fovy')]:
            if not np.allclose(np.fromstring(camera.attrib[attribute],sep=' '),workspace['cameras']['rgb'][key],rtol=0,atol=1e-6):
                raise ValueError(f'Rendered ego camera differs from corrected pose: episode {r["episode"]}')
        if r['replay_discrepancies']:discrepancies.append(r['episode'])
        frames+=r['frames'];duration+=r['duration_s'];bytes_total+=video.stat().st_size
    for sub in [output,output/'without-gate-fractions']:
        labels=json.loads((sub/'episodes.json').read_text())
        if {r['episode'] for r in labels}!=expected or len(labels)!=len(expected):
            raise ValueError('Clustering episode coverage mismatch')
        if not all((sub/r['video']).is_file() for r in labels):
            raise ValueError('Broken video links')
    result={'complete':True,'episodes':len(expected),'video_count':len(records),'total_frames':frames,
        'total_video_duration_s':duration,'total_video_bytes':bytes_total,
        'all_videos_fully_decoded_during_recording':True,'all_final_video_hashes_checked':True,
        'generated_frames_equal_encoded_frames':True,'all_clustering_video_links_exist':True,
        'replay_discrepancy_episodes':discrepancies,
        'corrected_ego_camera_episodes':len(records),'all_rendered_ego_camera_poses_checked':True,
        'camera_overrides_sha256':sha(OVERRIDES),
        'artifact_sha256':{str(p.relative_to(output)):sha(p) for p in [output/'input-manifest.json',
            output/'clustering-summary.json',output/'failure-features.csv',output/'without-gate-fractions/clustering-summary.json']}}
    (output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/hil194/failure-analysis')
    audit(p.parse_args().output)
