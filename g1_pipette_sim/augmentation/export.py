"""Read-only legacy import, deduplication, and immutable staged publication."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .contracts import atomic_json, dependencies, digest, file_hash, lock, seal, verify, tree_files
from .sources import import_seed

CAPTURE_FILES = ['accepted.json', 'config.json', 'commands.npz', 'verification.json', 'trajectory.npz',
                 'camera-poses.json', 'rgb.mp4', 'wrist_right.mp4', 'wrist_left.mp4']


def catalog(collections):
    entries = []; seen = {}; duplicates = []
    for collection in sorted(map(lambda p: Path(p).resolve(), collections)):
        for marker in sorted((collection/'candidates').glob('*/accepted.json')):
            folder = marker.parent
            c, ep, record = import_seed(folder)
            # New captures must carry the commit-time checksums. Legacy import
            # records present integrity; it cannot prove historical hashes.
            for name, expected in record.get('artifact_hashes', {}).items():
                if file_hash(folder/name) != expected:
                    raise ValueError(f'Corrupted accepted artifact: {folder/name}')
            files = {name: file_hash(folder/name) for name in CAPTURE_FILES}
            with np.load(folder/'trajectory.npz') as data:
                n = len(data['timestamp'])
                if n != record['validation']['frame_count'] or not all(np.isfinite(data[k]).all() for k in data.files):
                    raise ValueError('Invalid captured numeric data')
                for key, source in [('action','action.low_level'), ('action.hand_left','action.hand_left'), ('action.hand_right','action.hand_right')]:
                    expected = np.stack([np.interp(data['simulation_time'], ep.action_times, ep.arrays[source][:, j]) for j in range(ep.arrays[source].shape[1])], axis=1)
                    if not np.allclose(data[key], expected, rtol=0, atol=1e-10):
                        raise ValueError('Saved capture labels do not match dense commands')
            scene = json.loads((folder/'config.json').read_text())
            for key in ['pickup_layout', 'layout_search']:
                scene.pop(key, None)
            # Semantic fingerprint excludes archive compression and local paths.
            command_hashes = {key: digest(ep.arrays[key].tolist()) for key in ['action.low_level','action.hand_left','action.hand_right']}
            fingerprint = digest({'scene': scene, 'commands': command_hashes, 'times': ep.action_times.tolist(),
                                  'state': ep.arrays['observation.state'][0].tolist(), 'joint_names': ep.names,
                                  'source_files': c['source_files']})
            capture_id = c.get('capture_id', digest({'collection': str(collection), 'local_id': c['number'], 'files': files}))
            if fingerprint in seen:
                duplicates.append({'capture_id': capture_id, 'duplicate_of': seen[fingerprint]}); continue
            seen[fingerprint] = capture_id
            entries.append({'capture_id': capture_id, 'fingerprint': fingerprint, 'folder': str(folder),
                            'files': files, 'candidate': c, 'frames': n,
                            'integrity_evidence': 'commit_checksums' if 'artifact_hashes' in record else 'legacy_import_time_only'})
    return {'schema': 1, 'entries': entries, 'duplicates': duplicates}


def verify_catalog(value):
    for entry in value['entries']:
        for name, expected in entry['files'].items():
            if file_hash(Path(entry['folder'])/name) != expected:
                raise ValueError(f'Catalog source changed: {entry["folder"]}/{name}')
        for path, expected in {**entry['candidate']['source_files'], **entry['candidate'].get('model_assets', {})}.items():
            if file_hash(path) != expected:
                raise ValueError(f'Real source changed: {path}')


def publish(value, output, v3=False):
    from tools.package_pickup_augmentations import _package
    output = Path(output).resolve()
    verify_catalog(value)
    contract = {'catalog': value, 'dependencies': dependencies(), 'v3': v3}
    if v3:
        converter=Path('/home/ytung34/Projects/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py')
        contract['converter_sha256']=file_hash(converter)
        contract['lerobot_source']={str(p):file_hash(p) for p in sorted(converter.parents[2].rglob('*.py'))}
        import importlib.metadata
        contract['loader_versions']={name:importlib.metadata.version(name) for name in ['lerobot','torch','av']}
    with lock(output.parent / ('.'+output.name+'-publish')):
        if output.exists():
            if not verify(output, 'publication', contract):
                raise ValueError('Existing export has no matching publication receipt; choose a new version')
            return len(value['entries'])
        stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
        try:
            _package(None, stage, entries=value['entries'])
            atomic_json(stage/'meta/capture-catalog.json', value)
            if v3:
                import subprocess, sys
                script = Path('/home/ytung34/Projects/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py')
                custom = {p.name: p.read_bytes() for p in (stage/'meta').glob('*') if p.is_file() and p.name in ['augmentation.jsonl','training-contract.json','capture-catalog.json','modality.json','parent-splits.json']}
                reference = Path(tempfile.mkdtemp(prefix='augmentation-v2-audit-'))
                shutil.copytree(stage/'meta',reference/'meta')
                shutil.copytree(stage/'data',reference/'data')
                subprocess.run([sys.executable, str(script), '--repo-id', 'local/g1-pipette-sim-motor-aug', '--root', str(stage), '--push-to-hub', 'false'], check=True)
                for name, data in custom.items():
                    (stage/'meta'/name).write_bytes(data)
                from tools.audit_augmented_dataset import audit
                try:
                    audit(v2=reference,v3=stage,output=stage/'meta/dataset-audit.json')
                finally:
                    shutil.rmtree(reference,ignore_errors=True)
            info = json.loads((stage/'meta/info.json').read_text())
            if info['total_episodes'] != len(value['entries']) or info['total_frames'] != sum(e['frames'] for e in value['entries']):
                raise ValueError('Export count mismatch')
            verify_catalog(value)  # Reject input changes during export.
            if v3 and file_hash(converter)!=contract['converter_sha256']:
                raise ValueError('Converter changed during export')
            seal(stage, 'publication', contract, tree_files(stage))
            os.rename(stage, output)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return len(value['entries'])
