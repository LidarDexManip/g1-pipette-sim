"""Pinned replay of all command channels, including archived hand registers."""
import json
from pathlib import Path
import numpy as np
from g1_pipette_sim.dataset import Episode
from .contracts import file_hash

CHANNELS = {'action.low_level': 'motor_targets', 'action.hand_left': 'hand_left', 'action.hand_right': 'hand_right'}


def source_files(ep):
    info = ep.info
    paths = [ep.root / 'meta/info.json', ep.root / info['data_path'].format(
        episode_chunk=ep.index // info['chunks_size'], episode_index=ep.index)]
    notes = ep.root / 'meta/source_notes.json'
    if notes.exists():
        paths.append(notes)
    return {str(p.resolve()): file_hash(p) for p in paths}


def snapshot(candidate):
    if candidate.get('dataset_snapshot'):
        return candidate['dataset_snapshot']
    # Explicit compatibility with early captures. Never follow mutable HF main.
    provenance = Path(candidate['folder']).parents[1] / 'provenance.json'
    return json.loads(provenance.read_text())['dataset_snapshot']


def load_episode(candidate):
    for key in ['config', 'commands']:
        expected = candidate.get(key + '_sha256')
        if not expected or file_hash(candidate[key]) != expected:
            raise ValueError(f'Missing or changed {key} checksum')
    for path, expected in {**candidate.get('source_files', {}), **candidate.get('model_assets', {})}.items():
        if file_hash(path) != expected:
            raise ValueError(f'Source changed: {path}')
    ep = Episode(snapshot(candidate), candidate['parent_episode'])
    if candidate.get('joint_names', ep.names) != ep.names:
        raise ValueError('Joint order changed')
    with np.load(candidate['commands'], allow_pickle=False) as archive:
        if not np.array_equal(archive['times'], ep.action_times):
            raise ValueError('Command clock changed')
        if 'initial_state' in archive and not np.array_equal(archive['initial_state'], ep.arrays['observation.state'][0]):
            raise ValueError('Initial state changed')
        for target, source in CHANNELS.items():
            values = archive[source]
            if values.shape != ep.arrays[target].shape or not np.isfinite(values).all():
                raise ValueError(f'Invalid {source} shape/values')
            ep.arrays[target] = values.copy()
    return ep


def import_seed(folder):
    folder = Path(folder).resolve()
    record = json.loads((folder / 'accepted.json').read_text())
    if not record['independently_replayed'] or not record['validation']['pickup']:
        raise ValueError('Source must be an accepted reconstruction')
    candidate = dict(record['candidate'])
    candidate['folder'] = str(folder)
    candidate['dataset_snapshot'] = snapshot(candidate)
    missing = [key for key in ['config', 'commands'] if key+'_sha256' not in candidate]
    for key in missing:
        candidate[key+'_sha256'] = file_hash(candidate[key])
    if missing:
        candidate['legacy_import_note'] = 'Missing historical hashes measured at import: '+','.join(missing)
    provenance = folder.parents[1]/'provenance.json'
    if provenance.exists():
        from g1_pipette_sim.pickup_layouts import model_fingerprints
        old = json.loads(provenance.read_text())
        if old.get('model_fingerprints') and old['model_fingerprints'] != model_fingerprints():
            raise ValueError('Source model differs from current physics')
    ep = load_episode(candidate)
    candidate['source_files'] = source_files(ep)
    candidate['joint_names'] = ep.names
    candidate['model_assets'] = model_assets(json.loads(Path(candidate['config']).read_text()))
    saved_assets = json.loads((folder/'verification.json').read_text()).get('report', {}).get('asset_fingerprints')
    if saved_assets is not None:
        from .contracts import ROOT
        saved_assets = {str((ROOT/path).resolve()): value for path,value in saved_assets.items()}
        if saved_assets != candidate['model_assets']:
            raise ValueError('Model assets differ from the source validation report')
    return candidate, ep, record


def model_assets(cfg):
    from g1_pipette_sim.dynamics import build_dynamic_xml
    from g1_pipette_sim.pickup_layouts import asset_fingerprints_from_xml
    from .contracts import ROOT
    return {str((ROOT/path).resolve()): value for path,value in asset_fingerprints_from_xml(build_dynamic_xml(cfg)).items()}
