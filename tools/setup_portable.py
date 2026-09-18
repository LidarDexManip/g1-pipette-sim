"""Verify HIL194 and bind saved layouts to this checkout without changing physics."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import DEFAULT_DATASET, resolve_dataset_root
from g1_pipette_sim.augmentation.contracts import ROOT, atomic_json, file_hash
from g1_pipette_sim.pickup_layouts import model_fingerprints


def setup(dataset):
    dataset = resolve_dataset_root(dataset).resolve()
    identity = json.loads((ROOT/'config/hil194-data-identity.json').read_text())
    for relative, expected in identity['files'].items():
        if file_hash(dataset/relative) != expected:
            raise ValueError(f'Dataset differs from saved replay inputs: {relative}')
    manifest = json.loads((ROOT/'config/hil194-pickup-layouts.json').read_text())
    if manifest['model_fingerprints'] != model_fingerprints():
        raise ValueError('Physics changed since layout validation; revalidate layouts first')
    original_dataset = manifest['dataset_snapshot']
    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if value == original_dataset:
            return str(dataset)
        return value
    anchors = relocate(json.loads((ROOT/'config/hil194-scene-anchors.json').read_text()))
    anchor_path = ROOT/'.local/scene-anchors.json'
    atomic_json(anchor_path, anchors)
    base = relocate(json.loads((ROOT/'config/hil194.json').read_text()))
    def bind_anchors(value):
        if isinstance(value, dict):
            if 'registration_anchors' in value:
                value['registration_anchors'] = str(anchor_path)
                value['registration_anchors_sha256'] = file_hash(anchor_path)
            for child in value.values():
                bind_anchors(child)
        elif isinstance(value, list):
            for child in value:
                bind_anchors(child)
    bind_anchors(base)
    atomic_json(ROOT/'.local/hil194.json', base)
    manifest['dataset_snapshot'] = str(dataset)
    for index, entry in manifest['episodes'].items():
        if not entry.get('config'):
            continue
        source = ROOT/entry['config']
        if file_hash(source) != entry['sha256']:
            raise ValueError(f'Scene template changed: {source}')
        config = json.loads(source.read_text())
        config['layout_search']['dataset'] = str(dataset)
        target = ROOT/'.local/layouts'/f'episode-{int(index):03d}.json'
        atomic_json(target, config)
        entry['template_sha256'] = entry['sha256']
        entry['config'] = str(target.relative_to(ROOT))
        entry['sha256'] = file_hash(target)
    reviews = json.loads((ROOT/'config/hil194-recorded-reviews.json').read_text())
    reviews['dataset_snapshot'] = str(dataset)
    atomic_json(ROOT/'.local/recorded-reviews.json', reviews)
    atomic_json(ROOT/'.local/pickup-layouts.json', manifest)
    atomic_json(ROOT/'.local/dataset-identity.json', identity)
    print(f'Bound {len(manifest["episodes"])} layouts to verified dataset {dataset}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    setup(args.dataset)
