"""Generate an augmentation seed from a real replay; no historical outputs required.

Screen and independently record before accepting. Acceptance uses historical pickup
criteria; run the rack-contact audit before using captures as contact-clean training data.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode, DEFAULT_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.pickup_layouts import select_pickup_layout
from g1_pipette_sim.camera_views import apply_camera_views
from g1_pipette_sim.augmentation.contracts import ROOT, atomic_json, file_hash
from g1_pipette_sim.augmentation.sources import source_files, model_assets
from g1_pipette_sim.augmentation.runner import stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--episode', type=int, default=64)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/bootstrap')
    a = p.parse_args()
    root = a.output.resolve()
    if root.exists():
        raise ValueError('Choose a new output directory; existing captures are not overwritten')
    ep = Episode(a.dataset, a.episode)
    cfg, entry = select_pickup_layout(load_config(), ep)
    if entry is None:
        raise ValueError('Run tools/setup_portable.py for this dataset first')
    cfg = apply_camera_views(cfg, ep)
    attachment = next(e['time'] for e in entry['validation']['events'] if e['state']=='attached')
    folder = root/'candidates/000000'
    folder.mkdir(parents=True)
    atomic_json(folder/'config.json', cfg)
    np.savez_compressed(folder/'commands.npz', times=ep.action_times,
        motor_targets=ep.arrays['action.low_level'], hand_left=ep.arrays['action.hand_left'],
        hand_right=ep.arrays['action.hand_right'], initial_state=ep.arrays['observation.state'][0])
    candidate = dict(number=0, parent_episode=ep.index, kind='original_replay',
        cell=cfg['rack']['active_cell'], target_shift_m=[0.,0.], seed_attachment_s=attachment,
        config=str(folder/'config.json'), commands=str(folder/'commands.npz'), folder=str(folder),
        dataset_snapshot=str(ep.root.resolve()), joint_names=ep.names, source_files=source_files(ep),
        model_assets=model_assets(cfg), config_sha256=file_hash(folder/'config.json'),
        commands_sha256=file_hash(folder/'commands.npz'), run_id='bootstrap-original-replay-v1')
    atomic_json(folder/'candidate.json', candidate)
    result = stage(candidate)
    print('Screen:', result, flush=True)
    if not result['pickup']:
        raise RuntimeError('Seed failed current replay; diagnostic files retained, no acceptance')
    result = stage(candidate, record=True)
    if not result['pickup']:
        raise RuntimeError('Seed failed independent recording; no acceptance')
    atomic_json(root/'progress.json', {'accepted':[0], 'complete':True})
    # Timings must be selected for each source episode; these are for episode 64.
    recipes = [{'method':'well_only', 'cell':[r,c]} for r,c in
               [cfg['rack']['active_cell'], [3,4], [4,5]]]
    if ep.index == 64:
        recipes += [{'method':'joint_approach','amplitude_deg':3.,'approach_s':[1.5,3.,6.]},
                    {'method':'elbow_pulse','amplitude_deg':.3,'pulse_s':[6.02,7.02,7.42,8.62]}]
    atomic_json(root/'recipe.json', {'seed':20260918,'sources':[str(folder)],
        'proposals':len(recipes),'record':True,'placement_attempts':3,
        'infrastructure_attempts':2,'recipes':recipes})
    print(f'Accepted seed: {folder}\nGenerated recipe: {root / "recipe.json"}', flush=True)


if __name__ == '__main__':
    main()
