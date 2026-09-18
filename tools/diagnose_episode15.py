"""Replay episode 15 and compare nozzle alignment with measured-state kinematics.

Run from the project root. Recorded states are used only for offline comparison;
they do not modify the dynamic replay. Results go to outputs/episode15-diagnosis.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode, HGDAGGER_DATASET
from g1_pipette_sim.scene import load_config, rack_target
from g1_pipette_sim.episode_scene import resolve_scene
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.simulation import Simulation


def main():
    out = Path('outputs/episode15-diagnosis')
    out.mkdir(parents=True, exist_ok=True)
    episode = Episode(HGDAGGER_DATASET, 15)
    cfg, _ = resolve_scene(load_config('config/hgdagger-cameras.json'), episode)
    sim = DynamicSimulation(cfg, episode, source='low_level')
    measured = Simulation(cfg, episode, source='state')
    records = []
    while sim.time < episode.duration:
        sim.step()
        if records and sim.time - records[-1]['time'] < .01:
            continue
        tip = sim.data.body('tip')
        rotation = tip.xmat.reshape(3, 3)
        mouth = tip.xpos + rotation @ np.array([0, 0, cfg['pipette']['tip_length'] / 2])
        nozzle = sim.data.site('nozzle')
        error = rotation.T @ (nozzle.xpos - mouth)
        angle = np.degrees(np.arccos(np.clip(
            rotation[:, 2] @ nozzle.xmat.reshape(3, 3)[:, 2], -1, 1)))
        measured._pose(sim.time)
        state_error = measured.data.site('nozzle').xpos - rack_target(cfg)
        wrist = sim.data.body('right_wrist_yaw_link')
        local = wrist.xmat.reshape(3, 3).T @ (sim.data.body('pipette').xpos - wrist.xpos)
        records.append({
            'time': sim.time,
            'radial_mm': float(np.linalg.norm(error[:2]) * 1000),
            'depth_mm': float(-error[2] * 1000),
            'angle_deg': float(angle),
            'error_mm': (error * 1000).tolist(),
            'nozzle': nozzle.xpos.tolist(),
            'mouth': mouth.tolist(),
            'state_nozzle_error_mm': (state_error * 1000).tolist(),
            'tool_local': local.tolist(),
        })
    report = sim.report()
    report['scene_initialization'] = cfg.get('scene_initialization')
    for name, value in [('samples', records), ('report', report), ('config', cfg)]:
        (out / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(report['attachment_diagnostics'], indent=2))
    print(f'Events: {sim.events}; outputs: {out.resolve()}')


if __name__ == '__main__':
    main()
