"""Compare imported and reviewed prop presence over complete measured replays."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.environments.robot import build_robot_scene, MeasuredReplay
from g1_pipette_sim.environments.episode_scene import apply_episode_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', type=int, default=[0, 1, 126])
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/lidar-teleop/initial-alignment/contact-comparison.json'))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_string(build_robot_scene())
    data = mujoco.MjData(model)
    tube = model.body('env_02_cylinder').id
    rows = []
    for index in args.episodes:
        episode = Episode(args.dataset, index)
        row = {'episode': index, 'frames': len(episode.times)}
        for mode in ['imported', 'reviewed']:
            replay = MeasuredReplay(model, data, episode)
            if mode == 'imported':
                apply_episode_scene(model, args.dataset, -1)
            else:
                row['scene_state'] = replay.scene_state
            tube_frames = 0
            scene_frames = 0
            tube_max = 0.
            first = {}
            max_joint_error = 0.
            addresses = [model.joint(name).qposadr[0] for name in episode.names]
            for frame, t in enumerate(episode.times):
                replay.pose(t)
                max_joint_error = max(max_joint_error, float(np.max(np.abs(
                    data.qpos[addresses] - episode.arrays['observation.state'][frame, :29]))))
                penetrations = {}
                tube_depth = 0.
                for c in data.contact:
                    ga, gb = c.geom1, c.geom2
                    types = model.geom_contype[[ga, gb]]
                    if c.dist >= -.00001 or 2 not in types or np.all(types == 2):
                        continue
                    bodies = model.geom_bodyid[[ga, gb]]
                    key = ' / '.join(sorted(model.body(b).name for b in bodies))
                    penetrations[key] = max(penetrations.get(key, 0.), -float(c.dist)*1000)
                    if tube in bodies:
                        tube_depth = max(tube_depth, -float(c.dist)*1000)
                tube_frames += tube_depth > .01
                scene_frames += bool(penetrations)
                tube_max = max(tube_max, tube_depth)
                if frame == 0:
                    first = penetrations
            row[mode] = {'initial_penetrations_mm': first,
                         'robot_tube_penetration_frames': int(tube_frames),
                         'robot_tube_max_penetration_mm': tube_max,
                         'robot_scene_penetration_frames': int(scene_frames),
                         'measured_joint_max_error_rad': max_joint_error}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'episodes': rows,
        'threshold_mm': .01,
        'limitations': ['Presence review, not a robot placement or hand calibration.',
                        'Other scene contacts and static held-object mismatches remain.',
                        'Kinematic replay; no physical success evaluation.']}, indent=2)+'\n')


if __name__ == '__main__':
    main()
