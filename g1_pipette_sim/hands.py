"""Hand variants for the G1 model and playback of Inspire six-register commands.

The recorded demonstrations use Inspire hands. Their registers (0 closed, 1000 open)
are replayed on Inspire joints with the workspace mapping, or on Wuji Hand 2 joints
with an approximate synergy. Neither mapping is a calibrated hand model.

The Inspire mapping is the pipette workspace formula, unchanged. For thumb rotation the
Wuji synergy follows the register convention, 1000 = open hand with the thumb beside the
index finger (LIDAR head-camera frames show both hands gripping at 1000); the workspace
formula turns the Inspire thumb the other way.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HANDS = {
    'inspire': {'label': 'G1/Inspire', 'model': ROOT/'assets/robot/g1_inspire.xml',
                'tags': ('default', 'actuator', 'equality', 'sensor', 'contact'), 'prefix': {'left': 'L', 'right': 'R'}},
    'wuji': {'label': 'G1/Wuji Hand 2', 'model': ROOT/'assets/robot_wuji/g1_wuji.xml',
             'tags': ('default', 'actuator', 'equality', 'sensor', 'contact'), 'prefix': {'left': 'l', 'right': 'r'},
             'mapping': ROOT/'config/hands/wuji-from-inspire.json'},
}
REGISTER_FINGERS = ('pinky', 'ring', 'middle', 'index')


def hand_model(hand):
    path = HANDS[hand]['model']
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing; run python tools/prepare_wuji_assets.py first')
    return path


def load_mapping(hand):
    if hand == 'inspire':
        from .scene import load_config
        return load_config()['hand_mapping']
    return json.loads(HANDS[hand]['mapping'].read_text())


def inspire_pose(prefix, raw, mapping):
    u = np.clip(raw/mapping['register_max'], 0, 1); pose = {}
    for i, finger in enumerate(REGISTER_FINGERS):
        for segment in ['proximal', 'intermediate']:
            pose[f'{prefix}_{finger}_{segment}_joint'] = (1-u[i])*mapping['finger_closed_rad']
    bend = (1-u[4])*mapping['thumb_bend_closed_rad']
    pose.update({f'{prefix}_thumb_proximal_yaw_joint': u[5]*mapping['thumb_rotation_max_rad'],
                 f'{prefix}_thumb_proximal_pitch_joint': bend, f'{prefix}_thumb_intermediate_joint': bend*1.6,
                 f'{prefix}_thumb_distal_joint': bend*2.4})
    return pose


def wuji_pose(prefix, raw, mapping):
    u = np.clip(raw/mapping['register_max'], 0, 1); pose = {}
    for i, finger in enumerate(REGISTER_FINGERS):
        name = mapping['finger_names'][finger]
        for joint, closed in mapping['finger_closed_rad'].items():
            pose[f'{prefix}_{name}_{joint}'] = (1-u[i])*closed
        pose[f'{prefix}_{name}_mcp_abd'] = mapping['finger_abduction_rad']
    for joint, closed in mapping['thumb_bend_closed_rad'].items():
        pose[f'{prefix}_{joint}'] = (1-u[4])*closed
    for joint, rotated in mapping['thumb_rotation_rad'].items():
        pose[f'{prefix}_{joint}'] = pose.get(f'{prefix}_{joint}', 0.)+(1-u[5])*rotated
    return pose


def register_pose(hand, side, raw, mapping):
    """Joint targets for one hand from six recorded registers."""
    prefix = HANDS[hand]['prefix'][side]
    return (inspire_pose if hand == 'inspire' else wuji_pose)(prefix, raw, mapping)
