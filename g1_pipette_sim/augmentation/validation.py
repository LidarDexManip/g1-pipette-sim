"""Bounds on final composed commands; physics gates remain in fit_pickup_layouts."""
import numpy as np
import mujoco
from g1_pipette_sim.scene import build_xml
from tools.search_replay_layout import on_table


def preflight(ep, original, commands, cfg, max_delta=.35, max_velocity=.35):
    if not np.isfinite(commands).all() or commands.shape != original.shape:
        raise ValueError('Invalid final commands')
    dt = np.diff(ep.action_times)
    if np.any(dt <= 0):
        raise ValueError('Non-monotonic clock')
    arm = [i for i, name in enumerate(ep.names) if name.startswith(('right_shoulder_', 'right_elbow_', 'right_wrist_'))]
    if len(arm) != 7:
        raise ValueError('Expected seven named right arm joints')
    other = [i for i in range(29) if i not in arm]
    if not np.array_equal(commands[:, other], original[:, other]):
        raise ValueError('Untargeted joint changed')
    delta = commands - original
    velocity = float(np.max(abs(np.diff(delta, axis=0) / dt[:, None])))
    amplitude = float(np.max(abs(delta)))
    if amplitude > max_delta + 1e-10 or velocity > max_velocity + 1e-10:
        raise ValueError('Final composed perturbation exceeds bounds')
    model = mujoco.MjModel.from_xml_string(build_xml(cfg))
    for i in arm:
        joint = model.joint(ep.names[i]).id
        if model.jnt_limited[joint] and (np.any(commands[:, i] < model.jnt_range[joint, 0]) or np.any(commands[:, i] > model.jnt_range[joint, 1])):
            raise ValueError('Joint limit exceeded')
    if not on_table(cfg):
        raise ValueError('Refined scene is off table')
    return {'max_joint_delta_rad': amplitude, 'max_added_joint_velocity_rad_s': velocity}
