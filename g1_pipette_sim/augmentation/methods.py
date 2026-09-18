"""Proposal families using the previously tested algorithms and explicit fixed phases."""
import copy
import numpy as np
import mujoco
from g1_pipette_sim.scene import build_xml, rack_target
from g1_pipette_sim.simulation import set_robot_pose
from tools.collect_pickup_augmentations import placed_config, smoothstep, make_actions
from tools.search_larger_augmentations import actions as cartesian_actions
from .validation import preflight

METHODS = ('well_only', 'local_cartesian', 'retarget_well', 'cartesian', 'joint_approach', 'elbow_pulse', 'joint_persistent')


def generate(ep, cfg, recipe, seed, attachment):
    cfg = copy.deepcopy(cfg)
    original = ep.arrays['action.low_level'].copy()
    motor = original.copy()
    rng = np.random.default_rng(seed)
    method = recipe['method']
    if method not in METHODS:
        raise ValueError(f'Unknown method {method}')
    arm = [i for i, n in enumerate(ep.names) if n.startswith(('right_shoulder_', 'right_elbow_', 'right_wrist_'))]
    # Retained IK implementation uses this verified layout; never silently reorder.
    if arm != list(range(22, 29)):
        raise ValueError('Current IK implementation requires right-arm columns 22:29')
    if method in ['well_only', 'local_cartesian', 'retarget_well']:
        cell = recipe.get('cell', cfg['rack']['active_cell'])
        shift = np.asarray(recipe.get('shift_m', [0., 0.]), float)
        if method == 'retarget_well':
            moved = copy.deepcopy(cfg); moved['rack']['active_cell'] = cell
            shift = rack_target(moved)[:2] - rack_target(cfg)[:2]
        motor, _ = make_actions(ep, cfg, shift, seed, 'well_only' if method == 'well_only' else ('retarget_well' if method == 'retarget_well' else 'smooth_action_and_placement'))
        cfg = placed_config(cfg, cell, shift)
    elif method == 'cartesian':
        shift = np.asarray(recipe.get('shift_m', [0.005, 0.]), float)
        motor, _ = cartesian_actions(ep, cfg, shift)
        cfg = placed_config(cfg, cfg['rack']['active_cell'], shift)
    else:
        t = ep.action_times
        if method == 'elbow_pulse':
            begin, peak, returning, end = recipe.get('pulse_s', [10., 11., 11.4, 12.6])
            if not 0 <= begin < peak <= returning < end <= ep.duration:
                raise ValueError('Invalid elbow pulse window')
            envelope = smoothstep((t-begin)/(peak-begin)) * (1-smoothstep((t-returning)/(end-returning)))
            direction = np.zeros(7); direction[arm.index(ep.names.index('right_elbow_joint'))] = 1
            amplitude = recipe.get('amplitude_deg', .4)
        else:
            ramp, returning, end = recipe.get('approach_s', [3., 5., 10.])
            if not 0 < ramp <= returning < end < attachment:
                raise ValueError('Approach window must end before attachment')
            envelope = smoothstep(t/ramp)
            if method == 'joint_approach':
                envelope *= 1-smoothstep((t-returning)/(end-returning))
            elif not recipe.get('allow_persistent', False):
                raise ValueError('Persistent offsets require allow_persistent=true')
            direction = np.asarray(recipe.get('direction', rng.normal(size=7)), float)
            if direction.shape != (7,) or not np.isfinite(direction).all() or np.max(abs(direction)) == 0:
                raise ValueError('Invalid joint direction')
            direction /= np.max(abs(direction))
            amplitude = recipe.get('amplitude_deg', 3.)
        motor[:, arm] += envelope[:, None] * np.radians(amplitude) * direction
        recipe = {**recipe, 'direction': direction.tolist()}
        if method != 'elbow_pulse':
            # Couple the fixed scene to FK nozzle displacement at nominal contact.
            m = mujoco.MjModel.from_xml_string(build_xml(cfg)); d = mujoco.MjData(m)
            positions = []
            _, left, right = ep.sample(attachment, 'low_level')
            for targets in [original, motor]:
                body = np.array([np.interp(attachment, t, targets[:, j]) for j in range(29)])
                set_robot_pose(m, d, ep.names, body, left, right, cfg); mujoco.mj_forward(m, d)
                positions.append(d.site('nozzle').xpos.copy())
            shift = positions[1]-positions[0]
            if abs(shift[2]) > .04:
                raise ValueError('Table shift exceeds 40 mm')
            cfg = placed_config(cfg, cfg['rack']['active_cell'], shift[:2])
            cfg['table']['top_z'] += float(shift[2])
    bounds = preflight(ep, original, motor, cfg)
    return motor, cfg, {**recipe, **bounds}


def lift_revision(ep, cfg, motor, trim):
    revised = copy.deepcopy(ep)
    revised.arrays['action.low_level'] = motor.copy()
    return cartesian_actions(revised, cfg, np.zeros(2), lift_trim=trim)[0]
