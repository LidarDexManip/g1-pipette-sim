"""Physical regression for the HILR8 tool dropping at replay initialization."""
import copy
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.simulation import set_robot_pose


class GraspDynamicsTests(unittest.TestCase):
    def test_recorded_commands_hold_tool_and_open_hand_releases_it(self):
        for episode in (0, 1):
            with self.subTest(episode=episode):
                path = Path(f'outputs/hilr8-eerel/contact-timing/episode-{episode:03d}/restored-workspace.json')
                if not path.exists():
                    self.skipTest('HILR8 restored workspace and local dataset required')
                cfg = json.loads(path.read_text())
                cfg['base_position'][2] += .25  # Free-space test, clear of props.
                original = copy.deepcopy(cfg)
                ep = Episode(cfg['replay_audit_setup']['dataset'], episode)
                sim = DynamicSimulation(cfg, ep, source='low_level')
                model, data = sim.model, sim.data
                initial = data.qpos.copy()
                initial_ctrl = data.ctrl.copy()
                # Match desired targets to the unmodified dataset-to-joint mapping.
                ref = mujoco.MjData(sim.reference_model)
                checkpoints = []
                for frame in range(1000):
                    sim.step()
                    if frame in (199, 999):
                        set_robot_pose(sim.reference_model, ref, ep.names,
                                       *ep.sample(sim.time - model.opt.timestep, 'low_level'), cfg)
                        limits = model.jnt_range[sim.joints]
                        expected = np.clip(ref.qpos[sim.ref_qadr], limits[:, 0], limits[:, 1])
                        np.testing.assert_allclose(sim.desired, expected, atol=1e-10)
                        wrist = data.body('right_wrist_yaw_link')
                        local = wrist.xmat.reshape(3, 3).T @ (data.body('pipette').xpos - wrist.xpos)
                        error = local - np.array(cfg['pipette']['mount_position'])
                        self.assertLess(abs(error[2]), .003)
                        self.assertLess(np.linalg.norm(error), .009)
                        checkpoints.append(local.copy())
                self.assertLess(np.linalg.norm(checkpoints[1] - checkpoints[0]), .001)
                self.assertFalse(data.eq_active[sim.grip_id])
                self.assertFalse(data.eq_active[sim.attachment_id])
                forces = []
                for ci, contact in enumerate(data.contact):
                    names = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
                    if 'pipette' in names and any(n.startswith('R_') for n in names):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, force)
                        forces.append(force[0])
                self.assertGreater(sum(forces), .1)
                # Opening releases the finger pinch. The pipette finger-rest hook
                # can then catch on the palm, so this is not a free-fall test.
                z = data.body('pipette').xpos[2]
                sim.release_grip()
                for _ in range(600):
                    sim.step()
                self.assertGreater(z - data.body('pipette').xpos[2], .025)
                for i, joint in enumerate(sim.joints):
                    name = model.joint(int(joint)).name
                    if name.startswith('R_') and 'yaw' not in name:
                        self.assertLess(abs(data.qpos[sim.qadr[i]]), .01)
                for contact in data.contact:
                    names = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
                    self.assertFalse('pipette' in names and any(n.startswith('R_') for n in names))
                self.assertFalse(sim.report()['warnings'])
                sim.reset()
                np.testing.assert_allclose(data.qpos, initial, atol=1e-12)
                np.testing.assert_allclose(data.ctrl, initial_ctrl, atol=1e-12)
                self.assertEqual(cfg, original)
