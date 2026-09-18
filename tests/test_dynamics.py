import unittest
import numpy as np
import mujoco
from g1_pipette_sim.dataset import Episode,LEGACY_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dynamics import DynamicSimulation


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = Episode(LEGACY_DATASET,index=53)

    def make_sim(self, **kwargs):
        return DynamicSimulation(load_config('config/workspace.json'), self.episode, **kwargs)

    def test_every_movable_object_falls_when_unsupported(self):
        for joint in ['table_free', 'rack_free', 'waste_free', 'pipette_free', 'tip_free']:
            with self.subTest(joint=joint):
                sim = self.make_sim()
                # Setup perturbation only. Ordinary step must never reset it.
                sim.data.eq_active[:] = False
                j = sim.model.joint(joint).id
                adr = int(sim.model.jnt_qposadr[j])
                sim.data.qpos[adr:adr+3] += [1.4, .8, .3]
                mujoco.mj_forward(sim.model, sim.data)
                z0 = float(sim.data.qpos[adr+2])
                steps = round(.04/sim.model.opt.timestep)
                for _ in range(steps): sim.step()
                fall = z0-sim.data.qpos[adr+2]
                self.assertGreater(fall, .006)
                self.assertLess(fall, .011)
                self.assertFalse(sim.report()['warnings'])

    def test_tray_lands_on_table_and_responds_to_push(self):
        sim = self.make_sim()
        j = sim.model.joint('waste_free').id
        adr = int(sim.model.jnt_qposadr[j])
        sim.data.qpos[adr+2] += .06
        mujoco.mj_forward(sim.model, sim.data)
        for _ in range(round(.8/sim.model.opt.timestep)): sim.step()
        tray = sim.data.body('waste').xpos.copy()
        self.assertAlmostEqual(tray[2], sim.config['table']['top_z'], delta=.003)
        self.assertIn(('table_body', 'waste'), sim.contact_pairs)
        body = sim.model.body('waste').id
        sim.data.xfrc_applied[body, 0] = 1.0
        for _ in range(round(.1/sim.model.opt.timestep)): sim.step()
        self.assertGreater(sim.data.body('waste').xpos[0]-tray[0], .001)

    def test_decorative_tool_meshes_never_gain_contacts(self):
        sim = self.make_sim()
        pipe = sim.model.body('pipette').id
        visual = [i for i in range(sim.model.ngeom)
                  if sim.model.geom_bodyid[i] == pipe and sim.model.geom_contype[i] == 0]
        self.assertGreater(len(visual), 0)
        for operation in [sim.release_grip, sim.reset]:
            operation()
            self.assertTrue(np.all(sim.model.geom_contype[visual] == 0))
            self.assertTrue(np.all(sim.model.geom_conaffinity[visual] == 0))

    def test_joint_targets_do_not_teleport_live_robot(self):
        sim = self.make_sim()
        joint = sim.model.joint('right_elbow_joint').id
        adr = sim.model.jnt_qposadr[joint]
        sim.data.qpos[adr] += .1
        before = float(sim.data.qpos[adr])
        mujoco.mj_forward(sim.model, sim.data)
        sim.step()
        self.assertLess(abs(sim.data.qpos[adr]-before), .01)
        ref = sim.reference_model.jnt_qposadr[sim.reference_model.joint('right_elbow_joint').id]
        self.assertGreater(abs(sim.data.qpos[adr]-sim.reference.qpos[ref]), .08)
        self.assertTrue(np.all(np.abs(sim.data.actuator_force) <= sim.model.actuator_forcerange[:,1]+1e-8))

    def test_free_robot_base_and_release_reset(self):
        sim = self.make_sim(free_base=True)
        adr = sim.model.jnt_qposadr[sim.model.joint('robot_base').id]
        sim.release_grip()
        self.assertFalse(sim.data.eq_active[sim.grip_id])
        sim.data.qpos[adr+2] += .3
        z0 = sim.data.qpos[adr+2]
        mujoco.mj_forward(sim.model,sim.data)
        for _ in range(round(.04/sim.model.opt.timestep)): sim.step()
        self.assertLess(sim.data.qpos[adr+2],z0-.003)
        sim.reset()
        self.assertTrue(sim.data.eq_active[sim.grip_id])
        self.assertTrue(sim.data.eq_active[sim.retention_id])
        self.assertFalse(sim.data.eq_active[sim.attachment_id])

    def test_released_pipette_collides_with_table(self):
        sim = self.make_sim()
        sim.release_grip()
        adr = sim.model.jnt_qposadr[sim.model.joint('pipette_free').id]
        sim.data.qpos[adr:adr+7] = [.60, -.24, 1.05, 1, 0, 0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        for _ in range(round(1.0/sim.model.opt.timestep)): sim.step()
        self.assertIn(('pipette', 'table_body'), sim.contact_pairs)
        self.assertGreater(sim.data.body('pipette').xpos[2], sim.config['table']['top_z']-.03)
        self.assertFalse(sim.report()['warnings'])

    def test_dynamic_demo_uses_constraints_and_contacts(self):
        sim = self.make_sim(mode='demo')
        for _ in range(round(10/sim.model.opt.timestep)): sim.step()
        self.assertEqual([e['state'] for e in sim.events], ['attached', 'released', 'disposed'])
        self.assertIn(('rack', 'tip'), sim.contact_pairs)
        self.assertIn(('tip', 'waste'), sim.contact_pairs)
        self.assertGreater(sim.max_contacts, 0)
        self.assertFalse(sim.report()['warnings'])
        self.assertFalse(sim.data.eq_active[sim.attachment_id])
        self.assertFalse(sim.data.eq_active[sim.retention_id])
        self.assertTrue(sim.report()['finite'])


if __name__ == '__main__': unittest.main()

class FixedGraspPoseTests(unittest.TestCase):
    def test_hgdagger_pose_override_closes_fingers_without_rewriting_commands(self):
        cfg=load_config('config/hgdagger-cameras.json')
        sim=DynamicSimulation(cfg,Episode(LEGACY_DATASET,index=53),source='low_level')
        raw=cfg['hand_mapping']['pose_register_override']['R']
        for i,finger in enumerate(['pinky','ring','middle','index']):
            joint=sim.model.joint(f'R_{finger}_proximal_joint')
            expected=(1-raw[i]/1000)*cfg['hand_mapping']['finger_closed_rad']
            self.assertAlmostEqual(sim.data.qpos[joint.qposadr[0]],expected)
        np.testing.assert_allclose(sim.right,sim.motion.sample(0,'low_level')[2])
        self.assertFalse(sim.report()['grip_constraint_active'])
        self.assertLess(sim.report()['grip_position_error_m'],1e-6)
