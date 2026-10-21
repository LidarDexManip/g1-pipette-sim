import unittest
import numpy as np
import mujoco
from g1_pipette_sim.environments import ROOT, imported_xml
from g1_pipette_sim.environments.robot import build_robot_scene, load_registration, MeasuredReplay
from g1_pipette_sim.hands import HANDS, load_mapping, register_pose
from g1_pipette_sim.dataset import Episode

DATASET = ROOT/'.cache/huggingface/hub/datasets--jren313--g1-pipette-2view-teleop0925-5task-eerel/snapshots/c7adfc179e5e15ade41b06803542001a24959dda'
REGISTER_CASES = [np.full(6, 1000.), np.zeros(6), np.array([0, 250, 500, 750, 1000, 0.]), np.array([1000, 1000, 1000, 1000, 0, 1000.])]


class InspireMappingTests(unittest.TestCase):
    def test_register_pose_matches_legacy_inspire_formula(self):
        m = load_mapping('inspire')
        for raw in REGISTER_CASES:
            u = np.clip(raw/m['register_max'], 0, 1); bend = (1-u[4])*m['thumb_bend_closed_rad']
            pose = register_pose('inspire', 'right', raw, m)
            self.assertEqual(pose['R_index_intermediate_joint'], (1-u[3])*m['finger_closed_rad'])
            self.assertEqual(pose['R_thumb_proximal_yaw_joint'], u[5]*m['thumb_rotation_max_rad'])
            self.assertEqual(pose['R_thumb_distal_joint'], bend*2.4)
            self.assertEqual(len(pose), 12)


@unittest.skipUnless(HANDS['wuji']['model'].is_file(), 'Run tools/prepare_wuji_assets.py to build the Wuji model')
class WujiHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(build_robot_scene(hand='wuji'))

    def test_composition_keeps_scene_body_joints_and_adds_two_wuji_hands(self):
        m = self.model; names = [m.joint(i).name for i in range(m.njnt)]
        self.assertTrue(all(t != mujoco.mjtJoint.mjJNT_FREE for t in m.jnt_type))
        self.assertEqual(sum(n.startswith('l_') for n in names), 20)
        self.assertEqual(sum(n.startswith('r_') for n in names), 20)
        inspire = mujoco.MjModel.from_xml_string(build_robot_scene())
        body = {inspire.joint(i).name for i in range(inspire.njnt) if not inspire.joint(i).name.startswith(('L_', 'R_'))}
        self.assertTrue(body <= set(names))
        original = mujoco.MjModel.from_xml_string(imported_xml('lidar'))
        for i in range(original.ngeom):
            name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_GEOM, i)
            np.testing.assert_array_equal(original.geom_pos[i], m.geom_pos[m.geom(name).id])
        self.assertAlmostEqual(m.body('pelvis').pos[0], load_registration()['base_xy'][0])

    def test_register_extremes_stay_inside_wuji_joint_limits(self):
        m = self.model; mapping = load_mapping('wuji')
        for side in ('left', 'right'):
            open_pose = register_pose('wuji', side, np.full(6, 1000.), mapping)
            self.assertTrue(all(v == 0 for v in open_pose.values()))
            for raw in REGISTER_CASES:
                pose = register_pose('wuji', side, raw, mapping)
                self.assertEqual(len(pose), 20)
                for name, value in pose.items():
                    low, high = m.joint(name).range
                    self.assertTrue(low <= value <= high, (name, value))

    def test_measured_body_playback_is_unchanged_by_hand_choice(self):
        if not (DATASET/'meta/info.json').exists(): self.skipTest('Download demonstrations to test replay')
        data = mujoco.MjData(self.model); episode = Episode(DATASET, 0); replay = MeasuredReplay(self.model, data, episode, 'wuji')
        for frame in [0, len(episode.times)//2, len(episode.times)-1]:
            replay.pose(episode.times[frame])
            actual = [data.qpos[self.model.joint(n).qposadr[0]] for n in episode.names]
            np.testing.assert_array_equal(actual, episode.arrays['observation.state'][frame, :29])
        self.assertTrue(np.isfinite(data.qpos).all())


if __name__ == '__main__': unittest.main()
