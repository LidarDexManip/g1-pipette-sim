import json
import unittest
from pathlib import Path
import numpy as np
import mujoco
from g1_pipette_sim.environments import ROOT, imported_xml
from g1_pipette_sim.environments.robot import build_robot_scene,load_registration,MeasuredReplay
from g1_pipette_sim.dataset import Episode


class LidarRobotTests(unittest.TestCase):
    def test_composition_preserves_scene_geometry_and_adds_fixed_g1(self):
        original=mujoco.MjModel.from_xml_string(imported_xml('lidar'))
        model=mujoco.MjModel.from_xml_string(build_robot_scene())
        self.assertGreater(model.nq,29)
        self.assertTrue(all(t!=mujoco.mjtJoint.mjJNT_FREE for t in model.jnt_type))
        for i in range(original.ngeom):
            name=mujoco.mj_id2name(original,mujoco.mjtObj.mjOBJ_GEOM,i)
            j=model.geom(name).id
            np.testing.assert_array_equal(original.geom_pos[i],model.geom_pos[j])
            np.testing.assert_array_equal(original.geom_size[i],model.geom_size[j])
        self.assertAlmostEqual(model.body('pelvis').pos[0],load_registration()['base_xy'][0])

    def test_measured_body_playback_and_frozen_hand_fallback(self):
        root=ROOT/'.cache/huggingface/hub/datasets--jren313--g1-pipette-2view-teleop0925-5task-eerel/snapshots/c7adfc179e5e15ade41b06803542001a24959dda'
        if not (root/'meta/info.json').exists():self.skipTest('Download demonstrations to test replay')
        model=mujoco.MjModel.from_xml_string(build_robot_scene());data=mujoco.MjData(model)
        for i in [0,188]:
            episode=Episode(root,i);replay=MeasuredReplay(model,data,episode)
            for frame in [0,len(episode.times)//2,len(episode.times)-1]:
                replay.pose(episode.times[frame])
                actual=[data.qpos[model.joint(n).qposadr[0]] for n in episode.names]
                np.testing.assert_array_equal(actual,episode.arrays['observation.state'][frame,:29])
            self.assertEqual(replay.hand_sources['left'],'action.hand_left' if i==188 else 'observation.state.hand_left')

if __name__=='__main__':unittest.main()
