import copy
import unittest
import numpy as np
from g1_pipette_sim.dataset import Episode,DEFAULT_DATASET,HIL194_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.profiles import for_episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.pipette_mesh import parts


class Hil194Tests(unittest.TestCase):
    def test_defaults_and_frozen_clock_use_paired_rows(self):
        self.assertEqual(DEFAULT_DATASET,HIL194_DATASET)
        episode=Episode(index=46)
        self.assertEqual(episode.info['total_episodes'],194)
        self.assertTrue(episode.timing['action_timestamps_frozen'])
        self.assertEqual(episode.timing['command_alignment'],'paired_image_rows')
        np.testing.assert_array_equal(episode.action_times,episode.times)
        self.assertEqual(episode.frame_at((episode.times[0]+episode.times[1])/2),0)
        self.assertEqual(episode.frame_at(episode.times[1]),1)
        i=100;body,left,right=episode.sample(episode.times[i],'low_level')
        np.testing.assert_allclose(body,episode.arrays['action.low_level'][i])
        np.testing.assert_allclose(right,episode.arrays['action.hand_right'][i])
        self.assertIn('recorded',episode.hand_action_provenance)
        self.assertIn('reconstructed',Episode(index=15).hand_action_provenance)

    def test_session_selection_and_hand_command_mapping(self):
        cfg=load_config();old=for_episode(cfg,Episode(index=15));new=for_episode(cfg,Episode(index=46))
        self.assertEqual(old['camera_fit']['episode'],0)
        self.assertEqual(new['camera_fit']['episode'],46)
        self.assertEqual(new['cameras']['wrist_left']['parent_body'],'right_wrist_yaw_link')
        self.assertNotIn('parent_body',new['cameras']['wrist_right'])
        self.assertNotIn('pose_register_override',new['hand_mapping'])
        self.assertNotIn('red_tape',old['pipette'])
        self.assertTrue(new['pipette']['red_tape']['enabled'])
        self.assertNotIn('selected_session',cfg)

    def test_initialization_removes_overlap_without_welding_or_changing_targets(self):
        ep=Episode(index=46);sim=DynamicSimulation(load_config(),ep,source='low_level')
        self.assertGreater(sim.grasp_initialization['before_max_penetration_mm'],10)
        self.assertLessEqual(sim.grasp_initialization['after_max_penetration_mm'],.0201)
        self.assertFalse(sim.data.eq_active[sim.grip_id])
        adr=[sim.model.jnt_qposadr[sim.model.joint(n).id] for n in ep.names]
        np.testing.assert_allclose(sim.data.qpos[adr],ep.sample(0,'state')[0])
        target=sim.desired.copy();sim._reference(0);np.testing.assert_array_equal(sim.desired,target)
        actuator=sim.model.actuator('servo_R_index_proximal_joint').id
        joint=sim.model.actuator_trnid[actuator,0]
        self.assertLess(sim.data.qpos[sim.model.jnt_qposadr[joint]],sim.desired[actuator])
        self.assertFalse(sim.config['attachment']['automatic_ejection'])
        for _ in range(20):sim.step()
        self.assertTrue(sim.report()['finite'])

    def test_tape_wraps_blue_shaft_and_preserves_nozzle(self):
        cfg=for_episode(load_config(),Episode(index=46));p=cfg['pipette'];vertices=np.array(parts(p)['pipette_red_tape'][0])
        lo,hi=vertices[:,2].min(),vertices[:,2].max()
        self.assertAlmostEqual(lo,p['nozzle_z']+.0127)
        self.assertAlmostEqual(hi-lo,.020)
        profile=np.asarray(p['shaft_profile'])
        radius=np.linalg.norm(vertices[:,:2],axis=1)
        np.testing.assert_allclose(radius-np.interp(vertices[:,2],profile[:,0],profile[:,1]),.00002,atol=1e-12)
        bare=copy.deepcopy(p);bare.pop('red_tape')
        np.testing.assert_array_equal(parts(p)['pipette_nozzle'][0],parts(bare)['pipette_nozzle'][0])
        np.testing.assert_array_equal(parts(p)['pipette_lower_contact_0'][0],parts(bare)['pipette_lower_contact_0'][0])

    def test_additional_reference_recovers_new_session_scene(self):
        import tempfile
        from g1_pipette_sim.episode_scene import initialize
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_=initialize(load_config(),Episode(index=193),cache=tmp)
        rack=cfg['scene_initialization']['objects']['rack']
        self.assertEqual(rack['method'],'relative independently fitted episode reference')
        self.assertEqual(rack['reference_chain_depth'],1)
        self.assertLess(rack['fit_rms_px'],10)
        self.assertIn('primary_reference_failure',rack)
