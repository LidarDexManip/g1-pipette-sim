import unittest
import numpy as np
from g1_pipette_sim.dataset import Episode,LEGACY_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dynamics import DynamicSimulation

class CalibratedReplayTests(unittest.TestCase):
    def test_motor_command_replay_uses_observation_only_for_initialization(self):
        ep=Episode(LEGACY_DATASET,index=53);cfg=load_config('config/calibrated-physics.json')
        sim=DynamicSimulation(cfg,ep,source='low_level')
        qadr=[sim.model.jnt_qposadr[sim.model.joint(n).id] for n in ep.names]
        np.testing.assert_allclose(sim.data.qpos[qadr],ep.sample(0,'state')[0])
        sample=ep.sample
        def commands_only(t,source='state'):
            self.assertEqual(source,'low_level')
            return sample(t,source)
        ep.sample=commands_only
        for _ in range(100):sim.step()
        self.assertTrue(np.all(np.isfinite(sim.data.qpos)))
        for name in ep.names:
            a=sim.model.actuator('servo_'+name).id
            self.assertEqual(sim.model.actuator_gainprm[a,0],cfg['physics']['joint_servos'][name]['kp'])
            self.assertEqual(sim.bias_mask[a],0)
        self.assertAlmostEqual(sim.model.body('effective_wrist_payload').mass[0],cfg['effective_wrist_payload']['mass'])

if __name__=='__main__':unittest.main()
