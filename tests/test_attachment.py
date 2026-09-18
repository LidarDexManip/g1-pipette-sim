import unittest
from g1_pipette_sim.attachment import attachment_ready
from g1_pipette_sim.scene import load_config

class AttachmentTests(unittest.TestCase):
    def test_requires_seating_not_hovering_or_excessive_insertion(self):
        settings=load_config('config/hgdagger-cameras.json')['attachment']
        for z in (.004, 0., -.001, -.005, -.013):
            with self.subTest(z=z):self.assertFalse(attachment_ready([0,0,z],0,settings))
        for z in (-.010,-.011,-.012):
            with self.subTest(z=z):self.assertTrue(attachment_ready([0,0,z],0,settings))
        self.assertFalse(attachment_ready([.001,0,-.011],0,settings))
        self.assertFalse(attachment_ready([0,0,-.011],6,settings))
    def test_legacy_scenes_keep_symmetric_gate(self):
        settings=load_config('config/workspace.json')['attachment']
        self.assertTrue(attachment_ready([0,0,.004],0,settings))
    def test_preview_requires_continuous_hold_and_preserves_seating(self):
        import numpy as np
        from unittest.mock import patch
        from g1_pipette_sim.dataset import Episode,LEGACY_DATASET
        from g1_pipette_sim.simulation import Simulation
        from g1_pipette_sim.scene import rack_target
        cfg=load_config('config/hgdagger-cameras.json')
        sim=Simulation(cfg,Episode(LEGACY_DATASET,index=53))
        target=rack_target(cfg)
        error=np.array([0.,0.,-.011])
        # Prescribe a nozzle trajectory to isolate seating from IK/controller error.
        def pose(t):
            sim.data.site('nozzle').xpos[:]=target+error
            sim.data.site('nozzle').xmat[:]=np.eye(3).ravel()
        count=round(cfg['attachment']['dwell_seconds']/sim.model.opt.timestep)
        with patch.object(sim,'_pose',side_effect=pose):
            for _ in range(count-1):sim.step()
            self.assertEqual(sim.state,'in_rack')
            error[2]=.001;sim.step()
            self.assertEqual(sim.dwell,0.)
            error[2]=-.011
            for _ in range(count-1):sim.step()
            self.assertEqual(sim.state,'in_rack')
            for _ in range(2):sim.step()
            self.assertEqual(sim.state,'attached')
            self.assertAlmostEqual(sim.seated_depth,.011)
