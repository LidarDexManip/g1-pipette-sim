import copy
import unittest
import mujoco
import numpy as np
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.box_lid import pose
from tools.search_replay_layout import shifted_config,rack_penetrations


class BoxHingeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode=Episode(HGDAGGER_DATASET,15)
        cls.cfg=load_config('config/hgdagger-cameras.json')

    def test_opening_geometry_and_connection(self):
        for angle in [0.,45.,90.,180.,198.]:
            with self.subTest(angle=angle):
                cfg=copy.deepcopy(self.cfg);cfg['waste']['hinge']['initial_angle_deg']=angle
                sim=DynamicSimulation(cfg,self.episode,source='low_level')
                self.assertFalse(rack_penetrations(sim))
                self.assertEqual(mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_JOINT,'waste_free'),-1)
                self.assertGreater(sim.model.npair,0)
                origin,rotation=pose(cfg)
                np.testing.assert_allclose(sim.data.body('waste').xpos,origin+[0,0,.004],atol=1e-8)
                np.testing.assert_allclose(sim.data.body('waste').xmat.reshape(3,3),rotation,atol=1e-8)
                before=sim.data.body('waste').xpos.copy()
                sim.data.joint('rack_free').qpos[0]+=.03
                mujoco.mj_forward(sim.model,sim.data)
                np.testing.assert_allclose(sim.data.body('waste').xpos-before,[.03,0,0],atol=1e-8)

    def test_passive_hinge_moves_under_gravity_and_stays_connected(self):
        cfg=copy.deepcopy(self.cfg);cfg['waste']['hinge']['initial_angle_deg']=120.
        sim=DynamicSimulation(cfg,self.episode,source='low_level')
        joint=sim.model.joint('lid_hinge').id
        self.assertNotIn(joint,sim.model.actuator_trnid[:,0])
        start=float(sim.data.joint('lid_hinge').qpos[0])
        for _ in range(round(.3/sim.model.opt.timestep)):sim.step()
        self.assertGreater(float(sim.data.joint('lid_hinge').qpos[0]),start+.01)
        self.assertFalse(sim.report()['warnings'])
        self.assertFalse(rack_penetrations(sim,tolerance_m=.001))

    def test_layout_shift_preserves_hinge_relation(self):
        moved=shifted_config(self.cfg,[.03,-.02])
        a,ra=pose(self.cfg);b,rb=pose(moved)
        np.testing.assert_allclose(b-a,[.03,-.02,0])
        np.testing.assert_allclose(ra,rb)
        self.assertEqual(moved['waste']['hinge'],self.cfg['waste']['hinge'])
        with self.assertRaisesRegex(ValueError,'attached lid'):
            shifted_config(self.cfg,[.03,-.02],lid_placement='fixed')

    def test_open_lid_rests_on_table_beyond_180_degrees(self):
        sim=DynamicSimulation(self.cfg,self.episode,source='low_level')
        for _ in range(round(.8/sim.model.opt.timestep)):sim.step()
        angle=np.rad2deg(float(sim.data.joint('lid_hinge').qpos[0]))
        self.assertGreater(angle,180.)
        self.assertLess(angle,220.)
        self.assertIn(('table_body','waste'),sim.contact_pairs)
        self.assertFalse(rack_penetrations(sim))
        self.assertFalse(sim.report()['warnings'])
