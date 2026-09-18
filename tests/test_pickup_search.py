import copy,unittest
import numpy as np
from tools.measure_recorded_lift import estimate

class LiftReferenceTests(unittest.TestCase):
    def test_later_return_does_not_erase_pickup(self):
        t=np.arange(60)/60+1;z=np.r_[np.zeros(12),np.full(18,.01),np.full(30,-.005)];xyz=np.c_[z*0,z*0,z]
        r=estimate(t,xyz);self.assertAlmostEqual(r['target_m'],.01);self.assertAlmostEqual(r['required_lift_m'],.008)
    def test_monotonic_descent_does_not_invent_lift(self):
        t=np.arange(60)/60+1;z=-np.arange(60)*.001
        self.assertEqual(estimate(t,np.c_[z*0,z*0,z])['target_m'],0.)

class CandidateIsolationTests(unittest.TestCase):
    def test_proxy_does_not_change_config_or_full_model(self):
        from g1_pipette_sim.dataset import Episode
        from g1_pipette_sim.scene import load_config
        from g1_pipette_sim.profiles import for_episode
        from tools.fit_pickup_layouts import candidate_simulation
        ep=Episode(index=46);cfg=for_episode(load_config(),ep);original=copy.deepcopy(cfg)
        proxy=candidate_simulation(cfg,ep);full=candidate_simulation(cfg,ep,full=True)
        self.assertEqual(len(proxy.tip_geoms)*3,len(full.tip_geoms));self.assertEqual(cfg,original)
        self.assertNotIn('search_collision_segments',full.config['pipette'])

class WorldLiftTests(unittest.TestCase):
    def run_fake(self,tip_rises,rack_rises):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tools import fit_pickup_layouts as fit
        class Sim:
            def __init__(self):
                self.time=0.;self.index=-1;self.state='attached';self.events=[{'state':'attached','time':.05}];self.contact_diagnostics={};self.xml='<mujoco/>'
                self.model=SimpleNamespace(opt=SimpleNamespace(timestep=.05),body=lambda name:SimpleNamespace(id=1 if name=='rack' else 2))
                self.tip=SimpleNamespace(xpos=np.array([0.,0.,.009]),xmat=np.eye(3).reshape(-1));self.rack=SimpleNamespace(xpos=np.array([0.,0.,.01]))
                self.data=SimpleNamespace(ncon=0,body=lambda name:self.tip,site=lambda name:self.rack)
            def step(self):
                self.index+=1;self.time+=.05
                self.tip.xpos[2]=.009+(.01 if tip_rises and self.index else 0)
                self.rack.xpos[2]=.01+((.01 if rack_rises else -.01) if self.index else 0)
            def report(self):return {'finite':True,'warnings':{}}
        cfg={'pickup_fit_criteria':{'target_m':.01,'tolerance_m':.002,'required_lift_m':.008},'attachment':{'dwell_seconds':.1},'pipette':{'tip_length':.002}}
        with patch.object(fit,'candidate_simulation',side_effect=lambda *a:Sim()),patch.object(fit,'rack_penetrations',return_value=[]),patch.object(fit,'asset_fingerprints_from_xml',return_value={}):
            return fit.rollout(cfg,SimpleNamespace(duration=.2),full=True,lift_frame='world')[0]
    def test_falling_rack_cannot_invent_world_lift(self):
        result=self.run_fake(False,False)
        self.assertAlmostEqual(result['rack_clearance_m'],.01);self.assertEqual(result['lift_m'],0.);self.assertFalse(result['pickup'])
    def test_world_rise_is_measured_even_when_rack_moves_with_tip(self):
        result=self.run_fake(True,True)
        self.assertAlmostEqual(result['rack_clearance_m'],0.);self.assertAlmostEqual(result['lift_m'],.01);self.assertFalse(result['pickup'])

    def test_world_lift_with_tip_clear_of_rack_passes(self):
        result=self.run_fake(True,False)
        self.assertAlmostEqual(result["lift_m"],.01);self.assertTrue(result["pickup"])
