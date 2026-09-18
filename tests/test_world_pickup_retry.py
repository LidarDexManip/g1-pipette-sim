import unittest,copy
import numpy as np
from g1_pipette_sim.scene import load_config,rack_target
from tools.retry_world_pickups import move_well,placement_key,window_proposals
from tools.publish_world_pickup_layouts import config_for_row,proposals

class WorldRetryTests(unittest.TestCase):
    def test_changing_well_moves_rack_but_preserves_tip_target(self):
        base=load_config();origin={'delta':[.01,-.005,.002,3.],'active_cell':[2,4]};saved=copy.deepcopy(base)
        moved=move_well(base,origin,[6,9]);before=config_for_row(base,origin);after=config_for_row(base,moved)
        np.testing.assert_allclose(rack_target(before),rack_target(after),atol=1e-12)
        self.assertNotEqual(before['rack']['center_xy'],after['rack']['center_xy']);self.assertEqual(base,saved)
        self.assertEqual(before['physics'],after['physics']);self.assertEqual(before['attachment'],after['attachment'])
    def test_distinct_wells_are_not_deduplicated(self):
        a={'valid':True,'complete':True,'pickup':False,'score':-1.9,'delta':[0,0,0,0],'active_cell':[1,1],'evaluation_model':'full_48_sector_tip'}
        b={**a,'active_cell':[1,2],'pickup':True,'score':-1.8}
        rows=proposals({'trials':[a]},{'trials':[b]});self.assertEqual(len(rows),2);self.assertEqual(rows[0]['active_cell'],[1,2]);self.assertNotEqual(placement_key(a),placement_key(b))
    def test_window_proposals_respect_recorded_lift_and_dwell(self):
        base=load_config();base['pickup_fit_criteria']={'target_m':.02,'tolerance_m':.002,'reference_low_time_s':1.}
        origin={'delta':[0,0,0,0],'active_cell':base['rack']['active_cell']};mouth=rack_target(base);t=np.arange(300)*.005;z=np.where(t<1.15,mouth[2]-.011,mouth[2]+.009);trace=np.c_[t,np.full(len(t),mouth[0]),np.full(len(t),mouth[1]),z,np.tile(mouth,(len(t),1)),np.zeros(len(t))]
        rows=window_proposals(trace,base,origin);self.assertTrue(rows);self.assertGreaterEqual(rows[0]['proposal_time_s'],.6);self.assertLess(rows[0]['proposal_time_s'],1.15)
        self.assertLess(np.linalg.norm(rows[0]['delta'][:3]),1e-10)
