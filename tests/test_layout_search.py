import copy
import unittest

import numpy as np

from g1_pipette_sim.scene import load_config, rack_target
from tools.search_replay_layout import shifted_config, on_table, footprint, separated, rack_penetrations


class LayoutSearchTests(unittest.TestCase):
    def test_translation_preserves_tip_well_lid_and_physics(self):
        base = load_config('config/hgdagger-cameras.json')
        base['waste'].pop('hinge', None)
        original = copy.deepcopy(base)
        delta = np.array([.03, -.02])
        cfg = shifted_config(base, delta, lid_placement="fixed")
        np.testing.assert_allclose(rack_target(cfg) - rack_target(base), [*delta, 0])
        np.testing.assert_allclose(
            np.asarray(cfg['rack']['center_xy']) + cfg['waste']['offset_from_rack'],
            np.asarray(base['rack']['center_xy']) + base['waste']['offset_from_rack'])
        for name in ['table', 'pipette', 'physics', 'attachment', 'hand_mapping', 'cameras']:
            self.assertEqual(cfg[name], base[name])
        self.assertEqual(cfg['rack']['active_cell'], base['rack']['active_cell'])
        self.assertEqual(base, original)

    def test_table_support_checks_rotated_corners(self):
        cfg = load_config('config/hgdagger-cameras.json')
        cfg['waste'].pop('hinge', None)
        cfg['table']['center_xy'] = [0, 0]
        cfg['table']['size_xy'] = [.2, .2]
        cfg['rack']['center_xy'] = [0, 0]
        cfg['waste']['offset_from_rack'] = [0, 0]
        cfg['waste']['size_xy'] = [.01, .01]
        cfg['rack']['size_xy'] = [.18, .18]
        cfg['rack']['yaw_deg'] = 0
        self.assertTrue(on_table(cfg))
        cfg['rack']['yaw_deg'] = 45
        self.assertFalse(on_table(cfg))

    def test_carried_lid_has_clearance(self):
        base = load_config('config/hgdagger-cameras.json')
        base['waste'].pop('hinge', None)
        cfg = shifted_config(base, [.039, -.037])
        self.assertTrue(separated(footprint(cfg, 'rack', .006), footprint(cfg, 'waste', .006)))

    def test_actual_startup_collision_is_rejected(self):
        from g1_pipette_sim.dataset import Episode, HGDAGGER_DATASET
        from g1_pipette_sim.dynamics import DynamicSimulation
        base = load_config('config/hgdagger-cameras.json')
        base['waste'].pop('hinge', None)
        base['waste']['offset_from_rack'] = [.02, .02]
        episode = Episode(HGDAGGER_DATASET, 15)
        overlapping = DynamicSimulation(base, episode, source='low_level')
        self.assertTrue(any(x['bodies'] == ['rack', 'waste'] for x in rack_penetrations(overlapping)))
        cleared = DynamicSimulation(shifted_config(base, [0, 0]), episode, source='low_level')
        self.assertFalse(rack_penetrations(cleared))
