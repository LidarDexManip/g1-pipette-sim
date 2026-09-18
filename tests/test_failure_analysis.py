import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.cluster_replay_failures import outcome, gate_category, score, FEATURES, preprocess, fit


class FailureAnalysisTests(unittest.TestCase):
    def test_lift_categories_respect_tolerance(self):
        v={'attached':True,'lift_m':.019,'lift_target_m':.020,'lift_tolerance_m':.002}
        self.assertEqual(outcome({'validation':v}),'dwell_or_retention_failure')
        v['lift_m']=.017
        self.assertEqual(outcome({'validation':v}),'lift_shortfall')
        v['lift_m']=.023
        self.assertEqual(outcome({'validation':v}),'lift_overshoot')
        v['attached']=False
        self.assertEqual(outcome({'validation':v}),'no_attachment')

    def test_gate_categories_use_full_observation_window(self):
        diag={'samples':100,'gate_failures':{'radial':100,'depth':100,'angle':99}}
        self.assertEqual(gate_category(diag),'never_radial_and_depth')
        diag['gate_failures']={'radial':99,'depth':99,'angle':99}
        self.assertEqual(gate_category(diag),'gates_pass_individually_but_not_sustained_together')
        diag['samples']=0
        self.assertEqual(gate_category(diag),'no_gate_observations')

    def test_noise_not_scored_as_a_cluster(self):
        X=np.array([[0],[.1],[10],[10.1],[100]])
        r=score(X,np.array([0,0,1,1,-1]))
        self.assertEqual(r['clusters'],2)
        self.assertEqual(r['noise'],1)
        self.assertEqual(r['coverage'],.8)
        self.assertGreater(r['silhouette'],.9)
        self.assertIsNone(score(X,np.full(5,-1))['silhouette'])
        self.assertIsNone(score(X,np.zeros(5))['silhouette'])

    def test_missing_and_constant_features_are_finite(self):
        raw=np.tile(np.arange(len(FEATURES),dtype=float),(20,1))
        raw[:,0]=np.arange(20);raw[1,0]=np.nan
        X,metadata=preprocess(raw,'log_standard')
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.allclose(X[:,1:],0))
        for field in ['episode','outcome','recording_group','target_lift_mm','simulated_lift_mm']:
            self.assertNotIn(field,metadata['features'])

    def test_fixed_seed_reproduces_partition(self):
        X=np.random.default_rng(1).normal(size=(40,3))
        a,_=fit('kmeans',{'k':3},X,12)
        b,_=fit('kmeans',{'k':3},X,12)
        np.testing.assert_array_equal(a,b)

    def test_final_audit_rejects_partial_recordings(self):
        import json,tempfile
        from tools.audit_failure_recordings import audit
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'recordings.json').write_text(json.dumps({'complete':False,'errors':{},'recordings':[]}))
            (root/'input-manifest.json').write_text(json.dumps({'episodes':{'0':{'status':'no_pickup'}}}))
            with self.assertRaisesRegex(ValueError,'Incomplete recording set'):
                audit(root)
            self.assertFalse((root/'completion.json').exists())


if __name__=='__main__':unittest.main()
