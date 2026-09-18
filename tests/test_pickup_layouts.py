import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from g1_pipette_sim.pickup_layouts import select_pickup_layout,model_fingerprints,sha256

class PickupSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.ep=SimpleNamespace(index=46,root=self.root)
        self.config=self.root/'fit.json';self.config.write_text(json.dumps({'rack':{'center_xy':[.3,.1]},'layout_search':{'episode':46,'dataset':str(self.root),'source':'low_level','clock':'recorded'}}))
        self.manifest={'dataset_snapshot':str(self.root),'model_fingerprints':model_fingerprints(),'episodes':{'46':{'config':str(self.config),'sha256':sha256(self.config),'status':'verified_pickup'}}};self.path=self.root/'manifest.json'
    def tearDown(self):self.tmp.cleanup()
    def select(self):
        self.path.write_text(json.dumps(self.manifest));return select_pickup_layout({'original':True},self.ep,self.path)
    def test_episode_selection_and_other_dataset_isolation(self):
        cfg,entry=self.select();self.assertEqual(cfg['rack']['center_xy'],[.3,.1]);self.assertEqual(entry['status'],'verified_pickup')
        self.ep.root=self.root/'other';self.assertEqual(self.select(),({'original':True},None))
    def test_pending_episode_in_partial_manifest_uses_visual_default(self):
        self.manifest['complete']=False;self.ep.index=47
        self.assertEqual(self.select(),({'original':True},None))
    def test_rejects_modified_config_and_changed_model(self):
        self.config.write_text('{}')
        with self.assertRaisesRegex(ValueError,'modified'):self.select()
        self.manifest['model_fingerprints']['g1_pipette_sim/dynamics.py']='stale'
        with self.assertRaisesRegex(ValueError,'different model'):self.select()
    def test_asset_fingerprinting_is_readonly(self):
        from g1_pipette_sim.pickup_layouts import asset_fingerprints_from_xml
        asset=self.root/'mesh.obj';asset.write_text('hand-edited mesh')
        before=asset.stat().st_mtime_ns
        xml=f'<mujoco><compiler meshdir="{self.root}"/><asset><mesh file="mesh.obj"/></asset></mujoco>'
        fingerprints=asset_fingerprints_from_xml(xml)
        self.assertEqual(fingerprints[str(asset)],sha256(asset))
        self.assertEqual(asset.stat().st_mtime_ns,before)
        self.assertEqual(asset.read_text(),'hand-edited mesh')

    def test_mesh_edit_invalidates_unchanged_config(self):
        asset=self.root/'mesh.obj';asset.write_text('original')
        self.manifest['episodes']['46']['asset_fingerprints']={str(asset):sha256(asset)}
        self.select();asset.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Mesh asset changed'):self.select()

    def test_rejects_wrong_episode_even_with_valid_hash(self):
        cfg=json.loads(self.config.read_text());cfg['layout_search']['episode']=15;self.config.write_text(json.dumps(cfg));self.manifest['episodes']['46']['sha256']=sha256(self.config)
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):self.select()
    def test_unverified_status_is_preserved_and_missing_result_rejected(self):
        self.manifest['episodes']['46']['status']='attachment_only';self.assertEqual(self.select()[1]['status'],'attachment_only')
        self.ep.index=47
        with self.assertRaisesRegex(ValueError,'No pickup search'):self.select()
    def test_default_example_prefers_verified_attachment_during_motion(self):
        from g1_pipette_sim.pickup_layouts import recommended_episode
        self.manifest['episodes']={str(i):{'episode':i,'config':'example.json','status':status,'validation':{'events':[{'state':'attached','time':t}]}} for i,t,status in [(5,.15,'verified_pickup'),(14,12.,'verified_pickup'),(52,15.,'verified_pickup'),(0,8.,'attachment_only')]}
        self.path.write_text(json.dumps(self.manifest))
        self.assertEqual(recommended_episode(self.root,self.path),52)
        from unittest.mock import patch
        with patch("g1_pipette_sim.pickup_layouts.recorded_reviews",return_value={"52":{"exclude_from_default":True}}):
            self.assertEqual(recommended_episode(self.root,self.path),14)
        self.assertEqual(recommended_episode(self.root/'other',self.path),0)


class ReplayOptimizationTests(unittest.TestCase):
    def test_skipping_reference_forward_preserves_live_trajectory(self):
        import mujoco,numpy as np
        from g1_pipette_sim.dataset import Episode
        from g1_pipette_sim.dynamics import DynamicSimulation
        from g1_pipette_sim.scene import load_config
        class ReferenceForwardSimulation(DynamicSimulation):
            def _reference(self,t):
                super()._reference(t);mujoco.mj_forward(self.reference_model,self.reference)
        ep=Episode(index=46);cfg=load_config();new=DynamicSimulation(cfg,ep,source='low_level');old=ReferenceForwardSimulation(copy.deepcopy(cfg),ep,source='low_level')
        for _ in range(200):new.step();old.step()
        np.testing.assert_array_equal(new.data.qpos,old.data.qpos);np.testing.assert_array_equal(new.data.qvel,old.data.qvel);self.assertEqual(new.report(),old.report())
