import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tools import publish_world_pickup_layouts as publisher

class WorldPublicationTests(unittest.TestCase):
    def test_world_refinement_is_independently_replayed_and_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'episode-001';folder.mkdir()
            baseline={'pickup_fit_criteria':{'target_m':.01}}
            old={'valid':True,'complete':True,'score':-1.9,'delta':[0.,0.,0.,0.],'evaluation_model':'full_48_sector_tip'}
            refined={**old,'delta':[0.,0.,.001,0.],'lift_frame':'world','score':-2.}
            (folder/'baseline.json').write_text(json.dumps(baseline));(folder/'search.json').write_text(json.dumps({'trials':[old],'model_fingerprints':{},'baseline_scene_source':'test'}));(folder/'world-search.json').write_text(json.dumps({'trials':[refined],'model_fingerprints':{}}))
            result={'valid':True,'complete':True,'pickup':True,'attached':True,'score':-2.,'lift_frame':'world','lift_m':.01}
            with patch.object(publisher,'ROOT',root),patch.object(publisher,'model_fingerprints',return_value={}),patch.object(publisher,'Episode',return_value=SimpleNamespace(root=root)),patch.object(publisher,'candidate_config',side_effect=lambda base,delta:{**base,'selected':delta}),patch.object(publisher,'rollout',return_value=(result,np.empty((0,8)),{'asset_fingerprints':{}})) as replay:
                entry=publisher.finalize_world((1,str(root),str(root)))
                self.assertEqual(entry['status'],'verified_pickup');self.assertEqual(replay.call_args.kwargs,{'full':True,'lift_frame':'world'})
                self.assertEqual(json.loads((root/entry['config']).read_text())['selected'],refined['delta'])
                self.assertEqual(publisher.finalize_world((1,str(root),str(root))),entry);self.assertEqual(replay.call_count,1)
    def test_invalid_world_refinement_does_not_hide_original_proposals(self):
        old={'valid':True,'complete':True,'score':-1.9,'delta':[0.,0.,0.,0.],'evaluation_model':'full_48_sector_tip'}
        invalid={**old,'valid':False,'delta':[0.,0.,.001,0.],'score':-2.}
        self.assertEqual(publisher.proposals({'trials':[old]},{'trials':[invalid]}),[old])
    def test_unattached_replay_reuse_requires_identical_config_and_meshes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'episode-001';folder.mkdir();cfg={'a':1};config=folder/'validated-default.json';config.write_text(json.dumps(cfg));mesh=root/'mesh.obj';mesh.write_text('original')
            assets={'mesh.obj':publisher.sha256(mesh)};report={'events':[],'finite':True,'warnings':{},'asset_fingerprints':assets};(folder/'default-report.json').write_text(json.dumps(report))
            entry={'status':'no_pickup','config':str(config.relative_to(root)),'sha256':publisher.sha256(config),'asset_fingerprints':assets,'validation':{'valid':True,'complete':True,'attached':False,'events':[],'lift_m':0.}}
            (folder/'validated.json').write_text(json.dumps({'model_fingerprints':{},'entry':entry}))
            with patch.object(publisher,'ROOT',root):
                result=publisher.reuse_unattached_checkpoint(folder,cfg,{})
                self.assertEqual(result[0]['lift_frame'],'world');self.assertEqual(result[0]['lift_m'],0.)
                self.assertIsNone(publisher.reuse_unattached_checkpoint(folder,{'a':2},{}))
                mesh.write_text('changed');self.assertIsNone(publisher.reuse_unattached_checkpoint(folder,cfg,{}))
