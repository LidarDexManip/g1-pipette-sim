import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools import publish_pickup_layouts as publisher

class PublicationTests(unittest.TestCase):
    def test_better_full_replay_wins_over_optimistic_coarse_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'episode-001';folder.mkdir()
            known={'valid':True,'complete':True,'pickup':False,'attached':True,'score':-1.9,'evaluation_model':'full_48_sector_tip','lift_m':.027}
            state={'trials':[{'valid':True,'score':-2.,'delta':[0.], 'trial':0}, {**known,'delta':[1.],'trial':1}], 'baseline_scene_source':'test', 'verified_pickup':False}
            (folder/'search.json').write_text(json.dumps(state));(folder/'baseline.json').write_text('{}')
            worse={**known,'score':-1.5,'lift_m':.02}
            with patch.object(publisher,'ROOT',root),patch.object(publisher,'model_fingerprints',return_value={}),patch.object(publisher,'Episode',return_value=SimpleNamespace(root=root)),patch.object(publisher,'candidate_config',side_effect=lambda base,delta:{'chosen_delta':delta}),patch.object(publisher,'rollout',side_effect=[(worse,None,{'asset_fingerprints':{}}),(known,None,{'asset_fingerprints':{}})]) as replay:
                entry=publisher.finalize((1,str(root),str(root)))
                config=json.loads((root/entry['config']).read_text())
                self.assertEqual(config['chosen_delta'],[1.]);self.assertEqual(entry['validation']['lift_m'],.027)
                self.assertEqual(replay.call_count,2)
                self.assertEqual(publisher.finalize((1,str(root),str(root))),entry)
                self.assertEqual(replay.call_count,2)
