import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from g1_pipette_sim.replay_defaults import configure_default_replay


class ReplayDefaultsTests(unittest.TestCase):
    def args(self,**kwargs):
        return SimpleNamespace(**({'config':None,'dataset':None,'episode':None,'scene_layout':'auto'}|kwargs))

    def test_missing_cache_uses_bundled_matching_recipe(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'datasets--jren313--g1-pipette-3view-hil293';snapshot=cache/'snapshots/revision'
            (snapshot/'meta').mkdir(parents=True);(snapshot/'meta/info.json').write_text('{}')
            cfg=root/'config/replay/hil293-episode-011.json';cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({'replay_audit_setup':{'episode':11,'dataset':str(snapshot)}}))
            args=self.args();message=configure_default_replay(args,root,root/'missing',cache)
            self.assertEqual(args.dataset,snapshot);self.assertEqual(args.config,cfg)
            self.assertEqual(args.episode,11);self.assertEqual(args.scene_layout,'fixed')
            self.assertIn('HIL293',message)
            cfg.write_text(json.dumps({'replay_audit_setup':{'episode':12,'dataset':str(snapshot)}}))
            with self.assertRaisesRegex(ValueError,'does not match'):
                configure_default_replay(self.args(),root,root/'missing',cache)

    def test_explicit_inputs_never_substituted(self):
        for args in [self.args(dataset=Path('/missing-explicit')),self.args(config=Path('/explicit-config'))]:
            before=vars(args).copy()
            self.assertIsNone(configure_default_replay(args,preferred=Path('/missing'),fallback=Path('/also-missing')))
            self.assertEqual(vars(args),before)

    def test_existing_preferred_cache_and_missing_all(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.args()
            self.assertIsNone(configure_default_replay(args,preferred=root))
            self.assertIsNone(args.dataset)
            with self.assertRaisesRegex(FileNotFoundError,'Pass --dataset'):
                configure_default_replay(args,root,root/'missing',root/'also-missing')

if __name__=='__main__':unittest.main()
