import tempfile
import unittest
from pathlib import Path
from g1_pipette_sim.dataset import resolve_dataset_root

class DatasetRootTests(unittest.TestCase):
    def test_cache_uses_main_and_rejects_ambiguous_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('old','new'):
                meta=root/'snapshots'/name/'meta';meta.mkdir(parents=True);(meta/'info.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):resolve_dataset_root(root)
            (root/'refs').mkdir();(root/'refs/main').write_text('new\n')
            self.assertEqual(resolve_dataset_root(root),root/'snapshots/new')
            self.assertEqual(resolve_dataset_root(root/'snapshots/old'),root/'snapshots/old')
            (root/'refs/main').write_text('missing')
            with self.assertRaises(FileNotFoundError):resolve_dataset_root(root)
