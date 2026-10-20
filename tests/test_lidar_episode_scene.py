import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from g1_pipette_sim.environments.episode_scene import apply_episode_scene


class EpisodeSceneTests(unittest.TestCase):
    def test_absent_object_has_no_contacts_and_returns_in_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'meta').mkdir()
            (root/'meta/source_provenance.jsonl').write_text('reviewed dataset')
            config = root/'review.json'
            config.write_text(json.dumps({
                'source_provenance_sha256': hashlib.sha256(b'reviewed dataset').hexdigest(),
                'episodes': {'0': {'absent_objects': ['env_02_cylinder']}}}))
            model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
                <body name="env_02_cylinder"><geom type="sphere" size=".1"/></body>
                <body pos=".15 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
                </worldbody></mujoco>''')
            data = mujoco.MjData(model)
            rgba = model.geom_rgba.copy()
            with patch('g1_pipette_sim.environments.episode_scene.SCENE_REVIEW', config):
                result = apply_episode_scene(model, root, 0)
                mujoco.mj_forward(model, data)
                self.assertTrue(result['reviewed'])
                self.assertEqual(data.ncon, 0)
                self.assertEqual(model.geom_rgba[0, 3], 0)
                apply_episode_scene(model, root, 126)
                mujoco.mj_forward(model, data)
                self.assertGreater(data.ncon, 0)
                np.testing.assert_array_equal(model.geom_rgba, rgba)
                apply_episode_scene(model, root, 0)
                result = apply_episode_scene(model, root/'different_dataset', 0)
                mujoco.mj_forward(model, data)
                self.assertFalse(result['reviewed'])
                self.assertGreater(data.ncon, 0)


if __name__ == '__main__':
    unittest.main()
