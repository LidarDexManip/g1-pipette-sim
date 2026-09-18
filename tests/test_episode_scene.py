import copy,json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from g1_pipette_sim.episode_scene import corners,project,fit_pose,register_polygon,initialize,resolve_scene,paths
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET

class EpisodeSceneTests(unittest.TestCase):
    def test_known_planar_pose_is_recovered(self):
        cfg=load_config('config/hgdagger-cameras.json');pose=np.r_[np.array(cfg['rack']['center_xy'])+[.017,-.011],7.]
        uv=project(cfg,'wrist_right',corners(cfg,'rack',pose),(1080,1080,3));recovered,quality=fit_pose(cfg,'wrist_right','rack',uv,(1080,1080,3))
        np.testing.assert_allclose(recovered,pose,atol=1e-5)
        self.assertLess(quality['fit_rms_px'],1e-4)
    def test_texture_translation_and_blank_rejection(self):
        rng=np.random.default_rng(4);im=rng.integers(0,256,(240,320,3),dtype=np.uint8);im=cv2.GaussianBlur(im,(3,3),0)
        H=np.array([[1.,0,8],[0,1.,-5],[0,0,1.]])
        target=cv2.warpPerspective(im,H,(320,240));poly=np.array([[60,60],[260,60],[260,190],[60,190]],np.float32)
        predicted,q=register_polygon(im,target,poly);np.testing.assert_allclose(predicted,poly+[8,-5],atol=.4)
        with self.assertRaises(ValueError):register_polygon(np.zeros_like(im),np.zeros_like(im),poly)
    def test_cache_manual_objects_and_episode_guard(self):
        cfg=load_config('config/hgdagger-cameras.json');cfg['waste'].pop('hinge',None);e=Episode(HGDAGGER_DATASET,0);original=copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            resolved,path=initialize(cfg,e,tmp);self.assertEqual(cfg,original)
            self.assertEqual(resolved['scene_initialization']['objects']['lid']['status'],'assumed')
            before=path.stat().st_mtime_ns;cached,_=initialize(cfg,e,tmp);self.assertEqual(path.stat().st_mtime_ns,before)
            _,ann=paths(e,tmp);rack_pose=np.r_[cfg['rack']['center_xy'],0];lid_pose=np.r_[np.array(cfg['rack']['center_xy'])+[.025,-.08],8.]
            ann.write_text(json.dumps({'view':'wrist_right','frame':0,'resolution_wh':[1080,1080], 'rack_corners':project(cfg,'wrist_right',corners(cfg,'rack',rack_pose),(1080,1080,3)).tolist(),'lid_corners':project(cfg,'wrist_right',corners(cfg,'lid',lid_pose),(1080,1080,3)).tolist()}))
            corrected,_=initialize(cfg,e,tmp);self.assertEqual(corrected['scene_initialization']['objects']['lid']['status'],'estimated')
            np.testing.assert_allclose(np.array(corrected['rack']['center_xy'])+corrected['waste']['offset_from_rack'],lid_pose[:2],atol=1e-5)
            self.assertAlmostEqual(corrected['waste']['yaw_deg'],8.,places=4)
            self.assertNotEqual(corrected['scene_initialization']['signature'],cached['scene_initialization']['signature'])
            with self.assertRaises(ValueError):resolve_scene(corrected,Episode(HGDAGGER_DATASET,1))
            fixed,_=resolve_scene(cfg,e,'fixed');self.assertIs(fixed,cfg)
