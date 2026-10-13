import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from g1_pipette_sim.camera_views import apply_camera_views

class CameraViewTests(unittest.TestCase):
    def test_camera_only_correction_is_dataset_and_episode_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'views.json';pose={'position':[1,2,3],'quaternion':[1,0,0,0],'fovy':60};path.write_text(json.dumps({'dataset_snapshot':str(root),'episodes':{'14':{'rgb':pose}}}))
            original={'cameras':{'rgb':{'position':[0,0,1],'role':'fixed_ego'},'wrist_right':{'fovy':85}},'rack':{'center_xy':[.3,.1]},'physics':{'gravity':True},'layout_search':{'result':{'success':True}}};saved=copy.deepcopy(original)
            result=apply_camera_views(original,SimpleNamespace(root=root,index=14),path)
            self.assertEqual(original,saved);self.assertEqual(result['cameras']['rgb']['position'],[1,2,3])
            for key in ['rack','physics','layout_search']:self.assertEqual(result[key],saved[key])
            self.assertEqual(result['cameras']['wrist_right'],saved['cameras']['wrist_right'])
            self.assertIs(apply_camera_views(original,SimpleNamespace(root=root,index=15),path),original)
            self.assertIs(apply_camera_views(original,SimpleNamespace(root=root/'other',index=14),path),original)
    def test_non_camera_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'views.json';path.write_text(json.dumps({'dataset_snapshot':str(root),'episodes':{'14':{'rgb':{'physics':{}}}}}))
            with self.assertRaisesRegex(ValueError,'may only change'):apply_camera_views({'cameras':{'rgb':{}}},SimpleNamespace(root=root,index=14),path)
    def test_default_session_and_episode_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'views.json'
            pose={'position':[1,2,3],'quaternion':[1,0,0,0],'fovy':60}
            path.write_text(json.dumps({'dataset_snapshot':str(root),'defaults':{'rgb':pose},
                'sessions':[{'name':'second','start':46,'end':194,'views':{'rgb':{'position':[4,5,6]}}}],
                'episodes':{'52':{'rgb':{'fovy':55}}}}))
            original={'cameras':{'rgb':{'role':'fixed_ego'},'wrist_left':{'fovy':100}},'physics':{'gravity':True}}
            for index,position,fovy in [(0,[1,2,3],60),(45,[1,2,3],60),(46,[4,5,6],60),(52,[4,5,6],55),(193,[4,5,6],60)]:
                result=apply_camera_views(original,SimpleNamespace(root=root,index=index),path)
                self.assertEqual(result['cameras']['rgb']['position'],position)
                self.assertEqual(result['cameras']['rgb']['fovy'],fovy)
                self.assertEqual(result['physics'],original['physics'])
                self.assertEqual(result['cameras']['wrist_left'],original['cameras']['wrist_left'])
                self.assertEqual(len(result['camera_view_correction']['source_sha256']),64)
            self.assertEqual(original['cameras']['rgb'],{'role':'fixed_ego'})
    def test_overlapping_sessions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'views.json'
            path.write_text(json.dumps({'dataset_snapshot':str(root),'sessions':[
                {'name':'a','start':0,'end':50,'views':{}},{'name':'b','start':40,'end':60,'views':{}}]}))
            with self.assertRaisesRegex(ValueError,'Overlapping'):
                apply_camera_views({'cameras':{}},SimpleNamespace(root=root,index=46),path)


def test_recording_guard_rejects_stale_inherited_camera(tmp_path):
    from g1_pipette_sim.camera_views import assert_camera_views
    import pytest
    path=tmp_path/'overrides.json'
    path.write_text(json.dumps({'dataset_snapshot':str(tmp_path),'defaults':{'rgb':{'position':[1,2,3],'quaternion':[1,0,0,0],'fovy':60}}}))
    ep=SimpleNamespace(root=tmp_path,index=64)
    old={'cameras':{'rgb':{'position':[0,0,1],'quaternion':[1,0,0,0],'fovy':60}},'rack':{'x':1}}
    with pytest.raises(ValueError,match='Stale camera'):assert_camera_views(old,ep,path)
    corrected=apply_camera_views(old,ep,path)
    assert_camera_views(corrected,ep,path)
    assert old['cameras']['rgb']['position']==[0,0,1]
    assert corrected['rack']==old['rack']


def test_augmentation_applies_cameras_after_importing_seed(tmp_path,monkeypatch):
    import numpy as np
    from g1_pipette_sim.augmentation import runner
    path=tmp_path/'overrides.json'
    pose={'position':[1,2,3],'quaternion':[1,0,0,0],'fovy':60}
    path.write_text(json.dumps({'dataset_snapshot':str(tmp_path),'defaults':{'rgb':pose}}))
    old={'cameras':{'rgb':{'position':[0,0,1],'quaternion':[1,0,0,0],'fovy':60}}}
    cfg=tmp_path/'seed.json';cfg.write_text(json.dumps(old));source={'config':str(cfg)}
    ep=SimpleNamespace(root=tmp_path,index=64,arrays={'action.low_level':np.zeros((2,29))})
    monkeypatch.setattr(runner,'import_seed',lambda _: (source,ep,{'validation':{'events':[{'state':'attached','time':5}]}}))
    monkeypatch.setattr(runner,'apply_camera_views',lambda config,episode:apply_camera_views(config,episode,path))
    def check(episode,base,*args):
        assert base['cameras']['rgb']==pose
        raise ValueError('stop after camera check')
    monkeypatch.setattr(runner,'generate',check)
    result=runner.execute_task({'source':'unused','source_contract':source,'recipe':{},'seed':1})
    assert result['generation_rejected']
    assert json.loads(cfg.read_text())==old


def test_hil293_defaults_sessions_and_stale_export_guard(tmp_path):
    from g1_pipette_sim.camera_views import OVERRIDES,assert_camera_views
    from g1_pipette_sim.augmentation.export import assert_catalog_cameras
    import pytest
    spec=json.loads(OVERRIDES.read_text())
    root=Path(next(k for k in spec['dataset_profiles'] if 'hil293' in k))
    old={'cameras':{'rgb':{'position':[.473,.088,1.162],'quaternion':[1,0,0,0],'fovy':60},'wrist_left':{'role':'unchanged'}},'table':{'top_z':.648}}
    for index in [0,11,45,46,193,194,275,292]:
        ep=SimpleNamespace(root=root,index=index)
        expected=spec['defaults']['rgb'] if index<46 else spec['dataset_profiles'][str(root)]['sessions'][0]['views']['rgb']
        fixed=apply_camera_views(old,ep)
        assert fixed['cameras']['rgb']==expected
        assert fixed['table']==old['table'] and fixed['cameras']['wrist_left']==old['cameras']['wrist_left']
        with pytest.raises(ValueError,match='Stale camera'):assert_camera_views(old,ep)
        assert_camera_views(fixed,ep)
    folder=tmp_path/'capture';folder.mkdir();(folder/'config.json').write_text(json.dumps(old))
    value={'entries':[{'folder':str(folder),'candidate':{'dataset_snapshot':str(root),'parent_episode':11}}]}
    with pytest.raises(ValueError,match='Stale camera'):assert_catalog_cameras(value)
    (folder/'config.json').write_text(json.dumps(apply_camera_views(old,SimpleNamespace(root=root,index=11))))
    assert_catalog_cameras(value)


def test_recorder_rejects_stale_hil293_before_simulation():
    import pytest
    from g1_pipette_sim.camera_views import OVERRIDES
    from tools.collect_pickup_augmentations import RecordedSimulation
    spec=json.loads(OVERRIDES.read_text())
    root=Path(next(k for k in spec['dataset_profiles'] if 'hil293' in k))
    old={'cameras':{'rgb':{'position':[.473,.088,1.162],'quaternion':[1,0,0,0],'fovy':60}}}
    with pytest.raises(ValueError,match='Stale camera'):
        RecordedSimulation(old,SimpleNamespace(root=root,index=11),{'folder':'unused'})


def test_hil293_correction_survives_hf_cache_relocation():
    from g1_pipette_sim.camera_views import OVERRIDES
    spec=json.loads(OVERRIDES.read_text())
    root=Path(next(k for k in spec['dataset_profiles'] if 'hil293' in k))
    relocated=Path('/different/machine/hub')/root.parent.parent.name/'snapshots'/root.name
    old={'cameras':{'rgb':{}}}
    corrected=apply_camera_views(old,SimpleNamespace(root=relocated,index=11))
    assert corrected['cameras']['rgb']==spec['defaults']['rgb']
    # Another dataset/revision must not accidentally inherit this calibration.
    assert apply_camera_views(old,SimpleNamespace(root=relocated.parent/'different-revision',index=11)) is old
