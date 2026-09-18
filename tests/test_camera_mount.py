import unittest
import mujoco,numpy as np
from g1_pipette_sim.scene import load_config,build_xml

class CameraMountTests(unittest.TestCase):
    def test_camera_follows_its_mount_and_world_camera_stays_fixed(self):
        cfg=load_config();cfg['cameras']['test_wrist']={'parent_body':'right_wrist_yaw_link','position':[.04,.02,.03],'quaternion':[1,0,0,0],'fovy':60}
        model=mujoco.MjModel.from_xml_string(build_xml(cfg));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        fixed=data.camera('rgb').xpos.copy();before=data.camera('test_wrist').xpos.copy()
        j=model.joint('right_shoulder_pitch_joint').id;data.qpos[model.jnt_qposadr[j]]+=.3;mujoco.mj_forward(model,data)
        body=data.body('right_wrist_yaw_link');expected=body.xpos+body.xmat.reshape(3,3)@np.array([.04,.02,.03])
        np.testing.assert_allclose(data.camera('test_wrist').xpos,expected)
        self.assertGreater(np.linalg.norm(data.camera('test_wrist').xpos-before),.001)
        np.testing.assert_allclose(data.camera('rgb').xpos,fixed)
    def test_invalid_camera_mount_is_rejected(self):
        cfg=load_config();cfg['cameras']['rgb']['parent_body']='missing_body'
        with self.assertRaises(ValueError):build_xml(cfg)
    def test_hgdagger_has_two_fixed_views_and_one_right_wrist_view(self):
        cfg=load_config('config/hgdagger-cameras.json')
        model=mujoco.MjModel.from_xml_string(build_xml(cfg));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        before={n:(data.camera(n).xpos.copy(),data.camera(n).xmat.copy()) for n in ('rgb','wrist_right','wrist_left')}
        joint=model.joint('right_shoulder_pitch_joint');data.qpos[joint.qposadr[0]]+=.3;mujoco.mj_forward(model,data)
        for name in ('rgb','wrist_right'):
            np.testing.assert_allclose(data.camera(name).xpos,before[name][0])
            np.testing.assert_allclose(data.camera(name).xmat,before[name][1])
        self.assertGreater(np.linalg.norm(data.camera('wrist_left').xpos-before['wrist_left'][0]),.001)
        self.assertEqual(model.camera('wrist_left').bodyid[0],model.body('right_wrist_yaw_link').id)
