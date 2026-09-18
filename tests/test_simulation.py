import copy, unittest
import numpy as np
import mujoco
import cv2
from g1_pipette_sim.dataset import Episode,LEGACY_DATASET
from g1_pipette_sim.scene import load_config,fit_layout,rack_target
from g1_pipette_sim.simulation import Simulation,set_robot_pose
from g1_pipette_sim.calibrate_camera import calibrate

class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.episode=Episode(LEGACY_DATASET,index=53)

    def test_recorded_clock_and_joint_mapping(self):
        ep=self.episode;sim=Simulation(load_config('config/workspace.json'),ep)
        # Validate mapping at a real non-keyframe sample, not a zero default pose.
        sim._pose(5.123);expected=ep.sample(5.123)[0]
        actual=[sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(n).id]] for n in ep.names]
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        self.assertGreater(np.ptp(np.diff(ep.times)),.001)
        self.assertFalse(np.allclose(ep.arrays['action'][:,:29],ep.arrays['action.low_level']))
        self.assertGreater(np.ptp(ep.arrays['action.hand_right'][:,4]),0)

    def test_task_demo_and_released_tip_settles(self):
        sim=Simulation(load_config('config/workspace.json'),self.episode,mode='demo')
        for _ in range(5000):sim.step()
        self.assertEqual([e['state'] for e in sim.events],['attached','released','disposed'])
        self.assertLess(sim.motion.max_ik_error,.001)
        self.assertFalse(sim.report()['warnings'])
        self.assertLess(np.linalg.norm(sim.data.qvel[sim.tip_v:sim.tip_v+3]),.01)
        # Reset must restore the tip and disable its free-fall collision mode.
        sim.reset();self.assertEqual(sim.state,'in_rack');self.assertEqual(sim.events,[])
        self.assertEqual(sim.model.geom_contype[sim.tip_geom],0)
        np.testing.assert_allclose(sim.data.qpos[sim.tip_q:sim.tip_q+3],rack_target(sim.config)-[0,0,sim.config['pipette']['tip_length']/2])

    def test_no_attachment_far_from_rack(self):
        cfg=load_config('config/workspace.json');cfg['rack']['center_xy']=[1.5,1.5];sim=Simulation(cfg,self.episode)
        for _ in range(500):sim.step()
        sim.eject();self.assertEqual(sim.state,'in_rack');self.assertEqual(sim.events,[])

    def test_layout_seed_stays_finite_across_sessions(self):
        for index in [0,77]:
            ep=Episode(LEGACY_DATASET,index=index);cfg=fit_layout(load_config('config/workspace.json'),ep);sim=Simulation(cfg,ep)
            for t in np.linspace(0,ep.duration,15):
                sim.time=float(t);sim.step()
                self.assertTrue(sim.report()['finite'])
                self.assertFalse(sim.report()['warnings'])
            self.assertGreater(cfg['table']['top_z'],.3)
            self.assertLess(cfg['table']['top_z'],1)

    def test_camera_pose_recovery(self):
        # Known synthetic 3D landmarks exercise CV-to-MuJoCo rotation conventions.
        xyz=np.array([[x,y,z] for x in [-.15,.15] for y in [-.1,.1] for z in [0,.15]],float)
        K=np.array([[700,0,640],[0,700,360],[0,0,1]],float);rvec=np.array([.15,-.2,.1]);tvec=np.array([.03,-.04,1.2])
        uv,_=cv2.projectPoints(xyz,rvec,tvec,K,np.zeros(5))
        result=calibrate({'world_points_m':xyz.tolist(),'image_points_px':uv.reshape(-1,2).tolist(),'camera_matrix':K.tolist(),'resolution':[1280,720]})
        R,_=cv2.Rodrigues(rvec);np.testing.assert_allclose(result['position'],-R.T@tvec,atol=1e-5)
        from scipy.spatial.transform import Rotation
        recovered=Rotation.from_quat(np.roll(result['quaternion'],-1)).as_matrix()
        np.testing.assert_allclose(recovered,R.T@np.diag([1,-1,-1]),atol=1e-5)
        self.assertLess(result['calibration_report']['all_points_rms_px'],1e-4)

if __name__=='__main__':unittest.main()
