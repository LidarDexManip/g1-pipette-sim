import unittest
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.object_meshes import generate


class TipFitTests(unittest.TestCase):
    def test_photo_seating_covers_nozzle_without_crossing_bore(self):
        p=load_config('config/hgdagger-cameras.json')['pipette']
        depth=p['seated_insertion_depth'];nozzle=np.array(p['nozzle_profile']);inner=np.array(p['tip_inner_profile'])
        self.assertAlmostEqual(nozzle[-1,0]-p['nozzle_z']-depth,.0017)
        height=np.linspace(0,depth,100)
        radius=np.interp(p['nozzle_z']+height,nozzle[:,0],nozzle[:,1])
        bore=np.interp(p['tip_length']-depth+height,inner[:,0],inner[:,1])
        self.assertTrue(np.all(bore>=radius))
        self.assertLess(float(np.max(bore-radius)),.00015)
        paths=generate(load_config('config/hgdagger-cameras.json'))
        vertices=np.array([list(map(float,line.split()[1:])) for line in paths['tip_visual'].read_text().splitlines() if line.startswith('v ')])
        self.assertAlmostEqual(float(np.ptp(vertices[:,2])),p['tip_length'])
        self.assertAlmostEqual(float(vertices[:,2].max()),p['tip_length']/2)

    def test_attached_tip_keeps_overlap_under_load(self):
        cfg=load_config('config/hgdagger-cameras.json')
        cfg['rack']['center_xy']=[.65,.25]
        sim=DynamicSimulation(cfg,Episode(HGDAGGER_DATASET,15),source='low_level')
        p=cfg['pipette'];site=sim.data.site('nozzle');rotation=site.xmat.reshape(3,3).copy()
        sim.data.qpos[sim.tip_q:sim.tip_q+3]=site.xpos+rotation@np.array([0,0,p['seated_insertion_depth']-p['tip_length']/2])
        sim.data.qpos[sim.tip_q+3:sim.tip_q+7]=np.roll(Rotation.from_matrix(rotation).as_quat(),1)
        sim.data.eq_active[sim.retention_id]=False
        mujoco.mj_forward(sim.model,sim.data)
        sim._capture_weld(sim.attachment_id,'pipette','tip');sim.data.eq_active[sim.attachment_id]=True;sim.state='attached'
        sim.data.xfrc_applied[sim.model.body('tip').id,:3]=[.1,0,-.2]
        for _ in range(round(.3/sim.model.opt.timestep)):sim.step()
        tip=sim.data.body('tip');r=tip.xmat.reshape(3,3);mouth=tip.xpos+r[:,2]*p['tip_length']/2
        error=r.T@(sim.data.site('nozzle').xpos-mouth)
        self.assertAlmostEqual(float(-error[2]),p['seated_insertion_depth'],delta=.0003)
        self.assertLess(float(np.linalg.norm(error[:2])),.0003)
        self.assertFalse(sim.report()['warnings'])
