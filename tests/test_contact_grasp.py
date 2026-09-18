import unittest
import numpy as np
import mujoco
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dynamics import DynamicSimulation

class ContactGraspTests(unittest.TestCase):
    def test_supported_by_contacts_in_free_space_and_free_when_displaced(self):
        cfg=load_config('config/hgdagger-cameras.json');cfg['base_position'][2]+=.25
        sim=DynamicSimulation(cfg,Episode(HGDAGGER_DATASET,0),source='low_level')
        reference=sim._reference;sim._reference=lambda t:reference(0)
        mass=sum(sim.model.body_mass[sim.model.body(n).id] for n in ['pipette','ejector'])
        self.assertAlmostEqual(mass,.080,places=8)
        for _ in range(round(2/sim.model.opt.timestep)):sim.step()
        self.assertFalse(sim.data.eq_active[sim.grip_id])
        self.assertFalse(sim.data.eq_active[sim.attachment_id])
        self.assertLess(sim.report()['grip_position_error_m'],.025)
        forces=[]
        for contact_id,contact in enumerate(sim.data.contact):
            bodies=[sim.model.body(sim.model.geom_bodyid[g]).name for g in [contact.geom1,contact.geom2]]
            if 'pipette' in bodies and any(n.startswith('R_') for n in bodies):
                force=np.zeros(6);mujoco.mj_contactForce(sim.model,sim.data,contact_id,force);forces.append(force[0])
        self.assertGreater(sum(forces),.1)
        self.assertFalse(any('pipette' in p and any(n in p for n in ['table_body','rack','waste','world']) for p in sim.contact_pairs))
        sim.release_grip();sim.step()
        for i,j in enumerate(sim.joints):
            if sim.model.joint(int(j)).name=='R_index_proximal_joint':self.assertEqual(sim.desired[i],0.)
        # Move away from every supporting surface as a test setup perturbation:
        # an unheld tool must fall, rather than follow a hidden weld or pose reset.
        q=int(sim.model.joint('pipette_free').qposadr[0]);v=int(sim.model.joint('pipette_free').dofadr[0])
        sim.data.qpos[q:q+3]+=[1.2,0,.2];sim.data.qvel[v:v+6]=0.;mujoco.mj_forward(sim.model,sim.data)
        z=sim.data.body('pipette').xpos[2]
        for _ in range(round(.05/sim.model.opt.timestep)):sim.step()
        self.assertGreater(z-sim.data.body('pipette').xpos[2],.010)
        self.assertFalse(sim.report()['warnings'])
