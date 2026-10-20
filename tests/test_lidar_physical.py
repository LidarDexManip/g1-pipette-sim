import unittest
from pathlib import Path

import mujoco
import numpy as np

from g1_pipette_sim.environments import ROOT
from g1_pipette_sim.environments.physical import build_physical_scene, PhysicalReplay
from g1_pipette_sim.dataset import Episode


class PhysicalLidarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_string(build_physical_scene())
        root=ROOT/'.cache/huggingface/hub/datasets--jren313--g1-pipette-2view-teleop0925-5task-eerel/snapshots/c7adfc179e5e15ade41b06803542001a24959dda'
        cls.episode=Episode(root,0) if (root/'meta/info.json').exists() else None

    def setUp(self):
        self.data=mujoco.MjData(self.model)
        self.replay=PhysicalReplay(self.model,self.data,self.episode)

    def test_free_pipette_falls_under_gravity_without_pose_overwrite(self):
        m,d=self.model,self.data
        joint=m.joint('env_01_pipette_free');q=int(joint.qposadr[0]);v=int(joint.dofadr[0])
        d.qpos[q+2]+=.3
        start=d.qpos[q:q+3].copy()
        self.replay.advance(.05)
        self.assertLess(d.qpos[q+2],start[2]-.008)
        self.assertGreater(d.qpos[q+2],start[2]-.02)
        self.assertLess(d.qvel[v+2],-.4)
        self.assertAlmostEqual(m.body_mass[m.body('env_01_pipette').id],.08)
        self.assertTrue(np.all(m.actuator_forcelimited))
        self.assertFalse(any(t==mujoco.mjtEq.mjEQ_WELD for t in m.eq_type))

    def test_contact_support_prevents_free_fall_and_servos_are_force_limited(self):
        m,d=self.model,self.data
        initial=d.body('env_01_pipette').xipos.copy()
        self.replay.advance(.2)
        self.assertGreater(d.body('env_01_pipette').xipos[2],initial[2]-.03)
        pipette=m.body('env_01_pipette').id
        normal_force=0.
        for i,c in enumerate(d.contact):
            if pipette in m.geom_bodyid[[c.geom1,c.geom2]]:
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);normal_force+=force[0]
        self.assertGreater(normal_force,.1,'Pipette must be supported by contact forces')
        self.assertTrue(np.all(d.actuator_force<=m.actuator_forcerange[:,1]+1e-6))
        self.assertTrue(np.all(d.actuator_force>=m.actuator_forcerange[:,0]-1e-6))
        self.assertEqual(self.replay.steps,200)
        if self.episode:
            state=self.replay.reference.targets(d.time)
            target=np.array([state[n] for n in self.replay.names])
            self.assertGreater(np.max(np.abs(d.qpos[self.replay.addresses]-target)),.001,
                               'Live positions must be integrated, not overwritten by recorded poses')


if __name__=='__main__':unittest.main()
