import unittest
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dataset import Episode,HGDAGGER_DATASET
from g1_pipette_sim.dynamics import DynamicSimulation


def vertices(sim,g,body):
    m,d=sim.model,sim.data;mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh]
    v=m.mesh_vert[start:start+m.mesh_vertnum[mesh]]
    return (v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-d.body(body).xpos)@d.body(body).xmat.reshape(3,3)


def pair_contacts(sim,a,b):
    ids={sim.model.body(a).id,sim.model.body(b).id}
    return [(i,c) for i,c in enumerate(sim.data.contact) if {int(sim.model.geom_bodyid[c.geom1]),int(sim.model.geom_bodyid[c.geom2])}==ids]


def nozzle_at(sim,position):
    q=sim.model.jnt_qposadr[sim.model.joint('pipette_free').id]
    sim.data.qpos[q:q+7]=[*np.asarray(position)-np.array([0,0,sim.config['pipette']['nozzle_z']]),1,0,0,0]
    mujoco.mj_forward(sim.model,sim.data)


class CollisionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim=DynamicSimulation(load_config('config/hgdagger-cameras.json'),Episode(HGDAGGER_DATASET,15),source='low_level')

    def setUp(self):
        self.sim.reset()

    def test_compiled_meshes_share_visual_coordinate_frames(self):
        sim=self.sim;m=sim.model;p=sim.config['pipette']
        for name,key in [('pipette_lower','shaft_profile'),('pipette_nozzle','nozzle_profile')]:
            profile=np.asarray(p[key]);sets=[]
            for g in sim.tool_geom_ids:
                if m.geom(g).name==name or m.geom(g).name.startswith(name+'_'):
                    v=vertices(sim,g,'pipette');sets.append(v)
                    v=v[np.linalg.norm(v[:,:2],axis=1)>1e-6]  # cap centers are interior vertices
                    np.testing.assert_allclose(np.linalg.norm(v[:,:2],axis=1),np.interp(v[:,2],profile[:,0],profile[:,1]),atol=1e-8)
            self.assertEqual(len(sets),len(profile)-1)
            visual=next(g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name=='object_'+name and m.geom_group[g]==2)
            v=vertices(sim,visual,'pipette');c=np.vstack(sets)
            np.testing.assert_allclose([c.min(0),c.max(0)],[v.min(0),v.max(0)],atol=1e-8)
        visual=next(g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name=='object_rack_plate')
        rack_collision=[vertices(sim,g,'rack') for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g]==m.body('rack').id and m.mesh(m.geom_dataid[g]).name.startswith('object_well_sector_')]
        v=vertices(sim,visual,'rack');c=np.vstack(rack_collision)
        self.assertLess(cKDTree(c).query(v)[0].max(),1e-7)
        self.assertLess(cKDTree(v).query(c)[0].max(),1e-7)
        outer=np.array(p['tip_profile']);inner=np.array(p['tip_inner_profile'])
        for g in sim.tip_geoms:
            v=vertices(sim,g,'tip');z=v[:,2]+p['tip_length']/2;radius=np.linalg.norm(v[:,:2],axis=1)
            error=np.minimum(abs(radius-np.interp(z,outer[:,0],outer[:,1])),abs(radius-np.interp(z,inner[:,0],inner[:,1])))
            self.assertLess(error.max(),1e-8)
        self.assertAlmostEqual(float(m.body_mass[m.body('tip').id]),.0006,places=9)

    def test_hollow_tip_allows_axial_fit_and_blocks_sidewall(self):
        s=self.sim;s.data.eq_active[:]=False
        center=np.array([.65,-.3,1.]);s.data.qpos[s.tip_q:s.tip_q+7]=[*center,1,0,0,0]
        mouth=center+[0,0,s.config['pipette']['tip_length']/2]
        nozzle_at(s,mouth-[0,0,s.config['pipette']['seated_insertion_depth']])
        self.assertFalse(pair_contacts(s,'pipette','tip'))
        nozzle_at(s,mouth+[.001,0,-s.config['pipette']['seated_insertion_depth']])
        contacts=pair_contacts(s,'pipette','tip');self.assertTrue(contacts)
        self.assertTrue(any(c.dist<-.0001 for _,c in contacts))
        force=np.zeros(6);maximum=0.
        for i,c in contacts:
            mujoco.mj_contactForce(s.model,s.data,i,force);maximum=max(maximum,force[0])
        self.assertGreater(maximum,0.)
        for operation in [s.release_grip,s.eject,s.reset]:
            operation()
            self.assertTrue(np.all(s.model.geom_conaffinity[s.tip_geoms]&4))
            self.assertTrue(np.all(s.model.geom_conaffinity[s.tool_geom_ids]&8))

    def test_rack_web_blocks_nozzle_but_empty_well_is_open(self):
        s=self.sim;s.data.eq_active[:]=False
        s.data.qpos[s.tip_q:s.tip_q+3]=[1.5,1.5,1.]
        r=s.config['rack'];row,col=r['active_cell'];local=np.array([(col-(r['cols']-1)/2)*r['pitch'],(row-(r['rows']-1)/2)*r['pitch'],r['height']])
        rack=s.data.body('rack');rotation=rack.xmat.reshape(3,3)
        plate=rack.xpos+rotation@local
        nozzle_at(s,plate+[0,0,-.004]);self.assertFalse(pair_contacts(s,'pipette','rack'))
        nozzle_at(s,plate+rotation@np.array([r['pitch']/2,0,-.0003]))
        contacts=pair_contacts(s,'pipette','rack');self.assertTrue(contacts)
        self.assertTrue(any(c.dist<0 for _,c in contacts))
