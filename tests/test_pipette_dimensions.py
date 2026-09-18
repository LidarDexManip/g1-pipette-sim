import unittest
import numpy as np,mujoco
from g1_pipette_sim.scene import build_xml,load_config

class PipetteDimensionTests(unittest.TestCase):
    def test_measured_assembled_dimensions(self):
        m=mujoco.MjModel.from_xml_string(build_xml(load_config('config/hgdagger-cameras.json')));d=mujoco.MjData(m);mujoco.mj_forward(m,d);pipe=d.body('pipette');R=pipe.xmat.reshape(3,3);bounds=[]
        for g in range(m.ngeom):
         if m.body(m.geom_bodyid[g]).name not in ['pipette','ejector']:continue
         rot=R.T@d.geom_xmat[g].reshape(3,3);pos=R.T@(d.geom_xpos[g]-pipe.xpos);size=m.geom_size[g];kind=m.geom_type[g]
         if kind==mujoco.mjtGeom.mjGEOM_MESH:
          mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];v=m.mesh_vert[start:start+m.mesh_vertnum[mesh]]@rot.T+pos;bounds.extend([v.min(0),v.max(0)]);continue
         if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=size[0]*np.sqrt(rot[:,0]**2+rot[:,1]**2)+size[1]*abs(rot[:,2])
         elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=size[0]+size[1]*abs(rot[:,2])
         elif kind==mujoco.mjtGeom.mjGEOM_BOX:extent=abs(rot)@size
         else:raise ValueError(kind)
         bounds.extend([pos-extent,pos+extent])
        b=np.array(bounds);actual=b.max(0)-b.min(0);np.testing.assert_allclose(actual,[.0254,.0635,.254],atol=1e-6)
