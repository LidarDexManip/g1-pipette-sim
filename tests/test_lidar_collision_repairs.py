"""Geometry probes for openings that the original convex hulls blocked."""
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from g1_pipette_sim.environments import imported_xml


def fixture(body, position, radius=.002):
    root=ET.fromstring(imported_xml('lidar'));world=root.find('worldbody')
    for node in list(world):
        if node.tag=='body' and node.get('name')!=body:world.remove(node)
    probe=ET.SubElement(world,'body',name='probe',pos=' '.join(map(str,position)))
    ET.SubElement(probe,'freejoint')
    ET.SubElement(probe,'geom',name='probe_geom',type='sphere',size=str(radius),mass='.001')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data


class LidarCollisionRepairTests(unittest.TestCase):
    def test_tube_mouth_and_side_wall(self):
        model,data=fixture('env_02_cylinder',[.17125,-.210896,.87])
        self.assertEqual(data.ncon,0,'Tube mouth must remain open')
        data.qpos[0]+=.015
        mujoco.mj_forward(model,data)
        self.assertTrue(any(c.dist<0 for c in data.contact),'Tube wall must stop a probe')

    def test_tube_retains_dropped_probe(self):
        model,data=fixture('env_02_cylinder',[.17125,-.210896,.89])
        for _ in range(1000):mujoco.mj_step(model,data)
        z=data.body('probe').xpos[2]
        self.assertGreater(z,.744,'Tube must have a closed bottom')
        self.assertLess(z,.78,'Probe must enter the tube, not sit on an invisible cap')

    def test_rack_tube_hole_is_open(self):
        model,data=fixture('env_00_cylinder_base',[.17125,-.210896,.78])
        self.assertFalse(any(c.dist<-.0005 for c in data.contact),'Rack hole is filled')

    def test_stand_receiving_hole_is_open(self):
        model,data=fixture('env_06_pipette_stand',[.1915,.182,.827])
        self.assertFalse(any(c.dist<-.0005 for c in data.contact),'Stand opening is filled')
        data.qpos[0]+=.006
        mujoco.mj_forward(model,data)
        self.assertTrue(any(c.dist<0 for c in data.contact),'Stand wall must remain solid')

    def test_stand_planar_base_faces_are_not_dropped(self):
        model,data=fixture('env_06_pipette_stand',[.19,.18,.9])
        for y in (.15308,.15985,.16663):
            distances=[]
            for group in (1,3):
                mask=np.zeros(6,dtype=np.uint8);mask[group]=1
                distances.append(mujoco.mj_ray(model,data,np.array([.16042,y,.9]),
                                              np.array([0.,0.,-1.]),mask,1,-1,None))
            self.assertTrue(all(d>0 for d in distances))
            self.assertLess(abs(distances[0]-distances[1]),.001)

    def test_tip_rack_mouth_and_plate_hole(self):
        model,data=fixture('env_05_pin_rack',[.254,-.0226,.7657],radius=.0005)
        for x in .2288+.0084*np.arange(12):
            for y in -.0394+.0084*np.arange(8):
                for z in [.766,.7653]:
                    data.qpos[:3]=[x,y,z];mujoco.mj_forward(model,data)
                    self.assertFalse(any(c.dist<-.0001 for c in data.contact),
                                     f'Tip mouth or plate hole is capped at {x}, {y}, {z}')
        data.qpos[:3]=[.257,-.0226,.7653];mujoco.mj_forward(model,data)
        self.assertTrue(any(c.dist<0 for c in data.contact),'Tip rim must remain solid')

    def test_all_eight_centrifuge_wells_are_open(self):
        model,data=fixture('env_03_dome_object_open_assembly',[.057,.01,.829],radius=.001)
        mask=np.array([0,1,0,0,0,0],dtype=np.uint8)
        for angle in np.arange(8)*np.pi/4:
            x,y=.025+.032*np.cos(angle),.01+.032*np.sin(angle)
            distance=mujoco.mj_ray(model,data,np.array([x,y,.84]),np.array([0.,0.,-1.]),
                                  mask,1,-1,None)
            self.assertGreater(distance,0)
            floor=.84-distance
            for height in (.829,floor+.002):
                data.qpos[:3]=[x,y,height]
                mujoco.mj_forward(model,data)
                self.assertFalse(any(c.dist<-.0005 for c in data.contact),
                                 f'Centrifuge well at angle {angle} height {height} is capped')

    def test_table_supports_probe(self):
        model,data=fixture('env_07_table_new',[.25,.6,.8],radius=.02)
        for _ in range(750):mujoco.mj_step(model,data)
        self.assertAlmostEqual(data.body('probe').xpos[2],.74,delta=.002)
        self.assertTrue(np.isfinite(data.qpos).all())


if __name__=='__main__':unittest.main()
