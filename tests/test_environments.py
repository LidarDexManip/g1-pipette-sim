import copy
import json
from pathlib import Path
import unittest
import gzip
import hashlib
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from g1_pipette_sim.environments import ENVIRONMENTS, LAB_IDS, imported_xml, specification, environment_manifest, unpack_meshes
from g1_pipette_sim.scene import load_config, build_xml
from g1_pipette_sim.dynamics import build_dynamic_xml


class EnvironmentsTest(unittest.TestCase):
    def test_bundled_mesh_extraction_checks_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'meshes').mkdir()
            payload=b'v 0 0 0\n';mesh=root/'meshes/example.obj'
            packed=mesh.with_suffix('.obj.gz')
            packed.write_bytes(gzip.compress(payload))
            (root/'manifest.json').write_text(json.dumps({'artifacts':{
                'meshes/example.obj':{'sha256':hashlib.sha256(payload).hexdigest()}}}))
            unpack_meshes(root/'scene.xml')
            self.assertEqual(mesh.read_bytes(),payload)
            mesh.unlink();packed.write_bytes(gzip.compress(b'corrupted'))
            with self.assertRaisesRegex(ValueError,'does not match'):
                unpack_meshes(root/'scene.xml')
            self.assertFalse(mesh.exists())
            self.assertFalse(list((root/'meshes').glob('.unpack-*')))

    def test_default_is_unchanged_and_invalid_name_rejected(self):
        cfg=load_config(); original=copy.deepcopy(cfg)
        default=build_xml(cfg)
        explicit=copy.deepcopy(cfg);explicit['environment']='pipette'
        self.assertEqual(default,build_xml(explicit))
        self.assertEqual(cfg,original)
        self.assertEqual(specification(cfg)['name'],'pipette')
        with self.assertRaisesRegex(ValueError,'Unknown environment'):
            specification({'environment':'typo'})

    def test_all_variants_compile_without_changing_task_or_dynamic_state(self):
        cfg=load_config();base=mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg))
        for name in LAB_IDS:
            with self.subTest(environment=name):
                variant=copy.deepcopy(cfg);variant['environment']={'name':name,'seed':7}
                xml=build_dynamic_xml(variant);m=mujoco.MjModel.from_xml_string(xml)
                self.assertEqual((m.nq,m.nv,m.nu),(base.nq,base.nv,base.nu))
                np.testing.assert_array_equal(m.qpos0,base.qpos0)
                np.testing.assert_allclose(m.body_mass[m.body('table_body').id],base.body_mass[base.body('table_body').id])
                np.testing.assert_allclose(m.geom_size[m.geom('table').id],base.geom_size[base.geom('table').id])
                root=ET.fromstring(xml)
                self.assertIsNone(root.find(".//body[@name='table_body']/body[@name='environment_surroundings']"))
                body=root.find("./worldbody/body[@name='environment_surroundings']")
                self.assertIsNotNone(body)
                self.assertTrue(all(g.get('contype')=='0' and g.get('conaffinity')=='0' for g in body.findall('geom')))

    def test_imported_dimensions_and_table_collision(self):
        asset=ENVIRONMENTS['lidar']['asset']
        if not asset.exists():self.skipTest('Blender assets have not been exported')
        manifest=environment_manifest('lidar')
        self.assertEqual(len(manifest['objects']),7)
        self.assertIn('Environment_original',[o['name'] for o in manifest['excluded_objects']])
        self.assertNotIn('Environment_original',[o['name'] for o in manifest['objects']])
        self.assertFalse(any('environment_original' in name for name in manifest['artifacts']))
        self.assertFalse(list((asset.parent/'textures').glob('*')))
        self.assertNotIn('environment_original',imported_xml('lidar'))
        records={o['name'].strip():o for o in manifest['objects']}
        np.testing.assert_allclose(records['Table_new']['dimensions_world_m'],[.768,1.536,.72],atol=1e-6)
        pipette=next(o for o in records.values() if o['body'].endswith('_pipette'))
        self.assertAlmostEqual(pipette['dimensions_world_m'][2],.254,places=6)
        self.assertEqual(records['Table_new']['collision_parts'],5)
        root=ET.fromstring(imported_xml('lidar'));world=root.find('worldbody')
        body=ET.SubElement(world,'body',name='test_drop',pos='.25 .6 1.0')
        ET.SubElement(body,'freejoint');ET.SubElement(body,'geom',name='test_sphere',type='sphere',size='.02',mass='.1')
        m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
        for _ in range(750):mujoco.mj_step(m,d)
        self.assertAlmostEqual(d.body('test_drop').xpos[2],.74,delta=.002)
        # Under-table space remains empty, unlike a single table convex hull.
        mujoco.mj_forward(m,d);groups=np.array([0,0,0,1,0,0],dtype=np.uint8);hit=np.array([-1],dtype=np.int32)
        distance=mujoco.mj_ray(m,d,np.array([0.,0.,.2]),np.array([1.,0.,0.]),groups,1,-1,hit)
        self.assertEqual(distance,-1)

    def test_imported_scene_cannot_silently_replace_calibrated_task(self):
        cfg=load_config();cfg['environment']='lidar'
        with self.assertRaisesRegex(ValueError,'standalone scene'):
            build_xml(cfg)

if __name__=='__main__':unittest.main()
