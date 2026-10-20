"""Run with Blender: blender -b FILE.blend --python this_script.py -- --output DIR.

Bakes evaluated world transforms (including nonuniform scales) into OBJ meshes.
Materials/UV image textures are exported; connected-component convex hulls provide
static collision proxies for props. The raw Environment_original scan is excluded
by default. Only separately modeled objects are exported, on a clean floor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import bpy
import bmesh
import numpy as np
from mathutils import Vector


def text(values):
    return ' '.join(f'{float(x):.9g}' for x in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-object', action='append', default=['Environment_original'],
                        help='Blender object to omit; the raw room scan is always excluded by default.')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out = args.output.resolve(); (out/'meshes').mkdir(parents=True, exist_ok=True)
    (out/'textures').mkdir(exist_ok=True)
    root = ET.Element('mujoco', model='LIDAR Blender environment')
    ET.SubElement(root, 'compiler', angle='radian', meshdir='meshes', texturedir='textures')
    ET.SubElement(root, 'option', gravity='0 0 -9.81', timestep='0.002')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1920', offheight='1080')
    ET.SubElement(visual, 'headlight', ambient='.45 .45 .45', diffuse='.7 .7 .7')
    assets = ET.SubElement(root, 'asset'); world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 5', directional='true', dir='0 0 -1', diffuse='.7 .7 .7')
    deps = bpy.context.evaluated_depsgraph_get(); records=[]; excluded=[]; all_bounds=[]; textures={}; materials={}
    unit_scale = bpy.context.scene.unit_settings.scale_length
    def write_mesh(name, vertices, faces, uvs=None):
        path=out/'meshes'/f'{name}.obj'
        with path.open('w') as f:
            for v in vertices: f.write('v '+text(v)+'\n')
            if uvs is not None:
                for uv in uvs: f.write('vt '+text(uv)+'\n')
            for i, face in enumerate(faces):
                f.write('f '+' '.join(str(v+1)+(f'/{3*i+j+1}' if uvs is not None else '') for j,v in enumerate(face))+'\n')
        ET.SubElement(assets,'mesh',name=name,file=f'{name}.obj')
    def material(mat):
        key=mat.name if mat else 'default'
        if key in materials: return materials[key]
        name=f'env_material_{len(materials)}';materials[key]=name
        color=list(mat.diffuse_color) if mat else [.7,.7,.7,1]
        attrs={'name':name,'rgba':text(color),'specular':'.2','shininess':'.2'}
        if mat and mat.use_nodes:
            bsdf=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
            if bsdf:
                attrs['rgba']=text(bsdf.inputs['Base Color'].default_value)
                attrs['shininess']=str(1-float(bsdf.inputs['Roughness'].default_value))
                # Direct image-to-base-color links are the scanned-room texture.
                links=bsdf.inputs['Base Color'].links
                node=links[0].from_node if links else None
                if node and node.type=='TEX_IMAGE' and node.image:
                    img=node.image
                    if img.name not in textures:
                        tname=f'env_texture_{len(textures)}';textures[img.name]=tname
                        img.save_render(str(out/'textures'/f'{tname}.png'))
                        ET.SubElement(assets,'texture',name=tname,type='2d',file=f'{tname}.png')
                    attrs['texture']=textures[img.name]; attrs['rgba']='1 1 1 1'
                alpha=float(bsdf.inputs['Alpha'].default_value)
                rgba=attrs['rgba'].split();rgba[3]=str(alpha);attrs['rgba']=' '.join(rgba)
        ET.SubElement(assets,'material',**attrs)
        return name
    for idx,obj in enumerate(sorted(bpy.context.scene.objects,key=lambda o:o.name)):
        if obj.type!='MESH' or obj.hide_render: continue
        if obj.name in args.exclude_object:
            excluded.append({'name':obj.name,'reason':'Raw scan omitted; use separately modeled object meshes.'})
            continue
        evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
        verts=[tuple((obj.matrix_world@v.co)*unit_scale) for v in mesh.vertices]
        if not verts: evaluated.to_mesh_clear();continue
        slug=f'env_{idx:02d}_'+re.sub(r'[^a-zA-Z0-9]+','_',obj.name).strip('_').lower()
        body=ET.SubElement(world,'body',name=slug)
        bounds=[[min(v[i] for v in verts),max(v[i] for v in verts)] for i in range(3)];all_bounds.append(bounds)
        by_material={}
        for tri in mesh.loop_triangles: by_material.setdefault(tri.material_index,[]).append(tri)
        for mi,tris in by_material.items():
            name=f'{slug}_visual_{mi}'
            ids=sorted({v for t in tris for v in t.vertices});remap={v:i for i,v in enumerate(ids)}
            faces=[tuple(remap[v] for v in t.vertices) for t in tris]
            uv=mesh.uv_layers.active
            uvs=[tuple(uv.data[l].uv) for t in tris for l in t.loops] if uv else None
            write_mesh(name,[verts[v] for v in ids],faces,uvs)
            mat=material(mesh.materials[mi] if mi<len(mesh.materials) else None)
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,material=mat,contype='0',conaffinity='0',group='1')
        # Split topology before taking convex hulls: preserves gaps between table legs,
        # separate tubes and hardware. Each connected concave part is approximate.
        collision_count=0
        room=obj.name.lower().startswith('environment')
        is_table=obj.name=='Table_new'
        if is_table:
            # This authored table consists of five axis-aligned cuboids with
            # face-split vertices (24 per cuboid). Preserve the open leg space.
            if len(verts)%24:raise ValueError('Table topology changed; rebuild its collision decomposition')
            for ci,start in enumerate(range(0,len(verts),24)):
                points=np.asarray(verts[start:start+24]);lo=points.min(axis=0);hi=points.max(axis=0)
                if len(np.unique(np.round(points,6),axis=0))!=8:
                    raise ValueError('Table parts must be cuboids; inspect the new mesh')
                if not np.all(np.isclose(points,lo,atol=1e-6)|np.isclose(points,hi,atol=1e-6)):
                    raise ValueError('Rotated table parts need oriented box collision export')
                ET.SubElement(body,'geom',name=f'{slug}_collision_{ci}',type='box',pos=text((lo+hi)/2),
                              size=text((hi-lo)/2),group='3',rgba='.9 .2 .1 .25',contype='1',conaffinity='15',friction='.8 .005 .0001')
                collision_count+=1
        if not room and not is_table:
            # Weld duplicate positions across exported face/material seams first.
            parents=list(range(len(verts)))
            canonical={}
            for vi,v in enumerate(verts):
                key=tuple(round(x,6) for x in v)
                if key in canonical:parents[vi]=canonical[key]
                else:canonical[key]=vi
            def find(a):
                while parents[a]!=a: parents[a]=parents[parents[a]];a=parents[a]
                return a
            for edge in mesh.edges:
                a,b=map(find,edge.vertices);parents[a]=b
            components={}
            for t in mesh.loop_triangles: components.setdefault(find(t.vertices[0]),[]).append(t)
            for ci,tris in enumerate(components.values()):
                ids=sorted({v for t in tris for v in t.vertices})
                points=[verts[v] for v in ids]
                ext=[max(v[k] for v in points)-min(v[k] for v in points) for k in range(3)]
                if len(ids)<4 or min(ext)<1e-6: continue
                singular=np.linalg.svd(np.asarray(points)-np.mean(points,axis=0),compute_uv=False)
                if singular[-1]<1e-6:continue
                name=f'{slug}_collision_{ci}';remap={v:i for i,v in enumerate(ids)}
                # Export the actual hull rather than duplicating high-poly visuals.
                # MuJoCo would take the same convex hull for mesh collisions.
                bm=bmesh.new()
                for point in np.unique(np.round(points,8),axis=0):bm.verts.new(point)
                hull=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False)
                faces=[f for f in hull['geom'] if isinstance(f,bmesh.types.BMFace)]
                bmesh.ops.triangulate(bm,faces=faces)
                used=sorted({v for f in bm.faces for v in f.verts},key=lambda v:tuple(v.co));lookup={v:i for i,v in enumerate(used)}
                write_mesh(name,[tuple(v.co) for v in used],[tuple(lookup[v] for v in f.verts) for f in bm.faces])
                bm.free()
                ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,group='3',rgba='.9 .2 .1 .25',contype='1',conaffinity='15',friction='.8 .005 .0001')
                collision_count+=1
        records.append({'name':obj.name,'body':slug,'bounds_m':bounds,'dimensions_world_m':[b-a for a,b in bounds],
                        'vertices':len(verts),'triangles':len(mesh.loop_triangles),'visual_parts':len(by_material),
                        'collision_parts':collision_count,'collision':'visual-only concave room scan' if room else ('five exact cuboid table parts' if is_table else 'connected-component convex hulls; static approximate props')})
        evaluated.to_mesh_clear()
    lo=[min(b[k][0] for b in all_bounds) for k in range(3)]; hi=[max(b[k][1] for b in all_bounds) for k in range(3)]
    ET.SubElement(world,'geom',name='env_ground',type='plane',pos=text([0,0,lo[2]]),size='10 10 .02',rgba='.35 .35 .35 1',contype='1',conaffinity='15')
    ET.SubElement(root,'statistic',center=text([(a+b)/2 for a,b in zip(lo,hi)]),extent=str(max(b-a for a,b in zip(lo,hi))/2))
    ET.indent(root);ET.ElementTree(root).write(out/'scene.xml',encoding='unicode')
    source=Path(bpy.data.filepath)
    manifest={'schema_version':1,'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'blender_version':bpy.app.version_string,'unit_scale_m':unit_scale,'coordinates':'Blender world XYZ, Z up, meters; transforms baked; no resizing',
              'bounds_m':list(zip(lo,hi)),'objects':records,'excluded_objects':excluded,
              'floor':{'z_m':lo[2],'method':'Lowest point of retained modeled meshes; no object transforms changed'},'limitations':['Static furniture/props; no articulated lids or movable pipette in this scene asset.',
              'Prop collisions are convex hulls of disconnected mesh components, not validated contact surfaces.',
              'Raw room scan and its embedded furniture/robot imagery are excluded; no unmodeled room clutter or walls are inferred.',
              'Principled base colors and directly linked image textures exported; complex Blender shaders are not reproduced.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    referenced={m.get('file') for m in assets.findall('mesh')}
    for stale in (out/'meshes').glob('env_*.obj'):
        if stale.name not in referenced:stale.unlink()
    referenced_textures={t.get('file') for t in assets.findall('texture')}
    for stale in (out/'textures').glob('env_texture_*.png'):
        if stale.name not in referenced_textures:stale.unlink()
    manifest['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),'objects':len(records),'collision_parts':sum(r['collision_parts'] for r in records)}),flush=True)

if __name__=='__main__':main()
