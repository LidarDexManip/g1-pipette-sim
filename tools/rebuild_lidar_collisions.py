"""Generate concavity-preserving LIDAR colliders from unchanged visual assets.

Optional build dependencies: requirements-collision.txt. Runtime uses exported OBJs.
CoACD API: https://github.com/SarahWeiii/CoACD
Run --object env_00_cylinder_base to build one object; --apply installs its result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import signal
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT/'.analysis-deps'))
import trimesh
import coacd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

ASSET = ROOT/'assets/environments/lidar'


def thin_ring(piece):
    center=piece.vertices.mean(axis=0)
    _,basis=np.linalg.eigh(np.cov((piece.vertices-center).T))
    basis=basis[:,[1,2,0]]  # Smallest variance is the ring's thickness axis.
    local=(piece.vertices-center)@basis
    z0,z1=local[:,2].min(),local[:,2].max()
    radii=np.linalg.norm(local[:,:2],axis=1)
    inner,outer=radii.min(),radii.max()
    vertex=local[np.argmax(radii)]
    phase=np.arctan2(vertex[1],vertex[0])
    angles=np.linspace(phase,phase+2*np.pi,25)
    result=[]
    for a,b in zip(angles[:-1],angles[1:]):
        points=[[*(r*np.array([np.cos(t),np.sin(t)])),z]
                for z in (z0,z1) for r in (inner,outer) for t in (a,b)]
        result.append(trimesh.convex.convex_hull(np.asarray(points)@basis.T+center))
    return result


def open_cup(piece):
    """Give the single-sided centrifuge well 0.4 mm runtime walls and a bottom."""
    center=piece.bounds.mean(axis=0)[:2]
    profile=[]
    for z in np.unique(np.round(piece.vertices[:,2],7)):
        points=piece.vertices[np.abs(piece.vertices[:,2]-z)<1e-7]
        profile.append((z,np.linalg.norm(points[:,:2]-center,axis=1).max()))
    thickness=.0008
    angles=np.linspace(0,2*np.pi,33);pieces=[]
    for (z0,r0),(z1,r1) in zip(profile[:-1],profile[1:]):
        for a,b in zip(angles[:-1],angles[1:]):
            points=[[*(center+r*np.array([np.cos(t),np.sin(t)])),z]
                    for z,ro in [(z0,r0),(z1,r1)]
                    for r in (ro,ro-thickness) for t in (a,b)]
            pieces.append(trimesh.convex.convex_hull(points))
    cap=trimesh.creation.cylinder(radius=profile[0][1],height=thickness,sections=32)
    cap.apply_translation([*center,profile[0][0]+thickness/2]);pieces.append(cap)
    return pieces


def perforated_plate(piece):
    """Extrude convex unions of existing top triangles; never bridge a hole."""
    z0,z1=piece.bounds[:,2]
    faces=piece.faces[np.all(np.abs(piece.vertices[piece.faces,2]-z1)<1e-7,axis=1)]
    xy,inv=np.unique(np.round(piece.vertices[:,:2],7),axis=0,return_inverse=True)
    polygons={}
    for i,face in enumerate(faces):
        indices=list(inv[face])
        hull=ConvexHull(xy[indices])
        polygons[i]=(np.array(indices)[hull.vertices].tolist(),hull.volume)
    for _ in range(40):
        owners={};merged=0
        for key in list(polygons):
            if key not in polygons:continue
            vertices,_area=polygons[key]
            for edge in zip(vertices,vertices[1:]+vertices[:1]):
                edge=tuple(sorted(edge))
                other=owners.get(edge)
                if other is None or other not in polygons or other==key:
                    owners[edge]=key;continue
                first,area1=polygons[key];second,area2=polygons[other]
                union=list(set(first+second));hull=ConvexHull(xy[union])
                if abs(hull.volume-area1-area2)<1e-11:
                    polygons[key]=(np.array(union)[hull.vertices].tolist(),hull.volume)
                    del polygons[other];merged+=1
                owners[edge]=key
        if not merged:break
    output=[]
    for vertices,_area in polygons.values():
        points=np.vstack([np.c_[xy[vertices],np.full(len(vertices),z)] for z in (z0,z1)])
        output.append(trimesh.convex.convex_hull(points))
    print('Perforated plate:',len(faces),'top triangles ->',len(output),'convex prisms',flush=True)
    return output


def stand_sections():
    """Clip original surfaces into <=1 mm runtime height bands and 16 sectors.

    Sector origin lies inside the bore, inferred from visual downward-ray samples.
    Convexification is local to each band/sector, so it cannot cap the whole bore.
    """
    mesh=visual_mesh('env_06_pipette_stand')
    center=np.array([.383,.364,0.])
    levels=np.linspace(mesh.bounds[0,2],mesh.bounds[1,2],117)
    angles=np.linspace(0,2*np.pi,17)
    output=[]
    for i,(z0,z1) in enumerate(zip(levels[:-1],levels[1:])):
        band=mesh.slice_plane([0,0,z0],[0,0,1]).slice_plane([0,0,z1],[0,0,-1])
        for a,b in zip(angles[:-1],angles[1:]):
            section=band.slice_plane(center,[-np.sin(a),np.cos(a),0])
            section=section.slice_plane(center,[np.sin(b),-np.cos(b),0])
            if len(section.vertices)<3:continue
            # The modeled base includes zero-thickness faces. Preserve them as
            # thin walls rather than dropping planar fragments as zero-volume.
            offsets=np.vstack([np.eye(3),-np.eye(3)])*.0004
            points=(section.vertices[:,None,:]+offsets[None,:,:]).reshape(-1,3)
            hull=trimesh.convex.convex_hull(points)
            if hull.volume>1e-14:output.append(hull)
        if i%20==0:print('Stand section',i,'/',len(levels)-1,flush=True)
    return output


def tube_shell():
    """Revolve measured elliptical wall profiles; retain a closed conical bottom."""
    mesh=visual_mesh('env_02_cylinder')
    parts=mesh.split(only_watertight=False)
    outer=next(p for p in parts if len(p.faces)==1152)
    inner=next(p for p in parts if len(p.faces)==720)
    center=outer.bounds.mean(axis=0)[:2]
    def profile(piece):
        v=piece.vertices
        rows=[]
        for z in np.unique(np.round(v[:,2],7)):
            ring=v[np.abs(v[:,2]-z)<1e-7]
            rows.append([z,*np.max(np.abs(ring[:,:2]-center),axis=0)])
        return np.asarray(rows)
    op,ip=profile(outer),profile(inner)
    levels=np.unique(np.r_[op[:,0],ip[:,0]])
    angles=np.linspace(0,2*np.pi,33)
    pieces=[]
    def radius(p,z):
        return np.array([np.interp(z,p[:,0],p[:,i]) for i in (1,2)])
    def wedge(z0,z1,r0,r1,inner0,inner1,a,b):
        points=[]
        for z,ro,ri in [(z0,r0,inner0),(z1,r1,inner1)]:
            for rad in (ro,ri):
                for angle in (a,b):
                    points.append([*(center+rad*np.array([np.cos(angle),np.sin(angle)])),z])
        return trimesh.convex.convex_hull(np.asarray(points))
    for z0,z1 in zip(levels[:-1],levels[1:]):
        ri0=radius(ip,z0) if z0>=ip[0,0] else np.zeros(2)
        ri1=radius(ip,z1) if z1>=ip[0,0] else np.zeros(2)
        for a,b in zip(angles[:-1],angles[1:]):
            pieces.append(wedge(z0,z1,radius(op,z0),radius(op,z1),ri0,ri1,a,b))
    # Rims are separate closed solids. Preserve their radial cross sections in
    # short axial bands; decoration blocks retain their individual convex hulls.
    for part in parts:
        if len(part.faces) in (1152,720):continue
        if len(part.faces)!=1024:
            pieces.append(part.convex_hull);continue
        zlevels=np.unique(np.round(part.vertices[:,2],7))
        for z0,z1 in zip(zlevels[:-1],zlevels[1:]):
            rr=[]
            for z in (z0,z1):
                points=part.vertices[np.abs(part.vertices[:,2]-z)<1e-7,:2]-center
                # Elliptical radii: correct for the modeled Y/X scaling.
                ratio=op[:,2].max()/op[:,1].max()
                radii=np.linalg.norm(points/np.array([1,ratio]),axis=1)
                rr.append((radii.max()*np.array([1,ratio]),radii.min()*np.array([1,ratio])))
            for a,b in zip(angles[:-1],angles[1:]):
                pieces.append(wedge(z0,z1,rr[0][0],rr[1][0],rr[0][1],rr[1][1],a,b))
    return pieces


def visual_mesh(body):
    meshes=[trimesh.load(p, force='mesh', process=False)
            for p in sorted((ASSET/'meshes').glob(body+'_visual_*.obj'))]
    mesh=trimesh.util.concatenate(meshes)
    mesh.merge_vertices(merge_tex=True,merge_norm=True,digits_vertex=7)
    mesh.update_faces(mesh.nondegenerate_faces());mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def generate(body):
    if body=='env_02_cylinder':return tube_shell()
    if body=='env_06_pipette_stand':return stand_sections()
    mesh=visual_mesh(body)
    # Watertight disconnected solids can be decomposed independently. Nonmanifold
    # material seams must be remeshed together rather than convexified separately.
    edges=mesh.edges
    count,labels=connected_components(coo_matrix(
        (np.ones(len(edges)),(edges[:,0],edges[:,1])),
        shape=(len(mesh.vertices),len(mesh.vertices))),directed=False)
    face_labels=labels[mesh.faces[:,0]]
    pieces=[mesh.submesh([np.flatnonzero(face_labels==i)],append=True) for i in range(count)]
    output=[]
    coacd.set_log_level('info')
    for i,piece in enumerate(pieces):
        if len(piece.vertices)<4:continue
        # Watertight is not sufficient: inconsistent winding makes CoACD's
        # inside/outside tests invalid and can cause pathological subdivision.
        trimesh.repair.fix_normals(piece,multibody=True)
        if body=='env_03_dome_object_open_assembly' and len(piece.faces)==448:
            output.extend(open_cup(piece));continue
        if body=='env_05_pin_rack' and len(piece.faces)==192:
            output.extend(thin_ring(piece));continue
        if body=='env_05_pin_rack' and len(piece.faces)>10000 and piece.extents[2]<.004:
            output.extend(perforated_plate(piece));continue
        hull=piece.convex_hull
        if piece.is_volume and abs(piece.volume-hull.volume)<max(hull.volume*1e-5,1e-13):
            output.append(hull);continue
        if len(piece.faces)>30000:
            piece=piece.simplify_quadric_decimation(face_count=30000)
            trimesh.repair.fix_normals(piece,multibody=True)
        print(body,'part',i,'triangles',len(piece.faces),'watertight',piece.is_watertight,flush=True)
        result=coacd.run_coacd(
            coacd.Mesh(np.asarray(piece.vertices),np.asarray(piece.faces)),
            threshold=.001, real_metric=True, preprocess_mode='off' if piece.is_volume else 'on',
            preprocess_resolution=150, resolution=4000, mcts_nodes=15,
            mcts_iterations=60, mcts_max_depth=3, merge=True, seed=42)
        output.extend(trimesh.Trimesh(v,f,process=False) for v,f in result)
    return output


def save(body, pieces):
    base=ASSET/'meshes'/'collision_v2';base.mkdir(exist_ok=True)
    digest=hashlib.sha256()
    for piece in pieces:
        digest.update(np.asarray(piece.vertices,dtype=np.float64).tobytes())
        digest.update(np.asarray(piece.faces,dtype=np.int64).tobytes())
    # Building a new version must not overwrite meshes used by the active scene.
    folder=base/f'{body}-{digest.hexdigest()[:16]}';folder.mkdir(exist_ok=True)
    files=[]
    for i,piece in enumerate(pieces):
        path=folder/f'{body}_{i:04d}.obj'
        # No materials/UVs or visual asset rewriting.
        with path.open('w') as f:
            for vertex in piece.vertices:f.write('v '+' '.join(f'{v:.10g}' for v in vertex)+'\n')
            for face in piece.faces:f.write('f '+' '.join(str(v+1) for v in face)+'\n')
        files.append(str(path.relative_to(ASSET)))
    record={'body':body,'files':files,
            'method':('32-sector hollow elliptical wall profiles with closed bottom and separate rims'
                      if body=='env_02_cylinder' else
                      '16 angular sectors and <=1 mm height sections; 0.2 mm collision skin for zero-thickness faces'
                      if body=='env_06_pipette_stand' else
                      'Explicit thin annuli and extruded perforated plate; CoACD for remaining parts'
                      if body=='env_05_pin_rack' else
                      'Explicit 0.4 mm well walls and closed bottoms; winding-corrected CoACD for housing and lid'
                      if body=='env_03_dome_object_open_assembly' else 'Winding-corrected CoACD decomposition'),
            'coacd_threshold_source_m':None if body in ('env_02_cylinder','env_06_pipette_stand') else .001,
            'runtime_scale':.5,
            'contact_parameters':{'solref':[.004,1],'solimp':[.99,.99,.001,.5,2],'priority':1},
            'source_visual_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ASSET/'meshes').glob(body+'_visual_*.obj'))}}
    (base/f'{body}.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Generated',body,len(files),'convex parts',flush=True)


def apply(body):
    record=json.loads((ASSET/'meshes/collision_v2'/f'{body}.json').read_text())
    for name,expected in record['source_visual_sha256'].items():
        if hashlib.sha256((ASSET/'meshes'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Visual source changed since collision generation: {name}')
    root=ET.parse(ASSET/'scene.xml').getroot();asset=root.find('asset')
    node=root.find(f"./worldbody/body[@name='{body}']")
    old_meshes=set()
    for geom in list(node.findall('geom')):
        if int(geom.get('contype','1'))!=0:
            if geom.get('mesh'):old_meshes.add(geom.get('mesh'))
            node.remove(geom)
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name') in old_meshes:asset.remove(mesh)
    for i,file in enumerate(record['files']):
        name=f'{body}_collision_v2_{i}'
        ET.SubElement(asset,'mesh',name=name,file=str(Path(file).relative_to('meshes')))
        ET.SubElement(node,'geom',name=name,type='mesh',mesh=name,group='3',
                      rgba='.9 .2 .1 .25',contype='1',conaffinity='15',friction='.8 .005 .0001',
                      solref='.004 1',solimp='.99 .99 .001 .5 2',priority='1')
    # Compile before installing, so invalid or zero-volume hulls cannot break startup.
    import copy
    import mujoco
    check=copy.deepcopy(root)
    check.find('compiler').set('meshdir',str(ASSET/'meshes'))
    mujoco.MjModel.from_xml_string(ET.tostring(check,encoding='unicode'))
    ET.indent(root);ET.ElementTree(root).write(ASSET/'scene.xml',encoding='unicode')
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for obj in manifest['objects']:
        if obj['body']==body:obj.update(collision_parts=len(record['files']),collision=record['method'])
    for file in record['files']+['scene.xml']:
        data=(ASSET/file).read_bytes()
        manifest['artifacts'][file]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    (ASSET/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    objects=[o['body'] for o in json.loads((ASSET/'manifest.json').read_text())['objects']
             if o['body']!='env_07_table_new']
    parser.add_argument('--object',required=True,choices=objects,
                        help='The table retains its already validated five exact box colliders.')
    parser.add_argument('--apply',action='store_true',help='Install an already generated object')
    args=parser.parse_args()
    from g1_pipette_sim.environments import unpack_meshes
    unpack_meshes(ASSET/'scene.xml')
    if args.apply:apply(args.object)
    else:
        # CoACD runs native code; default SIGINT terminates it immediately rather
        # than deferring Python's KeyboardInterrupt until decomposition returns.
        signal.signal(signal.SIGINT,signal.SIG_DFL)
        save(args.object,generate(args.object))


if __name__=='__main__':main()
