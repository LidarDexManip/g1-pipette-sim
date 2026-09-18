"""Deterministic parametric meshes guided by the September 9 object photos.

All mesh coordinates are metres. These are measured-reference approximations,
not photogrammetric surfaces. Collision meshes remain separate from visuals.
"""
from pathlib import Path
import os
import tempfile
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / 'assets/objects'


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent episode workers must never observe a partially written OBJ.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
            temporary=Path(f.name)
            f.write('# Photo-guided parametric mesh; units: metres\n')
            for v in vertices: f.write('v ' + ' '.join(f'{x:.9g}' for x in v) + '\n')
            for face in faces: f.write('f ' + ' '.join(str(int(i)+1) for i in face) + '\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def lathe(profile, segments=48, caps=True):
    """Surface of revolution. Profile is ordered along its cross-section boundary."""
    vertices=[];faces=[]
    for z,r in profile:
        for a in np.linspace(0,2*np.pi,segments,endpoint=False):
            vertices.append([r*np.cos(a),r*np.sin(a),z])
    for j in range(len(profile)-1):
        for i in range(segments):
            a=j*segments+i;b=j*segments+(i+1)%segments;c=b+segments;d=a+segments
            faces.extend([[a,b,c],[a,c,d]])
    if caps:
        start=len(vertices);vertices.append([0,0,profile[0][0]])
        end=len(vertices);vertices.append([0,0,profile[-1][0]])
        for i in range(segments):
            faces.append([start,(i+1)%segments,i])
            a=(len(profile)-1)*segments+i;b=(len(profile)-1)*segments+(i+1)%segments
            faces.append([end,a,b])
    return vertices,faces


def plate_mesh(rows,cols,pitch,radius,thickness,segments=16):
    """Per-cell square-to-round annuli; open holes with real through surfaces."""
    vertices=[];faces=[];sectors=[]
    angles=np.linspace(0,2*np.pi,segments,endpoint=False)
    inner=np.c_[radius*np.cos(angles),radius*np.sin(angles)]
    unit=np.c_[np.cos(angles),np.sin(angles)]
    outer=unit*(pitch/2/np.max(np.abs(unit),axis=1))[:,None]
    # Each wedge is convex and is reused at all cell positions for contacts.
    for i in range(segments):
        nxt=(i+1)%segments
        quad=[inner[i],outer[i],outer[nxt],inner[nxt]]
        wedge=[[x,y,z] for z in [-thickness/2,thickness/2] for x,y in quad]
        f=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        sectors.append((wedge,f))
    for row in range(rows):
        for col in range(cols):
            offset=np.array([(col-(cols-1)/2)*pitch,(row-(rows-1)/2)*pitch,0])
            for wedge,sector_faces in sectors:
                start=len(vertices);vertices.extend((np.asarray(wedge)+offset).tolist())
                faces.extend([[start+i for i in f] for f in sector_faces])
    return vertices,faces,sectors


def tip_shell_parts(p, segments=48):
    """Convex annular-frustum sectors; the union retains the actual open bore."""
    length=p['tip_length']
    outer=np.asarray(p.get('tip_profile',[[0,.00045],[.004,.0009],[.025,.0025],[.029,.0031],[length,.0031]]),float)
    inner=np.asarray(p.get('tip_inner_profile',[[z,max(r-p.get('tip_wall',.00045),.0002)] for z,r in outer]),float)
    levels=np.unique(np.r_[outer[:,0],inner[:,0]])
    faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    parts=[]
    for low,high in zip(levels[:-1],levels[1:]):
        ro=np.interp([low,high],outer[:,0],outer[:,1]);ri=np.interp([low,high],inner[:,0],inner[:,1])
        if np.any(ri>=ro):raise ValueError('Tip bore must remain inside the outer profile')
        volume=np.pi*(high-low)/3*(ro[0]**2+ro[0]*ro[1]+ro[1]**2-ri[0]**2-ri[0]*ri[1]-ri[1]**2)/segments
        for i in range(segments):
            a,b=2*np.pi*np.array([i,i+1])/segments
            vertices=[]
            for z,r,inside in zip([low,high],ro,ri):
                vertices.extend([[inside*np.cos(a),inside*np.sin(a),z-length/2],
                                 [r*np.cos(a),r*np.sin(a),z-length/2],
                                 [r*np.cos(b),r*np.sin(b),z-length/2],
                                 [inside*np.cos(b),inside*np.sin(b),z-length/2]])
            parts.append((vertices,faces,volume))
    return parts


def generate(config):
    """Write config-specific meshes; filenames encode geometry to avoid stale reuse."""
    import hashlib,json
    shape={k:config[k] for k in ['rack','pipette']}
    shape['generator_version']=5
    key=hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest()[:12]
    out=ASSETS/key
    r=config['rack'];p=config['pipette'];result={}
    def save(name,v,f):
        path=out/(name+'.obj')
        if not path.exists():write_obj(path,v,f)
        result[name]=path
    v,f,sectors=plate_mesh(r['rows'],r['cols'],r['pitch'],r.get('hole_radius',.0035),r.get('plate_thickness',.002))
    save('rack_plate',v,f)
    for i,(v,f) in enumerate(sectors):save(f'well_sector_{i}',v,f)
    # Raised rims are a visible detail; bore shape is defined by the collision plate.
    v,f=lathe([[-.0003,r.get('hole_radius',.0035)],[.0008,r.get('hole_radius',.0035)],[.0008,r.get('hole_radius',.0035)+.0005],[-.0003,r.get('hole_radius',.0035)+.0005],[-.0003,r.get('hole_radius',.0035)]],segments=24,caps=False)
    save('well_rim',v,f)
    length=p['tip_length'];outer=p.get('tip_profile',[[z*length/.033,rad] for z,rad in [[0,.00045],[.004,.0009],[.025,.0025],[.029,.0031],[.033,.0031]]])
    outer=[[z-length/2,rad] for z,rad in outer]
    # The bore is explicitly an estimate: it cannot be measured in the supplied view.
    wall=p.get('tip_wall',.00045)
    inner=([[z-length/2,rad] for z,rad in reversed(p['tip_inner_profile'])] if 'tip_inner_profile' in p else [[z,max(rad-wall,.0002)] for z,rad in reversed(outer)])
    v,f=lathe(outer+inner+[outer[0]],caps=False);save('tip_visual',v,f)
    for i,(v,f,volume) in enumerate(tip_shell_parts(p)):save(f'tip_collision_{i}',v,f)
    # Body shape is partially hidden by the hand/tape; smooth outline only.
    handle=p.get('handle_profile',[[-.058,.008],[-.050,.012],[-.040,.014],[.035,.014],[.050,.012],[.057,.009]])
    v,f=lathe(handle)
    v=np.asarray(v);v[:,1]*=p.get('handle_y_scale',.8);save('pipette_handle',v.tolist(),f)
    nozzle=p['nozzle_z']
    v,f=lathe(p.get('shaft_profile',[[nozzle,.0011],[nozzle+.012,.0024],[-.080,.0044],[-.065,.006],[-.058,.008],[-.050,.008]]))
    save('pipette_lower',v,f)
    if p.get('photo_revision'):
        from .pipette_mesh import parts
        for name,(v,f) in parts(p).items():
            # Revised geometry has distinct cache keys via photo_revision.
            path=out/(name+'.obj');write_obj(path,v,f);result[name]=path
    return result
