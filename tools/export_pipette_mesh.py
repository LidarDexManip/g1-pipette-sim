"""Export the visible assembled pipette as OBJ/MTL, in metres and tool-local axes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np,mujoco
from g1_pipette_sim.scene import build_xml,load_config
from g1_pipette_sim.object_meshes import lathe
from g1_pipette_sim.pipette_mesh import loft

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('config/hil194.json'));parser.add_argument('--episode',type=int,default=46);parser.add_argument('--with-tip',action='store_true');parser.add_argument('--output',type=Path,default=Path('outputs/pipette-photo-refinement'));args=parser.parse_args()
cfg=load_config(args.config)
if cfg.get('dataset_profile',{}).get('sessions'):
    from g1_pipette_sim.dataset import Episode
    from g1_pipette_sim.profiles import for_episode
    cfg=for_episode(cfg,Episode(index=args.episode))
model=mujoco.MjModel.from_xml_string(build_xml(cfg));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
body=data.body('pipette');R=body.xmat.reshape(3,3);out=args.output;out.mkdir(parents=True,exist_ok=True)
if args.with_tip:
    nozzle=data.site('nozzle');rotation=nozzle.xmat.reshape(3,3);q=model.jnt_qposadr[model.joint('tip_free').id]
    from scipy.spatial.transform import Rotation
    data.qpos[q:q+3]=nozzle.xpos+rotation@np.array([0,0,cfg['pipette']['seated_insertion_depth']-cfg['pipette']['tip_length']/2])
    data.qpos[q+3:q+7]=np.roll(Rotation.from_matrix(rotation).as_quat(),1);mujoco.mj_forward(model,data)
obj=['# Photo-refined ONiLAB 0.5-10 uL. Units metres; assembled at rest.','mtllib pipette.mtl'];materials=[];offset=0
for g in range(model.ngeom):
    if model.body(model.geom_bodyid[g]).name not in (('pipette','ejector','tip') if args.with_tip else ('pipette','ejector')) or model.geom_group[g]==3:continue
    kind=model.geom_type[g];size=model.geom_size[g]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=model.geom_dataid[g];vstart=model.mesh_vertadr[mesh];fstart=model.mesh_faceadr[mesh]
        vertices=model.mesh_vert[vstart:vstart+model.mesh_vertnum[mesh]].copy();faces=model.mesh_face[fstart:fstart+model.mesh_facenum[mesh]].copy()
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        vertices,faces=lathe([[-size[1],size[0]],[size[1],size[0]]])
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        x,y,z=size;vertices,faces=loft([[[-x,-y,h],[x,-y,h],[x,y,h],[-x,y,h]] for h in [-z,z]])
    else:raise ValueError(f'Unsupported visible geometry {kind}')
    vertices=(np.asarray(vertices)@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-body.xpos)@R
    color=model.geom_rgba[g];material=f'part_{g}'
    materials.extend([f'newmtl {material}','Kd '+' '.join(map(str,color[:3])),f'd {color[3]}',''])
    obj.extend([f'o {model.geom(g).name or material}',f'usemtl {material}'])
    obj.extend('v '+' '.join(f'{x:.9g}' for x in v) for v in vertices)
    obj.extend('f '+' '.join(str(int(i)+offset+1) for i in face) for face in faces)
    offset+=len(vertices)
(out/'pipette.obj').write_text('\n'.join(obj)+'\n');(out/'pipette.mtl').write_text('\n'.join(materials))
print(out/'pipette.obj')
