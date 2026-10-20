"""Measure Euclidean surface errors at flagged rays (not vertical depth jumps)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.analysis-deps'))
import mujoco
import numpy as np
from trimesh.triangles import closest_point
from g1_pipette_sim.environments import imported_xml
from tools.audit_lidar_collisions import bounds


def triangles(model,data,ids):
    parts=[]
    for geom in ids:
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=model.geom_dataid[geom]
        v,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        f,k=model.mesh_faceadr[mesh],model.mesh_facenum[mesh]
        vertices=model.mesh_vert[v:v+n]@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        parts.append(vertices[model.mesh_face[f:f+k]])
    return np.concatenate(parts)


def distance(point,faces):
    best=float('inf')
    for start in range(0,len(faces),20000):
        chunk=faces[start:start+20000]
        closest=closest_point(chunk,np.broadcast_to(point,(len(chunk),3)))
        best=min(best,float(np.linalg.norm(closest-point,axis=1).min()))
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('outputs/lidar-teleop/collision-repair/after/report.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/lidar-teleop/collision-repair/surface-distance-checks.json'))
    args=parser.parse_args();report=json.loads(args.report.read_text())
    model=mujoco.MjModel.from_xml_string(imported_xml('lidar'));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data);groups=model.geom_group.copy();rows=[]
    for obj in report['geometry']:
        if not obj['examples'] and not obj['recessed_examples']:continue
        ids=np.flatnonzero(model.geom_bodyid==model.body(obj['body']).id)
        model.geom_group[:]=5;model.geom_group[ids]=groups[ids]
        top=max(bounds(model,data,int(g))[1,2] for g in ids)+.05
        faces={g:triangles(model,data,ids[groups[ids]==g]) for g in (1,3)}
        for category,key,source,target in [('inflated','examples',3,1),('recessed','recessed_examples',1,3)]:
            for sample in obj[key]:
                origin=np.array([sample['x']+1.234e-6,sample['y']+2.345e-6,top])
                mask=np.zeros(6,np.uint8);mask[source]=1
                hit=mujoco.mj_ray(model,data,origin,np.array([0.,0.,-1.]),mask,1,-1,None)
                if hit<0:continue
                point=origin-np.array([0.,0.,hit])
                error=distance(point,faces[target])*1000
                rows.append(dict(body=obj['body'],category=category,point_m=point.tolist(),surface_error_mm=error))
        print(obj['body'],max(x['surface_error_mm'] for x in rows if x['body']==obj['body']),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':'Flagged audit examples only; not a global Hausdorff bound.',
                                      'samples':rows},indent=2)+'\n')


if __name__=='__main__':main()
