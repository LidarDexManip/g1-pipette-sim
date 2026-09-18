"""Reproduce planar ego-camera candidates from saved world/pixel correspondences."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('landmarks',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=json.loads(a.landmarks.read_text());xyz=np.array(spec['world_points'],float);uv=np.array(spec['image_points'],float);K=np.array(spec['camera_matrix'],float)
    ok,rotations,translations,_=cv2.solvePnPGeneric(xyz,uv,K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
    if not ok:raise ValueError('No planar camera pose found')
    candidates=[]
    for rv,tv in zip(rotations,translations):
        R=cv2.Rodrigues(rv)[0];pixels=cv2.projectPoints(xyz,rv,tv,K,None)[0].reshape(-1,2)
        candidates.append({'position':(-R.T@tv.ravel()).tolist(),'quaternion':np.roll(Rotation.from_matrix(R.T@np.diag([1,-1,-1])).as_quat(),1).tolist(),'fovy':float(np.rad2deg(2*np.arctan(K[1,2]/K[1,1]))),'coordinate_rms_px':float(np.sqrt(np.mean((pixels-uv)**2)))})
    a.output.write_text(json.dumps({'candidates':candidates,'note':'Planar pose is ambiguous; compare occlusion against recorded images. Intrinsics and landmark geometry are assumed.'},indent=2)+'\n')
if __name__=='__main__':main()
