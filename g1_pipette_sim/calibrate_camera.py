"""Estimate a FIXED camera pose from measured 3D points and matching pixels.
Writes a centered, square-pixel pinhole view plus real-image rectification metadata.
Does not estimate scene scale, robot base pose, or moving-camera hand-eye transforms.
"""
import argparse,json,copy
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .scene import load_config


def calibrate(spec):
    xyz=np.asarray(spec['world_points_m'],dtype=np.float64)
    uv=np.asarray(spec['image_points_px'],dtype=np.float64)
    K=np.asarray(spec['camera_matrix'],dtype=np.float64)
    dist=np.asarray(spec.get('distortion_coefficients',[0,0,0,0,0]),dtype=float)
    width,height=spec['resolution'];lens=spec.get('lens_model','pinhole')
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<6 or uv.shape!=(len(xyz),2):raise ValueError('Supply at least six matching 3D/2D points')
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError('Invalid camera matrix')
    if np.linalg.matrix_rank(xyz-xyz.mean(0))<2:raise ValueError('Calibration points must not be collinear')
    if lens=='fisheye':
        uv_fit=cv2.fisheye.undistortPoints(uv[:,None,:],K,dist,P=K).reshape(-1,2);dist_fit=np.zeros(5)
    elif lens=='pinhole':uv_fit=uv;dist_fit=dist
    else:raise ValueError('lens_model must be pinhole or fisheye')
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv_fit,K,dist_fit,iterationsCount=1000,reprojectionError=3.0,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers)<6:raise ValueError('PnP failed: check point ordering, world units, intrinsics, and lens model')
    ids=inliers.ravel();rvec,tvec=cv2.solvePnPRefineLM(xyz[ids],uv_fit[ids],K,dist_fit,rvec,tvec)
    R,_=cv2.Rodrigues(rvec);camera_position=-R.T@tvec.reshape(3)
    if np.any((xyz@R.T+tvec.reshape(3))[:,2]<=0):raise ValueError('Calibration puts landmarks behind the camera')
    if lens=='fisheye':projected,_=cv2.fisheye.projectPoints(xyz[:,None,:],rvec,tvec,K,dist)
    else:projected,_=cv2.projectPoints(xyz,rvec,tvec,K,dist)
    errors=np.linalg.norm(projected.reshape(-1,2)-uv,axis=1)
    gl_rotation=R.T@np.diag([1,-1,-1]);quat=np.roll(Rotation.from_matrix(gl_rotation).as_quat(),1)
    f=float(K[1,1]);rectified=np.array([[f,0,width/2],[0,f,height/2],[0,0,1]])
    return {'position':camera_position.tolist(),'quaternion':quat.tolist(),'fovy':float(np.degrees(2*np.arctan(height/(2*f)))),'rectification':{'camera_matrix':K.tolist(),'distortion_coefficients':dist.tolist(),'rectified_camera_matrix':rectified.tolist(),'resolution':[width,height],'lens_model':lens},'calibration_report':{'inliers':ids.tolist(),'inlier_rms_px':float(np.sqrt(np.mean(errors[ids]**2))),'all_points_rms_px':float(np.sqrt(np.mean(errors**2))),'note':'Fixed camera pose from user correspondences. Calibration applies to the supplied cropped/resized image coordinates.'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('correspondences',type=Path);p.add_argument('--view',choices=['rgb','wrist_left','wrist_right'],required=True);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=load_config(a.config);camera=calibrate(json.loads(a.correspondences.read_text()));cfg['cameras'][a.view]=camera
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(camera['calibration_report'],indent=2));print(a.output)

if __name__=='__main__':main()
