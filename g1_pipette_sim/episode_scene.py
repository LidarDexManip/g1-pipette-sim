"""Episode-specific visual scene initialization; never uses replay actions to fit objects.

The fixed side view anchors planar poses. Other views provide independent overlays.
Camera/table calibration and unobserved objects retain their stated assumptions.
"""
import argparse,copy,hashlib,json,os
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .dataset import Episode,DEFAULT_DATASET
from .scene import load_config,ROOT

VERSION=4
DEFAULT_CACHE=ROOT/'outputs/episode-scenes'


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def initial_frame(episode,view):
    cap=cv2.VideoCapture(str(episode.video_path(view)))
    try:ok,frame=cap.read()
    finally:cap.release()
    if not ok:raise ValueError(f'Cannot read initial image: episode {episode.index}, {view}')
    return frame


def camera_image(episode,config,view):
    im=initial_frame(episode,view);rect=config['cameras'][view].get('rectification')
    if rect:
        K=np.array(rect['camera_matrix']);dist=np.array(rect['distortion_coefficients']);new=np.array(rect['rectified_camera_matrix'])
        if list(im.shape[1::-1])!=rect['resolution']:raise ValueError('Camera calibration/image size mismatch')
        im=cv2.fisheye.undistortImage(im,K,dist,Knew=new) if rect['lens_model']=='fisheye' else cv2.undistort(im,K,dist,None,new)
    return im


def project(config,view,points,shape):
    c=config['cameras'][view]
    if c.get('parent_body'):raise ValueError('Scene fitting requires a fixed camera')
    h,w=shape[:2];f=h/2/np.tan(np.deg2rad(c['fovy'])/2)
    R=Rotation.from_quat(np.roll(c['quaternion'],-1)).as_matrix()
    local=(np.asarray(points)-c['position'])@R
    if np.any(local[:,2]>=0):raise ValueError('Object projects behind camera')
    return np.c_[w/2+f*local[:,0]/-local[:,2],h/2-f*local[:,1]/-local[:,2]]


def corners(config,object_name,pose=None):
    from .box_lid import enabled, pose as lid_pose
    if object_name!='rack' and enabled(config):
        origin, rotation=lid_pose(config);x,y=np.array(config['waste']['size_xy'])/2;z=config['waste']['height']
        return np.array([[x,y,z],[x,-y,z],[-x,-y,z],[-x,y,z]])@rotation.T+origin
    if object_name=='rack':
        c=config['rack'];xy=c['center_xy'];yaw=c['yaw_deg'];z=config['table']['top_z']+c['height']
    else:
        c=config['waste'];xy=np.array(config['rack']['center_xy'])+c['offset_from_rack'];yaw=c.get('yaw_deg',0);z=config['table']['top_z']+c['height']
    if pose is not None:xy=pose[:2];yaw=pose[2]
    x,y=np.array(c['size_xy'])/2;local=np.array([[x,y],[x,-y],[-x,-y],[-x,y]])
    angle=np.deg2rad(yaw);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return np.c_[local@R.T+xy,np.full(4,z)]


def register_polygon(reference,current,polygon):
    """Forward/backward optical flow + robust homography, confined to the object."""
    gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY);target=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,np.asarray(polygon,np.int32),255)
    p=cv2.goodFeaturesToTrack(gray,400,.015,7,mask=mask)
    if p is None or len(p)<16:raise ValueError('Too few visible object features')
    options=dict(winSize=(31,31),maxLevel=4)
    q,ok,err=cv2.calcOpticalFlowPyrLK(gray,target,p,None,**options)
    back,bok,_=cv2.calcOpticalFlowPyrLK(target,gray,q,None,**options)
    keep=(ok.ravel()>0)&(bok.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<1.5)&(err.ravel()<30)
    if keep.sum()<16:raise ValueError('Object correspondence failed forward/backward consistency')
    H,inlier=cv2.findHomography(p[keep],q[keep],cv2.RANSAC,2.5)
    if H is None:raise ValueError('No stable object homography')
    ids=inlier.ravel().astype(bool);count=int(ids.sum());ratio=count/int(keep.sum())
    if count<16 or ratio<.45:raise ValueError(f'Weak visual consensus ({count} inliers, {ratio:.2f} fraction)')
    predicted=cv2.perspectiveTransform(np.asarray(polygon,np.float32)[None],H)[0]
    area=abs(cv2.contourArea(predicted));original=abs(cv2.contourArea(np.asarray(polygon,np.float32)))
    if not cv2.isContourConvex(predicted) or not .5<area/original<2:raise ValueError('Implausible object outline')
    if np.max(np.linalg.norm(predicted-polygon,axis=1))>180:raise ValueError('Object moved beyond trusted tracking range; annotate this episode')
    coverage=abs(cv2.contourArea(cv2.convexHull(p[keep][ids])))/original
    if coverage<.25:raise ValueError('Matches do not cover enough of the object')
    residual=cv2.perspectiveTransform(p[keep][ids],H)-q[keep][ids]
    return predicted,{'inliers':count,'inlier_fraction':ratio,'coverage':coverage,'tracking_rms_px':float(np.sqrt(np.mean(residual**2)))}


def register_from_anchor(config,episode,current,view):
    path=config.get('episode_scene',{}).get('registration_anchors')
    if not path:raise ValueError('No additional episode references configured')
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    payload=path.read_bytes()
    expected=config['episode_scene'].get('registration_anchors_sha256')
    if expected and hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError('Scene reference file changed; regenerate its configuration fingerprint')
    bank=json.loads(payload)
    if Path(bank['dataset_snapshot']).resolve()!=episode.root.resolve():
        raise ValueError('Scene references belong to a different dataset snapshot')
    anchors=[a for a in bank['anchors'] if a['reference_episode']==config['camera_fit']['episode'] and a['episode']!=episode.index]
    anchors=sorted(anchors,key=lambda a:abs(a['episode']-episode.index))[:8]
    errors=[]
    for anchor in anchors:
        ref=Episode(episode.root,anchor['episode']);reference=camera_image(ref,config,view)
        polygon=np.asarray(anchor['rack_corners'])
        try:
            observed,quality=register_polygon(reference,current,polygon)
            anchored=copy.deepcopy(config);pose=anchor['rack_pose_xy_yaw_deg'];anchored['rack']['center_xy']=pose[:2];anchored['rack']['yaw_deg']=pose[2]
            target=project(anchored,view,corners(anchored,'rack'),current.shape)+(observed-polygon)
            pose,metrics=fit_pose(config,view,'rack',target,current.shape)
            quality.update(method='relative independently fitted episode reference',anchor_episode=anchor['episode'],anchor_fit_rms_px=anchor['fit_rms_px'],reference_chain_depth=1)
            return observed,quality,pose,metrics
        except ValueError as error:errors.append(str(error))
    raise ValueError('Additional episode references failed: '+(errors[-1] if errors else 'none in this recording session'))


def fit_pose(config,view,object_name,uv,shape):
    c=config['rack'] if object_name=='rack' else config['waste']
    xy=c['center_xy'] if object_name=='rack' else np.array(config['rack']['center_xy'])+c['offset_from_rack']
    seed=np.r_[xy,c.get('yaw_deg',0.)]
    def residual(p):return (project(config,view,corners(config,object_name,p),shape)-uv).ravel()
    fit=least_squares(residual,seed,bounds=(seed-[.15,.15,35],seed+[.15,.15,35]),loss='soft_l1',f_scale=3,x_scale=[.02,.02,5])
    rms=float(np.sqrt(np.mean(residual(fit.x)**2)))
    if not fit.success or rms>10:raise ValueError(f'Planar pose fit rejected: {rms:.1f} px RMS; check corners/calibration')
    return fit.x,{'fit_rms_px':rms,'table_height_fixed':True}


def detect_tip(frame,polygon,config,view):
    mask=np.zeros(frame.shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.asarray(polygon,np.int32),255)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);green=cv2.inRange(hsv,np.array([43,65,55]),np.array([88,255,255]));green&=mask
    count,labels,stats,centers=cv2.connectedComponentsWithStats(green)
    candidates=[i for i in range(1,count) if 8<=stats[i,cv2.CC_STAT_AREA]<=900]
    if not candidates:return None,{'status':'unresolved','reason':'No isolated green tip visible; inherited active cell'}
    i=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA]);pixel=centers[i]
    return pixel,{'status':'detected','green_area_px':int(stats[i,cv2.CC_STAT_AREA])}


def assign_tip(config,view,pixel,polygon,shape):
    # Pixel-to-rack plane coordinates use the observed board, avoiding baseline camera fit bias.
    local=corners(config,'rack')[:,:2];H=cv2.getPerspectiveTransform(np.asarray(polygon,np.float32),local.astype(np.float32))
    xy=cv2.perspectiveTransform(np.asarray(pixel,np.float32).reshape(1,1,2),H)[0,0]
    # The visible tip mouth sits above the rack plate; undo its image parallax.
    z=config['table']['top_z']+config['rack']['height']
    displacement=project(config,view,[[*xy,z+.005]],shape)[0]-project(config,view,[[*xy,z]],shape)[0]
    corrected=np.asarray(pixel)-displacement
    xy=cv2.perspectiveTransform(np.asarray(corrected,np.float32).reshape(1,1,2),H)[0,0]
    rack=config['rack'];a=np.deg2rad(rack['yaw_deg']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    offset=R.T@(xy-rack['center_xy'])
    col=offset[0]/rack['pitch']+(rack['cols']-1)/2;row=offset[1]/rack['pitch']+(rack['rows']-1)/2
    cell=[int(np.clip(round(row),0,rack['rows']-1)),int(np.clip(round(col),0,rack['cols']-1))]
    if max(abs(row-cell[0]),abs(col-cell[1]))>1:raise ValueError('Detected green feature is too far from any rack well')
    return cell,{'cell_offset_fraction':[float(row-cell[0]),float(col-cell[1])],'pixel':np.asarray(pixel).tolist()}


def paths(episode,cache):
    key=hashlib.sha256(str(episode.root.resolve()).encode()).hexdigest()[:12]
    directory=Path(cache)/key;directory.mkdir(parents=True,exist_ok=True)
    return directory/f'episode_{episode.index:03}.json',directory/f'episode_{episode.index:03}.annotations.json'


def resolve_scene(config,episode,policy='auto',cache=DEFAULT_CACHE,refresh=False):
    from .profiles import for_episode
    config=for_episode(config,episode)
    if policy=='fixed':return config,None
    resolved=config.get('scene_initialization',{}).get('signature')
    if resolved:
        if resolved['episode']!=episode.index or resolved['snapshot']!=str(episode.root.resolve()):
            raise ValueError('This resolved scene belongs to a different episode/snapshot; use the base HG-DAgger config')
        return config,None
    if not config.get('episode_scene',{}).get('enabled',False):return config,None
    return initialize(config,episode,cache,refresh)


def initialize(config,episode,cache=DEFAULT_CACHE,refresh=False):
    from .profiles import for_episode
    config=for_episode(config,episode)
    output,annotation_file=paths(episode,cache)
    annotation=json.loads(annotation_file.read_text()) if annotation_file.exists() else {}
    reference_episode=Episode(episode.root,config['camera_fit']['episode'])
    video_state={f'{kind}:{view}':{'bytes':ep.video_path(view).stat().st_size,'mtime_ns':ep.video_path(view).stat().st_mtime_ns} for kind,ep in [('reference',reference_episode),('episode',episode)] for view in ['rgb','wrist_right']}
    signature={'videos':video_state,'version':VERSION,'base_config':fingerprint(config),'snapshot':str(episode.root.resolve()),'episode':episode.index,'annotations':fingerprint(annotation)}
    if output.exists() and not refresh:
        saved=json.loads(output.read_text())
        if saved.get('scene_initialization',{}).get('signature')==signature:return saved,output
    if Path(config.get('camera_fit',{}).get('dataset_snapshot','')).resolve()!=episode.root.resolve():
        raise ValueError('Episode initializer is tied to a different dataset snapshot; use --scene-layout fixed or calibrate a reference for this dataset')
    view='wrist_right'
    if config['cameras'][view].get('parent_body'):raise ValueError('Configured side camera must be fixed')
    reference=camera_image(reference_episode,config,view);current=camera_image(episode,config,view)
    if annotation:
        if annotation.get('view')!=view or annotation.get('frame')!=0 or annotation.get('resolution_wh')!=list(current.shape[1::-1]):raise ValueError('Annotations must match the initial fixed-side frame and its resolution')
        for name in ['rack_corners','lid_corners']:
            if name in annotation:
                points=np.asarray(annotation[name],np.float32)
                if points.shape!=(4,2) or not np.isfinite(points).all() or not cv2.isContourConvex(points):raise ValueError(f'{name} must be four finite corners in their original order')
    cfg=copy.deepcopy(config);report={'signature':signature,'reference_episode':reference_episode.index,'view':view,'frame':0,'objects':{},'observations':{},'limitations':['Relative planar image registration with estimated camera extrinsics, not measured 3D reconstruction','Table height, robot base, initial pipette mount and frozen hand pose remain assumed','Fixed overview is an independent check, not a calibrated triangulation constraint']}
    ann=config['camera_fit']['annotations'][view];h,w=current.shape[:2]
    ref_poly=np.array(ann['uv'],float)*[w/ann['wh'][0],h/ann['wh'][1]]
    if 'rack_corners' in annotation:
        observed=np.array(annotation['rack_corners'],float);target=observed;quality={'method':'manual initial-frame corners'}
    else:
        try:
            observed,quality=register_polygon(reference,current,ref_poly)
            target=project(config,view,corners(config,'rack'),current.shape)+(observed-ref_poly)
            quality['method']='relative reference-frame feature registration'
            rack_pose,metrics=fit_pose(config,view,'rack',target,current.shape)
        except ValueError as primary_error:
            if not config.get('episode_scene',{}).get('registration_anchors'):raise
            observed,quality,rack_pose,metrics=register_from_anchor(config,episode,current,view)
            quality['primary_reference_failure']=str(primary_error)
    if 'rack_corners' in annotation:
        rack_pose,metrics=fit_pose(config,view,'rack',target,current.shape)
    cfg['rack']['center_xy']=rack_pose[:2].tolist();cfg['rack']['yaw_deg']=float(rack_pose[2])
    report['objects']['rack']={**quality,**metrics,'status':'estimated','pose_xy_yaw_deg':rack_pose.tolist()};report['observations']['rack_corners']=observed.tolist()
    # Transparent lid matches can follow the background through the lid. Require manual corners
    # before calling its pose visually measured; otherwise carry the old relative arrangement explicitly.
    if cfg['waste'].get('hinge',{}).get('enabled'):
        report['objects']['lid']={'status':'hinged_assumed_angle','method':'attached to rack; configured hinge opening, not independent planar fitting','initial_angle_deg':cfg['waste']['hinge']['initial_angle_deg']}
        if 'lid_corners' in annotation:report['objects']['lid']['note']='Independent lid corners not applied to articulated geometry'
    elif 'lid_corners' in annotation:
        lid_pose,quality=fit_pose(config,view,'lid',np.array(annotation['lid_corners'],float),current.shape)
        cfg['waste']['offset_from_rack']=(lid_pose[:2]-rack_pose[:2]).tolist();cfg['waste']['yaw_deg']=float(lid_pose[2]);report['objects']['lid']={'status':'estimated','method':'manual initial-frame corners',**quality};report['observations']['lid_corners']=annotation['lid_corners']
    else:
        delta=np.deg2rad(rack_pose[2]-config['rack']['yaw_deg']);R=np.array([[np.cos(delta),-np.sin(delta)],[np.sin(delta),np.cos(delta)]])
        cfg['waste']['offset_from_rack']=(R@np.array(config['waste']['offset_from_rack'])).tolist();cfg['waste']['yaw_deg']=config['waste'].get('yaw_deg',0)+float(np.rad2deg(delta))
        report['objects']['lid']={'status':'assumed','method':'carried with rack using reference offset; transparent lid needs manual corners'}
    if 'tip_pixel' in annotation:tip=np.array(annotation['tip_pixel']);quality={'status':'annotated'}
    else:tip,quality=detect_tip(current,observed,cfg,view)
    if tip is not None:
        cell,cell_quality=assign_tip(cfg,view,tip,observed,current.shape);cfg['rack']['active_cell']=cell;quality.update(cell_quality)
        if max(abs(v) for v in cell_quality['cell_offset_fraction'])>.4:quality['status']='estimated_near_cell_boundary'
    quality['active_cell']=cfg['rack']['active_cell'];report['objects']['tip']=quality
    report['objects']['table']={'status':'fixed_reference','reason':'No depth/height calibration in episode'}
    # Track overview separately for review, without averaging inconsistent uncalibrated cameras.
    for other in ['rgb']:
        try:
            im=camera_image(episode,config,other);ref=camera_image(reference_episode,config,other);a=config['camera_fit']['annotations'][other];poly=np.array(a['uv'],float)*[im.shape[1]/a['wh'][0],im.shape[0]/a['wh'][1]]
            tracked,quality=register_polygon(ref,im,poly)
            expected=project(cfg,other,corners(cfg,'rack'),im.shape)-project(config,other,corners(config,'rack'),im.shape)
            report['overview_check']={**quality,'relative_disagreement_rms_px':float(np.sqrt(np.mean((tracked-poly-expected)**2))),'observed_corners':tracked.tolist()}
        except ValueError as error:report['overview_check']={'status':'unresolved','reason':str(error)}
    cfg['scene_initialization']=report;output.write_text(json.dumps(cfg,indent=2)+'\n')
    write_overlays(config,cfg,episode,output.with_suffix('.overlay.jpg'))
    return cfg,output


def write_overlays(base,cfg,episode,path):
    panels=[]
    for view in ['wrist_right','rgb']:
        im=camera_image(episode,cfg,view)
        for name in ['rack','lid']:
            for c,color in [(base,(0,150,255)),(cfg,(0,230,0))]:
                poly=project(c,view,corners(c,name),im.shape).round().astype(np.int32);cv2.polylines(im,[poly],True,color,2)
        obs=cfg['scene_initialization']['observations'] if view=='wrist_right' else {'rack_corners':cfg['scene_initialization'].get('overview_check',{}).get('observed_corners',[])}
        for key in ['rack_corners','lid_corners']:
            if len(obs.get(key,[]))==4:cv2.polylines(im,[np.asarray(obs[key],np.int32)],True,(255,255,0),2)
        if view=='wrist_right' and 'pixel' in cfg['scene_initialization']['objects']['tip']:
            pt=tuple(np.asarray(cfg['scene_initialization']['objects']['tip']['pixel'],int));cv2.circle(im,pt,9,(255,0,255),2)
        im=cv2.resize(im,(540,round(540*im.shape[0]/im.shape[1])));panel=np.zeros((585,540,3),np.uint8);panel[45:45+im.shape[0]]=im
        cv2.putText(panel,f'{view}: orange=before green=after cyan=tracked',(5,18),0,.36,(255,255,255),1)
        cv2.putText(panel,'Lid assumed unless annotated; camera calibration approximate',(5,36),0,.34,(255,255,255),1);panels.append(panel)
    cv2.imwrite(str(path),np.hstack(panels))


def preview_scene(cfg,episode,path):
    os.environ.setdefault('MUJOCO_GL','egl')
    from .simulation import Simulation
    from .camera_streams import CameraStreams
    from PIL import Image,ImageDraw
    sim=Simulation(cfg,episode,source='state');panels=[]
    for view in ['rgb','wrist_right','wrist_left']:
        real=initial_frame(episode,view);height=round(540*real.shape[0]/real.shape[1])
        with CameraStreams(sim.model,[view],540,height) as streams:rendered=streams.render(sim.data)[view]
        panel=Image.new('RGB',(1080,height+30),(25,25,25));draw=ImageDraw.Draw(panel)
        panel.paste(Image.fromarray(cv2.cvtColor(cv2.resize(real,(540,height)),cv2.COLOR_BGR2RGB)),(0,30));panel.paste(Image.fromarray(rendered),(540,30))
        draw.text((8,8),f'Real {view}, episode {episode.index}, initial frame',fill='white');draw.text((548,8),'Initialized simulation (estimated geometry/cameras)',fill='white');panels.append(panel)
    result=Image.new('RGB',(1080,sum(p.height for p in panels)));y=0
    for panel in panels:result.paste(panel,(0,y));y+=panel.height
    result.save(path)


def edit(config,episode,cache):
    """Drag projected corner handles in the initial fixed-side image, then save."""
    import tkinter as tk
    from PIL import Image,ImageTk
    from .profiles import for_episode
    config=for_episode(config,episode)
    output,annotation_file=paths(episode,cache)
    try:cfg,_=initialize(config,episode,cache)
    except ValueError:cfg=copy.deepcopy(config)
    view='wrist_right';im=camera_image(episode,cfg,view);scale=min(850/im.shape[1],750/im.shape[0])
    annotation=json.loads(annotation_file.read_text()) if annotation_file.exists() else {}
    report=cfg.get('scene_initialization',{})
    points={name:np.array(annotation.get(name+'_corners',report.get('observations',{}).get(name+'_corners',project(cfg,view,corners(cfg,name),im.shape))),float) for name in ['rack','lid']}
    tip=report.get('objects',{}).get('tip',{}).get('pixel',points['rack'].mean(0));points['tip']=np.array([annotation.get('tip_pixel',tip)],float)
    hinged=cfg['waste'].get('hinge',{}).get('enabled',False)
    if hinged:points.pop('lid',None)
    root=tk.Tk();root.title(f'Episode {episode.index}: drag object landmarks');canvas=tk.Canvas(root,width=round(im.shape[1]*scale),height=round(im.shape[0]*scale));canvas.pack()
    photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(im,None,fx=scale,fy=scale),cv2.COLOR_BGR2RGB)),master=root);canvas.create_image(0,0,image=photo,anchor='nw')
    selected=[None];colors={'rack':'yellow','lid':'cyan','tip':'magenta'}
    def draw():
        canvas.delete('handles')
        for name,poly in points.items():
            if len(poly)>1:canvas.create_polygon(*(poly*scale).ravel().tolist(),outline=colors[name],fill='',width=2,tags='handles')
            for i,(x,y) in enumerate(poly*scale):
                canvas.create_oval(x-6,y-6,x+6,y+6,fill=colors[name],tags='handles');canvas.create_text(x+15,y-10,text=f'{name}{i}',fill=colors[name],tags='handles')
    def press(event):
        options=[(np.linalg.norm(p*scale-[event.x,event.y]),name,i) for name,poly in points.items() for i,p in enumerate(poly)];distance,name,i=min(options);selected[0]=(name,i) if distance<35 else None
    def drag(event):
        if selected[0]:name,i=selected[0];points[name][i]=np.clip(np.array([event.x,event.y])/scale,[0,0],[im.shape[1]-1,im.shape[0]-1]);draw()
    def save():
        value={'view':view,'frame':0,'resolution_wh':list(im.shape[1::-1]),'rack_corners':points['rack'].tolist(),'tip_pixel':points['tip'][0].tolist()}
        if not hinged:value['lid_corners']=points['lid'].tolist()
        annotation_file.write_text(json.dumps(value,indent=2)+'\n');root.destroy()
    canvas.bind('<Button-1>',press);canvas.bind('<B1-Motion>',drag);draw()
    tk.Label(root,text=('Drag rack corners and active tip. Attached lid opening is set by waste.hinge.initial_angle_deg.' if hinged else 'Keep corner identities/order. Drag outlines onto the rack plate and lid top rim; magenta = active tip.')).pack();tk.Button(root,text='Save annotations and refit',command=save).pack();root.mainloop()
    return initialize(config,episode,cache,refresh=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--config',type=Path,default=ROOT/'config/hil194.json');p.add_argument('--episodes',nargs='+',default=['0'],help='Episode indices or all');p.add_argument('--cache',type=Path,default=DEFAULT_CACHE);p.add_argument('--refresh',action='store_true');p.add_argument('--preview',action='store_true',help='Render three real/sim initial-frame pairs for each prepared scene');p.add_argument('--edit',type=int,help='Interactively correct one episode on the fixed side image');a=p.parse_args()
    cfg=load_config(a.config)
    if a.edit is not None:
        _,path=edit(cfg,Episode(a.dataset,a.edit),a.cache);print(path);return
    first=Episode(a.dataset,0);indices=list(range(first.info['total_episodes'])) if a.episodes==['all'] else list(map(int,a.episodes));failures=[]
    for index in indices:
        try:
            episode=Episode(a.dataset,index);result,path=initialize(cfg,episode,a.cache,a.refresh)
            if a.preview:preview_scene(result,episode,path.with_suffix('.comparison.jpg'))
            print(index,result['rack']['center_xy'],result['rack']['active_cell'],path,flush=True)
        except ValueError as error:failures.append({'episode':index,'error':str(error)});print(index,'NEEDS ANNOTATION:',error,flush=True)
    failure_file=Path(a.cache)/'failures.json'
    prior=json.loads(failure_file.read_text()) if failure_file.exists() else []
    remaining=[failure for failure in prior if failure['episode'] not in indices]
    failure_file.write_text(json.dumps(sorted(remaining+failures,key=lambda item:item['episode']),indent=2)+'\n')
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
