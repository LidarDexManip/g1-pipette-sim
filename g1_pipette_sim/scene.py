from pathlib import Path
import copy, json, math, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'config/hil194.json'

def vec(values):
    return ' '.join(f'{float(x):.9g}' for x in values)

def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k:vec(v) if isinstance(v,(list,tuple,np.ndarray)) else str(v) for k,v in attrs.items()})

def load_config(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())

def rack_target(config):
    r=config['rack']; row,col=r['active_cell']
    local=np.array([(col-(r['cols']-1)/2)*r['pitch'],(row-(r['rows']-1)/2)*r['pitch'],0.0])
    rot=Rotation.from_euler('z',r['yaw_deg'],degrees=True).as_matrix()
    return np.r_[r['center_xy'],config['table']['top_z']+r['height']+.005]+rot@local

def camera_axes(position,target):
    z=np.asarray(position)-np.asarray(target);z=z/np.linalg.norm(z)
    up=np.array([0.,0.,1.])
    if abs(z@up)>.98: up=np.array([0.,1.,0.])
    x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    return np.r_[x,y]

def build_xml(config, robot_path=None):
    path=Path(robot_path or ROOT/'assets/robot/g1_inspire.xml')
    root=ET.parse(path).getroot();root.set('model','G1 Inspire pipette workspace')
    root.find('compiler').set('meshdir',str((path.parent/'meshes').resolve()))
    world=root.find('worldbody');pelvis=world.find("body[@name='pelvis']")
    for j in list(pelvis):
        if j.tag=='freejoint' or (j.tag=='joint' and j.get('type')=='free'):pelvis.remove(j)
    pelvis.set('pos',vec(config['base_position']));pelvis.set('quat',vec(config['base_quaternion']))
    # Prescribed-motion robot: contacts are diagnostic, not a force-controlled grasp.
    for geom in pelvis.iter('geom'):
        geom.set('contype','0');geom.set('conaffinity','0')
        if geom.get('group')!='1': geom.set('group','3')
        if geom.get('mesh','').startswith('Link') or 'hand_base' in geom.get('mesh',''):
            geom.set('rgba','0.075 0.085 0.095 1')
    for parent in pelvis.iter():
        for cam in list(parent.findall('camera')):parent.remove(cam)
    for tag in ['actuator','equality','contact','sensor']:
        node=root.find(tag)
        if node is not None:root.remove(node)
    element(root,'option',timestep=config['timestep'],gravity=[0,0,-9.81],integrator='implicitfast',iterations=50)
    element(root,'statistic',center=[.15,0,.7],extent=1.7)
    visual=element(root,'visual');element(visual,'global',offwidth=1920,offheight=1080)
    element(visual,'headlight',ambient=[.35,.35,.35],diffuse=[.65,.65,.65],specular=[.15,.15,.15])
    element(visual,'quality',shadowsize=2048)
    asset=root.find('asset')
    element(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1=[.11,.14,.17],rgb2=[.14,.17,.20],width=256,height=256)
    element(asset,'material',name='floor_mat',texture='floor_tex',texrepeat=[8,8],texuniform='true')
    element(world,'light',pos=[.4,-.5,2],dir=[0,0,-1],diffuse=[.8,.8,.8],directional='true')
    element(world,'geom',name='floor',type='plane',size=[3,3,.02],material='floor_mat',friction=[.8,.005,.0001])
    element(world,'geom',name='backdrop',type='box',pos=[2.5,0,1.2],size=[.02,3,1.2],rgba=[.035,.055,.075,1])
    t=config['table'];cx,cy=t['center_xy'];sx,sy=np.asarray(t['size_xy'])/2;top=t['top_z'];th=t['thickness']
    element(world,'geom',name='table',type='box',pos=[cx,cy,top-th/2],size=[sx,sy,th/2],rgba=[.57,.43,.25,1],friction=[.8,.005,.0001])
    for dx in [-sx+.04,sx-.04]:
        for dy in [-sy+.04,sy-.04]:
            element(world,'geom',type='box',pos=[cx+dx,cy+dy,(top-th)/2],size=[.018,.018,(top-th)/2],rgba=[.22,.24,.26,1])
    payload=config.get('effective_wrist_payload')
    if payload:
        wrist=pelvis.find(".//body[@name='right_wrist_yaw_link']")
        body=element(wrist,'body',name='effective_wrist_payload',pos=payload['position'])
        element(body,'inertial',pos=[0,0,0],mass=payload['mass'],diaginertia=[payload['mass']*.02**2]*3)
    from .photo_objects import add_objects
    add_objects(config, world, pelvis, asset)
    for name,c in config['cameras'].items():
        attrs={'name':name,'pos':c['position'],'fovy':c['fovy']}
        if 'quaternion' in c:attrs['quat']=c['quaternion']
        else:attrs['xyaxes']=camera_axes(c['position'],c['target'])
        parent_name=c.get('parent_body')
        parent=world if parent_name is None else world.find(f".//body[@name='{parent_name}']")
        if parent is None: raise ValueError(f'Unknown camera parent body: {parent_name}')
        element(parent,'camera',**attrs)
    ET.indent(root)
    return ET.tostring(root,encoding='unicode')

def fit_layout(config, episode):
    """Kinematic seed only: lowest nozzle samples estimate one active well."""
    import mujoco
    from .simulation import set_robot_pose
    cfg=copy.deepcopy(config);model=mujoco.MjModel.from_xml_string(build_xml(cfg));data=mujoco.MjData(model)
    points=[]
    for i in range(0,len(episode.times),max(1,len(episode.times)//250)):
        body,left,right=episode.sample(episode.times[i]);set_robot_pose(model,data,episode.names,body,left,right,cfg)
        mujoco.mj_forward(model,data);points.append(data.site('nozzle').xpos.copy())
    points=np.asarray(points);low=points[points[:,2]<=np.quantile(points[:,2],.12)];anchor=np.median(low,axis=0)
    # Endpoint touches tip mouth; rack mouth is 5 mm above rack grid.
    cfg['table']['top_z']=float(anchor[2]-.005-cfg['rack']['height'])
    old=rack_target(cfg);cfg['rack']['center_xy']=(np.asarray(cfg['rack']['center_xy'])+anchor[:2]-old[:2]).tolist()
    # Keep the estimated tabletop clear of the pelvis/thighs. The old visual
    # seed could put its near edge behind the robot when the well was close.
    near_edge=cfg['base_position'][0]+.13
    cfg['table']['center_xy']=[float(max(anchor[0]+.02,near_edge+cfg['table']['size_xy'][0]/2)),float(anchor[1]-.06)]
    for name in ['rgb','wrist_left','wrist_right']:
        cfg['cameras'][name]['target']=anchor.tolist()
    # Seed receiving tray from the first thumb closing edge after the low-height phase.
    # This is a geometric hypothesis, not a verified ejection annotation.
    bend=episode.arrays['action.hand_right'][:,4];threshold=cfg['attachment']['eject_thumb_bend_below']
    edges=np.flatnonzero((bend[:-1]>threshold)&(bend[1:]<=threshold))+1
    earliest=episode.times[min(int(np.argmin(points[:,2]))*max(1,len(episode.times)//250),len(episode.times)-1)]
    candidates=[i for i in edges if episode.action_times[i]>earliest]
    tray_note='default offset; no usable thumb edge'
    if candidates:
        event_time=episode.action_times[candidates[0]]
        body,left,right=episode.sample(event_time);set_robot_pose(model,data,episode.names,body,left,right,cfg);mujoco.mj_forward(model,data)
        delta=data.site('nozzle').xpos[:2]-np.asarray(cfg['rack']['center_xy'])
        axis=int(np.argmax(np.abs(delta)))
        if np.linalg.norm(delta)>.02:
            rack_half=(cfg['rack']['cols'] if axis==0 else cfg['rack']['rows'])*cfg['rack']['pitch']/2+.005
            clearance=rack_half+cfg['waste']['size_xy'][axis]/2+.008
            delta[axis]=np.sign(delta[axis])*max(abs(delta[axis]),clearance)
            cfg['waste']['offset_from_rack']=delta.tolist();tray_note=f'thumb threshold edge at {event_time:.3f}s; offset expanded to avoid rack overlap'
    cfg['fit_note']={'tray_method':tray_note,'method':'lowest 12 percent of model nozzle heights; uncalibrated kinematic seed, not image reconstruction','episode':episode.index,'anchor_m':anchor.tolist(),'base_pose_measured':False,'dimensions_measured':False}
    return cfg
