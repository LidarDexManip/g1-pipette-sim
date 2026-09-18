"""Rigid-body physics, torque-limited servos and breakable task constraints.

The reference model supplies targets only. Live qpos/qvel are never overwritten
between resets: gravity, contact and the actuator solver determine the motion.
"""
from .attachment import attachment_ready
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .scene import build_xml, element, vec, rack_target
from .simulation import Demo, set_robot_pose

DEFAULT_PHYSICS = {
    'table_mass': 18.0, 'rack_mass': .18, 'tray_mass': .08,
    'pipette_mass': .085, 'tip_mass': .0006,
    'body_kp': 180., 'body_kv': 18., 'body_torque_limit': 80.,
    'wrist_kp': 45., 'wrist_kv': 3., 'wrist_torque_limit': 5.,
    'grasp_mode': 'weld', 'grasp_friction': 1.2,
    'hand_kp': 3., 'hand_kv': .12, 'hand_torque_limit': 1.,
    'ejector_kp': 150., 'ejector_kv': 1., 'ejector_force_limit': 3.,
    'gravity_compensation': True, 'constraint_time_constant': .008, 'timestep': .0005,
    'rack_contact_time_constant': .002,
}


def physics_config(config):
    return {**DEFAULT_PHYSICS, **config.get('physics', {})}


def _rescale_mass(body, mass):
    """Scale geometry masses uniformly; MuJoCo computes inertia from geometry."""
    geoms=[g for g in body.findall('geom') if g.get('mass')!='0']
    # Relative mass follows primitive volume; exact material densities are unknown.
    volumes=[]
    for g in geoms:
        size=np.fromstring(g.get('size','0.01'),sep=' ')
        if g.get('mass') is not None: v=float(g.get('mass'))/1000
        elif g.get('type')=='box': v=8*np.prod(size)
        elif g.get('type')=='cylinder': v=2*np.pi*size[0]**2*size[1]
        elif g.get('type')=='capsule':
            ends=np.fromstring(g.get('fromto','0 0 0 0 0 0'),sep=' ')
            v=np.pi*size[0]**2*np.linalg.norm(ends[3:]-ends[:3])+4/3*np.pi*size[0]**3
        else: v=1.
        volumes.append(v)
    for geom,volume in zip(geoms,volumes):
        geom.attrib.pop('density',None);geom.set('mass',str(mass*volume/sum(volumes)))


def build_dynamic_xml(config, free_base=False):
    root=ET.fromstring(build_xml(config));world=root.find('worldbody')
    settings=physics_config(config);pelvis=world.find("body[@name='pelvis']")
    if free_base:element(pelvis,'freejoint',name='robot_base')
    # Collision bits: props/ground=1, robot=2, held tool=4, tip=8.
    # Visual duplicates never collide. Collision meshes stay in hidden group 3.
    for geom in pelvis.iter('geom'):
        if geom.get('group')=='3':
            geom.set('contype','2');geom.set('conaffinity','3')
            geom.set('friction','0.9 0.005 0.0001');geom.set('solref','0.008 1')
    # Move all five table geoms from the world into one physical rigid body.
    table=element(world,'body',name='table_body',pos=[0,0,.002])
    element(table,'freejoint',name='table_free')
    for geom in list(world.findall('geom')):
        if geom.get('name') not in ['floor','backdrop']:
            world.remove(geom);table.append(geom)
    _rescale_mass(table,settings['table_mass'])
    for name,mass in [('rack',settings['rack_mass']),('waste',settings['tray_mass'])]:
        body=root.find(f".//body[@name='{name}']")
        if name=='waste' and config['waste'].get('hinge',{}).get('enabled'):
            _rescale_mass(body,mass)
            continue
        pos=np.fromstring(body.get('pos'),sep=' ');pos[2]+=.004;body.set('pos',vec(pos))
        element(body,'freejoint',name=f'{name}_free');_rescale_mass(body,mass)
        for geom in body.findall('geom'):geom.set('friction','0.9 0.005 0.0001')
    rack=world.find("body[@name='rack']")
    for geom in rack.findall('geom'):
        if geom.get('mass')=='0':continue
        geom.set('priority','1');geom.set('solref',vec([settings['rack_contact_time_constant'],1]));geom.set('solimp','0.99 0.999 0.0001')
    yaw=Rotation.from_euler('z',config['rack']['yaw_deg'],degrees=True).as_matrix()
    target_local=yaw.T@(rack_target(config)-np.r_[config['rack']['center_xy'],config['table']['top_z']])
    element(rack,'site',name='rack_mouth',pos=target_local,size=.002,rgba=[0,0,0,0])
    wrist=pelvis.find(".//body[@name='right_wrist_yaw_link']")
    tool=wrist.find("body[@name='pipette']");wrist.remove(tool);world.append(tool)
    element(wrist,'site',name='grip_frame',pos=config['pipette']['mount_position'],quat=config['pipette']['mount_quaternion'],size=.001,rgba=[0,0,0,0])
    element(tool,'freejoint',name='pipette_free')
    # pipette_mass includes its moving ejector assembly.
    ejector_mass=sum(float(g.get('mass',0)) for g in tool.findall("body[@name='ejector']/geom"))
    _rescale_mass(tool,settings['pipette_mass']-ejector_mass if config['pipette'].get('product_model')=='onilab_0.5_10ul' else settings['pipette_mass'])
    for geom in tool.iter('geom'):
        if geom.get('mass')=='0': continue
        geom.set('contype','4');geom.set('conaffinity','11' if settings['grasp_mode']=='contact' else '9');geom.set('solref','0.008 1')
        if settings['grasp_mode']=='contact':
            geom.set('friction',vec([settings['grasp_friction'],.005,.0001]));geom.set('condim','4')
    tip=world.find("body[@name='tip']")
    pos=np.fromstring(tip.get('pos'),sep=' ');pos[2]+=.004;tip.set('pos',vec(pos))
    _rescale_mass(tip,settings['tip_mass'])
    for geom in tip.findall('geom'):
        if geom.get('mass')=='0': continue
        geom.set('contype','8');geom.set('conaffinity','7')
        geom.set('priority','1');geom.set('solref','0.003 1');geom.set('solimp','0.99 0.999 0.0001');geom.set('margin','0.000005')
    contact=element(root,'contact')
    from .box_lid import enabled, contact_pairs
    if enabled(config):contact_pairs(root,contact,element)
    # The stock palm includes mesh overlap around the two thumb roots.
    for side,hand in [('left','L'),('right','R')]:
        for thumb in ['thumb_proximal','thumb_proximal_base']:
            element(contact,'exclude',body1=f'{side}_wrist_yaw_link',body2=f'{hand}_{thumb}')
    # Tool/ejector overlap is intentional; they are connected by a slide joint.
    element(contact,'exclude',body1='pipette',body2='ejector')
    equality=element(root,'equality')
    for side in ['L','R']:
        for finger in ['index','middle','ring','pinky']:
            element(equality,'joint',joint1=f'{side}_{finger}_intermediate_joint',joint2=f'{side}_{finger}_proximal_joint',polycoef=[0,1,0,0,0],solref=[.006,1])
        for segment,ratio in [('intermediate',1.6),('distal',2.4)]:
            element(equality,'joint',joint1=f'{side}_thumb_{segment}_joint',joint2=f'{side}_thumb_proximal_pitch_joint',polycoef=[0,ratio,0,0,0],solref=[.006,1])
    tc=settings['constraint_time_constant']
    if config['rack'].get('taped_to_table',False):
        element(equality,'weld',name='rack_tape',body1='table_body',body2='rack',solref=[tc,1],torquescale=1)
    element(equality,'weld',name='grip',active='false' if settings['grasp_mode']=='contact' else 'true',body1='right_wrist_yaw_link',body2='pipette',relpose=[*config['pipette']['mount_position'],*config['pipette']['mount_quaternion']],solref=[tc,1],torquescale=1)
    element(equality,'weld',name='rack_retention',body1='rack',body2='tip',solref=[tc,1],torquescale=1)
    element(equality,'weld',name='tip_attachment',body1='pipette',body2='tip',active='false',solref=[config['attachment'].get('constraint_time_constant',.002),1],solimp=config['attachment'].get('constraint_impedance',[.99,.999,.0001]),torquescale=1)
    actuator=element(root,'actuator')
    for j in pelvis.iter('joint'):
        name=j.get('name')
        if name.startswith(('L_','R_')):
            if 'intermediate' in name or 'distal' in name:continue
            category='hand'
        elif 'wrist' in name:category='wrist'
        else:category='body'
        bound=settings[f'{category}_torque_limit']
        servo=settings.get('joint_servos',{}).get(name,{})
        element(actuator,'position',name=f'servo_{name}',joint=name,kp=servo.get('kp',settings[f'{category}_kp']),kv=servo.get('kv',settings[f'{category}_kv']),forcerange=[-bound,bound])
    bound=settings['ejector_force_limit']
    element(actuator,'position',name='servo_ejector',joint='ejector_slide',kp=settings['ejector_kp'],kv=settings['ejector_kv'],forcerange=[-bound,bound])
    # Larger contact buffer is useful with the rack grid and robot mesh contacts.
    root.find('option').set('iterations','100')
    root.find('option').set('timestep',str(settings['timestep']))
    ET.indent(root)
    return ET.tostring(root,encoding='unicode')


class DynamicSimulation:
    physics='dynamic'
    def __init__(self,config,episode,mode='replay',source='state',free_base=False):
        from .profiles import for_episode
        config=for_episode(config,episode)
        self.config=config;self.episode=episode;self.mode=mode;self.source=source;self.free_base=free_base
        self.settings=physics_config(config)
        self.reference_model=mujoco.MjModel.from_xml_string(build_xml(config))
        self.reference=mujoco.MjData(self.reference_model)
        self.motion=Demo(self.reference_model,episode,config) if mode=='demo' else episode
        self.xml=build_dynamic_xml(config,free_base)
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.joints=self.model.actuator_trnid[:,0].copy()
        self.qadr=self.model.jnt_qposadr[self.joints];self.vadr=self.model.jnt_dofadr[self.joints]
        self.ref_qadr=np.array([self.reference_model.jnt_qposadr[self.reference_model.joint(self.model.joint(int(j)).name).id] for j in self.joints])
        self.kp=self.model.actuator_gainprm[:,0].copy()
        self.bias_mask=np.array([self.settings.get('joint_servos',{}).get(self.model.joint(int(j)).name,{}).get('gravity_compensation',self.settings['gravity_compensation']) for j in self.joints],dtype=float)
        self.tip_q=int(self.model.jnt_qposadr[self.model.joint('tip_free').id]);self.tip_v=int(self.model.jnt_dofadr[self.model.joint('tip_free').id])
        self.tip_geom=self.model.geom('tip_geom').id
        self.tip_geoms=np.array([g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]==self.model.body('tip').id and self.model.geom_group[g]==3])
        self.grip_id=self.model.equality('grip').id;self.retention_id=self.model.equality('rack_retention').id;self.attachment_id=self.model.equality('tip_attachment').id
        self.initial_eq_data=self.model.eq_data.copy()
        self.tool_geom_ids=[i for i in range(self.model.ngeom) if self.model.geom_contype[i]==4 and self.model.geom_bodyid[i] in [self.model.body('pipette').id,self.model.body('ejector').id]]
        self._contact_body_names=tuple(self.model.body(int(b)).name or "world" for b in self.model.geom_bodyid)
        self._contact_pair_cache={}
        self._contact_force=np.zeros(6)
        self.reset()

    @property
    def duration(self):return self.motion.duration

    def _reference(self,t):
        body,left,self.right=self.motion.sample(t,self.source)
        set_robot_pose(self.reference_model,self.reference,self.episode.names,body,left,self.right,self.config)
        # Replay servos only read qpos. Reset explicitly computes FK for tool placement.
        if self.mode != "replay":mujoco.mj_forward(self.reference_model,self.reference)
        desired=self.reference.qpos[self.ref_qadr]
        limits=self.model.jnt_range[self.joints]
        self.desired=np.clip(desired,limits[:,0],limits[:,1])
        if self.settings['grasp_mode']=='contact' and self.grip_released:
            for i,j in enumerate(self.joints):
                name=self.model.joint(int(j)).name
                if name.startswith('R_'):
                    self.desired[i]=1.3 if name=='R_thumb_proximal_yaw_joint' else 0.

    def reset(self):
        mujoco.mj_resetData(self.model,self.data);self.model.eq_data[:]=self.initial_eq_data
        self.time=0.;self.state='in_rack';self.events=[];self.dwell=0.;self.settle_time=0.;self.was_open=False
        self.grip_released=False;self.release_time=None;self.contact_diagnostics={};self.max_contacts=0;self.max_tracking_error=0.;self.contact_pairs=set()
        self.attachment_diagnostics={'closest_approach':None,'minimum_radial_error_mm':None,'maximum_valid_dwell_s':0.,'gate_failures':{'radial':0,'depth':0,'angle':0},'samples':0,'maximum_grip_displacement_mm':0.}
        self.model.geom_contype[self.tip_geoms]=8;self.model.geom_conaffinity[self.tip_geoms]=7
        for gid in self.tool_geom_ids:self.model.geom_conaffinity[gid]=11 if self.settings['grasp_mode']=='contact' else 9
        self._reference(0)
        if self.mode=='replay':
            body,left,right=self.episode.sample(0,'state')
            set_robot_pose(self.reference_model,self.reference,self.episode.names,body,left,right,self.config)
            mujoco.mj_forward(self.reference_model,self.reference)
        # Initialization is the ONLY time the live robot/tool poses are written.
        for j in range(self.model.njnt):
            name=self.model.joint(j).name
            if self.model.jnt_type[j] not in [mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE]:continue
            ref=mujoco.mj_name2id(self.reference_model,mujoco.mjtObj.mjOBJ_JOINT,name)
            if ref>=0:
                self.data.qpos[self.model.jnt_qposadr[j]]=self.reference.qpos[self.reference_model.jnt_qposadr[ref]]
        tool=self.reference.body('pipette');q=self.model.jnt_qposadr[self.model.joint('pipette_free').id]
        self.data.qpos[q:q+7]=np.r_[tool.xpos,tool.xquat]
        self._reference(0)
        self.data.ctrl[:]=self.desired
        mujoco.mj_forward(self.model,self.data)
        self.grasp_initialization={'method':'configured hand pose'}
        if self.settings['grasp_mode']=='contact' and self.settings.get('initialize_contact_fingers',False):
            from .grasp_initialization import initialize_fingers
            self.grasp_initialization=initialize_fingers(self.model,self.data)
        self._capture_weld(self.retention_id,'rack','tip')
        if self.config['rack'].get('taped_to_table',False):
            self._capture_weld(self.model.equality('rack_tape').id,'table_body','rack')
        self._apply_controller()
        self.initial_objects={n:self.data.body(n).xpos.copy() for n in ['table_body','rack','waste','pipette']}

    def _capture_weld(self,eq_id,body1,body2):
        first=self.data.body(body1);second=self.data.body(body2);R=first.xmat.reshape(3,3)
        self.model.eq_data[eq_id,3:6]=R.T@(second.xpos-first.xpos)
        relative=R.T@second.xmat.reshape(3,3)
        self.model.eq_data[eq_id,6:10]=np.roll(Rotation.from_matrix(relative).as_quat(),1)

    def _apply_controller(self):
        self.data.ctrl[:]=self.desired
        if np.any(self.bias_mask):
            # Shift the servo equilibrium, so actuator force limits still apply
            # to the sum of tracking and model-based bias compensation.
            self.data.ctrl[:]+=self.bias_mask*self.data.qfrc_bias[self.vadr]/self.kp

    def _event(self,state):
        self.state=state;self.events.append({'time':round(self.time,4),'state':state,'nozzle_position':self.data.site('nozzle').xpos.tolist()})

    def step(self):
        dt=self.model.opt.timestep;self._reference(self.time);self._apply_controller()
        a=self.config['attachment'];nozzle=self.data.site('nozzle').xpos
        tip=self.data.body('tip');R=tip.xmat.reshape(3,3)
        mouth=tip.xpos+R@np.array([0,0,self.config['pipette']['tip_length']/2])
        error=R.T@(nozzle-mouth)
        tool_z=self.data.site('nozzle').xmat.reshape(3,3)[:,2]
        angle=math.degrees(math.acos(np.clip(R[:,2]@tool_z,-1,1)))
        if self.state=='in_rack':
            aligned=attachment_ready(error,angle,a)
            self.dwell=self.dwell+dt if aligned else 0.
            stats=self.attachment_diagnostics;radial=float(np.linalg.norm(error[:2])*1000);depth=float(-error[2]*1000);distance=float(np.linalg.norm(error)*1000)
            if stats['closest_approach'] is None or distance<stats['closest_approach']['distance_mm']:
                stats['closest_approach']={'time_s':self.time,'distance_mm':distance,'radial_error_mm':radial,'insertion_depth_mm':depth,'angle_deg':float(angle),'nozzle_world_m':nozzle.tolist(),'tip_mouth_world_m':mouth.tolist()}
            stats['minimum_radial_error_mm']=radial if stats['minimum_radial_error_mm'] is None else min(stats['minimum_radial_error_mm'],radial)
            stats['maximum_valid_dwell_s']=max(stats['maximum_valid_dwell_s'],self.dwell);stats['samples']+=1
            stats['gate_failures']['radial']+=int(radial>=a['radial_tolerance']*1000)
            depth_ok=(a['min_insertion_depth']*1000<=depth<=a['max_insertion_depth']*1000) if 'min_insertion_depth' in a else abs(depth)<a['axial_tolerance']*1000
            stats['gate_failures']['depth']+=int(not depth_ok);stats['gate_failures']['angle']+=int(angle>=a['angle_tolerance_deg'])
            if self.dwell>=a['dwell_seconds']:
                self.data.eq_active[self.retention_id]=False
                self._capture_weld(self.attachment_id,'pipette','tip')
                self.data.eq_active[self.attachment_id]=True
                self._event('attached');self.was_open=False
        if self.state=='attached' and (self.mode=='demo' or a.get('automatic_ejection',True)):
            if self.right[4]>=a['eject_thumb_bend_above']:self.was_open=True
            if self.was_open and self.right[4]<=a['eject_thumb_bend_below']:self.eject()
        mujoco.mj_step(self.model,self.data);self.time=float(self.data.time)
        mujoco.mj_forward(self.model,self.data)
        if self.state=='released':
            tray=self.data.body('waste');local=tray.xmat.reshape(3,3).T@(self.data.body('tip').xpos-tray.xpos)
            inside=np.all(np.abs(local[:2])<np.asarray(self.config['waste']['size_xy'])/2-.004)
            settled=inside and 0<local[2]<self.config['waste']['height']-.006 and np.linalg.norm(self.data.qvel[self.tip_v:self.tip_v+3])<.08
            self.settle_time=self.settle_time+dt if settled else 0.
            if self.settle_time>=.2:self._event('disposed')
        self.attachment_diagnostics['maximum_grip_displacement_mm']=max(self.attachment_diagnostics['maximum_grip_displacement_mm'],float(np.linalg.norm(self.data.body('pipette').xpos-self.data.site('grip_frame').xpos)*1000))
        self.max_contacts=max(self.max_contacts,self.data.ncon)
        self.max_tracking_error=max(self.max_tracking_error,float(np.max(np.abs(self.data.qpos[self.qadr]-self.desired))))
        for contact_index,c in enumerate(self.data.contact):
            key=(int(c.geom1),int(c.geom2))
            pair=self._contact_pair_cache.get(key)
            if pair is None:
                pair=tuple(sorted((self._contact_body_names[key[0]],self._contact_body_names[key[1]])))
                self._contact_pair_cache[key]=pair
            self.contact_pairs.add(pair)
            if 'pipette' in pair and any(name in pair for name in ['rack','tip','waste']):
                stats=self.contact_diagnostics.setdefault('/'.join(pair),{'samples':0,'maximum_penetration_mm':0.,'maximum_normal_force_N':0.})
                stats['samples']+=1;stats['maximum_penetration_mm']=max(stats['maximum_penetration_mm'],max(0.,-float(c.dist))*1000)
                force=self._contact_force;mujoco.mj_contactForce(self.model,self.data,contact_index,force)
                stats['maximum_normal_force_N']=max(stats['maximum_normal_force_N'],float(force[0]))
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise RuntimeError('Nonfinite dynamic state')

    def eject(self):
        if self.state!='attached':return
        self.data.eq_active[self.attachment_id]=False
        # A small impulse adds to the physically accumulated tip velocity.
        self.data.qvel[self.tip_v:self.tip_v+3]+=self.data.site('nozzle').xmat.reshape(3,3)@np.array([0,0,-self.config['attachment']['release_speed']])
        self.release_time=self.time;self._event('released')

    def release_grip(self):
        self.data.eq_active[self.grip_id]=False;self.grip_released=True
        for gid in self.tool_geom_ids:self.model.geom_conaffinity[gid]=11

    def report(self):
        warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(self.data.warning) if w.number}
        return {'physics':'dynamic','mode':self.mode,'episode':self.episode.index,'source':self.source,'robot_base':'free' if self.free_base else 'supported (fixed pelvis)',
                'time':self.time,'state':self.state,'events':self.events,'warnings':warnings,'finite':bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()),
                'dataset':str(self.episode.root),'timing':self.episode.timing,'hand_action_provenance':self.episode.hand_action_provenance,
                'attachment_diagnostics':self.attachment_diagnostics,
                'contact_diagnostics':self.contact_diagnostics,
                'lid':({'attachment':'hinge','angle_deg':float(np.rad2deg(self.data.joint('lid_hinge').qpos[0]))} if self.config['waste'].get('hinge',{}).get('enabled') else {'attachment':'free'}),
                'grasp_mode':self.settings['grasp_mode'],'grasp_initialization':self.grasp_initialization,
                'grip_constraint_active':bool(self.data.eq_active[self.grip_id]),
                'grip_position_error_m':float(np.linalg.norm(self.data.body('pipette').xpos-self.data.site('grip_frame').xpos)),
                'hand_pose_register_override':self.config['hand_mapping'].get('pose_register_override'),
                'physics_parameters':self.settings,'grip_released':self.grip_released,'max_contacts':self.max_contacts,'contact_body_pairs':[list(p) for p in sorted(self.contact_pairs)],'max_servo_tracking_error':self.max_tracking_error,
                'object_displacement_m':{n:(self.data.body(n).xpos-p).tolist() for n,p in self.initial_objects.items()},
                'max_demo_ik_error_m':getattr(self.motion,'max_ik_error',None),
                'limitations':['Estimated mass, friction, geometry and motor gains; not hardware-calibrated',
                               'DFQ hand proxy for recorded RH56DFTP; approximate register mapping',
                               ('Tool grasp uses finger/palm contacts with estimated friction and servo gains' if self.settings['grasp_mode']=='contact' else 'Tool grip uses a wrist weld'),
                               'Tip retention and attachment still use welds, not a calibrated press fit',
                               'Hollow tip shell and tapered shaft collisions remain enabled throughout insertion, attachment and release',
                               'Default pelvis support; free-base mode has no balance controller',
                               'Frozen hand feedback and invalid force/tactile channels are not used as force measurements']}
