from .attachment import attachment_ready
import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .scene import build_xml, rack_target


def set_robot_pose(model,data,names,body,left,right,config):
    for name,value in zip(names,body):
        j=model.joint(name).id;data.qpos[model.jnt_qposadr[j]]=value;data.qvel[model.jnt_dofadr[j]]=0
    mapping=config['hand_mapping']
    for side,raw in [('L',left),('R',right)]:
        # Explicit scene approximation for recordings with frozen hand channels.
        raw=mapping.get('pose_register_override',{}).get(side,raw)
        u=np.clip(np.asarray(raw)/mapping['register_max'],0,1)
        values={}
        for i,finger in enumerate(['pinky','ring','middle','index']):
            for segment in ['proximal','intermediate']:values[f'{finger}_{segment}_joint']=(1-u[i])*mapping['finger_closed_rad']
        bend=(1-u[4])*mapping['thumb_bend_closed_rad']
        values.update(thumb_proximal_yaw_joint=u[5]*mapping['thumb_rotation_max_rad'],thumb_proximal_pitch_joint=bend,thumb_intermediate_joint=bend*1.6,thumb_distal_joint=bend*2.4)
        for suffix,value in values.items():
            j=model.joint(f'{side}_{suffix}').id;data.qpos[model.jnt_qposadr[j]]=value;data.qvel[model.jnt_dofadr[j]]=0
    j=model.joint('ejector_slide').id
    data.qpos[model.jnt_qposadr[j]]=(1-np.clip(right[4]/mapping['register_max'],0,1))*.012
    data.qvel[model.jnt_dofadr[j]]=0


def solve_ik(model,data,names,target,seed,config):
    """Damped least squares, seven right-arm joints, downward tool orientation."""
    q=np.array(seed,copy=True);joint_ids=np.array([model.joint(n).id for n in names[22:29]])
    dofs=model.jnt_dofadr[joint_ids];jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    left=np.full(6,1000.);right=np.array([0,0,0,0,850,400.])
    for _ in range(65):
        set_robot_pose(model,data,names,q,left,right,config);mujoco.mj_forward(model,data)
        position=data.site('nozzle').xpos;orientation=data.site('nozzle').xmat.reshape(3,3)
        rotation_error=Rotation.from_matrix(orientation.T).as_rotvec()
        error=np.r_[target-position,.12*rotation_error]
        if np.linalg.norm(error)<2e-5:break
        mujoco.mj_jacSite(model,data,jp,jr,model.site('nozzle').id)
        jac=np.vstack([jp[:,dofs],.12*jr[:,dofs]])
        delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-5,error)
        q[22:29]=np.clip(q[22:29]+np.clip(delta,-.12,.12),model.jnt_range[joint_ids,0]+.002,model.jnt_range[joint_ids,1]-.002)
    set_robot_pose(model,data,names,q,left,right,config);mujoco.mj_forward(model,data)
    return q,float(np.linalg.norm(data.site('nozzle').xpos-target))


class Demo:
    """Scripted IK task; deliberately independent of the recorded success labels."""
    duration=10.0
    def __init__(self,model,episode,config):
        self.names=episode.names;self.times=np.linspace(0,self.duration,601)
        target=rack_target(config);w=config['waste'];wxy=np.asarray(config['rack']['center_xy'])+w['offset_from_rack']
        disposal=np.r_[wxy,config['table']['top_z']+w['height']+.09]
        if w.get('hinge',{}).get('enabled'):
            from .box_lid import pose
            origin,_=pose(config);disposal=origin+[0,0,w['height']+.09]
        knots=np.array([0,1.6,2.5,3.2,4.7,6.5,8.0,10.0])
        poses=np.array([target+[0,0,.11],target+[0,0,.025],target-[0,0,.002],target-[0,0,.002],target+[0,0,.11],disposal,disposal,disposal+[0,0,.06]])
        seed=episode.sample(0)[0];scratch=mujoco.MjData(model);self.q=[];self.max_ik_error=0
        for t in self.times:
            k=min(np.searchsorted(knots,t,side='right')-1,len(knots)-2);u=np.clip((t-knots[k])/(knots[k+1]-knots[k]),0,1);u=u*u*(3-2*u)
            point=(1-u)*poses[k]+u*poses[k+1]
            seed,error=solve_ik(model,scratch,self.names,point,seed,config)
            self.max_ik_error=max(self.max_ik_error,error);self.q.append(seed.copy())
        self.q=np.asarray(self.q)
        if self.max_ik_error>.01:raise ValueError(f'Demo IK cannot reach this layout (max error {self.max_ik_error:.3f} m). Adjust the layout/base/tool mount.')

    def sample(self,t,source='state'):
        i=min(max(int(t*60),0),len(self.q)-2);u=np.clip((t-self.times[i])*60,0,1)
        right=np.array([0,0,0,0,850 if t<7.3 else 80,400.])
        return (1-u)*self.q[i]+u*self.q[i+1],np.full(6,1000.),right


class Simulation:
    def __init__(self,config,episode,mode='replay',source='state'):
        from .profiles import for_episode
        config=for_episode(config,episode)
        self.config=config;self.episode=episode;self.source=source;self.mode=mode
        self.xml=build_xml(config);self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.motion=Demo(self.model,episode,config) if mode=='demo' else episode
        self.tip_joint=self.model.joint('tip_free').id
        self.tip_q=int(self.model.jnt_qposadr[self.tip_joint]);self.tip_v=int(self.model.jnt_dofadr[self.tip_joint]);self.tip_geom=self.model.geom('tip_geom').id
        self.tip_geoms=np.array([g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]==self.model.body('tip').id and self.model.geom_group[g]==3])
        self.reset()

    @property
    def duration(self):return self.motion.duration

    def reset(self):
        mujoco.mj_resetData(self.model,self.data);self.time=0.;self.state='in_rack';self.events=[];self.dwell=0.;self.settle_time=0.;self.was_open=False;self.previous_nozzle=None;self.seated_depth=0.
        self._pose(0);self._hold_tip();mujoco.mj_forward(self.model,self.data)

    def _pose(self,t):
        body,left,self.right=self.motion.sample(t,self.source)
        set_robot_pose(self.model,self.data,self.episode.names,body,left,self.right,self.config)
        mujoco.mj_forward(self.model,self.data)

    def _hold_tip(self):
        length=self.config['pipette']['tip_length']
        if self.state=='in_rack':
            pos=rack_target(self.config)-[0,0,length/2];quat=[1,0,0,0]
        elif self.state=='attached':
            mat=self.data.site('nozzle').xmat.reshape(3,3);pos=self.data.site('nozzle').xpos+mat@np.array([0,0,self.seated_depth-length/2])
            quat=np.roll(Rotation.from_matrix(mat).as_quat(),1)
        else:return
        self.data.qpos[self.tip_q:self.tip_q+7]=np.r_[pos,quat];self.data.qvel[self.tip_v:self.tip_v+6]=0
        self.model.geom_contype[self.tip_geoms]=0;self.model.geom_conaffinity[self.tip_geoms]=0

    def _event(self,state):
        self.state=state;self.events.append({'time':round(self.time,4),'state':state,'nozzle_position':self.data.site('nozzle').xpos.tolist()})

    def step(self):
        dt=self.model.opt.timestep
        self._pose(self.time)
        nozzle=self.data.site('nozzle').xpos.copy();a=self.config['attachment'];target=rack_target(self.config)
        angle=math.degrees(math.acos(np.clip(self.data.site('nozzle').xmat.reshape(3,3)[2,2],-1,1)))
        if self.state=='in_rack':
            aligned=attachment_ready(nozzle-target,angle,a)
            self.dwell=self.dwell+dt if aligned else 0
            if self.dwell>=a['dwell_seconds']:
                self.seated_depth=float(target[2]-nozzle[2]) if 'min_insertion_depth' in a else 0.
                self._event('attached');self.was_open=False
        if self.state=='attached' and (self.mode=='demo' or a.get('automatic_ejection',True)):
            if self.right[4]>=a['eject_thumb_bend_above']:self.was_open=True
            if self.was_open and self.right[4]<=a['eject_thumb_bend_below']:self.eject()
        self._hold_tip()
        # Only a released tip is a freely simulated task object. Robot/tool poses
        # are prescribed at each tick; their contacts and force transfer are disabled.
        mujoco.mj_step(self.model,self.data)
        self.time+=dt;self._pose(self.time);self._hold_tip();mujoco.mj_forward(self.model,self.data)
        if self.state=='released':
            tip=self.data.qpos[self.tip_q:self.tip_q+3];w=self.config['waste'];lid=self.data.body('waste');local=lid.xmat.reshape(3,3).T@(tip-lid.xpos)
            inside=np.all(np.abs(local[:2])<np.asarray(w['size_xy'])/2-.004)
            settled=inside and 0<local[2]<w['height']-.006 and np.linalg.norm(self.data.qvel[self.tip_v:self.tip_v+3])<.08
            self.settle_time=self.settle_time+dt if settled else 0.
            if self.settle_time>=.2:self._event('disposed')
        self.previous_nozzle=nozzle
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise RuntimeError('Nonfinite simulation state')

    def eject(self):
        if self.state!='attached':return
        self._hold_tip();self._event('released')
        self.model.geom_contype[self.tip_geoms]=1;self.model.geom_conaffinity[self.tip_geoms]=1
        mat=self.data.site('nozzle').xmat.reshape(3,3)
        self.data.qvel[self.tip_v:self.tip_v+3]=mat@np.array([0,0,-self.config['attachment']['release_speed']])

    def report(self):
        warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(self.data.warning) if w.number}
        return {'mode':self.mode,'episode':self.episode.index,'source':self.source,'time':self.time,'state':self.state,'events':self.events,'warnings':warnings,'finite':bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()),'max_demo_ik_error_m':getattr(self.motion,'max_ik_error',None),'limitations':['Fixed unmeasured base pose','Prescribed robot motion; no whole-body controller or robot contact dynamics','DFQ hand proxy, approximate register mapping','Estimated geometry and camera poses','Geometric attachment and thumb-threshold ejection; no calibrated press fit','Tactile and force channels are not used as physical force observations']}
