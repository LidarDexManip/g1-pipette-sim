"""Contact-driven imported-scene replay: targets command servos, never live qpos."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from . import environment_manifest, _vec
from .robot import build_robot_scene, MeasuredReplay, apply_joint_pose
from ..hands import HANDS


def build_physical_scene(name='lidar', registration=None, hand='inspire'):
    root=ET.fromstring(build_robot_scene(name, registration, hand))
    option=root.find('option')
    option.set('timestep','0.001');option.set('integrator','implicitfast')
    option.set('iterations','100');option.set('cone','elliptic')
    ET.SubElement(root,'size',memory='128M')
    objects={o['body']:o for o in environment_manifest(name)['objects']}
    for body_name,mass in [('env_01_pipette',.08),('env_02_cylinder',.015)]:
        body=root.find(f"./worldbody/body[@name='{body_name}']")
        bounds=np.asarray(objects[body_name]['bounds_m']);lengths=np.diff(bounds,axis=1).ravel()
        center=bounds.mean(axis=1)-np.fromstring(body.get('pos','0 0 0'),sep=' ')
        # Explicit approximate inertia avoids counting overlapping convex pieces
        # as duplicate mass. Pipette mass measured by the user; tube mass estimated.
        inertia=mass/12*np.array([lengths[1]**2+lengths[2]**2,
                                  lengths[0]**2+lengths[2]**2,
                                  lengths[0]**2+lengths[1]**2])
        ET.SubElement(body,'inertial',pos=_vec(center),mass=str(mass),diaginertia=_vec(inertia))
        ET.SubElement(body,'freejoint',name=body_name+'_free')
        for geom in body.findall('geom'):
            if geom.get('contype','1')!='0':
                geom.set('condim','4');geom.set('friction','1.0 .005 .0001')
    pelvis=root.find("./worldbody/body[@name='pelvis']")
    joints={j.get('name'):j for j in pelvis.iter('joint')}
    actuators=root.find('actuator');actuators.clear()
    finger_prefix=tuple(p+'_' for p in HANDS[hand]['prefix'].values())
    for joint_name,joint in joints.items():
        finger=joint_name.startswith(finger_prefix)
        if finger and hand=='inspire':
            joint.set('armature','.00001');joint.set('damping','.005');joint.set('frictionloss','.001')
            # Distal joints follow the original mechanical coupling equalities.
            if 'intermediate' in joint_name or 'distal' in joint_name:continue
        if finger and hand=='wuji':
            # Wuji joints are independently actuated. Keep the retargeting
            # repository's armature/damping, servo gains and URDF effort limits.
            kp,kv=2.,.06;limit=float(np.max(np.abs(np.fromstring(joint.get('actuatorfrcrange'),sep=' '))))
        else:
            kp,kv,limit=(3.,.12,1.) if finger else ((45.,3.,5.) if 'wrist' in joint_name else (180.,18.,80.))
        hardware=np.fromstring(joint.get('actuatorfrcrange',f'{-limit} {limit}'),sep=' ')
        limit=min(limit,float(np.max(np.abs(hardware))))
        ET.SubElement(actuators,'position',name='servo_'+joint_name,joint=joint_name,
                      kp=str(kp),kv=str(kv),forcerange=_vec([-limit,limit]))
    return ET.tostring(root,encoding='unicode')


class PhysicalReplay:
    """Fixed pelvis, free props, torque-limited servo replay with real contacts."""
    def __init__(self,model,data,episode=None,source='low_level',initial_pose=None,hand='inspire'):
        self.model=model;self.data=data;self.episode=episode;self.source=source
        self.reference=MeasuredReplay(model,data,episode,hand) if episode else None
        self.scene_state=self.reference.scene_state if episode else {}
        absent=self.scene_state.get('absent_objects',[])
        for name in ('env_01_pipette','env_02_cylinder'):
            # An absent prop has no collision/render geometry. Keep its unused
            # free body parked instead of integrating an invisible falling mass.
            model.body_gravcomp[model.body(name).id]=float(name in absent)
        self.hand_sources={s:f'action.hand_{s}' for s in ('left','right')} if episode else {}
        self.names=[model.joint(int(j)).name for j in model.actuator_trnid[:,0]]
        self.addresses=np.array([model.joint(n).qposadr[0] for n in self.names])
        self.hold=initial_pose or {}
        if self.reference:self.hold=self.reference.targets(0)
        # The only pose assignment: initialize from measured state, not commands.
        apply_joint_pose(model,data,self.hold)
        if not self.reference:self.hold={n:float(data.qpos[a]) for n,a in zip(self.names,self.addresses)}
        self.set_targets(0);mujoco.mj_forward(model,data)
        self.start_com={n:data.body(n).xipos.copy() for n in ('env_01_pipette','env_02_cylinder') if n not in absent}
        self.max_tracking_error=0.;self.max_penetration=0.;self.contact_steps=0
        self.pipette_robot_contact_steps=0;self.pipette_robot_max_penetration=0.
        self.pipette_robot_peak_normal_force=0.;self.first_pipette_robot_contact_s=None
        self.pipette_max_lift=0.
        self.initial_penetration=self.penetration()
        self.initial_contacts={}
        for c in data.contact:
            if c.dist>=-.001:continue
            key=' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name or 'world'
                                  for g in [c.geom1,c.geom2]))
            self.initial_contacts[key]=max(self.initial_contacts.get(key,0.),-float(c.dist)*1000)
        self.steps=0

    def set_targets(self,t):
        pose=self.reference.targets(t,self.source,hand_commands=True) if self.reference else self.hold
        self.data.ctrl[:]=[pose.get(n,self.data.qpos[a]) for n,a in zip(self.names,self.addresses)]

    def penetration(self):
        distances=self.data.contact.dist
        return max(0.,-float(distances.min())) if len(distances) else 0.

    def advance(self,t):
        m,d=self.model,self.data
        while d.time+1e-10<t:
            self.set_targets(d.time)
            previous=d.time
            mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all() or d.time<=previous or any(w.number for w in d.warning):
                raise RuntimeError('MuJoCo instability or capacity warning during physical replay; see simulator output.')
            self.steps+=1
            self.contact_steps+=int(d.ncon>0)
            self.max_penetration=max(self.max_penetration,self.penetration())
            self.max_tracking_error=max(self.max_tracking_error,float(np.max(np.abs(d.qpos[self.addresses]-d.ctrl))))
            pipette=m.body('env_01_pipette').id;touched=False
            pairs=d.contact.geom
            selected=np.flatnonzero(np.any(m.geom_bodyid[pairs]==pipette,axis=1)
                                    &np.any(m.geom_contype[pairs]==2,axis=1))
            for i in selected:
                c=d.contact[int(i)]
                force=np.zeros(6);mujoco.mj_contactForce(m,d,int(i),force)
                if force[0]<=0:continue
                touched=True
                self.pipette_robot_peak_normal_force=max(self.pipette_robot_peak_normal_force,float(force[0]))
                self.pipette_robot_max_penetration=max(self.pipette_robot_max_penetration,-float(c.dist))
            if touched and self.first_pipette_robot_contact_s is None:self.first_pipette_robot_contact_s=float(d.time)
            self.pipette_robot_contact_steps+=int(touched)
            self.pipette_max_lift=max(self.pipette_max_lift,float(d.body(pipette).xipos[2]-self.start_com['env_01_pipette'][2]))

    def report(self):
        return dict(steps=self.steps,initial_penetration_mm=1000*self.initial_penetration,
                    initial_penetration_by_body_mm=self.initial_contacts,
                    max_penetration_mm=1000*self.max_penetration,contact_steps=self.contact_steps,
                    max_actuated_joint_tracking_error_rad=self.max_tracking_error,
                    object_com_displacement_m={n:(self.data.body(n).xipos-p).tolist() for n,p in self.start_com.items()},
                    object_masses_kg={n:float(self.model.body_mass[self.model.body(n).id]) for n in self.start_com},
                    pipette_robot_contact_steps=self.pipette_robot_contact_steps,
                    first_pipette_robot_contact_s=self.first_pipette_robot_contact_s,
                    pipette_robot_max_penetration_mm=self.pipette_robot_max_penetration*1000,
                    pipette_robot_peak_normal_force_n=self.pipette_robot_peak_normal_force,
                    pipette_max_com_lift_m=self.pipette_max_lift,
                    grasp_constraint=False)
