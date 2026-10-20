"""Fit planar robot placement using hand-closure events; never fit actions to success."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.environments import environment_manifest
from g1_pipette_sim.environments.robot import build_robot_scene,load_registration,MeasuredReplay,DEFAULT_REGISTRATION


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/lidar-teleop/placement-fit.json'));a=p.parse_args()
    cfg=load_registration();reference=dict(cfg,base_xy=[0,0],base_yaw_deg=0)
    model=mujoco.MjModel.from_xml_string(build_robot_scene(registration=reference));data=mujoco.MjData(model)
    manifest=environment_manifest('lidar');objects={o['body']:o for o in manifest['objects']}
    targets={side:np.asarray(next(o['bounds_m'] for key,o in objects.items() if key.endswith(suffix))).mean(axis=1) for side,suffix in [('right','_pipette'),('left','_cylinder')]}
    notes=json.loads((a.dataset/'meta/source_notes.json').read_text());selection={side:[int(i) for i,t in notes['task_types']['by_episode_index'].items() if t==task][:5] for side,task in [('right','step1_pipette_pick'),('left','step2_tube_pick')]}
    records=[]
    for side,episodes in selection.items():
        for index in episodes:
            episode=Episode(a.dataset,index);raw=episode.arrays[f'action.hand_{side}'];signal=raw[:,:4].mean(axis=1) if side=='right' else raw[:,4]
            if np.ptp(signal)<100:continue
            frame=int(np.flatnonzero(signal<(signal.max()+signal.min())/2)[0]);time=float(episode.times[frame])
            replay=MeasuredReplay(model,data,episode);replay.pose(time);wrist=data.body(f'{side}_wrist_yaw_link')
            point=wrist.xpos+wrist.xmat.reshape(3,3)@np.array([.09,0,0])
            records.append({'episode':index,'side':side,'frame':frame,'time_s':time,'reference_palm_m':point.tolist(),'target_object_center_m':targets[side].tolist()})
    points=np.array([r['reference_palm_m'][:2] for r in records]);target=np.array([r['target_object_center_m'][:2] for r in records])
    def residual(x):
        yaw=np.deg2rad(x[2]);R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        return (points@R.T+x[:2]-target).ravel()
    fit=least_squares(residual,[.55,0,180],bounds=([.43,-.4,130],[.9,.4,230]),loss='soft_l1',f_scale=.02)
    # Reject a visually convenient fit that puts the hips through the table.
    table_id=model.body('env_07_table_new').id;pelvis=model.body('pelvis').id
    checks=[(Episode(a.dataset,index),index) for index in [r['episode'] for r in records]]
    checks=[(e,MeasuredReplay(model,data,e)) for e,index in checks]
    clearance_checks=[]
    for minimum_x in np.arange(float(fit.x[0]),.651,.005):
        fit=least_squares(residual,[minimum_x+.00001,float(fit.x[1]),float(fit.x[2])],bounds=([minimum_x,-.4,130],[.9,.4,230]),loss='soft_l1',f_scale=.02)
        model.body_pos[pelvis,:2]=fit.x[:2];theta=np.deg2rad(fit.x[2])/2;model.body_quat[pelvis]=[np.cos(theta),0,0,np.sin(theta)]
        penetration=0.
        for episode,replay in checks:
            for time in np.linspace(0,episode.duration,5):
                replay.pose(time)
                for contact in data.contact:
                    ga,gb=contact.geom1,contact.geom2
                    other=gb if model.geom_bodyid[ga]==table_id else (ga if model.geom_bodyid[gb]==table_id else None)
                    if other is not None and model.geom_contype[other]==2:
                        body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[other])) or ''
                        if any(part in body_name for part in ('pelvis','hip','knee','ankle','waist','torso')):
                            penetration=max(penetration,float(-contact.dist))
        clearance_checks.append({'base_x_m':float(fit.x[0]),'max_core_table_penetration_mm':1000*penetration})
        if penetration<=.0005:break
    else:raise RuntimeError('No collision-clear placement found in the bounded fit')
    cfg['base_xy']=fit.x[:2].tolist();cfg['base_yaw_deg']=float(fit.x[2]);cfg['placement_status']='Approximate XY/yaw fit to ten measured-joint hand-closure poses; 9 cm palm proxy, fixed pelvis height; sampled torso/leg table-clearance checked, manipulation contacts unvalidated'
    cfg['fit_dataset']={'repo_id':'jren313/g1-pipette-2view-teleop0925-5task-eerel','revision':a.dataset.name,'episodes':[r['episode'] for r in records]}
    errors=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1)
    report={'registration':cfg,'table_clearance_search':clearance_checks,'planar_error_mm':{'median':float(np.median(errors)*1000),'max':float(errors.max()*1000)},'events':records,
            'limitations':['Object centers are approximate grasp anchors, not measured contact points.','Palms use a 9 cm wrist offset; fingertip/grasp geometry not calibrated.','First five pickups of each hand only; other recording sessions need independent registration.','Z and geometry were not optimized to force a successful task.']}
    DEFAULT_REGISTRATION.write_text(json.dumps(cfg,indent=2)+'\n');a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
