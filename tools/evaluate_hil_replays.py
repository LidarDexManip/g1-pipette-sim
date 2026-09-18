"""Full dynamic replay and observed-joint validation for a complete HIL dataset.

Unresolved scene fits use the session reference ONLY when explicitly requested;
those rollouts are labeled unsuitable for object-success validation.
"""
import argparse,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode,DEFAULT_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.profiles import for_episode
from g1_pipette_sim.episode_scene import resolve_scene
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.simulation import set_robot_pose


def evaluate(job):
    index,root,config,source,output,cache,allow_unresolved=job
    ep=Episode(root,index);cfg=for_episode(load_config(config),ep);scene_error=None
    try:cfg,_=resolve_scene(cfg,ep,cache=cache)
    except ValueError as error:
        if not allow_unresolved:raise
        scene_error=str(error)
    sim=DynamicSimulation(cfg,ep,source=source);m,d=sim.model,sim.data
    adr=np.array([m.jnt_qposadr[m.joint(n).id] for n in ep.names]);obsdata=mujoco.MjData(sim.reference_model)
    times=[];actual=[];observed=[];nozzle=[];obs_nozzle=[];wrist_errors=[];drift=[];tip=[];next_sample=0.;lift=0.
    grip=d.site('grip_frame');tool=d.body('pipette');R0=grip.xmat.reshape(3,3).T@tool.xmat.reshape(3,3);p0=grip.xmat.reshape(3,3).T@(tool.xpos-grip.xpos)
    while sim.time<ep.duration-1e-9:
        sim.step()
        if sim.state=='attached':
            body=d.body('tip');mouth=body.xpos+body.xmat.reshape(3,3)[:,2]*cfg['pipette']['tip_length']/2
            lift=max(lift,float(mouth[2]-d.site('rack_mouth').xpos[2]))
        if sim.time+1e-10<next_sample:continue
        next_sample+=1/60
        q,left,right=ep.sample(sim.time,'state');set_robot_pose(sim.reference_model,obsdata,ep.names,q,left,right,cfg);mujoco.mj_forward(sim.reference_model,obsdata)
        times.append(sim.time);actual.append(d.qpos[adr].copy());observed.append(q);nozzle.append(d.site('nozzle').xpos.copy());obs_nozzle.append(obsdata.site('nozzle').xpos.copy());tip.append(d.body('tip').xpos.copy())
        wrist_errors.append(d.body('right_wrist_yaw_link').xpos-obsdata.body('right_wrist_yaw_link').xpos)
        grip=d.site('grip_frame');tool=d.body('pipette');R=grip.xmat.reshape(3,3).T@tool.xmat.reshape(3,3);p=grip.xmat.reshape(3,3).T@(tool.xpos-grip.xpos)
        drift.append([np.linalg.norm(p-p0),np.degrees(np.arccos(np.clip((np.trace(R0.T@R)-1)/2,-1,1)))])
    q=np.asarray(actual);obs=np.asarray(observed);err=q-obs;drift=np.asarray(drift);report=sim.report()
    validation={'joint_rmse_rad':np.sqrt(np.mean(err**2,axis=0)).tolist(),'right_arm_rmse_rad':float(np.sqrt(np.mean(err[:,22:29]**2))),
        'right_arm_p95_abs_error_rad':float(np.quantile(abs(err[:,22:29]),.95)),
        'wrist_position_rmse_mm':float(np.sqrt(np.mean(np.sum(np.asarray(wrist_errors)**2,axis=1)))*1000),
        'nozzle_position_rmse_vs_measured_joint_fk_mm':float(np.sqrt(np.mean(np.sum((np.asarray(nozzle)-obs_nozzle)**2,axis=1)))*1000),
        'maximum_grasp_translation_drift_mm':float(drift[:,0].max()*1000),'maximum_grasp_rotation_drift_deg':float(drift[:,1].max()),
        'scene_error':scene_error,'scene_source':'unresolved_session_reference' if scene_error else 'visual_episode_fit',
        'maximum_attached_lift_mm':lift*1000,'object_ground_truth':'Unavailable: FK nozzle comparison assumes the configured mount, not measured object poses.'}
    report['validation']=validation
    report['input_config_sha256']=hashlib.sha256(Path(config).read_bytes()).hexdigest()
    out=Path(output)/f'episode-{index:03}';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'workspace.json').write_text(json.dumps(cfg,indent=2))
    np.savez_compressed(out/'trace.npz',time=times,q=q,observed=obs,nozzle=nozzle,measured_joint_fk_nozzle=obs_nozzle,tip=tip,grasp_drift=drift)
    return {'episode':index,'duration_s':ep.duration,'source':source,'state':sim.state,'events':sim.events,'warnings':report['warnings'],
        'finite':report['finite'],'initial_grasp':sim.grasp_initialization,'contacts':sim.contact_diagnostics,**validation}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--config',type=Path,default=Path('config/hil194.json'));p.add_argument('--episodes',nargs='+',default=['all']);p.add_argument('--source',choices=['low_level','action'],default='low_level');p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('outputs/hil194/scenes'));p.add_argument('--workers',type=int,default=6);p.add_argument('--allow-unresolved-layout',action='store_true');a=p.parse_args()
    first=Episode(a.dataset,0);indices=range(first.info['total_episodes']) if a.episodes==['all'] else list(map(int,a.episodes));a.output.mkdir(parents=True,exist_ok=True)
    frozen=a.output/'input-config.json';frozen.write_text(Path(a.config).read_text())
    jobs=[(i,first.root,frozen,a.source,a.output,a.cache,a.allow_unresolved_layout) for i in indices];results=[];failures=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(evaluate,j):j[0] for j in jobs}
        for future in as_completed(pending):
            try:
                r=future.result();results.append(r);print(f"{len(results)+len(failures)}/{len(jobs)} ep {r['episode']}: {r['state']}, arm RMSE {r['right_arm_rmse_rad']:.4f} rad, scene={r['scene_source']}",flush=True)
            except Exception as error:failures.append({'episode':pending[future],'error':repr(error)});print('FAILED',failures[-1],flush=True)
            (a.output/'summary.json').write_text(json.dumps({'results':sorted(results,key=lambda r:r['episode']),'failures':failures},indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
