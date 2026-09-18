"""Open-loop recorded-command validation. Observations are used only for scoring."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .dataset import Episode,DEFAULT_DATASET
from .scene import load_config,fit_layout
from .dynamics import DynamicSimulation


def evaluate(job):
    index,config_path,root,source,output,fit=job
    ep=Episode(root,index);cfg=load_config(config_path)
    if fit:cfg=fit_layout(cfg,ep)
    sim=DynamicSimulation(cfg,ep,source=source)
    qadr=np.array([sim.model.jnt_qposadr[sim.model.joint(n).id] for n in ep.names]);times=[];actual=[];observed=[];nozzles=[];tips=[];lids=[];next_sample=0
    while sim.time<ep.duration-1e-9:
        sim.step()
        if sim.time>=next_sample:
            times.append(sim.time);actual.append(sim.data.qpos[qadr].copy());observed.append(ep.sample(sim.time,'state')[0]);nozzles.append(sim.data.site('nozzle').xpos.copy());tips.append(sim.data.body('tip').xpos.copy());lids.append(sim.data.body('waste').xpos.copy());next_sample+=1/60
    actual=np.array(actual);observed=np.array(observed);err=actual-observed;report=sim.report()
    report['validation']={'samples':len(times),'joint_rmse_rad':np.sqrt(np.mean(err**2,axis=0)).tolist(),'body_rmse_rad':float(np.sqrt(np.mean(err**2))),'right_arm_rmse_rad':float(np.sqrt(np.mean(err[:,22:29]**2))),'right_arm_p95_abs_error_rad':float(np.quantile(np.abs(err[:,22:29]),.95)),'object_trajectory_accuracy':'not measured: no calibrated object pose ground truth','layout_fit_uses_this_episode':fit}
    out=Path(output)/f'episode-{index:03}';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'workspace.json').write_text(json.dumps(cfg,indent=2)+'\n')
    np.savez_compressed(out/'trace.npz',time=times,q=actual,observed=observed,nozzle=nozzles,tip=tips,lid=lids)
    return {'episode':index,'source':source,'state':report['state'],'warnings':report['warnings'],**report['validation']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,nargs='+',required=True);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--source',choices=['state','action','low_level'],default='low_level');p.add_argument('--output',type=Path,required=True);p.add_argument('--fit-layout',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    jobs=[(i,a.config,a.dataset,a.source,a.output,a.fit_layout) for i in a.episodes];results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(evaluate,jobs):results.append(r);print(json.dumps(r),flush=True)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
