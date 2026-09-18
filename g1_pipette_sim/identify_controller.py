"""Fit PD gains to motor effort; never use held-out episode data for fitting.

Run as a module from the project root. Effort intercepts are diagnostics only;
they are not replayed as feed-forward forces. Full dynamics need separate validation.
"""
from g1_pipette_sim.dataset import Episode,LEGACY_DATASET
import pyarrow.parquet as pq,numpy as np,json
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
from pathlib import Path
import argparse
from .scene import load_config
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--train',type=int,nargs='+',default=[53,54,55,56])
parser.add_argument('--output',type=Path,default=Path('outputs/calibration/effort_fit.json'))
parser.add_argument('--config-output',type=Path,default=Path('config/identified.json'))
args=parser.parse_args()
records=[]
for i in args.train:
 e=Episode(LEGACY_DATASET,index=i);path=e.root/e.info['data_path'].format(episode_chunk=i//e.info['chunks_size'],episode_index=i)
 tau=np.array(pq.read_table(path,columns=['observation.effort'])['observation.effort'].to_pylist());q=e.arrays['observation.state'][:,:29];dt=np.median(np.diff(e.times));vel=savgol_filter(q,15,2,deriv=1,delta=dt,axis=0)
 records.append((e,q,vel,tau))
results={}
for source,key in [('action','action'),('low_level','action.low_level')]:
 fits=[]
 for j in range(29):
  best=None
  for delay in [0,.025,.05,.075,.1]:
   xx=[];yy=[]
   for e,q,v,tau in records:
    cmd=np.interp(e.times-delay,e.action_times,e.arrays[key][:,j]);mask=(e.times>.3)&(e.times<e.duration-.3)
    xx.append(np.c_[cmd[mask]-q[mask,j],-v[mask,j],np.ones(mask.sum())]);yy.append(tau[mask,j])
   X=np.concatenate(xx);Y=np.concatenate(yy);fit=lsq_linear(X,Y,bounds=([1,0,-10],[300,25,10]));pred=X@fit.x;rmse=float(np.sqrt(np.mean((pred-Y)**2)));r2=1-np.mean((pred-Y)**2)/max(np.var(Y),1e-10)
   rec=dict(joint=records[0][0].names[j],kp=float(fit.x[0]),kv=float(fit.x[1]),bias=float(fit.x[2]),delay=delay,rmse=rmse,r2=float(r2))
   if best is None or rmse<best['rmse']:best=rec
  fits.append(best)
 results[source]=fits
 print(source,json.dumps(fits[22:],indent=2),flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(results,indent=2)+'\n')
c=load_config('config/workspace.json');fits=results['low_level']
c['attachment']['eject_thumb_bend_above']=400.
c['physics']['joint_servos']={r['joint']:{'kp':r['kp'],'kv':r['kv'],'gravity_compensation':False} for r in fits}
c['calibration']={'training_episodes':args.train,'source':'low_level','method':'Bounded motor effort regression; fixed gains, no target correction or observed-state feedback','effort_fit':str(args.output)}
args.config_output.parent.mkdir(parents=True,exist_ok=True);args.config_output.write_text(json.dumps(c,indent=2)+'\n')
