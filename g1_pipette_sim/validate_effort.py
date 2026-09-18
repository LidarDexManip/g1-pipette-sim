"""Score identified PD torque predictions on held-out recorded motor data."""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from .dataset import Episode
from .scene import load_config

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('config/calibrated-physics.json'));p.add_argument('--output',type=Path,default=Path('outputs/calibration/heldout-effort.json'));a=p.parse_args();cfg=load_config(a.config);train=cfg['calibration']['training_episodes'];rows=[]
    for index in range(78):
        if index in train:continue
        ep=Episode(index=index);path=ep.root/ep.info['data_path'].format(episode_chunk=index//ep.info['chunks_size'],episode_index=index)
        tau=np.array(pq.read_table(path,columns=['observation.effort'])['observation.effort'].to_pylist());q=ep.arrays['observation.state'][:,:29];v=savgol_filter(q,15,2,deriv=1,delta=np.median(np.diff(ep.times)),axis=0)
        kp=np.array([cfg['physics']['joint_servos'][n]['kp'] for n in ep.names]);kv=np.array([cfg['physics']['joint_servos'][n]['kv'] for n in ep.names]);cmd=np.column_stack([np.interp(ep.times,ep.action_times,ep.arrays['action.low_level'][:,j]) for j in range(29)])
        pred=(cmd-q)*kp-v*kv;err=(pred-tau)[15:-15,22:29];actual=tau[15:-15,22:29]
        rows.append({'episode':index,'right_arm_effort_rmse_Nm':float(np.sqrt(np.mean(err**2))),'right_arm_joint_r2':(1-np.mean(err**2,axis=0)/np.maximum(np.var(actual,axis=0),1e-10)).tolist()})
    result={'training_episodes':train,'heldout_count':len(rows),'median_right_arm_effort_rmse_Nm':float(np.median([r['right_arm_effort_rmse_Nm'] for r in rows])),'median_joint_r2':np.median([r['right_arm_joint_r2'] for r in rows],axis=0).tolist(),'episodes':rows,'scope':'PD effort mapping only, not a forward-dynamics or object-motion validation'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))
if __name__=='__main__':main()
