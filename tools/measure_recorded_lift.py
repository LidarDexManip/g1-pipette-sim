"""Estimate per-episode lift from measured-joint FK; no object-pose ground truth."""
import argparse,json
from pathlib import Path
import numpy as np

def estimate(times,nozzle):
    mask=np.asarray(times)>.5;t=np.asarray(times)[mask];z=np.asarray(nozzle)[mask,2]
    if len(z)<6:raise ValueError('Too few samples after initialization')
    z=np.convolve(z,np.ones(6)/6,mode='valid');t=t[5:]
    # A later return below the rack must not erase an earlier recorded pickup.
    rise=np.maximum.accumulate(z[::-1])[::-1]-z;i=int(np.argmax(rise));j=i+int(np.argmax(z[i:]));target=float(rise[i]);tolerance=max(.002,.1*target)
    return {'target_m':target,'tolerance_m':tolerance,'required_lift_m':max(.002,target-tolerance),'reference_low_time_s':float(t[i]),'reference_high_time_s':float(t[j]),'method':'Maximum later rise from 100 ms smoothed measured-joint nozzle FK after first 0.5 s, assuming configured rigid tool mount; not measured object pose'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,default=Path('outputs/hil194/final-replays'));p.add_argument('--output',type=Path,default=Path('outputs/hil194/pickup-layouts/lift-targets.json'));a=p.parse_args();refs={}
    for path in sorted(a.baseline.glob('episode-*/trace.npz')):
        data=np.load(path);refs[str(int(path.parent.name.split('-')[-1]))]=estimate(data['time'],data['measured_joint_fk_nozzle'])
    a.output.write_text(json.dumps(refs,indent=2)+'\n')
if __name__=='__main__':main()
