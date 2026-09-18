"""Brief elbow-only insertion perturbations with a planned return to alignment."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import collect_pickup_augmentations as collection
from tools.search_larger_augmentations import exact_episode
from tools.fit_pickup_layouts import write_json
from g1_pipette_sim.pickup_layouts import sha256,model_fingerprints
from g1_pipette_sim.dataset import Episode
OUT=ROOT/'outputs/hil194/elbow-correction-study'
BASE=ROOT/'outputs/hil194/augmentation600/candidates/000000'


def evaluate(spec):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    number,amplitude,record=spec
    folder=OUT/'candidates'/f'{number:06}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'candidate.json'
    if path.exists():c=json.loads(path.read_text())
    else:
        prov=json.loads((BASE.parents[1]/'provenance.json').read_text())
        assert model_fingerprints()==prov['model_fingerprints']
        ep=Episode(prov['dataset_snapshot'],14);joint=ep.names.index('right_elbow_joint')
        with np.load(BASE/'commands.npz') as a:
            motor=a['motor_targets'].copy();times=a['times'].copy();left=a['hand_left'].copy();right=a['hand_right'].copy()
        return_start,return_end=(11.2,12.0) if number>=12 else (11.4,12.6)
        envelope=collection.smoothstep((times-10.)/1.)*(1-collection.smoothstep((times-return_start)/(return_end-return_start)))
        motor[:,joint]+=np.radians(amplitude)*envelope
        command=folder/'commands.npz';np.savez_compressed(command,times=times,motor_targets=motor,hand_left=left,hand_right=right)
        config=folder/'config.json';config.write_bytes((BASE/'config.json').read_bytes())
        c={'number':number,'parent_episode':14,'kind':'elbow_insertion_pulse','cell':json.loads(config.read_text())['rack']['active_cell'],'target_shift_m':[0.,0.],'elbow_offset_deg':amplitude,'pulse_ramp_in_s':[10.,11.],'pulse_ramp_out_s':[return_start,return_end],'correction':'preplanned return of elbow target; no feedback controller','noise':{'method':'direct elbow joint pulse','max_joint_delta_rad':float(np.radians(abs(amplitude)))},'seed_attachment_s':12.8515,'config':str(config),'config_sha256':sha256(config),'commands':str(command),'commands_sha256':sha256(command),'folder':str(folder),'dataset_snapshot':str(ep.root),'source_capture':str(BASE)}
        write_json(path,c)
    cached=folder/('accepted.json' if record else 'screen.json')
    if cached.exists():result=json.loads(cached.read_text())['validation' if record else 'result']
    else:
        result=collection.validate(c,record=record)
    return {'number':number,'amplitude_deg':amplitude,'result':result}


def main():
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');parser.add_argument('--extra',action='store_true');a=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    amplitudes=[.2,-.2,.4,-.4,.8,-.8,1.2,-1.2,.3,.35,.45,.5,.4,-.4,.8,-.8,1.2,-1.2]
    specs=[(i,x,a.record) for i,x in enumerate(amplitudes)]
    if a.extra:specs=[s for s in specs if s[0]>=8]
    elif not a.record:specs=[s for s in specs if s[0]<8]
    if a.record:
        specs=[s for s in specs if (OUT/'candidates'/f'{s[0]:06}'/'screen.json').exists() and json.loads((OUT/'candidates'/f'{s[0]:06}'/'screen.json').read_text())['result']['pickup']]
        specs=sorted(specs,key=lambda s:-abs(s[1]))[:2]
    rows=[]
    with ProcessPoolExecutor(max_workers=4 if not a.record else 2,max_tasks_per_child=1) as pool:
        futures={pool.submit(evaluate,s):s for s in specs}
        for f in as_completed(futures):
            r=f.result();rows.append(r);result=r['result']
            print(json.dumps({'candidate':r['number'],'elbow_deg':r['amplitude_deg'],'pickup':result['pickup'],'lift_mm':result.get('lift_m',0)*1000,'events':result.get('events'),'reason':result.get('invalid_reason')}),flush=True)
            write_json(OUT/('recording-results.json' if a.record else ('extra-screening-results.json' if a.extra else 'screening-results.json')),rows)
if __name__=='__main__':main()
