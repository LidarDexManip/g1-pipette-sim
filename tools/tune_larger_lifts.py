"""Trim only the post-attachment lift of larger translated arm demonstrations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import search_larger_augmentations as search
from tools.fit_pickup_layouts import write_json
from g1_pipette_sim.pickup_layouts import sha256


def run(spec):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    source,trim,number=spec
    path=search.OUT/'candidates'/f'{source:06}'
    c=json.loads((path/'candidate.json').read_text());ep=search.exact_episode(c);cfg=json.loads((path/'config.json').read_text())
    folder=search.OUT/'candidates'/f'{number:06}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'screen.json').exists():return number,json.loads((folder/'screen.json').read_text())['result']
    motor,noise=search.actions(ep,cfg,np.array([0.,0.]),lift_trim=trim)
    commands=folder/'commands.npz';np.savez_compressed(commands,times=ep.action_times,motor_targets=motor,hand_left=ep.arrays['action.hand_left'],hand_right=ep.arrays['action.hand_right'])
    write_json(folder/'config.json',cfg)
    c.update(number=number,folder=str(folder),config=str(folder/'config.json'),commands=str(commands),lift_trim_m=trim,refined_from=source)
    c['config_sha256']=sha256(c['config']);c['commands_sha256']=sha256(commands)
    with np.load(search.BASE/'commands.npz') as original:
        delta=motor[:,22:29]-original['motor_targets'][:,22:29]
    c['noise']['max_joint_delta_rad']=float(np.max(abs(delta)));c['noise']['post_attachment_lift_trim_m']=trim
    write_json(folder/'candidate.json',c)
    result=search.collection.validate(c);return number,result


def main():
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    specs=[]
    for p in sorted((search.OUT/'candidates').glob('*/screen.json')):
        number=int(p.parent.name)
        if number>=200:continue
        r=json.loads(p.read_text())['result']
        if r.get('attached') and r.get('valid') and r.get('seed_phase_timing_passed') and not r['pickup']:
            extra=r['lift_m']-r['lift_target_m']
            if extra>0:
                for j,trim in enumerate([extra,extra*.7]):specs.append((number,trim,200+number*2+j))
    results=[]
    with ProcessPoolExecutor(max_workers=4,max_tasks_per_child=1) as pool:
        for number,r in pool.map(run,specs):
            results.append({'number':number,'result':r});write_json(search.OUT/'lift-refinement-summary.json',results)
            print(json.dumps({'candidate':number,'pickup':r['pickup'],'lift_mm':r['lift_m']*1000,'events':r.get('events')}),flush=True)
if __name__=='__main__':main()
