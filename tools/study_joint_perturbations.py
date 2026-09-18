"""Bounded direct joint-noise study: 24 trajectories, unchanged acceptance gates.

Persistent offsets and approach-only bumps are compared separately. Placement
retries reuse identical saved commands and do not count as new perturbations.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import collect_pickup_augmentations as collection
from tools.search_larger_augmentations import exact_episode
from tools.fit_pickup_layouts import write_json
from tools.search_replay_layout import on_table
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.scene import build_xml
from g1_pipette_sim.simulation import set_robot_pose
from g1_pipette_sim.pickup_layouts import sha256,model_fingerprints
OUT=ROOT/'outputs/hil194/joint-perturbation-study'
BASE=ROOT/'outputs/hil194/augmentation600/candidates/000000'
SEED=20260914


def specification(index):
    family='persistent' if index<12 else 'approach_only'
    amplitude=[3.,7.,12.][(index%12)//4]
    direction=np.random.default_rng(np.random.SeedSequence([SEED,index])).normal(size=7)
    direction/=np.max(abs(direction))
    return family,amplitude,direction


def prepare(index):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    folder=OUT/'perturbations'/f'{index:03}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'proposal.json').exists():return json.loads((folder/'proposal.json').read_text())
    family,amplitude,direction=specification(index)
    cfg=json.loads((BASE/'config.json').read_text());prov=json.loads((BASE.parents[1]/'provenance.json').read_text())
    if model_fingerprints()!=prov['model_fingerprints']:raise ValueError('Baseline model changed')
    ep=Episode(prov['dataset_snapshot'],14)
    with np.load(BASE/'commands.npz') as a:
        original=a['motor_targets'].copy();hands={side:a['hand_'+side].copy() for side in ['left','right']}
        assert np.array_equal(a['times'],ep.action_times)
    envelope=collection.smoothstep(ep.action_times/3.)
    if family=='approach_only':envelope*=1-collection.smoothstep((ep.action_times-5.)/5.)
    correction=envelope[:,None]*np.radians(amplitude)*direction
    motor=original.copy();motor[:,22:29]+=correction
    model=mujoco.MjModel.from_xml_string(build_xml(cfg));data=mujoco.MjData(model)
    joints=np.array([model.joint(name).id for name in ep.names[22:29]])
    checks=[]
    if np.any(motor[:,22:29]<model.jnt_range[joints,0]) or np.any(motor[:,22:29]>model.jnt_range[joints,1]):checks.append('joint_limit')
    velocity=float(np.max(abs(np.diff(correction,axis=0)/np.diff(ep.action_times)[:,None])))
    if velocity>.35:checks.append('added_velocity_above_0.35_rad_s')
    # Fit a fixed scene placement to the nozzle displacement at nominal insertion.
    attach=json.loads((BASE/'accepted.json').read_text())['validation']['events'][0]['time']
    positions=[]
    for targets in [original,motor]:
        body=np.array([np.interp(attach,ep.action_times,targets[:,j]) for j in range(29)])
        left=np.array([np.interp(attach,ep.action_times,hands['left'][:,j]) for j in range(6)])
        right=np.array([np.interp(attach,ep.action_times,hands['right'][:,j]) for j in range(6)])
        set_robot_pose(model,data,ep.names,body,left,right,cfg);mujoco.mj_forward(model,data)
        positions.append(data.site('nozzle').xpos.copy())
    shift=positions[1]-positions[0]
    # A large vertical move can put the fixed initial robot inside the table.
    # Keep generation bounded and count these rejections explicitly.
    if abs(shift[2])>.04:checks.append('table_height_shift_above_40_mm')
    cfg=collection.placed_config(cfg,cfg['rack']['active_cell'],shift[:2]);cfg['table']['top_z']+=float(shift[2])
    if not on_table(cfg):checks.append('off_table')
    commands=folder/'commands.npz';np.savez_compressed(commands,times=ep.action_times,motor_targets=motor,hand_left=hands['left'],hand_right=hands['right'])
    write_json(folder/'base-config.json',cfg)
    proposal={'perturbation_id':index,'family':family,'amplitude_deg':amplitude,'joint_direction':direction.tolist(),'joint_names':ep.names[22:29],'max_joint_delta_deg':float(np.degrees(np.max(abs(correction)))),'max_added_velocity_rad_s':velocity,'ramp_in_s':3,'ramp_out_s':[5,10] if family=='approach_only' else None,'nominal_tip_shift_m':shift.tolist(),'dataset_snapshot':str(ep.root),'commands':str(commands),'commands_sha256':sha256(commands),'seed':SEED,'generation_rejections':checks,'base_config':str(folder/'base-config.json')}
    write_json(folder/'proposal.json',proposal);return proposal


def candidate(proposal,attempt=0,xy=(0.,0.)):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    number=proposal['perturbation_id']*10+attempt
    folder=OUT/'candidates'/f'{number:06}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'candidate.json'
    if path.exists():return json.loads(path.read_text())
    cfg=json.loads(Path(proposal['base_config']).read_text())
    cfg=collection.placed_config(cfg,cfg['rack']['active_cell'],xy)
    config=folder/'config.json';write_json(config,cfg)
    c={'number':number,'parent_episode':14,'kind':'direct_joint_'+proposal['family'],'perturbation_id':proposal['perturbation_id'],'placement_attempt':attempt,'cell':cfg['rack']['active_cell'],'target_shift_m':proposal['nominal_tip_shift_m'][:2],'placement_refinement_m':list(xy),'noise':{'method':'direct additive joint offsets; no IK or null-space projection','max_joint_delta_rad':float(np.radians(proposal['max_joint_delta_deg'])),'amplitude_deg':proposal['amplitude_deg'],'direction':proposal['joint_direction'],'family':proposal['family']},'seed_attachment_s':12.8515,'config':str(config),'config_sha256':sha256(config),'commands':proposal['commands'],'commands_sha256':proposal['commands_sha256'],'folder':str(folder),'dataset_snapshot':proposal['dataset_snapshot'],'source_capture':str(BASE)}
    # Each accepted capture is self-contained for packaging and replay.
    import shutil
    shutil.copy2(proposal['commands'],folder/'commands.npz');c['commands']=str(folder/'commands.npz')
    write_json(path,c);return c


def evaluate(spec):
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    index,attempt,xy,record=spec
    proposal=prepare(index)
    if proposal['generation_rejections']:return {'perturbation_id':index,'generation_rejections':proposal['generation_rejections']}
    c=candidate(proposal,attempt,xy);folder=Path(c['folder'])
    cached=folder/('accepted.json' if record else 'screen.json')
    if cached.exists():result=json.loads(cached.read_text())['validation' if record else 'result']
    else:
        result=collection.validate(c,record=record)
    return {'perturbation_id':index,'candidate':c['number'],'attempt':attempt,'result':result}


def summarize():
    rows=[];total_replays=0
    for p in sorted((OUT/'perturbations').glob('*/proposal.json')):
        proposal=json.loads(p.read_text());i=proposal['perturbation_id'];trials=[]
        for folder in sorted((OUT/'candidates').glob('*')):
            c=json.loads((folder/'candidate.json').read_text())
            if c['perturbation_id']!=i:continue
            if (folder/'screen.json').exists():
                r=json.loads((folder/'screen.json').read_text())['result'];trials.append({'candidate':c['number'],'pickup':r['pickup'],'attached':r.get('attached',False),'lift_m':r['lift_m'],'invalid_reason':r.get('invalid_reason'),'recorded':(folder/'accepted.json').exists()})
        total_replays+=len(trials)
        rows.append({'perturbation_id':i,'family':proposal['family'],'amplitude_deg':proposal['amplitude_deg'],'generation_rejections':proposal['generation_rejections'],'screened':bool(trials),'screen_success':any(t['pickup'] for t in trials),'recorded_success':any(t['recorded'] for t in trials),'trials':trials})
    report={'seed':SEED,'proposed_perturbations':len(rows),'generation_rejections':sum(bool(r['generation_rejections']) for r in rows),'distinct_perturbations_screened':sum(r['screened'] for r in rows),'placement_trials':total_replays,'screen_successes':sum(r['screen_success'] for r in rows),'independently_recorded_successes':sum(r['recorded_success'] for r in rows),'perturbations':rows}
    write_json(OUT/'results.json',report);return report


def main():
    from g1_pipette_sim.augmentation.contracts import legacy_write_disabled
    legacy_write_disabled()
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','refine','record'],default='screen');p.add_argument('--workers',type=int,default=4);p.add_argument('--perturbations',type=int,nargs='+');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.phase=='screen':specs=[(i,0,[0.,0.],False) for i in range(24)]
    elif a.phase=='refine':
        specs=[]
        reference=json.loads((BASE/'verification.json').read_text())['report']['attachment_diagnostics']['closest_approach']
        bias=np.array(reference['nozzle_world_m'])[:2]-np.array(reference['tip_mouth_world_m'])[:2]
        for p in sorted((OUT/'candidates').glob('*/screen.json')):
            c=json.loads((p.parent/'candidate.json').read_text());r=json.loads(p.read_text())
            if c['placement_attempt']!=0 or r['result']['pickup']:continue
            closest=r['report'].get('attachment_diagnostics',{}).get('closest_approach')
            if closest is None:continue
            correction=np.array(closest['nozzle_world_m'])[:2]-np.array(closest['tip_mouth_world_m'])[:2]-bias
            if np.linalg.norm(correction)>.02:continue
            for attempt,scale in [(1,.5),(2,1.)]:specs.append((c['perturbation_id'],attempt,(correction*scale).tolist(),False))
    else:
        report=summarize();specs=[]
        for r in report['perturbations']:
            good=[t for t in r['trials'] if t['pickup']]
            if not good:continue
            c=json.loads((OUT/'candidates'/f"{good[0]['candidate']:06}"/'candidate.json').read_text())
            specs.append((c['perturbation_id'],c['placement_attempt'],c['placement_refinement_m'],True))
    if a.perturbations is not None:specs=[s for s in specs if s[0] in a.perturbations]
    errors=[]
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(evaluate,s):s for s in specs}
        for f in as_completed(futures):
            try:
                r=f.result();result=r.get('result',{})
                print(json.dumps({'phase':a.phase,'perturbation':r['perturbation_id'],'candidate':r.get('candidate'),'pickup':result.get('pickup'),'lift_mm':result.get('lift_m',0)*1000,'generation_rejections':r.get('generation_rejections')}),flush=True)
            except Exception as e:errors.append({'spec':futures[f],'error':repr(e)});print('ERROR',repr(e),flush=True)
    write_json(OUT/(a.phase+'-errors.json'),errors)
    report=summarize();print(json.dumps({k:v for k,v in report.items() if k!='perturbations'}),flush=True)
if __name__=='__main__':main()
