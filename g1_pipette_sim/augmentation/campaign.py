"""Balanced, immutable multi-parent pilot with compact failure retention."""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time
import traceback
import tarfile
import numpy as np
from .contracts import ROOT, atomic_json, bind_run, dependencies, digest, file_hash, lock, verify, seal
from .sources import import_seed
from .store import Ledger
from .runner import task, stage, candidate_inputs
from .campaign_report import report, clustering_export
from .storage import write_storage, inventory

BASELINES={6:4,13:5,14:0,24:6,30:7,35:20,40:9,41:10,45:47,64:25,68:50,144:3}


def phase_window(folder, cfg):
    """Freeze a conservative geometric proximity boundary from the saved baseline."""
    import mujoco
    from g1_pipette_sim.dynamics import build_dynamic_xml
    model=mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg))
    address=int(model.jnt_qposadr[model.joint('tip_free').id])
    with np.load(Path(folder)/'trajectory.npz') as data:
        w,x,y,z=data['qpos'][:,address+3:address+7].T
        axis=np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],axis=1)
        mouth=data['tip_position']+axis*cfg['pipette']['tip_length']/2
        distance=np.linalg.norm(data['nozzle_position']-mouth,axis=1)
        near=np.flatnonzero(distance<.005)
        near_time=float(data['timestamp'][near[0]]) if len(near) else 0.
    end=np.floor((near_time-.5)*10)/10
    eligible=end>=1.
    return {'eligible':bool(eligible),'approach_s':[round(end*.25,3),round(end*.5,3),round(end,3)] if eligible else None,
            'first_5mm_proximity_s':near_time,'guard_s':.5,'reason':None if eligible else 'insufficient_pre_contact_approach',
            'method':'First saved nozzle-to-tip-mouth proximity below 5 mm, with 0.5 s guard; frozen before sampling. This is a geometric proxy, not contact-onset ground truth.'}


def build_sources():
    collection=ROOT/'outputs/hil194/augmentation600'
    split_path=ROOT/'outputs/hil194/starvla-motor30/preparation-contract.json'
    old=json.loads(split_path.read_text());held=set(old['held_out_real_episodes'])
    sources=[]
    for parent,number in BASELINES.items():
        folder=collection/'candidates'/f'{number:06}'
        candidate,ep,record=import_seed(folder)
        if str(ep.root.resolve())!=str(Path(old['real_snapshot']).resolve()):raise ValueError('Existing split snapshot differs')
        cfg=json.loads(Path(candidate['config']).read_text())
        sources.append({'parent_episode':parent,'folder':str(folder),'candidate':candidate,'partition':'val' if parent in held else 'train',
                        'source_video':str(ep.video_path('rgb')),'source_videos':{view:str(ep.video_path(view)) for view in ['rgb','wrist_right','wrist_left']},
                        'phase':phase_window(folder,cfg),'config':cfg,
                        'source_artifacts':{name:file_hash(folder/name) for name in ['accepted.json','trajectory.npz','verification.json','config.json','commands.npz']}})
    return sources,{'path':str(split_path),'sha256':file_hash(split_path),'held_out_real_episodes':sorted(held),'snapshot':old['real_snapshot']}


def schedule(sources, seed, root, campaign_id, draws=12):
    if not 1 <= draws <= min(s["config"]["rack"]["rows"]*s["config"]["rack"]["cols"]-1 for s in sources):
        raise ValueError("Draw count exceeds available distinct rack cells")
    specs=[]
    # Draw-major interleaving gives every source both families in each round.
    for draw in range(draws):
        for source in sources:
            parent=source['parent_episode'];cfg=source['config'];cell=cfg['rack']['active_cell']
            rng=np.random.default_rng(np.random.SeedSequence([seed,parent,0,draw]))
            cells=[(r,c) for r in range(cfg['rack']['rows']) for c in range(cfg['rack']['cols']) if [r,c]!=cell]
            permutation=np.random.default_rng(np.random.SeedSequence([seed,parent,100])).permutation(len(cells))
            chosen=list(cells[int(permutation[draw])])
            xy=[0.,0.] if draw<4 else rng.uniform(-.0005,.0005,2).tolist()
            direction=np.random.default_rng(np.random.SeedSequence([seed,parent,1,(draw//12)*6+draw%6])).normal(size=7)
            direction/=np.max(abs(direction))
            recipes=[('placement_only',{'method':'well_only','cell':chosen,'shift_m':xy}),
                     ('joint_approach',{'method':'joint_approach','amplitude_deg':1. if draw%12<6 else 3.,
                                        'direction':direction.tolist(),'approach_s':source['phase']['approach_s']})]
            for family,recipe in recipes:
                number=len(specs);eligible=family=='placement_only' or source['phase']['eligible']
                spec={'number':number,'run_id':campaign_id,'campaign_id':campaign_id,'source':source['folder'],
                      'source_contract':source['candidate'],'parent_episode':parent,'family':family,'draw':draw,
                      'partition':source['partition'],'source_video':source['source_video'],'recipe':recipe,
                      'seed':seed+number,'output':str(Path(root)/'runs'),'record':True,'placement_attempts':3,
                      'eligible':eligible,'skip_reason':None if eligible else source['phase']['reason']}
                spec['id']=digest(spec);specs.append(spec)
    return specs


def execute(spec):
    started=time.monotonic();root=Path(spec['output']);folder=root/'proposals'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'proposal-spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Proposal spec changed')
    atomic_json(path,spec)
    timestamp=time.time_ns()
    try:
        result=task(spec)
    except Exception as error:
        result={'state':'error','error':repr(error),'traceback':traceback.format_exc()}
    result.update(proposal_id=spec['id'],parent_episode=spec['parent_episode'],family=spec['family'],draw=spec['draw'],
                  partition=spec['partition'],worker_seconds=time.monotonic()-started,
                  peak_worker_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    atomic_json(folder/f'execution-{timestamp}.json',result)
    return result


def run(root, seed=20260915, workers=4, draws=12):
    root=Path(root).resolve()
    protected=ROOT/'outputs/hil194/augmentation600'
    if root==protected or root.is_relative_to(protected):raise ValueError('Campaign must be separate from augmentation600')
    with lock(root):
        sources,split=build_sources()
        contract={'schema':1,'seed':seed,'sources':sources,'split':split,'dependencies':dependencies(),
                  'quota_per_parent_family':draws,'placement_attempts':3,'infrastructure_attempts':2,
                  'failure_retention':'20 Hz state and 200 Hz pre-attachment alignment, commands/configs/reports; no extra failed-screen videos'}
        bind_run(root,contract)
        campaign_id=digest(contract);specs=schedule(sources,seed,root,campaign_id,draws)
        if not verify(root,'code-snapshot',contract['dependencies']):
            with tarfile.open(root/'code-snapshot.tar.gz','w:gz') as archive:
                for relative in contract['dependencies']['files']:
                    if relative.endswith(('.py','.json')):archive.add(ROOT/relative,arcname=relative)
            seal(root,'code-snapshot',contract['dependencies'],[root/'code-snapshot.tar.gz'])
        atomic_json(root/'campaign.json',{'campaign_id':campaign_id,'proposals':len(specs),'sources':sources,'split':split})
        # The full registry keeps all original videos visible, including nonparticipants.
        manifest=json.loads((ROOT/'config/hil194-pickup-layouts.json').read_text())
        selected={s['parent_episode']:s for s in sources};eligibility=[]
        for parent in range(len(manifest['episodes'])):
            entry=manifest['episodes'][str(parent)];source=selected.get(parent)
            eligibility.append({'original_episode':parent,'selected':source is not None,'partition':'val' if parent in split['held_out_real_episodes'] else 'train',
                                'status':'ready' if source else 'baseline_capture_not_selected' if entry.get('validation',{}).get('pickup') else 'no_passing_published_reconstruction',
                                'joint_eligible':source['phase']['eligible'] if source else False,'phase':source['phase'] if source else None})
        atomic_json(root/'source-eligibility.json',eligibility)
        atomic_json(root/'proposal-schedule.json',specs)
        ledger=Ledger(root)
        for spec in specs:
            ledger.add(spec['id'])
            if not spec['eligible']:
                atomic_json(root/'runs/proposals'/spec['id']/'proposal-spec.json',spec)
                ledger.finish(spec['id'],'skipped',{'reason':spec['skip_reason']})
        for path in (root/'runs/candidates').glob('*/candidate.json'):
            c=json.loads(path.read_text())
            if not verify(path.parent,'proposal',c['lineage'][-1]):
                if (path.parent/'accepted.json').exists():raise ValueError('Acceptance lacks proposal receipt')
                continue  # Incomplete unaccepted proposal is deterministically regenerated by its task.
            for name in ['screen','record','record-failed']:verify(path.parent,name,candidate_inputs(c))
            if (path.parent/'accepted.json').exists():stage(c,True)
        report(root,specs,ledger)
        started=time.time();calibration_complete=False
        for retry_round in range(2):
            pending=[s for s in specs if s['eligible'] and ledger.claim(s['id'],2)]
            batches=[pending[:2],pending[2:]] if pending and not calibration_complete else [pending]
            for batch_index,batch in enumerate(batches):
                if not batch:continue
                width=2 if not calibration_complete and batch_index==0 else workers
                with concurrent.futures.ProcessPoolExecutor(max_workers=width,max_tasks_per_child=1,mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures={pool.submit(execute,s):s for s in batch}
                    for future in concurrent.futures.as_completed(futures):
                        spec=futures[future]
                        try:result=future.result()
                        except Exception as error:result={'state':'error','error':repr(error)}
                        ledger.finish(spec['id'],result['state'],result)
                        summary,_=report(root,specs,ledger)
                        atomic_json(root/'run-timing.json',{'started_epoch':started,'updated_epoch':time.time(),'elapsed_seconds':time.time()-started,'workers':width})
                        print(json.dumps({'proposal':spec['number'],'parent':spec['parent_episode'],'family':spec['family'],'state':result['state'],
                                          'accepted':summary['totals']['accepted'],'worker_seconds':result.get('worker_seconds'),'rss_GiB':result.get('peak_worker_rss_bytes',0)/2**30}),flush=True)
                if not calibration_complete:
                    write_storage(root,root/'reports/storage.json',1);calibration_complete=True
        summary,attempts=report(root,specs,ledger)
        clustering_export(root,attempts)
        if dependencies()!=contract['dependencies']:raise ValueError('Implementation changed during campaign; refusing finalization')
        for source in sources:
            for name,expected in source['source_artifacts'].items():
                if file_hash(Path(source['folder'])/name)!=expected:raise ValueError('Original source changed during collection')
        if summary['status']=='complete':
            summary=finalize(root,specs,ledger)
        write_storage(root,root/'reports/storage.json',1)
        ledger.close()
        if summary['status']!='complete':raise RuntimeError('Campaign incomplete; inspect proposal/error reports before resuming')
        return summary


def finalize(root,specs,ledger):
    from .export import catalog
    value=catalog([root/'runs'])
    old=json.loads((ROOT/'outputs/hil194/augmentation-catalog-v1.json').read_text())
    old_by_fingerprint={e['fingerprint']:e['capture_id'] for e in old['entries']}
    kept=[];duplicates=list(value['duplicates'])
    for entry in value['entries']:
        previous=old_by_fingerprint.get(entry['fingerprint'])
        if previous:duplicates.append({'capture_id':entry['capture_id'],'duplicate_of':previous,'scope':'existing_catalog'})
        else:kept.append(entry)
    separate={'schema':1,'entries':kept,'duplicates':duplicates,'collection':'separate_campaign_only','source_catalog':str(ROOT/'outputs/hil194/augmentation-catalog-v1.json')}
    atomic_json(root/'accepted-catalog.json',separate)
    dedup={'new_proposal_ids':[e['candidate']['proposal_id'] for e in kept],'unique_new_accepted':len(kept),'duplicates':duplicates,
           'train':sum(e['candidate'].get('partition')=='train' for e in kept),'validation':sum(e['candidate'].get('partition')=='val' for e in kept)}
    summary,attempts=report(root,specs,ledger,dedup)
    atomic_json(root/'runs/progress.json',{'status':'collected','accepted':[e['candidate']['number'] for e in value['entries']],'target':len(value['entries'])})
    clustering_export(root,attempts)
    return summary
