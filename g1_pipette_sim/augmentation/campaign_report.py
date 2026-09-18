"""Source/family attribution and portable failure features; no clustering claims."""
import csv,json
from pathlib import Path
import numpy as np
from .contracts import atomic_json
from .failure_data import atomic_npz


def write_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def failure_reason(result):
    if result.get('pickup'):return 'success'
    if result.get('initial_overlaps'):return 'initial_overlap'
    if result.get('invalid_reason'):return result['invalid_reason']
    if not result.get('valid',False):return 'invalid_simulation'
    if not result.get('attached',False):return 'no_attachment'
    if not result.get('seed_phase_timing_passed',True):return 'attachment_timing'
    if result.get('lift_m',0)>result.get('lift_target_m',0)+result.get('lift_tolerance_m',0):return 'excessive_lift'
    if result.get('lift_dwell_s',0)<.1:return 'insufficient_lift_or_dwell'
    if not result.get('complete'):return 'incomplete_rollout'
    return 'final_attachment_or_lift_failure'


def report(root, specs, ledger, dedup=None):
    root=Path(root);rows=ledger.rows();states={r['id']:r for r in rows};groups={};proposals=[];attempts=[]
    for spec in specs:
        record=states[spec['id']];result=json.loads(record['result']) if record['result'] else {}
        key=(spec['parent_episode'],spec['family']);group=groups.setdefault(key,{'original_episode':key[0],'family':key[1],'partition':spec['partition'],
            'proposed':0,'skipped':0,'generation_rejected':0,'screened_proposals':0,'physics_attempts':0,'first_screen_passes':0,
            'refined_screen_passes':0,'recording_failures':0,'accepted':0,'errors':0,'worker_seconds':0.,'new_unique_accepted':None})
        group['proposed']+=1;group['skipped']+=record['state']=='skipped';group['generation_rejected']+=bool(result.get('generation_rejected'))
        group['errors']+=record['state']=='error';group['accepted']+=record['state']=='accepted';group['worker_seconds']+=result.get('worker_seconds',0.)
        trial_rows=[]
        for revision in range(2):
            for attempt in range(3):
                number=spec['number']*100+revision*10+attempt;folder=root/'runs/candidates'/f'{number:06}'
                path=folder/'screen.json'
                if not path.exists():continue
                screen=json.loads(path.read_text());r=screen['result'];c=json.loads((folder/'candidate.json').read_text())
                row={'proposal_id':spec['id'],'original_episode':spec['parent_episode'],'partition':spec['partition'],'family':spec['family'],
                     'candidate':number,'placement_attempt':attempt,'revision':revision,'pickup':r['pickup'],
                     'independently_recorded':(folder/'accepted.json').exists(),'reason':failure_reason(r),
                     'amplitude_deg':spec['recipe'].get('amplitude_deg',0),'draw':spec['draw'],'folder':str(folder),
                     'source_video':spec['source_video'],'seed':spec['seed'],'placement_adjusted':bool(np.linalg.norm(c['lineage'][-1]['offset_m'])>0),
                     'lift_m':r.get('lift_m'),'lift_target_m':r.get('lift_target_m'),'lift_dwell_s':r.get('lift_dwell_s')}
                trial_rows.append(row);attempts.append(row)
        group['physics_attempts']+=len(trial_rows);group['screened_proposals']+=bool(trial_rows)
        first=next((r for r in trial_rows if r['placement_attempt']==0 and r['revision']==0),None)
        group['first_screen_passes']+=bool(first and first['pickup'])
        group['refined_screen_passes']+=bool(any(r['pickup'] for r in trial_rows) and not (first and first['pickup']))
        group['recording_failures']+=sum('recorded' in r and not r['recorded'] for r in result.get('trials',[]))
        if dedup is not None:
            if group['new_unique_accepted'] is None:group['new_unique_accepted']=0
            group['new_unique_accepted']+=spec['id'] in dedup['new_proposal_ids']
        proposals.append({'proposal_id':spec['id'],'original_episode':spec['parent_episode'],'partition':spec['partition'],'family':spec['family'],
                          'draw':spec['draw'],'amplitude_deg':spec['recipe'].get('amplitude_deg',0),'state':record['state'],
                          'infrastructure_attempts':record['attempts'],'reason':result.get('reason',result.get('error','')),
                          'worker_seconds':result.get('worker_seconds',0.),'candidate':result.get('candidate')})
    for group in groups.values():
        eligible=group['proposed']-group['skipped'];group['eligible']=eligible
        group['yield_over_scheduled']=group['accepted']/group['proposed']
        group['yield_over_eligible']=group['accepted']/eligible if eligible else None
    totals={k:sum(g[k] for g in groups.values()) for k in ['proposed','eligible','skipped','generation_rejected','screened_proposals','physics_attempts','accepted','errors','worker_seconds']}
    status='complete' if all(r['state'] in ['accepted','rejected','skipped'] for r in rows) else 'running'
    if any(r['state']=='error' for r in rows):status='incomplete'
    value={'status':status,'totals':totals,'by_original_and_family':list(groups.values()),'deduplication':dedup}
    value['by_family']=[{'family':family,**{k:sum(g[k] for g in groups.values() if g['family']==family) for k in totals}} for family in sorted({g['family'] for g in groups.values()})]
    value['by_original']=[{'original_episode':parent,**{k:sum(g[k] for g in groups.values() if g['original_episode']==parent) for k in totals}} for parent in sorted({g['original_episode'] for g in groups.values()})]
    atomic_json(root/'reports/summary.json',value)
    for name,items in [('by-original-and-family',list(groups.values())),('by-family',value['by_family']),('by-original',value['by_original']),('proposals',proposals),('attempts',attempts)]:
        if items:write_csv(root/'reports'/f'{name}.csv',items,list(items[0]))
    (root/'reports/attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in attempts))
    lines=['# Augmentation campaign results','',f"Status: {status}. Counts below are new campaign attempts, separate from augmentation600.",'',
           '| Original | Partition | Family | Proposed | Skipped | Physics attempts | Accepted | New unique |',
           '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for g in sorted(groups.values(),key=lambda g:(g['original_episode'],g['family'])):
        lines.append(f"| {g['original_episode']} | {g['partition']} | {g['family']} | {g['proposed']} | {g['skipped']} | {g['physics_attempts']} | {g['accepted']} | {g['new_unique_accepted'] if g['new_unique_accepted'] is not None else 'pending audit'} |")
    lines+=['',f"Accepted {totals['accepted']} / {totals['eligible']} eligible proposals; {totals['skipped']} of {totals['proposed']} scheduled proposals skipped.",
            'Placement retries do not increase the proposal count. Accepted means independent recording passed. Failed attempts are not training episodes.']
    (root/'reports/summary.md').write_text('\n'.join(lines)+'\n')
    return value,attempts


def clustering_export(root, attempts):
    """64 phase-aligned samples, validity masks, and separate outcome metadata."""
    root=Path(root);features=[];masks=[];records=[];phase=np.linspace(-3.,2.,64)
    names=['tip_local_x_mm','tip_local_y_mm','tip_local_z_mm','angle_deg','running_max_force_N','running_max_penetration_mm']
    for row in attempts:
        folder=Path(row['folder']);path=folder/'screen-state.npz'
        if not path.exists():continue
        candidate=json.loads((folder/'candidate.json').read_text())
        with np.load(path) as data:
            times=data['time_s'];valid=(phase+candidate['seed_attachment_s']>=times[0])&(phase+candidate['seed_attachment_s']<=times[-1])
            signals=np.column_stack([data['nozzle_relative_tip_m']*1000,data['angle_deg'],data['maximum_contact_force_N'],data['maximum_penetration_mm']])
            x=np.stack([np.interp(phase+candidate['seed_attachment_s'],times,signals[:,j]) for j in range(len(names))],axis=1)
            finite=np.isfinite(x)&valid[:,None];x[~finite]=np.nan
            features.append(x.astype(np.float32));masks.append(finite);records.append({**row,'feature_row':len(features)-1})
    if features:
        atomic_npz(root/'reports/clustering-features.npz',features=np.stack(features),valid=np.stack(masks),phase_seconds=phase)
    (root/'reports/clustering-index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    atomic_json(root/'reports/clustering-schema.json',{'version':1,'feature_channels':names,'shape':['attempt',64,len(names)],
        'time_reference':'baseline attachment; interval -3 to +2 seconds','missing_values':'NaN with explicit boolean valid mask; never treat missing values as zero',
        'labels':'Outcomes, source IDs, family, and partition are in clustering-index.jsonl, not the numeric feature tensor.',
        'comparisons':'Includes screened successes and failures. Generation/preflight/infrastructure failures are separate in proposals.csv.',
        'future_use':'Cluster training-partition failures only; inspect representative saved states/reports and source-video segments before requesting targeted new demonstrations.',
        'limitation':'These features summarize simulator behavior and do not establish a real-world demonstration deficit.'})
