"""Export the installed per-episode pickup results as CSV, JSON and a plot."""
import csv,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.pickup_layouts import MANIFEST,ROOT,recorded_reviews

root=ROOT/'outputs/hil194/pickup-layouts';m=json.loads(MANIFEST.read_text());targets=json.loads((root/'lift-targets.json').read_text());reviews=recorded_reviews(m['dataset_snapshot']);rows=[]
for i in range(194):
    entry=m['episodes'].get(str(i),{});v=entry.get('validation',{});contacts=v.get('contacts',{})
    row={'episode':i,'status':entry.get('status','pending'),'recorded_lift_estimate_mm':targets[str(i)]['target_m']*1000,'simulated_lift_mm':v['lift_m']*1000 if 'lift_m' in v else None,'attachment_time_s':next((e['time'] for e in v.get('events',[]) if e['state']=='attached'),None),'trials':entry.get('trials',0),'config':entry.get('config'),'max_tool_penetration_mm':max((s['maximum_penetration_mm'] for s in contacts.values()),default=None),'max_contact_normal_force_N':max((s['maximum_normal_force_N'] for s in contacts.values()),default=None)}
    report_name='default-rejected-report.json' if entry.get('status')=='no_valid_layout' else 'default-report.json'
    report_path=ROOT/entry['report'] if entry.get('report') else root/f'episode-{i:03}'/report_name
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    config_path=ROOT/entry['config'] if entry.get('config') else None
    config=json.loads(config_path.read_text()) if config_path and config_path.exists() else {}
    delta=config.get('layout_search',{}).get('delta_xy_z_yaw')
    for axis,name in enumerate(['rack_delta_x_mm','rack_delta_y_mm','table_height_delta_mm','rack_delta_yaw_deg']):
        row[name]=delta[axis]*(1000 if axis<3 else 1) if delta is not None else None
    review=reviews.get(str(i),{})
    row['lift_frame']=v.get('lift_frame','rack' if v else None)
    row['rack_clearance_mm']=v.get('rack_clearance_m',v.get('lift_m',0))*1000 if v else None
    row['excluded_from_default_example']=review.get('exclude_from_default',False)
    row['recorded_review_note']=review.get('note')
    row['recorded_initial_tip_state']=review.get('initial_tip_state')
    row['recorded_final_tip_state']=review.get('final_tip_state')
    row['recorded_pickup_visible']=review.get('recorded_pickup_visible')
    row['external_object_intervention_observed']=review.get('external_object_intervention_observed')
    row['recording_group']='inherited_HG_DAgger' if i<46 else 'new_HIL_collection'
    row['hand_action_provenance']=report.get('hand_action_provenance')
    diagnostics=report.get('attachment_diagnostics',{})
    row['maximum_grasp_translation_drift_mm']=diagnostics.get('maximum_grip_displacement_mm')
    closest=diagnostics.get('closest_approach') or {}
    for key in ['time_s','radial_error_mm','insertion_depth_mm','angle_deg']:
        row['closest_approach_'+key]=closest.get(key)
    row['invalid_reason']=v.get('invalid_reason') or entry.get('reason')
    failures=diagnostics.get('gate_failures',{});samples=diagnostics.get('samples',0)
    for gate in ['radial','depth','angle']:
        row[gate+'_gate_failure_fraction']=failures.get(gate,0)/samples if samples else None
    row['lift_difference_mm']=row['simulated_lift_mm']-row['recorded_lift_estimate_mm'] if row['simulated_lift_mm'] is not None else None;rows.append(row)
with (root/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'complete':m.get('complete',False),'counts':dict(Counter(r['status'] for r in rows)),'verified_episodes':[r['episode'] for r in rows if r['status']=='verified_pickup'],'valid_attachments_with_at_least_2mm_max_lift':[r['episode'] for r in rows if r['status'] in ['verified_pickup','attachment_only','pickup_lift_mismatch'] and r['simulated_lift_mm'] is not None and r['simulated_lift_mm']>=2],'verified_with_attachment_before_1s':[r['episode'] for r in rows if r['status']=='verified_pickup' and r['attachment_time_s'] is not None and r['attachment_time_s']<1],'unresolved_episodes':[r['episode'] for r in rows if r['status']!='verified_pickup'],'matched_lift_excluded_from_default_example':[r['episode'] for r in rows if r['status']=='verified_pickup' and r['excluded_from_default_example']],'matched_lift_and_sampled_recorded_pickup':[r['episode'] for r in rows if r['status']=='verified_pickup' and r['recorded_pickup_visible'] and not r['excluded_from_default_example']],'recorded_video_reviews':reviews,'scope':'Unchanged-command, trajectory-conditioned simulation fitting. Recorded lift is an estimate from measured-joint FK, not object-pose ground truth. Final simulated lift uses world-frame tip rise from the first attached sample; rack clearance is separate. Attachment timing is not fitted.','criteria':m['criteria']}
(root/'results-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(12,5));ids=np.arange(194);ax.plot(ids,[r['recorded_lift_estimate_mm'] for r in rows],color='#888888',linewidth=1,label='Recorded-motion lift estimate')
reference=np.array([r['recorded_lift_estimate_mm'] for r in rows]);tolerance=np.maximum(2.,.1*reference)
ax.fill_between(ids,np.maximum(0,reference-tolerance),reference+tolerance,color='#888888',alpha=.15,label='Acceptance band')
for status,color in [('verified_pickup','#277a4d'),('attachment_only','#c48326'),('no_pickup','#b85454'),('pickup_lift_mismatch','#c48326')]:
    selected=[r for r in rows if r['status']==status]
    if selected:ax.scatter([r['episode'] for r in selected],[r['simulated_lift_mm'] for r in selected],color=color,s=18,label={'verified_pickup':'Matched recorded lift','attachment_only':'Different or insufficient lift','pickup_lift_mismatch':'Different lift','no_pickup':'No pickup'}[status])
ax.set(xlabel='Episode',ylabel='Maximum lift (mm)',title='Full-model validation of fitted replay placements');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'lift-results.png',dpi=150)
print(json.dumps(summary,indent=2))
