"""Summarize completed frozen-config HIL194 validation and export review artifacts."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/hil194')
summary=json.loads((root/'final-replays/summary.json').read_text())
results=summary['results'];videos=json.loads((root/'video-integrity.json').read_text())
if len(results)!=194 or summary['failures']:raise SystemExit('Final replay validation is incomplete')
if videos['checked']!=videos['expected']:raise SystemExit('Full video validation is incomplete')
inputs={json.loads((root/'final-replays'/f"episode-{r['episode']:03}"/'report.json').read_text())['input_config_sha256'] for r in results}
if len(inputs)!=1:raise SystemExit('Replay inputs differ; do not aggregate mixed model configurations')
metrics={}
for key in ['right_arm_rmse_rad','wrist_position_rmse_mm','nozzle_position_rmse_vs_measured_joint_fk_mm','maximum_grasp_translation_drift_mm','maximum_grasp_rotation_drift_deg']:
    a=np.array([r[key] for r in results]);metrics[key]={'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
attached=[r['episode'] for r in results if any(e['state']=='attached' for e in r['events'])]
failed_scenes=[r['episode'] for r in results if r['scene_error']]
peaks=[]
for r in results:
    for pair,stats in r['contacts'].items():peaks.append({'episode':r['episode'],'pair':pair,**stats})
peaks.sort(key=lambda r:r['maximum_penetration_mm'],reverse=True)
action=json.loads((root/'final-action-replays/summary.json').read_text())
if len(action['results'])!=7 or action['failures']:raise SystemExit('Paired action comparison incomplete')
by_id={r['episode']:r for r in results}
comparison=[{'episode':r['episode'],'low_level_arm_rmse_deg':math.degrees(by_id[r['episode']]['right_arm_rmse_rad']),'action_arm_rmse_deg':math.degrees(r['right_arm_rmse_rad'])} for r in action['results']]
report={'episodes':len(results),'frames':245898,'input_config_sha256':next(iter(inputs)),'finite_replays':sum(r['finite'] for r in results),'replays_with_warnings':[r['episode'] for r in results if r['warnings']],
        'attachments':attached,'visual_scene_fits':194-len(failed_scenes),'unresolved_scene_episodes':failed_scenes,
        'video_files_fully_decoded':videos['checked'],'video_failures':[r for r in videos['results'] if not r['ok']],
        'metrics':metrics,'initial_finger_penetration_max_mm':max(r['initial_grasp'].get('after_max_penetration_mm',0) for r in results),'worst_tool_contacts':peaks[:10],'paired_action_comparison':comparison,
        'scope':'Recorded-command playback, not VLA inference or validated sim-to-real task reproduction. Unresolved layouts use a labeled session reference for motion diagnostics only.'}
(root/'validation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
fields=['episode','duration_s','scene_source','right_arm_rmse_rad','wrist_position_rmse_mm','nozzle_position_rmse_vs_measured_joint_fk_mm','maximum_grasp_translation_drift_mm','maximum_grasp_rotation_drift_deg','maximum_attached_lift_mm','state','scene_error']
with (root/'replay-metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in results)
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
ids=np.array([r['episode'] for r in results]);known=np.array([r['scene_error'] is None for r in results])
for ax,key,label,scale in zip(axes,['right_arm_rmse_rad','maximum_grasp_translation_drift_mm','maximum_grasp_rotation_drift_deg'],['Right-arm RMSE (degrees)','Maximum tool translation in hand (mm)','Maximum tool rotation in hand (degrees)'],[180/np.pi,1,1]):
    values=np.array([r[key] for r in results])*scale
    ax.scatter(ids[known],values[known],s=14,label='Visual scene estimate',color='#246a9d')
    ax.scatter(ids[~known],values[~known],s=20,marker='x',label='Unresolved scene; reference used',color='#b95b28')
    ax.axvline(45.5,color='gray',linestyle='--',linewidth=1);ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(loc='upper left',fontsize=9);axes[-1].set_xlabel('HIL194 episode index (new capture starts at 46)')
fig.suptitle('HIL194: full dynamic replay with unchanged recorded motor targets\nJoint tracking and grasp stability are separate validation requirements')
fig.tight_layout();fig.savefig(root/'replay-discrepancies.png',dpi=150)
print(json.dumps(report,indent=2))
