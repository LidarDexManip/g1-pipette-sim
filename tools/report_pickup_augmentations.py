"""Audit accepted augmentation captures and summarize count/diversity/provenance."""
import argparse,json,hashlib,csv
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
 root=Path(root);progress=json.loads((root/'progress.json').read_text());rows=[];hashes=set()
 for file in sorted((root/'candidates').glob('*/accepted.json')):
  accepted=json.loads(file.read_text());c=accepted['candidate'];v=accepted['validation'];folder=file.parent
  first=json.loads((folder/'screen.json').read_text())['result'];second=json.loads((folder/'verification.json').read_text())['result']
  if not all(r['pickup'] and r['complete'] and r['seed_phase_timing_passed'] for r in [first,second]):raise ValueError(f'Invalid accepted capture {folder}')
  cfg=json.loads((folder/'config.json').read_text());commands=np.load(folder/'commands.npz')
  digest=hashlib.sha256(json.dumps({'rack':cfg['rack'],'table':cfg['table']},sort_keys=True).encode()+commands['motor_targets'].tobytes()).hexdigest()
  if digest!=c['content_hash'] or digest in hashes:raise ValueError('Duplicate or altered capture')
  hashes.add(digest)
  with np.load(folder/'trajectory.npz') as data:
   n=len(data['timestamp'])
   if n!=v['frame_count'] or data['action'].shape!=(n,29):raise ValueError('Bad captured action shape')
   if not np.isfinite(data['action']).all() or not np.isfinite(data['observation.state']).all():raise ValueError('Nonfinite training data')
   if np.max(abs(data['simulation_time']-data['timestamp']))>.000501:raise ValueError('Camera/state timestamps diverged')
   duration=float(data['simulation_time'][-1])
  rows.append({'candidate':c['number'],'parent_episode':c['parent_episode'],'method':c['kind'],'row':c['cell'][0],'col':c['cell'][1],'target_dx_mm':c['target_shift_m'][0]*1000,'target_dy_mm':c['target_shift_m'][1]*1000,'maximum_joint_delta_rad':c['noise']['max_joint_delta_rad'],'world_lift_mm':v['lift_m']*1000,'reference_lift_mm':v['lift_target_m']*1000,'frames':n,'duration_s':duration})
 out=root/'report';out.mkdir(exist_ok=True)
 if not rows:return
 with (out/'demonstrations.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 methods=Counter(r['method'] for r in rows);parents=Counter(r['parent_episode'] for r in rows)
 summary={'accepted':len(rows),'target':progress['target'],'collection_status':progress['status'],'by_method':dict(methods),'by_parent':dict(parents),'distinct_wells':len({(r['row'],r['col']) for r in rows}),'unique_geometry_action_hashes':len(hashes),'camera_streams':3*len(rows),'total_frames_per_camera':sum(r['frames'] for r in rows),'total_duration_s':sum(r['duration_s'] for r in rows),'screened':progress['screened'],'rejected':progress['rejected'],'execution_errors':progress['errors'],'audit':'All accepted records have two complete passing validations, unique scene/action hashes, finite 29-D actions/states, and synchronized sampled state/camera timestamps.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 grid=np.zeros((8,12),int)
 for r in rows:grid[r['row'],r['col']]+=1
 fig,axes=plt.subplots(1,2,figsize=(12,4));im=axes[0].imshow(grid,cmap='Blues',origin='lower');axes[0].set(xlabel='Rack column',ylabel='Rack row',title='Accepted demonstrations by occupied well');fig.colorbar(im,ax=axes[0])
 ids=sorted(parents);axes[1].bar([str(i) for i in ids],[parents[i] for i in ids]);axes[1].set(xlabel='Real parent episode',ylabel='Accepted variants',title='Parent coverage');fig.tight_layout();fig.savefig(out/'coverage.png',dpi=140);plt.close(fig)
 lines=['# Pipette augmentation collection','',f"**{len(rows)} / {progress['target']} successful additional simulator demonstrations collected.** Status: {progress['status']}.",'', '| Method | Accepted |','|---|---:|']
 for method,count in methods.items():lines.append(f'| {method} | {count} |')
 lines+=['',f"Coverage: {len(parents)} parent episodes, {summary['distinct_wells']} occupied rack wells, {len(hashes)} unique geometry/action combinations. {summary['camera_streams']} camera videos; {summary['total_duration_s']/3600:.2f} hours of task trajectories.",'',
 'Each capture passed independent full-physics replay and full three-camera video decoding. The audit checks the two passing reports, geometry/action hashes, finite observations/actions and frame synchronization. These are simulator successes, not independently collected real demonstrations or evidence of sim-to-real transfer.','',
 'The action-changing samples are small, correlated arm-target perturbations with matching tip/rack placement. They use 29 motor targets plus separately stored hand commands; high-level mimic-goal labels are not fabricated. The initial well-only variants retain the original actions.','',
 'Known limits: approximate camera/tool/hand calibration, fixed pelvis, geometric tip attachment, and strong correlation among variants. Some seed attachment timing has not been fully reviewed against real video. Keep real parents and simulated descendants in the same evaluation split.','',
 '[All accepted variants and lift heights](demonstrations.csv) · [Coverage plot](coverage.png) · [Collection instructions](../../../../docs/PICKUP_AUGMENTATION.md)','']
 (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',nargs='?',default='outputs/hil194/augmentation600');a=p.parse_args();report(a.root)
