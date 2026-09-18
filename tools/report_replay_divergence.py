"""Build a reviewable report and plots from controlled replay diagnostics."""
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.diagnose_replay_divergence import decompose
from g1_pipette_sim.pickup_layouts import ROOT


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--inputs',nargs='+',type=Path,default=[ROOT/'outputs/hil194/replay-divergence',ROOT/'outputs/hil194/replay-divergence-corrections'])
 p.add_argument('--output',type=Path,default=ROOT/'outputs/hil194/replay-divergence-report')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 results=[];errors=[]
 for folder in a.inputs:
  summary=json.loads((folder/'summary.json').read_text());errors.extend(summary['errors'])
  for result in summary['results']:results.append((folder,result))
 fields=['episode','mode','wrist_rmse_mm','arm_tracking_nozzle_rmse_mm','grasp_nozzle_rmse_mm','world_lift_mm','recorded_fk_lift_mm','height_and_dwell_match_only','tool_contact_screen_passed']
 with (a.output/'comparisons.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(r for _,r in sorted(results,key=lambda x:(x[1]['episode'],x[1]['mode'])))
 lines=['# Controlled replay divergence investigation','',f'{len(results)} full diagnostic replays; {len(errors)} execution errors. No diagnostic mode is automatically published as a successful pickup.','',
 'All comparisons freeze each episode’s current fitted layout. Those layouts were optimized for command replay and can compensate for tracking or grasp errors. A counterfactual failing on that layout does not prove its underlying model is worse.','',
 '## Interpretation','',
 '- `command`: unchanged recorded motor targets and current production contact grasp.',
 '- `state_targets`: measured body joints used as servo targets; the live robot still responds dynamically. This is not measured-position playback.',
 '- `measured_arm`: impose the 29 measured body joint positions and velocities each step. The hand and objects remain dynamic; unknown reaction forces are injected.',
 '- `rigid_tool`: command replay with a wrist weld, only to test grasp sensitivity.',
 '- `measured_arm_rigid_tool`: measured-arm diagnostic plus wrist weld. The weld is compliant, not an exact measured tool pose.',
 '- `compliant_tip`: command replay with rack-retention rotational weld scale reduced from 1 to 0.01. This is an uncalibrated mechanics hypothesis.',
 '- `effort_bias`: command replay with constant torque offsets from the existing regression on episodes 46, 80 and 120; 160 and 193 are held out. Targets and placement are unchanged. Offsets might include sensor bias rather than physical torque.','',
 '## Results','', '| Episode | Mode | Wrist RMSE (mm) | Tool/mount nozzle RMSE (mm) | Lift / FK target (mm) | Height + dwell | Tool contact screen |', '|---|---|---:|---:|---:|---|---|']
 for folder,r in sorted(results,key=lambda x:(x[1]['episode'],x[1]['mode'])):
  lines.append(f"| {r['episode']} | {r['mode']} | {r['wrist_rmse_mm']:.2f} | {r['grasp_nozzle_rmse_mm']:.2f} | {r['world_lift_mm']:.2f} / {r['recorded_fk_lift_mm']:.2f} | {r['height_and_dwell_match_only']} | {r['tool_contact_screen_passed']} |")
 lines+=['','The table’s height/dwell flag is only a diagnostic metric; it is not the full production validation or evidence of matched real attachment timing. Tool/mount error is relative to the assumed fixed mount, not measured 3D pipette ground truth.','',
 '## Error decomposition','',
 'Actual nozzle minus measured-joint FK nozzle equals (actual-arm FK minus measured-joint FK) plus (actual nozzle minus actual-arm FK). The first vector isolates arm tracking under the assumed mount; the second measures deviation from that mount. The norms can cancel and must not be added.','',
 '## Recorded evidence','',
 'Episode 46’s fixed-camera samples show the insertion approach and a lifted green tip in the final 21.3 s frame. Intermediate frames near the rack do not establish the exact first attachment time. No exact 3D tip lift or camera-to-robot calibration is inferred from these images.',
 '[Recorded insertion frames](../replay-divergence/episode046-recorded-insertion.jpg) · [Attachment timing review](../replay-divergence/episode046-timing-review.jpg)','',
 '## Scope and remaining calibration','',
 'No production controller, grasp, insertion gate, mesh, rack layout or successful default config was changed by these diagnostics. Exact camera registration and wrist-to-nozzle calibration remain unresolved; current camera fits and mounts are approximate. The existing FK lift target remains an estimate and should be replaced by tracked object motion only after camera calibration and visibility/occlusion checks.','']
 for folder,r in results:
  if r['mode']!='command':continue
  index=r['episode'];d=np.load(folder/f'episode-{index:03}'/'command'/'trace.npz');cfg=json.loads((folder/f'episode-{index:03}'/'command'/'config.json').read_text());settings=cfg['attachment']
  total,control,grasp=decompose(d['actual_nozzle'],d['arm_fk_nozzle'],d['measured_fk_nozzle']);t=d['time']
  fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
  for label,x in [('Total vs measured-joint FK',total),('Arm tracking contribution',control),('Tool deviation from mount',grasp)]:axes[0].plot(t,np.linalg.norm(x,axis=1)*1000,label=label)
  axes[0].set_ylabel('Nozzle error (mm)');axes[0].legend(fontsize=8)
  axes[1].plot(t,d['radial']*1000,label='Radial error (mm)');axes[1].plot(t,d['depth']*1000,label='Insertion depth (mm)');axes[1].axhspan(settings['min_insertion_depth']*1000,settings['max_insertion_depth']*1000,alpha=.15,color='green',label='Allowed depth');axes[1].axhline(settings['radial_tolerance']*1000,color='black',linestyle=':',label='Radial limit');axes[1].set_ylabel('Distance (mm)');axes[1].legend(fontsize=8)
  axes[2].plot(t,d['angle'],label='Nozzle vs tip angle');axes[2].plot(t,np.rad2deg(np.arccos(np.clip(d['measured_axis'][:,2],-1,1))),label='Measured-joint FK vs vertical');axes[2].axhline(settings['angle_tolerance_deg'],color='black',linestyle=':',label='Angle limit');axes[2].set_ylabel('Angle (degrees)');axes[2].set_xlabel('Recorded time (s)');axes[2].legend(fontsize=8)
  for ax in axes:
   ax.grid(alpha=.2)
   for event in r['events']:ax.axvline(event['time'],color='red',linestyle='--',alpha=.7)
  fig.suptitle(f'Episode {index}: unchanged command replay; red line = simulated attachment')
  fig.tight_layout();fig.savefig(a.output/f'episode-{index:03}.png',dpi=140);plt.close(fig)
  lines.append(f'[Episode {index} error/gate plot](episode-{index:03}.png)')
 experiment=ROOT/'outputs/hil194/bias-corrected-layouts/summary.json'
 if experiment.exists():
  trial_results=json.loads(experiment.read_text())
  lines+=['','## Placement refit with controller-bias hypothesis','',
          'These trials refit placement after changing the controller, using full collision geometry and the unchanged world-lift/contact criteria. They are experimental and are not automatically installed.','',
          '| Episode | Trials completed | Valid attached trials | Lift-matching candidates |', '|---|---:|---:|---:|']
  for index in sorted({r['episode'] for r in trial_results['results']}):
   rows=[r for r in trial_results['results'] if r['episode']==index]
   lines.append(f"| {index} | {len(rows)} | {sum(r['valid'] and r['attached'] for r in rows)} | {sum(r['pickup'] for r in rows)} |")
  lines+=['',f"Trial execution errors: {len(trial_results['errors'])}. See [full trial results](../bias-corrected-layouts/summary.json)."]
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(a.output/'REPORT.md')
if __name__=='__main__':main()
