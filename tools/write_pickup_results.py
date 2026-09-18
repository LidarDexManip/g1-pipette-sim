"""Write a readable report from the final installed pickup validation results."""
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from g1_pipette_sim.pickup_layouts import recommended_episode
folder=ROOT/'outputs/hil194/pickup-layouts'
summary=json.loads((folder/'results-summary.json').read_text())
manifest=json.loads((ROOT/'config/hil194-pickup-layouts.json').read_text())
counts=summary['counts']
default_episode=recommended_episode(manifest['dataset_snapshot'])
lines=['# HIL194 pickup placement results','',
       f"Manifest complete: **{summary['complete']}**. Dataset: `{manifest['dataset_snapshot']}`.",'',
       '| Independent full-model result | Episodes |','|---|---:|']
for status,count in counts.items():lines.append(f'| {status} | {count} |')
lines.extend(['','Simulated lift is maximum world-frame tip-mouth rise from the first attached sample; rack clearance is reported separately in the CSV. `verified_pickup` means this simulated attached-tip lift matches the recorded-joint FK estimate within ±2 mm or ±10%, whichever is larger. It does not establish matched attachment timing or a faithful real-world replay. `attachment_only` includes pickups whose height or dwell fails that criterion. `no_valid_layout` means no publishable collision-screened layout was found within this search budget, not that no possible layout exists.','',
              '## Lift matches','', '| Episode | Simulated lift (mm) | Recorded estimate (mm) | Attachment (s) | Recorded-video finding |','|---|---:|---:|---:|---|'])
reviews=summary['recorded_video_reviews']
for i in summary['verified_episodes']:
    e=manifest['episodes'][str(i)];v=e['validation'];t=next(ev['time'] for ev in v['events'] if ev['state']=='attached');note=reviews.get(str(i),{}).get('note','Not manually reviewed; attachment timing is unverified.').replace('|','/')
    lines.append(f"| {i} | {v['lift_m']*1000:.2f} | {v['lift_target_m']*1000:.2f} | {t:.3f} | {note} |")
lines.extend(['','## Run and inspect','', '```bash','./run.sh                       # automatically selected verified example',f'./run.sh --episode {default_episode}','~/miniconda3/envs/lerobot-act/bin/python tools/pickup_status.py','```','',
 'The default dynamic, low-level, recorded-clock replay loads the matching episode config. Unsuccessful diagnostic layouts print a warning; episodes without a valid layout require an explicit diagnostic choice such as `--scene-layout visual`. Explicit configs and different action sources do not inherit this validation.','',
 '[All episode heights and discrepancies](../outputs/hil194/pickup-layouts/results.csv) · [Machine-readable summary](../outputs/hil194/pickup-layouts/results-summary.json) · [Lift plot](../outputs/hil194/pickup-layouts/lift-results.png) · [Search settings and validation](PICKUP_LAYOUTS.md) · [Failure diagnostics](PICKUP_DISCREPANCIES.md)','',
 '## What remains unresolved','',
 'The search changes rack/tip XY placement, rack yaw, and common table/object height. It preserves robot commands, hand commands, tool mounting, collision geometry, contact parameters and the insertion gate. Initial candidate screening used a 16-sector tip shell; the later world-lift retry and published validation use the normal 48-sector shell. The retry also permits changing the active physical rack well. The first pass optimized rack clearance as a proposal metric; final acceptance uses world-coordinate lift, with at least 2 mm clearance during its 100 ms dwell. The bounded first pass allows up to 24 trials and can defer distant approaches after 12; additional targeted retries and exact-model refinements are recorded per episode.','',
 'These placements are conditioned on the replay trajectory. They can compensate for tracking errors and are not recovered real object poses. The lift target assumes a rigid wrist-to-tool transform and a fixed pelvis; actual pipette and tip poses were not measured. The reference is the largest subsequent upward displacement, not a manually annotated pickup interval.','',
 'The remaining discrepancies include pipette translation/rotation within the approximate grasp, uncertain hand register mapping and wrist-to-tool transform, camera/object registration, and contact/retention mechanics. The original baseline had a median 9.43 mm wrist-position RMSE and a median maximum 23.61 mm pipette translation relative to the hand. These baseline values are separate from the fitted-layout results.','',
 'Recorded-video checks are sampled, not exhaustive annotations. Episode 104 includes a human tip removal absent from robot commands. Episodes 5, 7, 8 and 81 have documented early simulated attachment. Review notes exclude these from default example selection; matching a height alone does not resolve those discrepancies.','',
 'For VLA fine-tuning, keep real and simulated variants of each source episode in the same data split. Preserve the high-level `action` versus motor `action.low_level` distinction and the unusual camera names. These replay artifacts are not a validated synthetic LeRobot training dataset. See [the dataset audit and training contract](HIL194.md).',''])
(ROOT/'docs/PICKUP_RESULTS.md').write_text('\n'.join(lines))
print(ROOT/'docs/PICKUP_RESULTS.md')
