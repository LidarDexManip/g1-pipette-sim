# HIL194 pickup placement results

Manifest complete: **True**. Dataset: `/home/ytung34/.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hil194/snapshots/643ece1b2e93f15022c83832302848c9978d5f05`.

| Independent full-model result | Episodes |
|---|---:|
| attachment_only | 87 |
| verified_pickup | 16 |
| no_pickup | 91 |

Simulated lift is maximum world-frame tip-mouth rise from the first attached sample; rack clearance is reported separately in the CSV. `verified_pickup` means this simulated attached-tip lift matches the recorded-joint FK estimate within ±2 mm or ±10%, whichever is larger. It does not establish matched attachment timing or a faithful real-world replay. `attachment_only` includes pickups whose height or dwell fails that criterion. `no_valid_layout` means no publishable collision-screened layout was found within this search budget, not that no possible layout exists.

## Lift matches

| Episode | Simulated lift (mm) | Recorded estimate (mm) | Attachment (s) | Recorded-video finding |
|---|---:|---:|---:|---|
| 2 | 17.07 | 18.37 | 0.188 | Not manually reviewed; attachment timing is unverified. |
| 5 | 20.75 | 19.18 | 0.159 | The real pipette is empty and the tip remains in the rack at 0.2 and 1 s, but simulation attaches at 0.159 s. Later recorded frames show a pickup. Lift height matches, but attachment timing does not. |
| 6 | 22.19 | 21.00 | 17.714 | Not manually reviewed; attachment timing is unverified. |
| 7 | 20.10 | 21.92 | 0.174 | The nozzle begins near/in the racked tip, but the real pipette is visibly empty and separate from the tip at 4–6 s. Later frames show insertion and a lifted green tip. Simulated attachment at 0.1745 s is too early, despite the matching lift height. |
| 13 | 26.79 | 24.57 | 15.690 | Not manually reviewed; attachment timing is unverified. |
| 14 | 34.09 | 31.28 | 12.838 | Sampled frames show an initially empty pipette, insertion at 12.8–14.3 s, and a later lifted green tip. Exact real attachment time is not measured. |
| 24 | 29.34 | 31.77 | 15.046 | Not manually reviewed; attachment timing is unverified. |
| 30 | 27.91 | 29.63 | 17.620 | Not manually reviewed; attachment timing is unverified. |
| 35 | 22.96 | 25.44 | 6.098 | Not manually reviewed; attachment timing is unverified. |
| 40 | 33.01 | 33.36 | 25.224 | Not manually reviewed; attachment timing is unverified. |
| 41 | 27.97 | 28.69 | 17.084 | Not manually reviewed; attachment timing is unverified. |
| 45 | 26.32 | 28.68 | 2.811 | Not manually reviewed; attachment timing is unverified. |
| 64 | 39.33 | 40.08 | 8.834 | Not manually reviewed; attachment timing is unverified. |
| 68 | 30.04 | 32.74 | 17.938 | Not manually reviewed; attachment timing is unverified. |
| 121 | 38.02 | 42.08 | 0.174 | Not manually reviewed; attachment timing is unverified. |
| 144 | 34.48 | 36.42 | 14.128 | Not manually reviewed; attachment timing is unverified. |

## Run and inspect

```bash
./run.sh                       # automatically selected verified example
./run.sh --episode 14
~/miniconda3/envs/lerobot-act/bin/python tools/pickup_status.py
```

The default dynamic, low-level, recorded-clock replay loads the matching episode config. Unsuccessful diagnostic layouts print a warning; episodes without a valid layout require an explicit diagnostic choice such as `--scene-layout visual`. Explicit configs and different action sources do not inherit this validation.

[All episode heights and discrepancies](../outputs/hil194/pickup-layouts/results.csv) · [Machine-readable summary](../outputs/hil194/pickup-layouts/results-summary.json) · [Lift plot](../outputs/hil194/pickup-layouts/lift-results.png) · [Search settings and validation](PICKUP_LAYOUTS.md) · [Failure diagnostics](PICKUP_DISCREPANCIES.md)

## What remains unresolved

The search changes rack/tip XY placement, rack yaw, and common table/object height. It preserves robot commands, hand commands, tool mounting, collision geometry, contact parameters and the insertion gate. Initial candidate screening used a 16-sector tip shell; the later world-lift retry and published validation use the normal 48-sector shell. The retry also permits changing the active physical rack well. The first pass optimized rack clearance as a proposal metric; final acceptance uses world-coordinate lift, with at least 2 mm clearance during its 100 ms dwell. The bounded first pass allows up to 24 trials and can defer distant approaches after 12; additional targeted retries and exact-model refinements are recorded per episode.

These placements are conditioned on the replay trajectory. They can compensate for tracking errors and are not recovered real object poses. The lift target assumes a rigid wrist-to-tool transform and a fixed pelvis; actual pipette and tip poses were not measured. The reference is the largest subsequent upward displacement, not a manually annotated pickup interval.

The remaining discrepancies include pipette translation/rotation within the approximate grasp, uncertain hand register mapping and wrist-to-tool transform, camera/object registration, and contact/retention mechanics. The original baseline had a median 9.43 mm wrist-position RMSE and a median maximum 23.61 mm pipette translation relative to the hand. These baseline values are separate from the fitted-layout results.

Recorded-video checks are sampled, not exhaustive annotations. Episode 104 includes a human tip removal absent from robot commands. Episodes 5, 7, 8 and 81 have documented early simulated attachment. Review notes exclude these from default example selection; matching a height alone does not resolve those discrepancies.

For VLA fine-tuning, keep real and simulated variants of each source episode in the same data split. Preserve the high-level `action` versus motor `action.low_level` distinction and the unusual camera names. These replay artifacts are not a validated synthetic LeRobot training dataset. See [the dataset audit and training contract](HIL194.md).
