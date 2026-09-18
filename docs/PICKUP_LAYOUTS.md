# Per-episode pickup layouts

For default dynamic HIL194 replay, `run.sh` automatically selects a published, episode-specific config. With no `--episode`, it chooses a verified example and prefers attachment during recorded movement and the newer recording session (selected from the current world-lift manifest). Explicit episode selections are always honored:

```bash
./run.sh --episode 14
```

Startup reports the simulated lift and recorded-joint lift estimate for the selected config. During fitting, verified configs are published as they finish. Episodes still pending use the original visual initializer. See [the completed results](PICKUP_RESULTS.md) and list the available configs with:

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/pickup_status.py
```

To inspect the original image-based placement, use `--scene-layout visual`. An explicit `--config` takes precedence over automatic pickup selection. Automatic selection applies only to dynamic, supported-base replay using `low_level` and the recorded clock. Other action sources, kinematic playback, demos and free-base runs do not inherit the pickup validation.

## Record video

```bash
# Selected simulator camera, saved at 30 fps:
./run.sh --episode 14 --headless --video --output outputs/episode14-video

# All three recorded/simulated camera pairs:
./run.sh --episode 14 --headless --compare --output outputs/episode14-comparison
```

Each command writes `simulation.mp4` inside its output directory. Remove `--headless` to keep the interactive viewer open. Add `--duration 10` to limit the export to ten simulation seconds. Interactive `--video` follows the viewer camera, including mouse zoom, pan and rotation. Press `F` for the free overview if a fixed sensor camera is selected. With `--headless`, the export uses the selected `--camera` view (default `overview`). `--compare` always exports the fixed real/sim camera pairs.

## What is fitted

The search translates the rack and its active tip together. The lid stays connected to the rack. The XY mode translates within 120 mm of the baseline. The height mode also permits common table/object height changes up to 20 mm and rack yaw changes up to 10 degrees. The current all-episode pass uses height mode; each trial records its mode. The original pass held the active well fixed. The later full-model retry may choose a different physical well and translate the rack to keep the proposed tip location. Tool mounting, hand mapping, gravity, friction, servo settings, robot commands and insertion thresholds remain unchanged in the published layouts.

The original candidate screening used a coarser 16-sector tip shell to reduce computation. The later world-lift retry used the full 48-sector shell for every trial. Every published pickup is independently replayed with the unchanged 48-sector shell used by `run.sh`. Candidate simulations run in fresh processes to bound native model memory.

These are **trajectory-conditioned simulation placements**, not recovered real object poses. They can compensate for robot/tool tracking errors. They are not held-out policy successes or validated sim-to-real demonstrations.

## Pickup validation

A result must satisfy the existing geometric attachment gate and then raise the attached tip mouth by the episode-specific recorded-motion target for 100 ms. Final validation measures **world-coordinate rise from the first attached sample**. The tip mouth must also clear the current rack-mouth site by at least 2 mm during that dwell, so moving the rack with the tip does not count. It must remain attached at the end of a separate full replay, with finite states and no numerical warnings. Rack clearance is reported separately.

Earlier candidate searches optimized height above the rack mouth. That is not identical to world-coordinate rise: the tip starts slightly below that site, and the rack can move. Those historical results are retained as proposal diagnostics; final default selection uses the corrected world-lift validation.

The target is the largest later upward displacement of the nozzle computed from measured joints and the assumed rigid tool mount, smoothed over 100 ms and excluding the first 0.5 s. This is an **FK estimate**, not measured pipette motion. The acceptance band is ±2 mm or ±10% of that estimate, whichever is larger; both target and simulated height are reported. The first exploratory pass used a blanket 20 mm target; it is archived separately and does not establish a recorded-lift match.

Generate these targets from the audited reference traces with `python tools/measure_recorded_lift.py`.

The additional contact screen rejects:

- Initial rack overlap with robot/tool/lid deeper than 0.1 mm.
- Rack/lid penetration deeper than 0.1 mm during replay.
- Pipette/rack, pipette/tip or pipette/lid penetration deeper than 0.5 mm.
- Reported normal force above 100 N in those tool/object contacts.

These conservative screening limits reject obvious artifacts; they do not calibrate real insertion forces or prove physical fidelity. The simulated grasp and tip attachment mechanism retain their documented limitations.

Some episodes can attach without matching the recorded-motion lift. These are labeled `attachment_only`, not `verified_pickup`. A config that fails independent full validation is not published as a successful pickup. Search failure is not a proof that no layout exists.

## Recorded-video checks

Sampled video review confirms real pickup sequences in episodes 14, 46 and 52; episode 14 also passes the corrected world-lift criterion; episode 52 exceeds its recorded-lift tolerance. Episodes 5 and 7 match the lift metric but attach too early in simulation (0.159 and 0.175 s respectively; the real pipettes are still empty at 1 s and 4–6 s). Episodes 8 and 81 also have documented early attachment. Episode 104 starts with an attached tip that a person removes around 2 s; robot commands cannot reproduce this intervention. Its small FK rise is not evidence of a real pickup. Episodes with these documented discrepancies are excluded from automatic example selection. Startup displays review notes from `config/hil194-recorded-reviews.json`. This is sampled review, not exhaustive annotation of all videos.

## Artifacts and reproducibility

`config/hil194-pickup-layouts.json` is the atomic lookup manifest. It records the dataset snapshot, model fingerprints, config hashes, actual model mesh checksums and validation status. The simulator rejects stale or mismatched validation rather than silently claiming it still applies.

Each `outputs/hil194/pickup-layouts/episode-NNN/` contains the baseline, trial history, best candidate, and validation reports. The manifest points to the config to run (`validated-world-default.json` for the current world-lift validation). `best-*` files are proposal diagnostics and may reflect a provisional candidate; use the manifest config and its independent validation report for the reported outcome. When present, `world-default-approach-trace.npy` contains the full-model approach samples before attachment. Search state is saved after every trial; only a partially running trial is lost on interruption.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
~/miniconda3/envs/lerobot-act/bin/python tools/fit_pickup_layouts.py \
  --workers 10 --max-evaluations 24 --phase xy

# Additional search for episodes not yet verified:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
~/miniconda3/envs/lerobot-act/bin/python tools/fit_pickup_layouts.py \
  --workers 10 --max-evaluations 24 --phase height

# Publish newly verified results while the search is running:
~/miniconda3/envs/lerobot-act/bin/python tools/watch_world_pickup_layouts.py --once

# After every episode has a search result, validate and install the full manifest:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
~/miniconda3/envs/lerobot-act/bin/python tools/publish_world_pickup_layouts.py \
  --workers 10 --install
```

`--screening-evaluations 12` defers further refinement after 12 trials when the best insertion score remains above 3. The current pass uses this bounded screening option; promising cases retain the 24-trial budget. This is not an exhaustive feasibility test.

`--max-evaluations` is an additional per-episode trial budget for that invocation. `--total-budget` counts trials already completed in the selected phase against that budget; `--prioritize` checks short episodes with suitable baseline lift first. Already verified pickups are skipped. A full manifest may include valid but unsuccessful diagnostic layouts; startup prints their status explicitly. Episodes with no collision-screened layout require an explicit diagnostic choice (`--scene-layout visual`).

For controlled comparisons of arm tracking, grasp and insertion mechanics, see [replay divergence diagnostics](REPLAY_DIVERGENCE.md).
