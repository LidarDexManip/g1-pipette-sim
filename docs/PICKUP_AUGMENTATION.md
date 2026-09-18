# Validated motor-target augmentation collection

The collection target is **600 additional successful demonstrations**, including the two pilot captures. These are simulator-validated motor-target demonstrations, kept separate from real high-level action labels. No dataset is uploaded.

## Completed collection

All **600** demonstrations are collected and verified: **64** well-only placements, **534** smooth action-and-placement variants, and **2** fixed-rack alternate-well variants. They cover 94 wells and contain 425,815 frames per camera (1,800 source camera videos). Each capture passed two full simulation replays; the final audit checked every numeric frame in both LeRobot formats and every episode’s camera boundaries and future-action padding.

Use `outputs/hil194/augmentation600/lerobot-v3/` with the installed LeRobot v3 loader, or `lerobot-v21/` with a compatible v2.1 reader. The [final report](../outputs/hil194/augmentation600/report/REPORT.md) and [demonstration table](../outputs/hil194/augmentation600/report/demonstrations.csv) list parent episodes, placements, simulated lift heights, and reference lift heights. `completion.json` and `dataset-audit.json` record the final verification.

## Progress and files

```bash
cd ~/Projects/g1_pipette_sim
~/miniconda3/envs/lerobot-act/bin/python tools/augmentation_status.py
```

Collection: `outputs/hil194/augmentation600/`.

- `progress.json` and `collector.log`: collection progress and new successes, including lift heights.
- `candidates/NNNNNN/accepted.json`: only independently replayed, video-audited successes receive this file.
- Each accepted folder includes three MP4s, the exact scene config, dense augmented commands, sampled states, full model states, camera poses, and two physics reports.
- `lerobot-v21/`: incrementally refreshed LeRobot v2.1 dataset.
- `lerobot-v3/`: produced locally when collection completes, then checked with the native LeRobot loader. Every numeric action/state/hand row is compared against its validated capture in both v2.1 and v3.0. Native camera pixels and action-chunk padding are checked at every episode boundary.
- `export-progress.json`: packaged count and final loader verification.

A screened proposal is not counted until independent replay and all three complete video decodes pass. Candidate numbers differ from exported dataset episode numbers; `meta/augmentation.jsonl` maps them.

## Augmentation methods

1. **Well-only:** choose another well and translate the connected rack/lid so the tip stays at its original world location. Robot actions are unchanged. The first eight rounds cover nearby alternate wells for each seed.
2. **Smooth actions plus placement:** sample an XY nozzle offset in a mixture of ±0.5 mm and ±1.5 mm ranges. Translate the rack so the selected physical well is at the corresponding tip location. Map the offset through the arm Jacobian with orientation constraints, ramp it in over two seconds, and add a small smooth null-space perturbation. Hand commands and the initial body state remain unchanged. These are small local motion variations; they are not broad demonstrations of arbitrary reaching.
3. **Fixed-rack retargeting:** every fifth later round tries an adjacent well, approximately 10.5 mm away, while keeping the rack fixed. Apply the corresponding smooth arm-target correction. Rejection is expected when the altered approach collides or no longer seats the tip.

The command trajectories are fully generated before simulation. No object is teleported during replay, and no observed-state forcing or wrist-weld diagnostic is used. The tip remains in a physical rack well. Fixed seeds, saved commands, content hashes and model fingerprints preserve provenance.

Seed episodes: 14, 64, 68, 144, 6, 13, 24, 30, 35, 40, 41 and 45. The early-attachment seeds 2, 5, 7 and 121 are excluded. Not all remaining seeds have been fully aligned to real attachment timing; synthetic success does not establish faithful real timing.

## Acceptance

The normal full-resolution collision model and existing gravity, grasp, contact and insertion settings are preserved. Each candidate must pass the existing initial-overlap, penetration, force, finite-state and world-lift checks. Lift must match the parent's FK reference within its existing ±2 mm/±10% tolerance, satisfy dwell and clearance, and remain attached at the end. Attachment must occur after the first second and within 0.75 s of the seed's validated simulated attachment event.

The same candidate is then independently replayed while capturing all cameras. Every saved video is decoded completely and checked against the trajectory frame count. This is deterministic repeat validation, not evidence of robustness to unmodeled dynamics. Near-neighbor robustness and acceptance distributions should be considered when selecting training samples.

## Action and observation contract

- `action`: **29 motor-joint targets in radians**, not the real dataset's 35-dimensional high-level mimic goals.
- `action.hand_left`, `action.hand_right`: six raw hand-register commands each. Concatenating all three action fields gives 41 outputs; selecting only arms/hands requires an explicit matching data/deployment configuration.
- `observation.state`: 29 actual simulated body joint positions in the declared joint order.
- `observation.hand_joint_positions`: 12 actual simulated actuated finger angles, in radians. These are not equivalent to the frozen hand-register feedback in the real dataset.
- `rgb`: fixed ego; `wrist_right`: fixed side; `wrist_left`: right wrist.
- Export rate: **30 Hz**. Ego images are 512×288; side and wrist images are 384×384. Aspect ratios are preserved.
- `diagnostics.*`: privileged task/physics fields, excluded from normal image-and-joint policy inputs.

The exported actions are sampled from the dense replay command stream. Replaying them with a 30 Hz zero-order hold is not guaranteed to reproduce the original dense, interpolated commands; use each capture's `commands.npz` for exact replay. The training contract records this distinction.

Use the real dataset's `action.low_level` and hand-command fields when constructing a matching real-data training set. Do not combine these motor labels with the existing high-level action column. Match sampling, action horizon, state selection, units and normalization. A GROOT checkpoint with a high-level action head needs a matching motor-target data configuration and output adaptation before training/deployment.

The metadata `train` range is a storage index, not a proposed evaluation split. Group every real source episode and all of its simulated descendants together, using `split_group`/`parent_episode`. Synthetic variants are correlated and do not count as independent task coverage.

## Replay a successful capture

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/replay_pickup_augmentation.py \
  --candidate 0
```

This uses the saved augmented motor commands. Running the original parent episode with only the new scene config would omit action noise. The default overview allows mouse navigation. Add `--headless` for a physics-only replay.

## Collection and packaging commands

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
~/miniconda3/envs/lerobot-act/bin/python tools/collect_pickup_augmentations.py \
  --target 600 --workers 6 --render-workers 4

~/miniconda3/envs/lerobot-act/bin/python tools/watch_augmentation_export.py
```

Do not start another collector while the existing one holds its lock. Completed captures are recovered on restart; coordinator recovery was tested by a clean worker-pool shutdown. A changed model, seed manifest or collector version requires explicit provenance handling or a new output directory.

The pipeline is simulation-only: approximate camera registration, hand/tool geometry, fixed pelvis and geometric tip retention remain limitations for real deployment.

For the review of the additional Cartesian, direct-joint, and elbow-pulse experiments and the proposed path beyond 600 captures, see [Augmentation integration plan](AUGMENTATION_INTEGRATION_PLAN.md).

The integrated runner, immutable export commands, and training-copy workflow are documented in [AUGMENTATION_PIPELINE.md](AUGMENTATION_PIPELINE.md). Use these commands for new collections; earlier experimental writers are retired.
