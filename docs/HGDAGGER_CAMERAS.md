# September 4 HG-DAgger camera initialization

`config/hgdagger-cameras.json` initializes cameras against snapshot
`e74439f3eb2257661145571571b2dfaa5592f734` of `jren313/g1-pipette-3view-hgdagger-20260904`.
The dataset contains 46 episodes, 54,222 frames, and three videos per episode at nominal 60 FPS.
The task is tip attachment. This configuration does not establish successful dynamic replay of these episodes.

| Dataset / simulator key | Inferred physical role | Native image size | Assumed vertical FOV |
|---|---|---|---|
| `rgb` | Fixed overview / ego | 1280 × 720 | 60° |
| `wrist_right` | Fixed close side view | 1080 × 1080 | 85° |
| `wrist_left` | Right wrist, looking along pipette | 1080 × 1080 | 105° |

The wrist assignment is inferred from image motion and the tool remaining near the same pixels; it has not been confirmed by the operator. The dataset source notes also describe a repair of swapped wrist video ports. Preserve these keys when comparing images; do not infer physical mounting from their names. The third camera is parented to `right_wrist_yaw_link`; both other cameras stay fixed in world coordinates. Positions are meters; quaternions use MuJoCo wxyz, camera axes +x right, +y up, −z forward. Wrist transforms are relative to the wrist body, not world.

## Live camera monitor

`./run.sh` opens the main simulator plus a second window with all three simulated sensor views: overview (`rgb`), fixed side (`wrist_right`), and right wrist (`wrist_left`). Labels retain the dataset keys and configured roles. Camera aspect ratios are preserved, and no simulation steps occur between rendering the three images. The monitor targets 15 updates/second and continues servicing controls while paused.

Close the monitor with Escape or its close button without stopping the main simulator. Use `./run.sh --no-camera-window` to disable it. Headless runs do not create a monitor. The monitor is for live display; clean RGB/video export still uses `camera_streams` as described below.

## Run

Replay now initializes per-episode rack/tip placement by default. See [episode scene initialization](EPISODE_SCENES.md) for overlays and landmark editing. Add `--scene-layout fixed` to reproduce the original shared-layout camera previews below.

From `/home/ytung34/Projects/g1_pipette_sim`:

```bash
./run.sh \
  --dataset ~/.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hgdagger-20260904 \
  --config config/hgdagger-cameras.json --episode 0 \
  --physics kinematic --source state --camera rgb
```

Use `--camera wrist_right` for the fixed side or `--camera wrist_left` for the moving right-wrist view. Kinematic state replay is intentional for camera inspection. The cache path resolves through `refs/main`; pass the snapshot directory explicitly to pin a revision. Do not add `--fit-layout` when reproducing this camera fit.

Export synchronized, clean RGB videos at native aspect ratios and sizes:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.camera_streams \
  --dataset ~/.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hgdagger-20260904 \
  --config config/hgdagger-cameras.json --episode 0 \
  --physics kinematic --source state --native-resolution \
  --duration 5 --fps 30 --output outputs/hgdagger-cameras/streams
```

This produces three MP4s, first-frame PNGs, and `streams.json` with timestamps and camera poses. Rendering does not step between cameras. These are camera exports, not a LeRobot demonstration recorder. Without `--native-resolution`, all views use the requested `--width` and `--height`, which can change framing.

## Evidence and limits

`outputs/hgdagger-cameras/comparison-episode-{000,015,030,045}.jpg` contains real/sim pairs at the start, midpoint, and 90% of each episode. Columns are overview, fixed side, right wrist. `tools/preview_hgdagger_cameras.py` regenerates them using measured robot state, without changing camera poses per episode.

`tools/fit_hgdagger_cameras.py` reproduces the initial fit using four manually annotated rack corners in episode 0, assumed focal lengths, centered principal points, and zero distortion. The config preserves the annotations and assumptions. Rack placement/table height come from the prior kinematic nozzle estimate; active well row 2, column 4 is an approximate visual choice. Planar fitting admits ambiguous poses, and field of view was chosen to avoid severe robot mesh occlusion. Its small corner residuals are fitting residuals in resized images, **not evidence of accurate camera calibration**.

The fixed side and wrist views reproduce the general rack/tool framing. The overview's arm placement still differs substantially. The room, table extent, hand posture, lens distortion, and object appearance remain approximate. Recorded hand commands/feedback are constant, despite a visibly grasped and taped tool. The scene now uses an explicit static feedback-pose approximation, documented below; it cannot recover missing finger motion. Recorded commands remain unchanged. Existing physics parameters were inherited and have not been validated on this dataset.

Before using these images to train a deployment policy, calibrate each camera's intrinsics including the square crop, then solve the two fixed extrinsics and wrist-to-camera transform from a measured fiducial board visible across diverse robot poses. Measure the real pipette mount and account for the actual grasp posture. Validate projected rack corners and tool landmarks on held-out frames, then address appearance and distortion. This initialization alone is not sufficient visual sim-to-real validation.

## Tip seating requirement

This scene now requires the nozzle endpoint to be **2–5 mm below the tip mouth**, with less than **3 mm lateral error** and **12° angular error**, continuously for **0.10 seconds** before attachment. Hovering above the mouth cannot attach. Missing any condition resets the hold timer. Dynamic simulation measures this relative to the live tip pose; kinematic preview uses the fixed rack mouth. The thresholds are under `attachment` in `config/hgdagger-cameras.json`: `min_insertion_depth`, `max_insertion_depth`, `radial_tolerance`, `angle_tolerance_deg`, and `dwell_seconds` (meters, degrees, seconds).

This is a stricter geometric seating rule, not a measured insertion-force or friction-fit model. Existing replay trajectories may now fail to seat the tip; commands are not adjusted to force success. Older configs retain their original symmetric `axial_tolerance` gate. In this scene, explicit insertion limits supersede that legacy field.

## Frozen hand pose and dynamic replay (September 10 update)

All 46 episodes have constant hand feedback: left `[995,999,996,1000,499,467]`, right `[96,0,0,0,670,636]`, while both hand command vectors are `[1000,1000,1000,1000,1000,1000]`. These are six Inspire register values, not individual joint angles in radians. `hand_mapping.pose_register_override` now uses those feedback values as a fixed pose approximation for this scene. It does not establish that the telemetry was live or reconstruct finger motion. Remove that field to restore command-driven hand posing. The recorded command arrays and ejector command remain unchanged. The current default now uses finger/palm contacts instead of the wrist weld; see [the contact-grasp update](ONILAB_PIPETTE.md).

For dynamic command replay use `--physics dynamic --source low_level`. Using `--source state` instead asks the simulated servo to track already-measured joint positions; those differ from the commands that generated them and can result in gravity-induced sag. For exact measured robot poses use `--physics kinematic --source state`.

A check of episode 0 over eight seconds with the prior open-hand pose found tool–wrist displacement below 0.3 mm for both sources: the weld remained active. With `state`, the arm/tool sagged roughly 6 cm; with `low_level`, tool height changed roughly 6 mm. These are reproduction diagnostics, not a validation of dynamics against all real episodes. Reports now include `grip_constraint_active` and `grip_position_error_m` to distinguish arm sag from an actual released grip. In the legacy weld mode, `G` disables the wrist weld. In current contact mode, it commands the fingers open; `R` resets the grasp.
