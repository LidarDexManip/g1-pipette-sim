# Environments

There are **two base scenes** (the original pipette workspace and the LIDAR lab)
plus **eight procedural visual variants** of the pipette workspace: **10 entries**.
They are not ten independently calibrated robot tasks.

```bash
./run.sh --list-environments
./run.sh --environment lidar                     # G1 + clean calibrated lab
./view_lidar.sh --episode 0                      # measured-joint demonstration
./view_lidar.sh --episode 126 --speed 2
./view_lidar.sh --episode 0 --headless --compare \
  --output outputs/lidar-teleop/comparison
./run.sh --environment pipette                   # original pipette task
./run.sh --environment lab-open-shelves           # original task + decoration
```

Mouse: left drag orbits, right drag pans, wheel zooms. LIDAR supports `--snapshot`,
`--video`, `--compare`, `--duration`, `--speed`, `--headless`, and
`--show-collisions`. Video/snapshot exports are 1920×1080; video is 30 fps.
Without a dataset, the G1 holds its initial pose. With `--dataset`, it plays the
measured body joints. **LIDAR playback is kinematic with static props**, not a
physical grasp-success test. `--physics dynamic`, non-state sources, free base,
and scripted demo mode are rejected for this environment.

No environment selection means the original pipette workspace. If its default
HIL194 cache is absent, the launcher uses HIL293 with a matching saved replay
config (episode 11 when unspecified), fixed rack/tip placement, and corrected
cameras. A portable recipe is included in `config/replay/hil293-episode-011.json`.
Explicit dataset/config inputs are never silently substituted.

## Clean LIDAR geometry and calibration

The source `LIDAR Environment.blend` contains seven modeled objects and a separate
raw room scan. **The scan and scan texture are excluded by default**, including on
re-export. The clean scene contains the table, open benchtop instrument, pin rack,
pipette stand, cylinder, cylinder base, and pipette, plus a plain floor.

The source pipette is 0.508 m tall. The user's measured pipette length is 10 inches
(0.254 m), so the runtime scene applies a common **0.5 scale** and puts the table
feet at floor Z=0. Resulting dimensions:

| Object | Calibrated dimensions |
|---|---|
| Table, including legs | 0.768 × 1.536 × 0.720 m |
| Pipette | 0.254 m tall |

This is a metric scale anchor, not proof that every generated object shape is
accurate. The original pipette task's dimensions and cameras are unaffected.
The robot stays at its real model scale; only the imported environment is scaled.

| File | Purpose |
|---|---|
| `assets/environments/lidar/scene.xml` | Raw, unscaled Blender export, without scan |
| `assets/environments/lidar/meshes/` | Raw mesh coordinates, retained for reproducibility |
| `assets/environments/lidar/manifest.json` | Raw source dimensions, hashes, excluded objects |
| `config/environments/lidar-scene.json` | Runtime scale and floor normalization |
| `config/environments/lidar-robot.json` | Robot XY/yaw fit, fixed pelvis height, overview camera |
| Run output `scene.xml` | Effective calibrated scene **including the robot** |
| Run output `environment.json` | Effective calibrated object dimensions |

The original `.blend` and exported source meshes are preserved; scene loading
applies calibration consistently to visual meshes and collision shapes. A Blender
re-export does not reset the runtime scale. Loading the raw asset XML directly
bypasses calibration; use `run.sh` or `build_robot_scene()` for the calibrated model.

Physics scope:

- The table has five separate cuboid colliders, preserving open leg space.
- Other props use static connected-component convex proxies. Concave openings
  within a component can be filled; these are not validated grasp surfaces.
- The floor collides. No scanned surfaces, unmodeled clutter, or walls are present.
- The G1 has a fixed pelvis. Joints are prescribed during diagnostics. Props are
  static; masses, movable-object initialization, and grasp/release physics remain
  to be implemented and validated for this five-step task.

Regenerate source assets with Blender 5.2 or newer:

```bash
blender -b -noaudio 'LIDAR Environment.blend' \
  --python tools/export_blender_environment.py -- \
  --output assets/environments/lidar
```

Blender is needed only for conversion. The exporter preserves evaluated transforms,
material colors and supported image textures, records checksums, and removes stale
generated meshes/textures. The raw scan is always excluded by its object name.

## Demonstration dataset and checks

Dataset: `jren313/g1-pipette-2view-teleop0925-5task-eerel`, pinned to revision
`c7adfc179e5e15ade41b06803542001a24959dda`. The full download contains **447 episodes,
272,236 frames at 60 Hz, and 894 videos**. It occupies about 17.2 GB and lives under
`.cache/huggingface/hub/datasets--jren313--g1-pipette-2view-teleop0925-5task-eerel/`
in this repository; the cache is excluded from Git.

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/download_lidar_dataset.py
```

| Task | Episodes | Initial replay example |
|---|---:|---:|
| Pick pipette | 93 | 0 |
| Pick tube | 91 | 126 |
| Aim pipette at tube | 79 | 188 |
| Return tube | 91 | 127 |
| Return pipette | 93 | 1 |

Both recorded video views retain their dataset names (`rgb`, `wrist_left`); those
names alone do not establish their physical mounts. `--compare` exports the sim
**overview** at left and synchronized real feeds at right. It is not a calibrated
camera-pair comparison. The 30 fps inspection video is not a training dataset;
replay follows the original image-row timestamps and measured joints.

Hand feedback is used when it varies. Entirely constant feedback falls back to
that hand's recorded commands, recorded explicitly in `replay-report.json`.
117 left-hand and 46 right-hand episodes have constant feedback. The mapping from
registers to Inspire joint angles remains approximate. The pelvis is fixed because
absolute base translation/yaw is not measured in the dataset.

Initial checks replay all 3,780 recorded frames of the five examples above. Joint
assignment matches the measured data exactly, which verifies the playback path,
**not physical fidelity or task success**. Static objects do not follow the hands;
`task_success` is explicitly null. Later workflow steps require correct held-object
initialization, and some props differ in appearance/state between recordings.

The native-scale diagnostic first exposed palms 45–56 cm below the table. After
metric scaling, `tools/fit_lidar_robot_placement.py` fits XY/yaw using the first five
pipette pickups and first five tube pickups. Hand-closure events and a 9 cm wrist
palm proxy provide approximate object-center anchors. A bounded clearance search
rejects torso/leg placement through the table. This is an initial geometric fit,
not a precise fingertip-contact calibration or a held-out success result.

The fitted base is `(0.485, -0.0393)` m with yaw `167.45°`; pelvis height is
0.793 m above the floor. The ten approximate palm-to-object XY anchors have
73.1 mm median and 121.4 mm maximum residual, so this registration still needs
refinement. Do not treat these as measured grasp positions.

The calibrated full-frame checks found:

| Episode | Task | Frames with robot/table penetration >1 mm | Maximum penetration |
|---|---|---:|---:|
| 0 | Pipette pickup | 0 / 936 | 0 mm |
| 126 | Tube pickup | 410 / 659 | 55.1 mm |
| 188 | Aim | 0 / 668 | 0 mm |
| 127 | Tube return | 421 / 728 | 56.5 mm |
| 1 | Pipette return | 0 / 789 | 0 mm |

The penetrations involve the left fingers and wrist, not the torso or legs.
They indicate unresolved hand mapping/geometry and scene registration; the
diagnostic does not isolate which contributes most. No physical task success
rate is reported: all props remain static, and other object contacts are not
included in the table-only check.

Reports in `outputs/lidar-teleop/`:

- `download-verification.json`: all Parquet row counts and video headers match
  metadata (resolution, frame count, FPS); not exhaustive decoding of all videos.
- `dataset-audit.json`: all 447 episodes have finite replay channels and strictly
  increasing image timestamps; timing and hand-feedback details.
- `placement-fit.json`: fitting episodes, closure events, residuals, clearance search.
- `replay-audit.json`: historical native-scale diagnostic.
- `calibrated-replay-audit.json`: five full trajectories after calibration, including
  robot/table penetration diagnostics.
- `calibrated-comparison-000/simulation.mp4`: example with two real video feeds.

Before dynamic task validation: refine object/camera registration per recording
session, replace unsuitable convex proxies around grasp openings, give pipette
and tube free bodies with measured mass/friction, initialize held objects for
later steps, and validate grasps/releases before replaying low-level motor targets.
The dataset's 35-value main action is a high-level controller input, not identical
to its 29-value low-level motor targets.

## Code organization

The largest visual OBJ is committed as `.obj.gz` to stay below GitHub's file-size
limit. The environment loader extracts it automatically on first use and checks
its SHA-256 against the export manifest. This is lossless compression; the
extracted OBJ is ignored by Git. Blender/GLB source files, downloaded datasets,
and `outputs/` are not required in the repository to run the exported scene.

- `g1_pipette_sim/environments/__init__.py`: registry, scene calibration, manifests,
  procedural injection, imported asset loading.
- `environments/workspace.py`: original room/table construction.
- `environments/procedural.py`: deterministic lab styles shared with video scripts.
- `environments/robot.py`: G1 composition and measured-body/hand playback.
- `environments/viewer.py`: interactive viewer and real-video comparison export.
- `tools/audit_lidar_demonstrations.py`, `verify_lidar_dataset.py`: dataset checks.
- `tools/fit_lidar_robot_placement.py`, `check_lidar_replays.py`: placement and replay checks.

Add another imported scene by registering its asset XML and optional calibration
file in `ENVIRONMENTS`. Procedural decorations stay under a separate static body
with collisions disabled, so they cannot become part of the dynamic task table.
Original task configs accept `"environment": {"name": "lab-wire-racks", "seed": 7}`.
LIDAR's `--config` instead selects its robot registration file.
