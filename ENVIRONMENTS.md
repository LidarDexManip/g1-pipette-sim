# Environments

There are **two base scenes** (the original pipette workspace and the LIDAR lab)
plus **eight procedural visual variants** of the pipette workspace: **10 entries**.
They are not ten independently calibrated robot tasks.

```bash
./run.sh --list-environments
./run.sh --environment lidar                     # G1 + clean calibrated lab
./view_lidar.sh --episode 0                      # physical motor-target replay
./view_lidar.sh --episode 0 --physics kinematic   # measured-pose diagnostic
./view_lidar.sh --episode 126 --speed 2
./view_lidar.sh --episode 0 --headless --compare \
  --output outputs/lidar-teleop/comparison
./run.sh --environment pipette                   # original pipette task
./run.sh --environment lab-open-shelves           # original task + decoration
```

Mouse: left drag orbits, right drag pans, wheel zooms. LIDAR supports `--snapshot`,
`--video`, `--compare`, `--duration`, `--speed`, `--headless`, and
`--show-collisions`. Video/snapshot exports are 1920×1080; video is 30 fps.
LIDAR now defaults to **dynamic physics and low-level motor targets**. The robot
starts at the measured pose; force-limited position servos track subsequent targets
while MuJoCo integrates motion at 1 kHz. Hand servos use recorded hand commands.
The pipette (80 g) and tube (estimated 15 g) are free bodies with gravity and contact
friction; the pelvis, table, racks, stand, and centrifuge remain fixed. There is
no hand/tool weld, attachment trigger, or per-frame pose overwrite. Without a
dataset, servos hold the configured initial robot pose while props remain physical.

Use `--physics kinematic` for the original exact measured-pose diagnostic with
static props. Dynamic `--source state` tracks measured positions as servo targets;
it is not exact playback. `--source action` bypasses the real high-level controller
and is not equivalent to low-level replay. Free-base and scripted demo modes remain
unsupported. Physical interaction does not imply a successful or calibrated grasp.
Controller gains, approximate box inertias, and friction are simulation estimates.
The report includes contact forces, penetration, tracking error, and prop displacement.

Initial physical validation of episode 0 completed 15,662 steps (15.662 seconds)
without simulator warnings. Maximum pipette COM rise was 145.8 mm; maximum
hand/pipette proxy penetration was 1.11 mm, with no grasp weld. First force-bearing
hand/pipette contact was at 0.089 seconds, earlier than the real approach, so this
is evidence of working dynamics rather than calibrated task reproduction.
The maximum penetration across all contacts was 5.54 mm. Initial pipette/stand
overlap remains 1.63 mm. Results: `outputs/lidar-teleop/physical-episode-000/replay-report.json`.

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
- Other props use the repaired convex decompositions described below, including
  explicit hollow tube and rack openings.
- The floor collides. No scanned surfaces, unmodeled clutter, or walls are present.
- The G1 has a fixed pelvis. Dynamic joints are actuated; only kinematic diagnostics
  prescribe their positions. Held-object initialization for later task steps and
  grasp/release success still require calibration and validation.

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
replay follows the recorded timing. Dynamic mode uses motor-target timestamps when
available and paired image rows when the recorded command clock is frozen.

In kinematic mode, hand feedback is used when it varies. Entirely constant feedback falls back to
that hand's recorded commands, recorded explicitly in `replay-report.json`.
117 left-hand and 46 right-hand episodes have constant feedback. The mapping from
registers to Inspire joint angles remains approximate. The pelvis is fixed because
absolute base translation/yaw is not measured in the dataset.

The historical kinematic checks replay all 3,780 recorded frames of the five examples above. Joint
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

The initial broader `tools/audit_lidar_collisions.py` audit checked robot/scene contacts,
static object overlaps, visual-versus-collision top surfaces, and self contacts
in a separate diagnostic model. Run it with `--dataset <snapshot directory>`.
Its default output is `outputs/lidar-teleop/collision-audit/`, containing a
summary and per-frame contact records for episodes 0, 126, 188, 127, and 1.
All five have robot/object penetration over 1 mm, including the three that
passed the table-only check. Outer visual and collision bounds agree, but
convex hulls fill the tube rack holes, tube opening, and pipette stand opening.
Static tube/rack and pipette/stand proxies overlap by approximately 39 and
38 mm, respectively. The tip rack also overlaps the tabletop by 4.4 mm.

All 50 robot mesh colliders match their corresponding visual mesh transforms;
this does not validate their convex collision surfaces. Robot self collisions
were disabled by the initial masks. The separate diagnostic enabled them with
the original robot asset's four exclusions and detected no self penetration
over 1 mm in these five episodes. The repaired composition now copies those four
exclusions and enables robot self-collision detection.
No recorded joint exceeded the modeled limits in these examples. Repair the
concave object colliders before refining placement or enabling dynamic grasps.

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

Before claiming dynamic task success: refine object/camera registration per recording
session, identify mass/inertia/friction and servo gains, initialize held objects for
later steps, and validate grasps/releases. Free-body dynamics and servo replay are
implemented in `g1_pipette_sim/environments/physical.py`.
The dataset's 35-value main action is a high-level controller input, not identical
to its 29-value low-level motor targets.

## Wuji Hand 2 option

`--hand wuji` replaces both Inspire hands with Wuji Hand 2 (Beta 2) in the LIDAR
scene. The G1 + Wuji model comes from
[g1-wuji-retargeting](https://github.com/LidarDexManip/g1-wuji-retargeting), so
trajectories retargeted there use the same assembly and provisional wrist mounts.
Build it there, then copy the robot into `assets/robot_wuji/`:

```bash
cd ~/Projects/g1-wuji-retargeting && ./fetch_assets.sh && python build.py
cd ~/Projects/g1-pipette-sim && python tools/prepare_wuji_assets.py
./view_lidar.sh --episode 126 --hand wuji
./view_lidar.sh --episode 126 --hand wuji --headless --compare \
  --output outputs/lidar-teleop/wuji-comparison-126
python tools/check_lidar_replays.py --dataset "$DATASET" --hand wuji \
  --output outputs/lidar-teleop/wuji-replay-audit.json
```

Both the default physical replay and `--physics kinematic` accept `--hand wuji`;
`replay-report.json` records `robot`/`hand` and Wuji-specific limitations. The 29 G1
joints use the same targets and servos as with Inspire. The recorded hands are
Inspire, so their six registers drive the 20 Wuji joints per hand through the
approximate synergy in `config/hands/wuji-from-inspire.json` (finger curl, thumb
rotation via `thumb_cmc_flex`, thumb bend via `thumb_cmc_abd`, `mcp`, `ip`). Thumb
rotation follows the register convention (1000 = open hand, thumb beside the index
finger). This is a playback proxy, not a retargeting result. In physical replay all
20 Wuji joints per hand are independent position servos with the retargeting
repository's gains (kp 2, kv 0.06), armature/damping and URDF effort limits.

The robot placement was fitted with the Inspire palm; the Wuji hand is longer. With
the current global placement, physical episode 0 lifts the pipette 9.7 mm and episode
126 starts with the left thumb 9.8 mm inside the tube without lifting it, so Wuji
grasp success is not expected yet. `--hand inspire` (the default) builds a scene
identical to before and replays identically; the pipette workspace remains
Inspire-only because its grasp, ejector and contact settings are tuned for Inspire.

## Code organization

Collision-repair tools:

```bash
# Optional build dependencies, using the simulator's Python environment:
python -m pip install -r requirements-collision.txt
python tools/rebuild_lidar_collisions.py --object env_00_cylinder_base
python tools/rebuild_lidar_collisions.py --object env_00_cylinder_base --apply

# Probe openings, walls, closed bottoms, and table support:
PYTHONPATH=.:tests python -m unittest test_lidar_collision_repairs

# Inspect all objects and audit measured replay contacts:
MUJOCO_GL=egl python tools/render_lidar_collision_comparison.py
python tools/audit_lidar_collisions.py --dataset /path/to/snapshot \
  --output outputs/lidar-teleop/collision-repair/after
```

Generated collision assets live in `assets/environments/lidar/meshes/collision_v2/`.
Each object's JSON record identifies its source visual hashes and generated pieces.
Visual meshes, scene scale, table height, and object placement are unchanged.
The table retains its five exact boxes. Thin-wall contacts use `solref=".004 1"`,
`solimp=".99 .99 .001 .5 2"`, and priority 1; these settings passed small-probe
checks but are not a real-world friction/contact calibration.

The tube uses explicit hollow elliptical profiles; its open source surface seams
are closed into solid walls and a bottom. The stand uses angular sectors and
height bands no larger than 1 mm at runtime, with a 0.2 mm skin on each side of
surface fragments so zero-thickness base faces remain collidable. The tip rack uses annular pieces and
convex extrusions of its perforated top surface, because a generic decomposition
can cap very thin rings even with a small error threshold. Other irregular
surfaces use CoACD decomposition after correcting face winding in the collision
builder's working copy. Watertightness alone does not establish consistent winding.
The centrifuge's single-sided well inserts receive 0.4 mm walls and closed bottoms.
None of these changes makes props movable or
turns measured-position playback into a dynamic task rollout.

Compare the saved baseline `outputs/lidar-teleop/collision-repair/before.json`
with the new `after/report.json`. Penetration depth is **per convex piece**:
thinner pieces can report smaller depths even when overlap remains. Use the
opening/wall tests, contact locations, and contact-frame counts as well; do not
interpret a reduced depth alone as a successful repair or successful task.
The probe tests cover all 96 tip-rack positions at two depths, all eight centrifuge
wells, a tube-rack opening, the tube mouth/wall/bottom, the stand bore/wall/base,
and table support. Re-exporting the Blender scene resets the base collision
definitions; regenerate and apply the repaired objects afterward.

The repaired meshes are approximate, not an exact surface match. The follow-up
`tools/check_lidar_surface_error.py` measures Euclidean distance at flagged audit
examples, because vertical ray differences near cavity edges can be misleading.
Checked discrepancies reach 2.24 mm on the tube rack and 1.28 mm on the reconstructed
tube rim; the flagged centrifuge and stand samples are below 0.2 mm. This is a
sampled check, not a global error bound.

Object poses were deliberately preserved. The contact audit still finds initial
tube/rack, pipette/stand, and tip-rack/table overlaps of approximately 2.2, 1.6,
and 4.4 mm respectively. Five measured replay episodes also retain robot/scene
penetration, including roughly 55–57 mm hand/table penetration. Kinematic replay
prescribes positions regardless of contact; these results require registration,
object seating, and replay calibration before physical rollout validation.
They must not be interpreted as successful dynamic demonstrations.

Measured replay now applies reviewed episode scene presence from
`config/environments/lidar-episode-scenes.json`. For the pinned two-view dataset,
episodes 0 and 1 have an empty blue rack in the real recording: the imported tube
is therefore absent from rendering and collision detection in those replays.
This is an object-presence correction, not collision suppression for a present
object or a change to robot/hand joint angles. Other episodes restore the original
prop properties; overrides only apply when the dataset provenance hash matches.
The static scene preview without a dataset continues to show all modeled objects.
The viewer prints the applied review and saves it in `replay-report.json`.

`tools/check_lidar_episode_scenes.py --dataset /path/to/snapshot` compares the
original and reviewed scene across every frame of episodes 0, 1, and 126. It
reports initial contacts, full-trajectory tube contacts, and measured-joint
assignment error. Unreviewed episodes, held-object motion, and alignment remain
unresolved; do not infer empty-rack initialization from a task label alone.

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
