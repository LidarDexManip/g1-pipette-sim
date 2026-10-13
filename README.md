# G1 / Inspire pipette workspace simulation

**New machine:** follow [the portable quickstart](README_SETUP.md) for installation, the pinned real dataset, simulation, augmentation collection, and illustration videos. No historical outputs are included.

Saved pickup acceptance below refers to historical attachment/lift criteria. The newer rack-contact audit can reject these captures; see the quickstart before using them for training.

Default HIL194 replays now load available per-episode pickup configs automatically. Use `--scene-layout visual` to inspect the original image-based placement.

A runnable, offline MuJoCo environment. `./run.sh` defaults to `jren313/g1-pipette-3view-hil194`, a verified pickup example selected from the current world-lift manifest, dynamic physics, and low-level command replay. It loads the matching saved layout; `--episode N` selects a specific episode. Valid indices are 0–193.

The default ONiLAB 0.5–10 µL pipette has an 80 g complete assembly and a contact-based grasp: the wrist weld is disabled. Fingers and palm support the tool through collision/friction forces. Press **G** to command the right fingers open (the finger rest may still hang on the palm).

```bash
cd ~/Projects/g1_pipette_sim
./run.sh
# Recorded-pose camera inspection:
./run.sh --physics kinematic --source state --camera rgb
```

It contains the G1 with two Inspire hand models, a table, an open yellow tip rack with its hinged clear lid, a pipette with an animated ejector, and a detachable green tip. The HIL194 task is attachment only; automatic ejection is disabled for this profile. Episodes 46–193 use the photographed red-tape pipette. It can replay any dataset episode, perform a scripted attachment/ejection task, and export simulation video or three-view comparisons.

**Dynamic physics is now the default.** Robot joints are driven by torque-limited servos. The table, rack, tray, pipette, and tip have mass, inertia, gravity, and collision dynamics. Their poses are initialized once at reset and are then integrated by MuJoCo. Robot collision meshes and self-collision are enabled, apart from adjacent/overlapping mechanism parts and constrained grasp pairs.

The robot's pelvis is supported by default, like a fixed test stand. `--free-base` removes that support and gives the whole robot gravity and floor contacts; no balance controller is included. The floor and backdrop are static fixtures. Geometry, masses, friction and controller gains remain estimates, so this is not yet a hardware-calibrated sim-to-real environment. The original prescribed-motion preview is retained as `--physics kinematic`.

## Per-episode object placement

Default HIL194 replay loads a cached visual layout for the selected episode, or estimates one from its initial images. Rack position/yaw and the active tip can differ between episodes; the lid's unannotated pose remains explicitly assumed. Use `--scene-layout fixed` to keep the configured layout, and `--refresh-scene` to recompute. Estimates that fail confidence checks are rejected with correction instructions.

## Calibrated recorded-command replay

Episode 53 now completes attachment, release and disposal with unchanged **low-level motor commands**, identified PD gains and an effective wrist payload. It does not replay observed joint positions into the live robot. Photo-derived object dimensions remain unchanged.

```bash
cd ~/Projects/g1_pipette_sim
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --episode 53 --source low_level --config config/episode53-calibrated.json

# Shared physical parameters, with an approximate layout seed for another episode:
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --episode 57 --source low_level --config config/calibrated-physics.json --fit-layout
```

These older calibrated profiles require the older dataset explicitly. The original demo settings remain available as `config/workspace.json`. `config/episode53-calibrated.json` freezes the successful scene; omit `--fit-layout` with that file. `config/calibrated-physics.json` shares gains and payload across episodes. Its layout still needs initialization. Improved joint tracking does not establish matched object dynamics or guarantee pickup/disposal in another episode. `--source action` bypasses the unmodelled high-level-to-low-level controller and is not equivalent to the calibrated motor replay.

Replay videos and snapshots are generated locally under `outputs/`; historical recordings are not included in a clone.

## Run on this machine

Complete [installation](README_SETUP.md) first. `run.sh` uses `G1_SIM_PYTHON` when set, otherwise an existing `lerobot-act` environment or `python` on PATH. The commands below require the older dataset at the specified location.

```bash
cd ~/Projects/g1_pipette_sim

# Watch the 10-second scripted attach / lift / eject demonstration.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --mode demo

# Replay recorded episode 53, seeding the layout from its tool trajectory.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --fit-layout

# Start from another episode. Valid indices: 0 through 77.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 77 --fit-layout --camera rgb
```

Interactive runs also open a **camera monitor** showing synchronized real and simulated `rgb`, `wrist_right`, and `wrist_left` views during replay (simulation views only during demos), labeled with their configured physical roles. Each view keeps the dataset aspect ratio; all three render the same simulation state at up to 15 updates/second (actual rate depends on rendering speed). The monitor stays responsive while paused and is independent of the main viewer's mouse-controlled camera. Space/R/E/G/C/F also work when the monitor has keyboard focus. Escape or its window close button closes only the monitor. Both windows close when the run ends. Use `--no-camera-window` to open only the main viewer; `--headless` never opens either window. The monitor uses Tkinter/Pillow, available in the configured `lerobot-act` environment.

The viewer closes when the episode/demo finishes. Space pauses/resumes, **R** resets, **E** ejects an attached tip, **G** releases the pipette grip in dynamic mode, **C** cycles the camera presets, and **F** returns to the free overview. Close the window to exit. Reset is disabled during video recording to keep the video timeline consistent. The overview now starts as a free camera: **left-drag** orbits, **right-drag** pans, and the **scroll wheel** zooms. Hold Shift while dragging to change the movement axis. Press Space to pause the task while exploring. The `rgb` and wrist presets are fixed sensor views; press F to return to mouse navigation. Camera navigation works while paused. Interactive video export follows your mouse-controlled view; headless exports keep their specified camera.

`--speed 0.5` runs at half speed; `--duration 5` stops after five simulation seconds. Full help:

```bash
./run.sh --help
```

## Headless operation and video export

```bash
# Physics/replay without a window or renderer.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --headless --episode 53 --fit-layout --output outputs/replay-53

# Export the scripted demo at 30 FPS.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --headless --mode demo --video --snapshot --output outputs/my-demo

# Compare all three real camera videos with their simulated counterparts.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --headless --episode 53 --fit-layout --compare --duration 5 \
  --snapshot --output outputs/my-comparison
```

A run writes:

- `workspace.json`: resolved scene configuration, including the layout seed when requested.
- `scene.xml`: generated MuJoCo scene, with absolute paths to this project's robot meshes.
- `report.json`: events, numerical warnings, run settings, and limitations.
- `simulation.mp4`: when `--video` or `--compare` is selected.
- `snapshot.png`: first frame when `--snapshot` is selected.
- `final.png`: final frame when rendering is enabled.

The default output directory is `outputs/latest`; rerunning into the same directory replaces matching output files. Use different `--output` paths to preserve runs. Launch through `run.sh`, not by opening `scene.xml` alone: the Python loop initializes body poses and supplies controller targets and attachment logic.

Headless rendering defaults to EGL. If EGL is unavailable, try `MUJOCO_GL=osmesa ./run.sh --headless ...` with OSMesa installed. Headless runs without images do not require a rendering context. A desktop display is required for the interactive viewer.

The new physical demo is in `outputs/dynamic-demo-video/`. Earlier `outputs/demo/` and `outputs/comparison/` files show the original kinematic version; their reports identify the mode.

## What is reconstructed and what is assumed

| Component | Implementation / provenance |
|---|---|
| G1 body | Existing local G1 29-joint model and meshes, copied without changing the source project |
| Hands | Existing Inspire **DFQ** geometry; recording schema says **RH56DFTP**, so exact geometry/mounting still needs verification |
| Hand commands | Six raw registers mapped to joint angles; intermediate finger joints are coupled; mapping is configurable and approximate |
| Robot root | Supported pelvis by default; optional free base. No measured global root trajectory is available |
| Table | One rigid body for tabletop and legs, with a free joint, estimated 18 kg mass, and floor contacts |
| Tip rack | Rigid body, estimated 0.18 kg; 8×12 open wells, one active cell, optional tape weld to the table (enabled by default) |
| Pipette | Separate free body, estimated 0.085 kg plus 0.005 kg ejector, held by a solver weld that can be released with G |
| Green tip | 0.0006 kg free body with a tapered collision mesh; rack retention and nozzle attachment are switchable weld constraints |
| Clear tray | Free open container, estimated 0.08 kg, with bottom/wall collision shapes |
| Cameras | Three fixed pinhole estimates plus overview; names preserve dataset keys and do not assert actual wrist mounting |

The table/rack/pipette are procedural geometry based on the observed objects, not photogrammetric meshes. The default scene has one active tip, and one attachment/ejection cycle per reset. The tip visual is hollow and the rack has round open wells. The tip contact proxy is solid; pipette internals, spring forces, and compliant interference fits are not modeled.

`--fit-layout` keeps the estimated near table edge clear of the robot and uses the lowest 12% of estimated nozzle positions to seed one rack well and tabletop height. It can also seed the receiving tray from a later thumb-closing threshold crossing, expanding the offset to avoid overlap with the rack. These are **kinematic hypotheses**, not recovered metric ground truth or independently verified task labels. The method depends on the assumed root and tool mount. It will not reliably identify every layout or unsuccessful attempt.

For a measured scene, edit a configuration and run with `--config`; omit `--fit-layout` so your measurements are retained:

```bash
./run.sh --episode 53 --config config/my_measured_workspace.json
```

All dimensions are metres, angles in `yaw_deg` are degrees, other joint angles are radians, and quaternions use **w x y z**. World axes are x forward from the robot, y to its left, z up. `pipette.mount_position` and `mount_quaternion` are relative to `right_wrist_yaw_link`. The nozzle points along the pipette's local negative z axis.

## How robot actions are generated

The `run.sh` demo/replay entry point does not run a learned policy or the G1 whole-body controller.

| Selection | Source of robot target |
|---|---|
| `--mode replay --source state` | Dataset `observation.state[:29]`: measured G1 joint positions, used as reference targets |
| `--mode replay --source action` | Dataset `action[:29]`: recorded high-level joint-position targets |
| `--mode replay --source low_level` (default) | Dataset `action.low_level`: recorded low-level joint-position targets |
| `--mode demo` | Scripted tool waypoints solved by right-arm inverse kinematics; hand/thumb commands are scripted |

Both hands use the recorded `action.hand_left/right` registers in replay, because right-hand feedback is frozen. The mapping expands six channels into the Inspire joints; physical joint equality constraints couple the passive finger segments. The body's six root-command fields are not applied, and there is no locomotion policy. `--source` has no effect on the scripted demo.

In **dynamic** mode, MuJoCo position servos produce approximately:

```text
torque = clip(kp * (target - position) - kv * velocity + bias_compensation,
              -torque_limit, +torque_limit)
```

Bias compensation uses the simulator's current model-based joint bias forces and can be disabled in the configuration. Limits apply to the total servo force. Contact forces and gravity affect the resulting motion, so it will differ from the demonstration. Live joint positions are not overwritten after each step. Original recorded torque estimates are not used as motor commands.

In **kinematic** mode, positions are set directly as before, robot collision forces are disabled, and only the released tip is dynamically simulated. Use it for exact pose visualization or to compare against dynamic tracking.

```bash
# Replay the recorded action targets through physical servos.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --fit-layout --source action

# Replay the original low-level joint targets through these simulator servos.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --fit-layout --source low_level

# Free robot: no pelvis support, and no balance controller.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --mode demo --free-base

# Original exact-pose preview.
./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --fit-layout --physics kinematic
```

## Task dynamics

The state sequence is `in_rack → attached → released → disposed`.

1. A solver weld holds the physical tip in the physical rack. It transmits gravity/contact loads and follows the rack if it moves.
2. Attachment requires actual nozzle-to-tip radial/axial proximity, acceptable angular alignment, and continuous dwell. The rack weld is disabled and a pipette-to-tip weld is enabled at their current relative pose. Neither body is teleported.
3. Ejection requires the thumb-bend command to exceed the upper threshold and then fall below the lower threshold, or the E key. It disables the attachment weld and adds a small downward release impulse to the tip's existing velocity.
4. The released tip falls and collides with the tray, table, robot and floor. The disposal test uses the **moving tray's** coordinate frame and requires 0.2 seconds of slow motion inside it.

In the default HIL194 scene the wrist weld is inactive: closed fingers and palm support the pipette through physical contacts. G commands the right fingers open; the curved finger rest can still hang on the palm. Older configurations use a six-axis wrist weld, which G disables. Tip retention/attachment remains an approximation of the real press fit.

Hand–pipette contact is enabled from initialization in the default contact-grasp scene. Older weld-grasp scenes filter those pairs while constrained. In dynamic mode, pipette-to-tip contact stays enabled during insertion, attachment and release. Convex sectors represent the hollow tip wall, and separate tapered shaft sections follow the visual profile. Use `--show-collisions` to inspect collision geometry in the main viewer and exports. Robot self-contact remains enabled, with exclusions for adjacent/overlapping thumb roots and the pipette/ejector mechanism. Visual duplicate meshes never collide.

`physics` in `config/workspace.json` controls masses, servo gains, force/torque limits, bias compensation, and solver timestep. Dynamic mode uses **0.5 ms substeps** to resolve the thin tip and tray; recorded targets are interpolated on the original timestamps. The rest of `attachment` controls the geometric thresholds and release impulse. These are not inferred force measurements.

The scripted demo's success verifies simulator logic. It does not validate real grasp forces, precise press-fit mechanics, or real-world transfer. High target tracking error or displaced props can be real outcomes of an inaccurate scene or controller, even when MuJoCo reports no numerical warnings.

## Replay semantics and data caveats

- Default `--source low_level` replays the 29 recorded motor targets. `--source state` maps the 29 measured G1 joints by name.
- `--source action` replays the first 29 high-level target values.
- `--source low_level` replays the separately recorded low-level targets. These differ from high-level targets in every inspected episode.
- Root command tails are not interpreted as measured robot position.
- `--clock recorded` uses the image timestamps for measured body poses and action timestamps for command streams. Strictly increasing action timestamps are required for separate interpolation; otherwise the image timeline is used. `--clock nominal` uses frame number / 60.
- Frozen right-hand feedback is ignored; both hands use commanded registers.
- Hand-force streams are constant within episodes and are not used.
- Tactile samples exceed the documented range; they are not interpreted as forces.
- Body-state acquisition time is unavailable. Per-camera synchronization and actual mounting remain unverified.
- Two square views were center-cropped during conversion. Intrinsics must correspond to the cropped images.

The original dataset and existing simulation repositories are never modified.

## Calibrate a fixed camera

Use measured landmarks in the **same world frame as the simulation** and their pixel coordinates in the exported dataset image. Intrinsics must already be known. Prefer landmarks spread across the image and at more than one height. At least six point pairs are required. For planar landmarks, camera pose can be ambiguous; verify against independent landmarks.

Create a JSON file with these fields (replace all values with your measurements):

```text
{
  "resolution": [image_width, image_height],
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "lens_model": "pinhole",
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "world_points_m": [[X1, Y1, Z1], ... at least six points ...],
  "image_points_px": [[u1, v1], ... corresponding pixels ...]
}
```

For an OpenCV fisheye model, set `lens_model` to `fisheye` and provide its four distortion coefficients. Do not mix these with the pinhole distortion convention.

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.calibrate_camera \
  my_rgb_correspondences.json --view rgb \
  --config outputs/replay-53/workspace.json --output config/calibrated.json

./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/calibrated.json --headless --episode 53 \
  --compare --duration 5 --output outputs/calibrated-comparison
```

Repeat for each view, using the previous output as the next `--config`. For HIL194, pass `--camera-pose config` to use your calibrated poses instead of the dataset’s default rendering corrections. The utility estimates a fixed camera pose using PnP/RANSAC and reports reprojection residuals. It saves a centered pinhole projection and metadata to rectify the real image during comparison. Comparisons preserve each video's aspect ratio. Low residuals do not validate incorrect physical dimensions, landmark associations, or a moving camera; a moving camera needs a separately calibrated body-relative transform.

MuJoCo references: [Python viewer and rendering](https://mujoco.readthedocs.io/en/stable/python.html), [camera conventions](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera).

## Another machine / Python environment

Copy this project including `assets/robot/`, and make the dataset available locally. No GPT API key, network service, or robot hardware connection is used.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
G1_SIM_PYTHON="$PWD/.venv/bin/python" ./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --mode demo
```

The tested MuJoCo version is 3.8.1. If assets are absent, `prepare_assets.py --source /path/to/g1_inspire_model.xml` can copy them from an existing local model with a sibling `meshes` folder. `assets/robot/provenance.json` records source paths and mesh checksums. The copied upstream robot assets retain their original terms; this project does not grant a new license to them.

## Validation

```bash
~/miniconda3/envs/lerobot-act/bin/python -m unittest discover -s tests -v
```

Tests exercise named joint mapping and timing, layout initialization across recording sessions, rejection of distant attachment, complete kinematic and dynamic task cycles, reset, and synthetic camera pose recovery. Physical tests also verify free fall for every movable prop and the free robot base, tray/table contact, response to an applied force, pipette/table collision, and that servo control does not teleport joints. Full episode 53 dynamic replay, headless rendering/video, three-view comparison, and desktop viewer launch are exercised. Dynamic reports include active contact pairs, peak servo tracking error (radians for hinges, metres for the ejector), object displacement, and resolved physics parameters. Inspect each run's `report.json` for its actual result; recorded success labels are not used to force simulation success.

## September 9 photo-based object update

The default workspace now uses the photos in `assets/Media`: a ribbed transparent box with 96 round open wells, a lid-shaped receiving tray, a profiled green tip (about 33 mm long), and a revised white/blue pipette. Dimensions are approximate ruler estimates, not a calibrated scan. Gravity and collision remain enabled; visual meshes and contact proxies are separate. The box uses a fixed constraint to approximate the tape visible in the photos (`rack.taped_to_table: true`; set false for a loose box). The HIL194 profile uses a passive lid hinge; the obscured pipette mount remains estimated.

Run `./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --mode demo` for generated IK motion, or `./run.sh --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53 --fit-layout` for dataset target replay. Omit older `--config` exports to use the new dimensions. To regenerate the close-up preview, run `~/miniconda3/envs/lerobot-act/bin/python tools/preview_objects.py`.

## GR00T camera observations

`g1_pipette_sim.camera_streams.CameraStreams` returns clean, synchronized RGB arrays without viewer overlays. To export separate view videos and camera-pose metadata:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.camera_streams \
  --episode 46 --native-resolution --duration 5 --output outputs/camera-streams
```

Camera configs now accept an optional `parent_body` for head/torso/wrist mounting; positions and quaternions then use the parent frame. The HIL194 profile has two fixed views and one camera on the right wrist, with separate approximate references for each capture session. Use native resolution to preserve the dataset camera aspects.

The default HIL194 tip-box lid is connected by a passive hinge and can open beyond 180 degrees to rest on the table.

HIL194 ego views now use the corrected camera pose by default for both recording sessions, in `run.sh` and the camera-stream exporter. Use `--camera-pose config` to retain the selected config’s poses.
