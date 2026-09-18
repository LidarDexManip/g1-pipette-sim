# GR00T checkpoint interface and camera streams

For the current HIL194 dataset, see [the dataset audit and action-stage limits](HIL194.md). The checkpoint-specific discussion below concerns older piston models and does not establish the HIL194 training or deployment contract. Its `modality.json` selects `action` labels; motor replay uses `action.low_level`. The local TWIST2 real server can convert a 35-dimensional mimic goal through a learned controller into 29 motor targets, so direct joint-target replay must not be mistaken for that control pipeline.

## Verified local N1.7 metadata

Inspected `~/gr00t_checkpoints/g1-inspire-starvla-n1d7-wristcam/` (README, config.yaml, config.full.yaml and training data registry) and the config for `~/gr00t_checkpoints/starvla-piston-full-finetune/`. Both configs select `unitree_g1_piston_wristcam_n1d7`, absolute actions, a 30-dimensional action and 30-step horizon. This was a metadata/source-code audit, not an inference run. The original piston training dataset's raw modality mapping was not available locally, so its exact historical source-column provenance has not been independently reconstructed.

The wristcam checkpoint contract lists:

- Cameras in order: `video.ego_view`, `video.left_wrist_view`, `video.right_wrist_view`.
- State: left arm 7, right arm 7, left hand 6, right hand 6, waist 3 = 29.
- Action: left arm 7, right arm 7, left hand 6, right hand 6, base height 1, navigation 3 = 30.
- Absolute arm joint targets, not Cartesian end-effector commands or torques. Action values are unnormalized by the serving pipeline.

The local deployment bridge packs these targets into a whole-body-controller goal. `InterpolationPolicy` interpolates/rate-limits the targets; `G1DecoupledWholeBodyPolicy` copies the resulting upper-body targets into the full command and combines them with lower-body commands. The arm path does not require an additional Cartesian IK conversion. Thus "joint-position targets" is correct, but treating the complete policy output as the pipette dataset's 29-element `action.low_level` motor vector is incorrect. Preserve chunk scheduling, interpolation, hand conversion and lower-body behavior when integrating the policy.

For the current pipette dataset specifically, `meta/modality.json` maps arm action slices to `action`, not `action.low_level`. Selecting low-level labels for a new fine-tune requires an explicit data mapping change. The old piston policy's hand conventions must not be assumed to equal the pipette dataset's raw 0–1000 registers. The existing single-camera pipette training example also uses a different action/state layout; it is not an automatic drop-in for this 30-dimensional checkpoint.

The second camera's *key* says left wrist. That does not establish its physical mounting. Identify the live streams before assigning ego/side/right-wrist roles. If the actual view changes for the new fine-tune, keep the new three-view contract identical in real data, synthetic data and serving. Checkpoint architecture alone does not determine action or camera semantics.

## Export clean streams

From the simulation project directory:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.camera_streams \
  --dataset ~/Datasets/g1-pipette-3view-20260819-plus21 \
  --episode 53 --config config/episode53-calibrated.json \
  --duration 5 --fps 30 --output outputs/camera-streams
```

This writes separate `rgb.mp4`, `wrist_left.mp4`, `wrist_right.mp4`, first-frame PNGs, and `streams.json` with simulation timestamps and camera-to-world poses. Videos have no HUD. These are the existing **uncalibrated, world-fixed** presets, not automatically the physical cameras implied by the legacy names. The default 640×480 is a preview resolution, not a claim about the policy's native imaging geometry.

In a simulation/policy loop, use arrays directly:

```python
# For headless use, set MUJOCO_GL=egl before importing mujoco.
from g1_pipette_sim.camera_streams import CameraStreams

with CameraStreams(sim.model, names=("rgb", "wrist_left", "wrist_right"),
                   width=640, height=480) as cameras:
    images = cameras.render(sim.data)
    # images[name]: uint8 RGB, shape (480, 640, 3), owned copies, no overlays
    timestamp = float(sim.data.time)
```

Call this on the simulation thread. Do not step physics between rendering the views. For different native view dimensions, instantiate one CameraStreams per camera at its calibrated resolution and render each from the same unchanged sim.data. Use the checkpoint's exact image preparation and camera order afterwards. A 224×224 network input does not imply rendering a square camera frustum: reproduce the native projection/crop before resizing. No ZMQ/ROS publisher or GR00T policy connection is implemented by this helper.

## Place cameras correctly

Each camera config accepts `parent_body`. If omitted, `position` and `quaternion` are world-relative, preserving old configs. If present, they are relative to that MuJoCo body. A calibrated right-wrist camera should normally use `right_wrist_yaw_link`; the ego camera should use its actual head/torso mounting link. The side camera can remain world-fixed if the real camera is fixed.

MuJoCo camera coordinates are +x right, +y up, -z forward; quaternions are wxyz. OpenCV optical coordinates are +x right, +y down, +z forward. Convert orientation with `R_mount_camera_gl = R_mount_camera_cv @ diag(1,-1,-1)`; translation is the same camera origin. The camera pose evolves as `T_world_camera = T_world_mount @ T_mount_camera`.

Recommended acquisition:

1. Identify each physical stream using serial numbers and temporarily covering one lens at a time. Record actual resolution, crop, lens settings and capture timestamps.
2. Print a ChArUco board on a rigid flat surface and measure its square size. Calibrate each camera's intrinsics/distortion using roughly 20–30 sharp views with varied tilts, distances and coverage. This count is a practical starting point, not a guarantee.
3. For the fixed side camera, establish the board pose in the robot/workspace frame and estimate camera pose by PnP. Seeing a board only gives camera-to-board pose; it does not automatically establish robot-to-board pose.
4. For the right-wrist camera, keep the board fixed and collect roughly 20–30 robot poses with rotations around multiple axes. Save synchronized images and measured joint states. Compute wrist-to-base poses with FK and use hand–eye calibration to estimate camera-to-wrist. Translation-only motion is insufficient. If the robot base moves, include that motion or keep it fixed during calibration.
5. Treat a moving head/torso ego camera as another body-mounted camera. A fixed pose estimated in one robot posture will not remain correct when that link moves.
6. Validate on images/robot poses excluded from fitting by projecting board corners, rack landmarks and tool features. Report pixel residuals *and* physical alignment near the tip. Check timing and motion-dependent error, not just one attractive overlay.

`g1_pipette_sim.calibrate_camera` currently handles known-intrinsics **fixed-camera PnP**, including image rectification metadata. It does not calibrate intrinsics or solve hand–eye calibration. Its simulated projection is centered and square-pixel: the real stream must use the matching rectification/resolution, or the renderer must be extended to reproduce the full raw intrinsics/distortion. Crop/resize changes must be reflected in the intrinsics.

References: [OpenCV calibration and hand–eye API](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html), [ChArUco calibration](https://docs.opencv.org/5.0/tutorials/objdetect/aruco_calibration/aruco_calibration.html), [MuJoCo Python rendering](https://mujoco.readthedocs.io/en/latest/python.html).

## Validation

A one-second smoke test exported 30 decodable 640×480 frames for each of three cameras. Two tests confirm that a body-mounted camera follows its link while a world camera remains fixed, and that invalid mount names fail clearly. No real camera calibration has been inferred or claimed from these tests.
