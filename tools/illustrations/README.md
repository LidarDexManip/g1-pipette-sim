# Augmentation illustration videos

Run from the simulator repository. Outputs go to `outputs/augmentation-illustration/`.
The launcher selects the existing `lerobot-act` Python environment and EGL rendering.
Override Python with `G1_SIM_PYTHON` if needed. MuJoCo, NumPy, Pillow, OpenCV and FFmpeg are required.
These tools use the locally saved accepted captures; they do not collect new demonstrations.

| Video | Generator | Command |
|---|---|---|
| Lab environments, episode 64 | `lab_environments.py` | `./tools/illustrations/run.sh lab` |
| Rack/tip placement cycling, episode 64 | `rack_tip_placements.py` | `./tools/illustrations/run.sh rack` |
| Original/perturbed arm comparison, episode 64 | `capture_comparison.py` | `./tools/illustrations/run.sh joint` |
| Elbow correction close-up, episode 64 | `capture_comparison.py` | `./tools/illustrations/run.sh elbow` |
| Older multi-arm illustration, episode 14 | `arm_perturbations.py` | Run directly with simulator Python |
| Older blended scene comparisons, episode 14 | `scene_overlays.py` | Run directly; supports `--study`, `--single-max`, `--joint` |

The two older episode-14 generators may both write `arm-perturbations.mp4`; they are alternative presentations. The four launcher presets have separate output names.


All four episode-64 presets now render at **3840×2160, 30 fps** by default, with scaled text and native-resolution scene rendering. Pass `--resolution 1080p` to any preset for a smaller file. Centered display titles are **Visual Data Augmentation** (rack/tip), **Joint Perturbation** (both arm and elbow videos), and **Lab Environment Augmentation** (lab video). Episode numbers remain in filenames/provenance only. The elbow video omits the bottom “Elbow perturbation” status text, retaining timing, playback speed, and later attachment/return statuses.

## Lab environments

```bash
./tools/illustrations/run.sh lab
./tools/illustrations/run.sh lab --swap-seconds 0.5 --name lab-slower
./tools/illustrations/run.sh lab --duration 2.4 --name lab-preview
```

Default: candidate 25 (successful episode 64), 3840x2160 (4K), 30 fps, full 505-frame / 16.83-second capture, surroundings change every 0.3 seconds. Eight deterministic layouts vary shelf positions, angles, footprints, heights and tier counts using open steel shelves, wire racks, mobile carts and cabinets. Neutral steel, cabinet, cardboard and plastic material colors stay fixed across scenes. Warm, neutral and cool illumination varies the actual scene lights' direction, intensity and color together across the robot, task objects and furniture; there is no image tint filter or random furniture recoloring. Nearby shelves and consumables boxes remain visible in the policy camera strip. A mini centrifuge and two sample-tube racks also occupy the task table. These are visual-only surroundings; the original trajectory is preserved. `--seed` controls layouts/order. A closer presentation camera shows the robot and task table above a synchronized strip of ego, fixed-side and right-wrist cameras. The strip uses the configured policy observation cameras, including the current ego correction; it preserves each camera’s aspect ratio before policy preprocessing. The lab surroundings are rendered into all four views, with no state advance between them. Labels are outside camera images.

The robot, pipette, rack, tip, table and lid follow their original saved state trajectory without cuts or changes. Decorative geometry is added only to the renderer, so it has no collisions and cannot alter the task. This is a visual augmentation illustration, not a fresh dynamics validation or evidence of learned invariance. Training augmentation should render scene variants through the policy's three observation cameras and evaluate held-out environments.

Use `--resolution 1080p` for a smaller 1920x1080 video; 4K is the default.

Outputs: `.mp4`, midpoint `.png`, eight-layout contact sheet `-layouts.jpg`, per-frame world camera poses `-camera-poses.json`, and `.json` with source hashes, exact procedural layouts, swap schedule, and frame verification. Every MP4 frame is decoded before publishing the file.

## Other generators and compatibility

`capture_comparison.py` accepts `--base`, `--capture`, `--name`, `--title`, `--zoom`, `--start`, `--end`, `--repeat`, and camera controls. Run a preset with `--help` for its generator's arguments.

Original `tools/illustrate_pickup_augmentation.py`, `tools/illustrate_lab_environments.py`, `tools/illustrate_arm_perturbations.py`, `tools/overlay_augmented_videos.py`, and `tools/render_capture_comparison.py` remain compatibility entry points. Implementations now live in this directory. Generation/study/validation scripts remain in their existing locations; only presentation renderers were moved.

## Text-free real/simulation comparison

```bash
MUJOCO_GL=egl ~/miniconda3/envs/lerobot-act/bin/python tools/illustrations/real_sim_comparison.py
```

Rebuilds `outputs/episode14-comparison-ego-fixed/simulation.mp4` from original real camera footage and a fresh dynamic replay of the folder’s saved configuration. Default output is 3240×2160 at 30 fps, with 1080-pixel camera columns and original aspect ratios; square real views retain their native resolution. Camera order is ego, right wrist, fixed side; real views are above simulator views. No labels or label bands. Ego panels retain aspect-ratio padding. All frames are decoded and replay events, final state and contact diagnostics must match the saved report before replacing the video. `highres-render.json` records provenance and frame alignment. Use `--folder` for another saved dynamic comparison or `--cell-width` to choose another size.

## All original augmentation-parent baselines

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ~/miniconda3/envs/lerobot-act/bin/python tools/illustrations/original_baselines.py --workers 4
```

Replays original recorded `low_level` commands for episodes 6, 13, 14, 24, 30, 35, 40, 41, 45, 64, 68 and 144 using the fitted baseline configurations pinned by `config/hil194-pickup-layouts.json`. No augmented action files are used. Current corrected camera poses are applied without changing physical layouts. Saved full-replay event, final-state and contact diagnostics must match exactly before a video is published.

Outputs: `outputs/hil194/original-baseline-comparisons/episode-NNN/simulation.mp4`, per-episode provenance/verification records, `progress.json`, and a linked `index.html` listing baseline rack-contact verdicts. Videos use the same text-free 3240×2160 format above. Rerunning resumes verified video files whose config and video hashes match.

### Measured-position source comparisons

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ~/miniconda3/envs/lerobot-act/bin/python tools/illustrations/original_baselines.py --source state --workers 4
```

Writes a separate set under `outputs/hil194/original-baseline-comparisons-state/`. Uses `observation.state` as body-joint servo targets with dynamic physics and the original recorded hand commands. It does not impose measured poses kinematically; tracking and grasp discrepancies can remain. The original low-level comparisons are preserved. Layout, camera views, image timing and output resolution match the target-driven comparisons. `report.json` remains the original reference; the actual new state-replay report is saved inside `highres-render.json` as `replay_report`, with differences from the reference recorded explicitly. The browser's contact verdict is taken from the new replay, not inherited from the target-driven reference. Changing source is expected to change physics results, so only same-source rerenders require exact reference equality.

### Kinematic measured-position diagnostic

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ~/miniconda3/envs/lerobot-act/bin/python tools/illustrations/original_baselines.py --source state --physics kinematic --workers 4
```

Writes `outputs/hil194/original-baseline-comparisons-kinematic-state/`, leaving both dynamic sets intact. Prescribes measured body positions and uses the kinematic model’s fixed tool mount/hand approximation and geometric tip attachment. Existing fitted layouts and camera poses are held constant for this comparison; those layouts were originally optimized for dynamic motor-target replay, not independently reconstructed from images.

Each `kinematic-diagnostics.json` records body-joint agreement and signed pipette/ejector–rack collision-geometry distances at every rendered frame (30 Hz). Distance checks ignore normal collision masks and use conservative bounding spheres only to prune definitely distant pairs. Negative distances indicate sampled geometric overlap, not physical contact forces. Distances above 2 mm are capped. This sampling can miss intersections between video frames. Kinematic attachment is not a task-success validation. Real-video pipette landmarks have not been independently annotated; these videos support visual diagnosis, not a claim of measured reconstruction accuracy.
