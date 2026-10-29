# LIDAR pipette pickup on a B200

This package implements real-data behavior cloning, a privileged-state PPO
teacher, teacher-guided recurrent ego-camera student training (DAgger), and visual
PPO with a separate real-data BC loss. It is independent of the old Qwen tip
attachment experiments and does not import RLinf, starVLA, or VLAPolicyBridge.

The GPU path uses MuJoCo Warp 3.8.1 and a parity-tested batched TorchScript TWIST2
controller. The CPU path is a reference/debug implementation. Local tests are not
B200 throughput or sim-to-real validation. Run the GPU test and benchmark below
on the actual server before committing a large compute allocation.

## Remote requirements

- Linux x86-64, Python 3.10 or 3.11, a B200 exposed to CUDA, and an NVIDIA driver
  compatible with CUDA 12.9 (R575 or newer is the recommended baseline). Warp
  1.12.0 bundles CUDA 12.9, while the selected PyTorch wheels bundle CUDA 12.8.
  Use an administrator-provided working driver; this
  repository does not change drivers. `nvidia-smi` must work inside the training
  environment/container. The B200 is compute capability 10.0.
- PyTorch 2.7.1 / torchvision 0.22.1 CUDA 12.8 wheels and Warp 1.12.0 (CUDA 12 wheel), then the pinned physics and
  vision packages in `requirements-lidar-ppo.txt`. These wheels support Blackwell.
  A full CUDA toolkit is not needed for the normal pip installation. Keep the
  environment separate from existing robot inference installations.
- Fast local storage for real videos, prepared BC shards, encoder weights, and
  checkpoints. Budget at least 100 GB initially; the supplied dataset alone is
  about 17.2 GB and additional datasets increase this. Prefer 128 GB host RAM;
  actual needs depend on workers, shard sizes, and simulation batch size.
- The real dataset(s), including Parquet, MP4, and `meta/` files. Do not copy only
  the five reproduced episodes. Hugging Face authentication is needed if access
  is restricted; authenticate normally, never commit tokens.
- The portable scene bundle exported below: scene XML, all referenced meshes,
  controller, hand mapping, reset snapshots, and checksums. It contains the exact
  local experimental scene and must be transferred separately from Git.
- Network access or a local Hugging Face cache for `facebook/dinov2-base`.
  Training checkpoints record the resolved encoder revision. Use that same
  revision at inference; exported actors do not contain the encoder weights.
- For CPU camera diagnostics, EGL/OpenGL runtime libraries. GPU training uses
  Warp's renderer and does not require a desktop/X server or Blender.

```bash
git clone --branch feature/pick-up-ppo git@github.com:LidarDexManip/g1-pipette-sim.git
cd g1-pipette-sim
python3.10 -m venv .venv-lidar-ppo
source .venv-lidar-ppo/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lidar-ppo.txt
python -m lidar_ppo preflight --device cuda:0 --vision
python -m pytest -q tests/test_lidar_ppo.py tests/test_lidar_ppo_pipeline.py
```

The GPU test checks CPU/Warp dynamics and contact reductions on a small fixture.
It must pass on the server. Then validate the actual LIDAR model: mesh-heavy
contact capacity and rendering cost are scene-specific. The CLI raises on contact
or constraint overflow rather than silently training through truncated contacts.
The exported current scene has 12,337 geoms and 65 velocity DoFs: do not infer
thousands-of-worlds throughput from benchmarks of simple locomotion scenes.

## Choose a writable data directory

Run this in each terminal used for the commands below, on both the workstation
and server. The default is inside your home directory and needs no administrator
permissions. You can instead select a writable server scratch/data mount.

```bash
export LIDAR_DATA_ROOT="$HOME/Datasets/g1-pickup"
mkdir -p "$LIDAR_DATA_ROOT"
```

`/data` is not required. Do not use `sudo` to run the exporter or training.

## Export the current local scene once

Run this on the workstation that has the current LIDAR replay checkout, dataset,
and TWIST2 checkpoint. The exporter imports that checkout explicitly, captures its
physical scene, and rewrites all mesh paths into the bundle. It does not depend
on committing the source checkout's unrelated working changes. It requires the
export dependencies from `requirements-lidar-ppo.txt`.

```bash
python -m lidar_ppo export-bundle \
  --source-repo /home/ytung34/Projects/g1_pipette_sim \
  --dataset /home/ytung34/Projects/g1_pipette_sim/.cache/huggingface/hub/datasets--jren313--g1-pipette-2view-teleop0925-5task-eerel/snapshots/c7adfc179e5e15ade41b06803542001a24959dda \
  --controller-root /home/ytung34/Projects/TWIST2 \
  --episodes 28 108 30 352 357 \
  --output "${LIDAR_DATA_ROOT}/lidar-bundle" &&
python -m lidar_ppo audit-bundle --bundle "${LIDAR_DATA_ROOT}/lidar-bundle"
```

The audit runs only if export succeeds. A failed export does not produce a usable
`checksums.json`; fix the export error before running the audit.

Use a fresh output directory for each export. Copy `$LIDAR_DATA_ROOT/lidar-bundle` to the
remote computer with `rsync`/`scp`; keep the same layout. The remote training
process does not need the original TWIST2 checkout or ONNX model once the bundle
exists. The bundle is a data artifact and is intentionally not committed to Git.

`audit-bundle` writes `reset-audit.json` and fills the physically qualified reset
indices in `review.json`. Episode 108 is reserved for simulation validation,
matching the real-data session split below. Training requires valid ordinary
starts in BOTH splits, not just successful late-grasp snapshots.

**Current scene limitation:** the original five replay successes include initial
penetrations and large contact forces. The default 3 mm penetration / 100 N hand
force limits are initial engineering screening values, not measured robot limits.
The existing experimental platform is 4 cm below the measured platform. A lift
label alone does not certify a valid reset. If the audit rejects ordinary starts,
correct scene registration/contact geometry and re-export; do not simply loosen
the thresholds to reproduce historical success. BC pretraining can run while
this calibration work is performed. `benchmark` explicitly uses diagnostic resets
and its outputs must not be reported as qualified pickup performance.

Verify the bundled `egoview` camera against the actual ego video before visual
training. Its inherited mount/intrinsics are approximate. The config renders
320x180 (16:9, matching the 1280x720 real video), then letterboxes the full frame
to 224x224. It never treats the overview camera or wrist image as the ego view.

## Download and prepare all real pickup demonstrations

```bash
hf download jren313/g1-pipette-2view-teleop0925-5task-eerel \
  --repo-type dataset --revision c7adfc179e5e15ade41b06803542001a24959dda \
  --local-dir "${LIDAR_DATA_ROOT}/teleop0925"
python tools/make_lidar_bc_manifest.py \
  --dataset "${LIDAR_DATA_ROOT}/teleop0925" --output "${LIDAR_DATA_ROOT}/real-pickups".json
python -m lidar_ppo prepare-data --manifest "${LIDAR_DATA_ROOT}/real-pickups".json \
  --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" --output "${LIDAR_DATA_ROOT}/pickup-bc"
# Recommended: encode real images once while DINO is frozen.
python -m lidar_ppo cache-features --data "${LIDAR_DATA_ROOT}/pickup-bc" --output "${LIDAR_DATA_ROOT}/pickup-features"
```

The manifest generator selects recorded successes of the pickup task using dataset
task metadata, independently of simulator results. It yields 69 training and 24
validation episodes for the pinned dataset, holding out session `20260924_2016`.
These are dataset labels, not independently reannotated success labels. The
source's older generic task text is not used for selection.

Use `$LIDAR_DATA_ROOT/pickup-features` instead of `$LIDAR_DATA_ROOT/pickup-bc` in the training commands
below to avoid repeated real-image encoding and decompression. Cached features
are FP16 and carry the exact encoder revision; mismatched caches are rejected.
Recompute features if the encoder or image preprocessing changes. Online simulated
images still pass through the encoder on each visual rollout.

To add datasets, add entries to the manifest; see
`config/lidar_ppo/real-data.example.json`. The implemented adapter supports LeRobot
v2 per-episode Parquet/MP4 and the 29-joint + 6-root + 6-hand-register command
contract. Different embodiments, EE-relative actions, units, or LeRobot v3 storage
need an explicit conversion. The loader rejects mismatched joint order/schema.
Do not concatenate unrelated task actions into an unconditioned pickup policy.
Sessions are globally named to avoid accidental train/validation overlap.

Nonmonotonic command timestamps in the supplied data use the explicitly declared
`paired_image_rows` alignment. This preserves existing replay semantics but does
not recover sensor/command latency. Frames are selected by recorded image times;
body velocities are causal differences. Hand commands are used instead of unreliable
feedback. Measured hardware velocities and latency still need parity validation.

The action is seven right-arm increments relative to the previous COMMANDED
reference, plus six absolute normalized hand targets. The arm scale is 0.04 rad
per 40 ms; hands have an 0.08-full-range per-step slew limit. Demonstration targets
outside the arm action range are masked, not silently relabeled as achievable.
These are starting command envelopes, not hardware-certified speed limits.

You can mask only unwanted finger targets while keeping valid arm targets:

```json
{"episode": 28, "session": "20260924_1614", "task": "pipette_pickup",
 "exclude_bc": [{"start_s": 4.0, "end_s": 4.4, "channels": [7,8,9,10,11,12]}]}
```

This interval is a FORMAT EXAMPLE, not a label for episode 28. Review actual
videos before using exclusions. Idle rows receive reduced weight. Sampling is
balanced by episode, and sequences never cross episode boundaries. The first half
of each real sequence reconstructs recurrent state without a loss.

## Train

All commands use new output directories unless `--resume` is supplied. Logs are
JSONL; checkpoints contain optimizer state, configuration, and provenance.
`--updates` means ADDITIONAL updates when resuming. Resume starts fresh simulation
episodes; it is optimizer continuation, not bitwise continuation of an interrupted
physics rollout. Choose checkpoints using evaluation, not training reward alone.

```bash
# Real-data pretraining can start before simulation reset calibration is complete.
./run_lidar_ppo.sh bc --data "${LIDAR_DATA_ROOT}/pickup-bc" --output runs/bc --updates 10000

# Benchmark before scaling; this measures physics+controller, not vision/learner.
./run_lidar_ppo.sh benchmark --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" --num-envs 256 --steps 100
./run_lidar_ppo.sh benchmark --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" --num-envs 1024 --steps 100

# PPO state teacher: 25 Hz actions, 50 Hz TWIST2, 1 kHz physics.
./run_lidar_ppo.sh teacher --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" --num-envs 1024 \
  --output runs/teacher --updates 1000
./run_lidar_ppo.sh evaluate --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" \
  --checkpoint runs/teacher/latest.pt --num-envs 128 --episodes 100 \
  --output runs/teacher/evaluation.json

# Student visits its own states and gets teacher labels, plus 25% real-BC loss.
./run_lidar_ppo.sh distill --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" --data "${LIDAR_DATA_ROOT}/pickup-bc" \
  --teacher runs/teacher/latest.pt --student runs/bc/best.pt --num-envs 128 \
  --output runs/student --updates 1000

# Optional visual PPO refinement; separate real BC batch on each optimizer step.
./run_lidar_ppo.sh visual-ppo --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" --data "${LIDAR_DATA_ROOT}/pickup-bc" \
  --student runs/student/latest.pt --num-envs 128 \
  --output runs/visual-ppo --updates 500
./run_lidar_ppo.sh evaluate --bundle "${LIDAR_DATA_ROOT}/lidar-bundle" \
  --checkpoint runs/visual-ppo/latest.pt --num-envs 128 --episodes 100 \
  --output runs/visual-ppo/evaluation.json
./run_lidar_ppo.sh export-policy --checkpoint runs/visual-ppo/latest.pt \
  --output exports/pickup-v1
```

Start with `--updates 2` to check each stage. Compare 256/512/1024/2048 teacher
worlds and 32/64/128/256 visual worlds. More worlds are not always faster. Frozen
DINO patch features are pooled to a spatial 2x2 grid; only those features enter
the PPO buffer, avoiding storage of all raw rollout images. Encoder inference
uses BF16 on CUDA; controller, PPO probability ratios, rewards, and physics stay
FP32. The visual actor is a 256-unit GRU. Recurrent PPO minibatches retain sequence
order and reset hidden states at episode boundaries. True terminal transitions
do not bootstrap; time limits bootstrap the terminal observation before reset.

The state teacher uses a privileged MLP. The visual PPO critic also uses simulator
state, but the deployed actor receives only ego RGB, body state, and command
history. Old real demonstrations are NEVER inserted into PPO rollouts. The BC
loss and simulator PPO loss use separate datasets and provenance.

Current randomization covers friction, servo gains, initial arm references,
proprioceptive noise, and image brightness. Object geometry/mass, camera extrinsics,
lighting/backgrounds, and hardware delay calibration are not automatically solved
by this release; use independently calibrated bundle variants and extend measured
randomization ranges before making sim-to-real claims. A frozen pretrained encoder
does not remove the contact or dynamics gap. The teacher is fixed-base only.

Use separate processes on different B200s for seeds/ablations:
`CUDA_VISIBLE_DEVICES=1 ... --device cuda:0`. This release is single-GPU per run;
it does not implement DDP or pretend that multiple GPUs are used automatically.
Pin a different `seed` in a copied config for each run. Keep output directories
and validation sets separate from training artifacts.

The exported `lidar_ppo.policy.PickupPolicy` is an inference interface, not a
robot transport. Integrate its action contract with the existing TWIST2 host,
verify command/observation timing and camera calibration, and evaluate physical
pickup retention before deployment. No real-robot commands are sent by this repo.

References: [PyTorch installation](https://pytorch.org/get-started/locally/),
[Warp installation/driver compatibility](https://nvidia.github.io/warp/v1.12/user_guide/installation.html),
[MuJoCo Warp](https://mujoco.readthedocs.io/en/3.8.0/mjwarp/),
[DemoStart](https://arxiv.org/abs/2409.06613),
[DINOv2](https://arxiv.org/abs/2304.07193).
