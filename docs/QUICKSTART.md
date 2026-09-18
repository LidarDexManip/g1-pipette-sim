# Run from a fresh clone

Linux, Python 3.12, and MuJoCo 3.8.1 are the tested baseline. Install system packages
`ffmpeg`, `python3-tk`, and `fonts-dejavu-core` for video tools, the camera monitor,
and captions. Interactive playback needs a desktop; headless rendering uses EGL.

```bash
git clone https://github.com/yi-shiuan-tung/g1-pipette-sim.git
cd g1-pipette-sim
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export G1_SIM_PYTHON="$PWD/.venv/bin/python"
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Download the exact real dataset used by the saved layouts. Authenticate with
`hf auth login` first if access requires it. Dataset files stay in the HF cache,
not in Git.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jren313/g1-pipette-3view-hil194', repo_type='dataset',
                  revision='643ece1b2e93f15022c83832302848c9978d5f05')
PY
export DATASET="$HOME/.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hil194/snapshots/643ece1b2e93f15022c83832302848c9978d5f05"
python tools/setup_portable.py --dataset "$DATASET"
```

Setup verifies metadata and all episode parquet checksums, then writes local
path bindings under `.local/`. Physics fingerprints and scene-template checksums
must match. Object meshes regenerate on demand; reference photos are unnecessary.
Re-run setup after moving the checkout or changing the dataset location.

## Simulator

```bash
./run.sh --dataset "$DATASET" --episode 64
./run.sh --dataset "$DATASET" --episode 64 --physics kinematic --source state \
  --config .local/layouts/episode-064.json
./run.sh --dataset "$DATASET" --episode 64 --headless --compare \
  --output outputs/episode64-comparison
./run.sh --help
```

Dynamic replay consumes commanded motor targets. Kinematic/state playback sets
measured body-joint positions directly, with approximate hand/tool reconstruction.
Saved pickup layouts are historical attachment/lift fits, not certificates that
an episode passes the stricter no-rack-contact criterion.

## Generate augmentation seeds and new demonstrations

No historical `outputs/` files are needed. Bootstrap screens the original real
trajectory and independently replays it for camera/state recording. It refuses
to overwrite a previous collection or accept a failed replay.

```bash
python tools/bootstrap_augmentation_seed.py --dataset "$DATASET" \
  --episode 64 --output outputs/bootstrap
python tools/run_augmentations.py --config outputs/bootstrap/recipe.json \
  --output outputs/augmentations --workers 2
```

The generated recipe contains three rack/well configurations, a joint approach
perturbation, and an elbow pulse. Increase proposals and adjust recipes to collect
more variations; acceptance is not guaranteed. Method support includes `well_only`,
`local_cartesian`, `retarget_well`, `cartesian`, `joint_approach`, `elbow_pulse`, and
explicitly enabled `joint_persistent`. See [pipeline details](AUGMENTATION_PIPELINE.md).
Existing historical recipes under `config/augmentation/` retain their old input
paths; use the locally generated recipe for a fresh clone.

**Acceptance uses historical pickup/lift criteria.** Audit rack contacts before
labeling exported episodes as contact-clean training demonstrations. Visual-only
augmentation retains original action labels; action-changing data uses motor
joint targets and must match the deployment interface.

## Illustration videos

Lab surroundings can be rendered from the new seed without archived captures:

```bash
python tools/illustrations/lab_environments.py --collection outputs/bootstrap \
  --candidate 0 --resolution 4k --name lab-environments
```

The rack-placement presentation expects a collection with `progress.json` listing
accepted integer candidate IDs and `candidates/NNNNNN/` captures. The bootstrap
collection is a valid one-layout smoke test:

```bash
python tools/illustrations/rack_tip_placements.py --collection outputs/bootstrap \
  --candidate 0 --resolution 4k --name rack-placement
```

To visualize multiple generated placements, create a collection index with
`python tools/index_illustration_collection.py --collection outputs/augmentations`;
then pass that collection and an accepted candidate ID to the same command.
Choose captures from the same real parent episode.

Joint/elbow videos accept explicit capture paths. Select an accepted generated
candidate listed in the run's progress report; directories are numbered by
proposal and placement attempt (they are not consecutive demonstration IDs).

```bash
python tools/illustrations/capture_comparison.py \
  --base outputs/bootstrap/candidates/000000 \
  --capture outputs/augmentations/candidates/000300 \
  --title 'Joint Perturbation' --name joint-perturbation --resolution 4k
```

The example capture path is only valid if that attempt was accepted. Use `--zoom`
for a close-up. These presentation videos are not validated training rollouts.
`tools/illustrations/run.sh` preserves historical video defaults; its old joint
and elbow paths require those historical captures. Prefer the explicit commands
above on a new machine.

## Tests

```bash
python -m unittest discover -s tests -p 'test_pickup_layouts.py'
python -m unittest discover -s tests -p 'test_dataset_root.py'
```

Some historical integration tests require older real datasets or previous run
artifacts; those are intentionally excluded from the repository.
