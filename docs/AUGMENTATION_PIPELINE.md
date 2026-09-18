# Integrated augmentation pipeline

The original collections are read-only inputs. New runs, catalogs, simulation exports,
and training copies have separate versioned output paths. The original 600 captures
and illustration videos are not modified.

## Collection

```bash
PY=/home/ytung34/miniconda3/envs/lerobot-act/bin/python
$PY tools/run_augmentations.py \
  --config config/augmentation/smoke.json \
  --output outputs/hil194/integrated-smoke-final --workers 3

$PY tools/run_augmentations.py \
  --config config/augmentation/pilot-episode14.json \
  --output outputs/hil194/joint-cartesian-pilot-v1 --workers 4
```

The smoke recipe reproduces a baseline, the earlier 3-degree approach example,
and the 0.5-degree elbow pulse. The pilot recipe proposes 24 trajectories with
3/7-degree joint variations and 5-mm Cartesian translations. Each proposal has
at most three placement attempts; optional lift correction creates a separate
trajectory revision. Persistent offsets require explicit opt-in.

Supported methods: `well_only`, `local_cartesian`, `retarget_well`, `cartesian`,
`joint_approach`, `elbow_pulse`, and opt-in `joint_persistent`. Recipes specify
amplitude/direction, shift, cell, and explicit phase windows. The final composed
commands and scene undergo the same bounds check after refinement. Full physics
acceptance thresholds are unchanged, and successful screens are independently
replayed for recording.

As requested, the tested absolute timing is retained: approach ramp-in ends at
3 s, return spans 5–10 s, elbow pulses use approximately 10–12.6 s, and lift
correction starts at 14 s. Joint and elbow methods default to the parent-14 timing. Other parents require
explicit `approach_s` or `pulse_s` windows; episode 64 illustration recipes now
exercise this path. Fixed post-attachment lift refinement remains restricted to
parent 14. Well/local Cartesian methods can use other accepted parents. Automatic
phase inference and a multi-parent trajectory pilot remain deferred.

`progress.json` distinguishes proposals, generation rejections, physics attempts,
screen successes, independent recording acceptances, and infrastructure errors.
Placement attempts are not counted as new perturbations. The SQLite ledger owns
retry counts. Repeating the identical command resumes unfinished work. Changing
code, source data, recipe, runtime package versions, or model dependencies requires
a new output directory. Infrastructure failures return nonzero after bounded retries.

## Cache and interruption contract

- A coordinator lock prevents two coordinators from owning a run. Per-task locks
  also protect against a surviving worker from an interrupted coordinator.
- Run contracts bind deterministic recipes, RNG seeds, source snapshots/parquets,
  command archives, joint order, model files, mesh hashes, and implementation files.
- Replay restores motor targets and both saved hand command arrays. New archives
  include the initial observed state. Hand reconstruction/imputation provenance is
  retained; simulated finger state is a separate quantity.
- Result reuse requires content receipts. Every referenced file is hashed again;
  mtime and row-count matches are not considered evidence of integrity.
- A physics verification file is not a completed recording. Missing acceptance
  triggers a fresh recording attempt, even when physics verification succeeded.
- Acceptance commits only after videos decode completely and capture artifacts
  have hashes. A crash between acceptance and its receipt can recover by checking
  those commit-time hashes. Corrupt accepted data raises an error, not an overwrite.
- Uncommitted proposals are regenerated deterministically. Unversioned output
  directories cannot be adopted as new caches.
- Old experiment writers now fail with a migration message. Their numeric helper
  functions remain available to the shared registry and illustration tools.

Legacy import computes missing historical hashes **at import time** and labels
that limitation. It does not claim that previously unhashed bytes were historically
verified. No source files or acceptance markers are rewritten.

## Catalog and exports

```bash
$PY tools/export_augmentation_catalog.py \
  --collections outputs/hil194/augmentation600 \
    outputs/hil194/larger-augmentation-study \
    outputs/hil194/joint-perturbation-study \
    outputs/hil194/elbow-correction-study \
  --catalog outputs/hil194/augmentation-catalog-v1.json

$PY tools/export_augmentation_catalog.py \
  --catalog outputs/hil194/augmentation-catalog-v1.json \
  --output outputs/hil194/merged-v1/lerobot-v21

$PY tools/export_augmentation_catalog.py \
  --catalog outputs/hil194/augmentation-catalog-v1.json \
  --output outputs/hil194/merged-v1/lerobot-v3 --v3
```

Catalog entries have collection-independent IDs, full method/source metadata,
artifact hashes, and root-parent provenance. Exact semantic duplicates are removed
using commands, scene, clock, initial state, schema, and source identity. Nearby
trajectories are not considered duplicates merely because they look similar.

Publication builds a new staging directory, checks source integrity again, seals
all output files, and atomically renames the result. Videos are copied rather than
hard-linked to mutable sources. An existing export must match its entire receipt;
a growing catalog requires a new export version. v3 publication checks every
native episode's numeric labels, first/last camera frames, and action padding.

## GROOT / motor training copies

```bash
$PY tools/prepare_starvla_motor_mix.py \
  --catalog outputs/hil194/augmentation-catalog-v1.json \
  --split-manifest outputs/hil194/motor-parent-splits-v1.json \
  --output outputs/hil194/motor-training-v1 --workers 4
```

The builder accepts arbitrary catalog sizes and routes episode 1000 onward to the
correct chunks. It writes separate `real_train`, `real_val`, `sim_train`, and
`sim_val` datasets with `meta/modality.json`. Real-parent membership is frozen in
the split manifest and does not depend on the number of synthetic descendants.
All descendants of a held-out parent remain held out. The real source currently
has 194 episodes; counts are read from metadata, not hard-coded to 194 or 600.

Actions are **41 dimensions: 29 motor joint targets in radians plus two six-register
hand commands**. State is 29 measured body joints. This differs from the original
35-dimensional high-level action column. GROOT requires a matching custom
embodiment/modality configuration and a matching deployment interface; this builder
does not train or deploy a model.

For the local GR00T N1.7 checkout, multiple dataset paths are colon-separated.
`--ds-weights-alpha 0` gives equal weight to each dataset, so passing only
`real_train:sim_train` provides a 50/50 domain mixture. The builder's
`suggested-mixture.json` records a starting recommendation; it does not silently
configure a trainer. Keep validation paths out of the training mixture.

A 1,000-sim + 200-real **episode** experiment is reasonable. GR00T does not impose
a fixed demonstration-count target; training samples are temporal windows, and
batch size/sequence dimensions govern GPU memory. For 20-second episodes at 30 Hz,
1,200 episodes contain about 720,000 frames, with highly correlated overlapping
training windows. This is different from 1,200 individual frames.

Compare a real-only baseline, a 50/50 sampled real/sim mixture, and natural-frequency
sampling (roughly 5:1 sim/real if lengths match). These are proposed experiments,
not an established optimum. Measure held-out real task success. Repeated descendants
of 12 parents, or especially one parent, do not provide 1,000 independent task
contexts. Preserve held-out parents before generating or selecting augmentations.

References: [NVIDIA custom-embodiment fine-tuning guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md),
[NVIDIA GR00T repository/data format](https://github.com/NVIDIA/Isaac-GR00T), and
[NVIDIA's real-plus-synthetic training results](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/).

## Implementation verification — 2026-09-14

- 22 targeted tests passed (cache inputs/content, source channels/initial state,
  exclusive ownership, interruption recovery, bounded retries, frozen parent splits,
  retired writer rejection, and chunk routing).
- Final full-physics smoke: 3 proposed, 0 generation rejections, 3 physics screens,
  3 screen passes, 3 independent recording acceptances, 0 infrastructure errors.
  This is a baseline plus two known successful variants, not an unbiased acceptance-rate estimate.
- Earlier smoke regression against the saved examples had exactly zero difference
  in all recorded qpos and motor actions for the baseline, small joint example,
  and elbow example (642 frames each).
- Measured final motion relative to the baseline: joint example 3.057 degrees /
  16.119 mm maximum nozzle displacement; elbow example 0.448 degrees / 6.556 mm.
- A real filesystem recovery check removed one final recording receipt, resumed
  the run, and recovered the receipt with unchanged trajectory bytes.
- Final v2/v3 exports: 3 episodes, 1,926 frames. Native v3 audit checked all numeric
  labels, all episode boundary images, and future-action padding. Identical v3
  publication was subsequently reused only after receipt verification.
- Legacy catalog: **617 unique accepted captures, 436,729 frames, no exact
  duplicates**. Catalog source integrity was checked again after creation.
- Disposable full-builder fixture: 2 real + **1,001 simulation episodes**;
  simulation output has two chunks. The local GR00T N1.7 episode loader loaded
  episodes 999 and 1000, including all three videos and split action modalities.
  Identical export reuse passed; deliberately corrupted output was rejected.
  These fixture episodes are test data under `/tmp`, not demonstrations.

Evidence is under `outputs/hil194/integrated-smoke-final/`, including
`progress.json`, `catalog.json`, immutable v2/v3 publications, and
`lerobot-v3/meta/dataset-audit.json`. The combined legacy catalog is
`outputs/hil194/augmentation-catalog-v1.json`.

The 24-proposal exploratory pilot, full 617-capture merged video export, full
real/sim training conversion, and model fine-tuning have not been launched.
The implementation and commands support those next steps. New phase-dependent
methods remain restricted to parent 14 as agreed; broadening real-parent coverage
requires a subsequent phase-aware pilot.

Episode 64 illustration recipes: `config/augmentation/episode64-illustrations.json`. The comparison renderer accepts `--base` to keep the original and augmented clips aligned to the same real parent, and optional camera overrides for a clear tip close-up.

For the episode 64 placement illustration, run `tools/illustrate_pickup_augmentation.py --candidate 25 --seed 64 --swap-seconds 0.3 --name episode64-rack-tip-placements`. It cycles through the 43 accepted layouts while keeping baseline robot/pipette state continuous. Layout-source hashes are checked against the catalog. This visual composite is explicitly marked as presentation output, not a newly validated training rollout.
