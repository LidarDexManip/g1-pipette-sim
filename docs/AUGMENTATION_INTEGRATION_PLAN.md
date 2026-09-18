# Augmentation review and integration plan

Status: implementation is now tracked in [AUGMENTATION_PIPELINE.md](AUGMENTATION_PIPELINE.md). The review below is the pre-implementation baseline; fixed episode/timing assumptions were explicitly retained by user request. Reviewed the current local files on 2026-09-14. The workspace has no Git metadata, so this is a review of the current implementation, not a commit diff.

## Decision

Reuse the existing full-physics validation, capture format, camera recorder, and action contract. Move proposal generation, source loading, phase selection, placement refinement, and run management into a shared augmentation package. Keep the original 600 collection immutable and produce versioned, deduplicated exports that can include additional runs.

The new methods exist as experiment scripts, but are **not integrated into the main collector**. Do not scale by changing `--target` alone or by copying the experiment loops into the current collector.

## What exists today

| Collection / method | Independently accepted captures | Export state | Integration state |
| --- | ---: | --- | --- |
| Original collection: 64 well-only, 534 local Cartesian/action variants, 2 fixed-rack well retargets | 600 | v2.1 and v3; saved audit | Main collector, 12 source parents |
| Larger Cartesian IK translations with matched placement and optional lift correction | 6 | Separate v2.1; numeric audit | `search_larger_augmentations.py`, `tune_larger_lifts.py`; parent 14 only |
| Direct joint approach perturbations | 9 | Separate v2.1; numeric audit | `study_joint_perturbations.py`; parent 14 only |
| Elbow insertion pulses with planned realignment | 2 | Capture folders only | `study_elbow_correction.py`; parent 14 only |

There are **617 accepted capture markers across four collections**, not a unified 617-episode dataset. Seventeen are outside the original collection. Deduplication and provenance reconciliation must precede counting them as unique additions. The new experiments all descend from original accepted capture 0 / real episode 14.

Shared pieces already reused:

- `collect_pickup_augmentations.placed_config`, `smoothstep`, `validate`, and `RecordedSimulation`.
- `fit_pickup_layouts.rollout` and its full collision, initial-overlap, contact force/penetration, lift, finite-state, dwell, and final-attachment checks.
- Dense `commands.npz`, sampled/full-state `trajectory.npz`, three source camera videos, camera poses, screening and recording reports, and `accepted.json`.
- `package_pickup_augmentations.package` for the two larger experimental datasets.
- A better snapshot-pinned, three-command-channel loader in `search_larger_augmentations.exact_episode`, currently installed by monkeypatch rather than used by the shared replay path.

The visualization tools render or blend saved states. Their overlays and slow-motion frames are presentation artifacts and must never enter training exports as camera observations.

## Review findings

### R1 — Experiments are tied to one seed and fixed wall-clock phases (P1 for scaling)

Evidence: `tools/study_joint_perturbations.py:43-48,86`; `tools/study_elbow_correction.py:26-34`; `tools/search_larger_augmentations.py:42-43,69`; `tools/tune_larger_lifts.py`.

The experiments hard-code parent 14, source capture 0, joint slices, and absolute windows. Joint approach noise ends at 10 s; elbow pulses occur around 10–12.6 s; lift trimming starts at 14 s. Current manifest attachment times span **2.811 s (episode 45) to 25.2245 s (episode 40)**. Episode 35 attaches at 6.098 s. Reusing the current approach window on these seeds would perturb contact instead of just the approach; the elbow pulse could happen after task completion.

Required: resolve arm joints by semantic names, accept arbitrary approved seed references, and derive per-seed approach/contact/attachment/lift windows. Preserve the full contact interval, not just the final attachment timestamp. Explicitly skip a method when a seed lacks enough time for its envelope.

### R2 — Training and v3 export do not support growth safely (P1 for the dataset expansion)

Evidence: `tools/prepare_starvla_motor_mix.py:119,141`; `tools/watch_augmentation_export.py:21-37`; `tools/package_pickup_augmentations.py:81-85,104`.

The training builder reads only `augmentation600` and asserts exactly 600 captures. It cannot include the new runs. The watcher reuses an existing v3 dataset whenever its version is v3.0, even if the v2 collection has grown; it then overwrites custom provenance and can mark the mismatched export complete. Candidate integers are only local to a collection, so merging manifests by `candidate_number` would collide. Exported provenance also omits important method details such as joint direction, envelope, placement-attempt identity, and lift correction, relying on absolute local source paths.

Required: collection-independent capture IDs, explicit multi-run dataset manifests, versioned exports with content/count checks, full method lineage in exported metadata, and an arbitrary-size training builder. Freeze real-parent train/validation membership; do not re-sample it when the set of augmented parents changes. Audit chunk boundaries when the combined dataset exceeds 1,000 episodes; the current training-copy writer always uses chunk 000.

### R3 — Existing experiment files bypass cache/provenance validation (P1 before unattended reuse)

Evidence: `tools/study_joint_perturbations.py:37-42,93-102`; `tools/search_larger_augmentations.py:62-68,83-88`; `tools/study_elbow_correction.py:20-25,36-40`.

If a proposal/candidate file exists, preparation returns it before validating the current recipe or model. If a screen/acceptance file exists, evaluation returns it without invoking the exact loader. Changing a seed, amplitude schedule, source command archive, or implementation while reusing an output directory can silently return the old candidate/result. Recorded study provenance is documentation, not an enforced run contract. Experiments also lack the main collector's exclusive lock; overlapping runs can write the same candidate or summary. Several experiment drivers log exceptions but still exit successfully, so automation cannot infer completeness from exit status.

Required: enforce immutable run and candidate fingerprints before any cache return; use transactional task ownership and atomic publication; distinguish infrastructure errors from physical rejection; return an explicit incomplete/failure status when work errors remain. Record the sampler version as well as the random seed.

### R4 — Exact replay fix is not in the shared loader (P2)

Evidence: `tools/collect_pickup_augmentations.py:74-80`; `tools/search_larger_augmentations.py:22-30`; `tools/study_joint_perturbations.py:101`; `tools/replay_pickup_augmentation.py`.

The original loader still opens the current default dataset and restores only motor targets. Experimental workers monkeypatch in `exact_episode`, which restores both hand arrays and pins the snapshot, but ordinary replay and the original collector do not use it. This is particularly important given the original taped-grasp collection and reconstructed/imputed hand commands: preserve archived commands and explicit grasp provenance rather than treating changed hand labels as interchangeable.

Required: one strict loader for generation, validation, replay, and export. Verify the source snapshot/file hashes, joint order, clocks, initial observed state, all three command arrays, configuration, model assets, and relevant code versions. Archive enough initial/reference metadata to avoid silently depending on mutable dataset state. Keep an explicit compatibility path for early captures with missing historical hash fields; newly computed import hashes are not historical verification evidence.

### R5 — Interrupted recordings can be skipped permanently (P2)

Evidence: `tools/collect_pickup_augmentations.py:100-105,225-227`.

`verification.json` is written before trajectory saving and video auditing. Restart queues a successful screen only if verification is absent. A failure after verification but before acceptance therefore leaves a successful candidate that is never recorded again. The experiments retry when acceptance is absent, but do not provide a shared durable task state or concurrency protection.

Required: accepted output is committed only after every required artifact is present and audited. A successful physics report is not a completed capture. Resume incomplete recordings; retain terminal physical failures; retry bounded infrastructure failures. Preserve candidate identity and append-only export identity across restarts.

### R6 — Shared validation and preflight are implicit, and method metadata is incomplete (P2)

Evidence: `tools/collect_pickup_augmentations.py:87-94`; `tools/study_joint_perturbations.py:53-56,79-89`; `tools/study_elbow_correction.py:27-35`; `tools/tune_larger_lifts.py:19-29`; `tools/package_pickup_augmentations.py:104`.

Recording replaces the module-global simulation factory without restoring it. Current one-task-per-child workers contain this side effect, but a reusable library or persistent worker pool would carry it into later calls. Proposal, refinement, and composition checks differ by script: joint candidates do not repeat `on_table` after refinement, elbow generation has no shared finite/limit/velocity preflight, and composed lift corrections need final-trajectory bounds and metadata recomputed against the root seed. Testing only an incremental correction is insufficient for a chain of transformations.

Required: inject the simulation/recording factory explicitly; apply common validation to the final composed commands and scene after every transformation; version acceptance criteria separately from proposal bounds; persist both individual transforms and root-relative commanded/measured deviations.

### Test coverage and scope

Eight existing augmentation/publication/dataset-root tests passed during this review. `tests/test_pickup_augmentation.py` covers placement algebra and smoothstep, not the new generators, caching, interruptions, phase generalization, or multi-run exports. The saved study reports and accepted markers were inventoried; the review did not rerun all physical trajectories or re-decode the original 600 videos.

## Which methods to promote

| Method | Evidence | Initial rollout policy |
| --- | --- | --- |
| Well-only and local Cartesian + small null-space variation | Original 600 across 12 parents | Preserve as baseline strategies |
| Joint approach perturbation | 9/12 approach proposals accepted after placement fitting; 3°, 7°, 12° tested on parent 14 | Primary new strategy for a multi-parent pilot; conservative per-seed amplitudes, zero perturbation before contact |
| Larger Cartesian IK translation | 6 accepted captures from 50 screening candidates; 5–10 mm translations; no tested 20 mm success | Secondary strategy with bounded placement/lift refinement |
| Fixed-rack alternate-well retargeting | Two original successes | Retain as low-budget exploratory strategy |
| Elbow pulse near insertion | 3/18 screened successes, 2 independently recorded; 0.4–0.5° recorded | Opt-in, small per-seed quota; report as planned realignment, not reactive recovery |
| Persistent random joint offset | 0 successes among 6 simulated / 12 proposed persistent variants | Disabled in the default production mix; keep only as an explicitly requested experiment |

The joint study proposed 24 trajectories: 6 generation rejections, 18 physics-screened, 9 independently accepted, and 38 placement tests. Different random directions were sampled at each amplitude, so the results are not a controlled amplitude sweep. All new-method evidence is from one real parent. Do not extrapolate a global acceptance rate or claim more independent task coverage from these pilot counts.

## Proposed shared architecture

Introduce `g1_pipette_sim/augmentation/` with a small typed API, leaving CLI scripts as thin entry points:

| Module | Responsibility |
| --- | --- |
| `types.py` | Seed reference, named joint schema, phase windows, proposal recipe, placement attempt, immutable command archive, capture/result manifests |
| `sources.py` | Pinned source loading, archived motor/hand commands, initial state and clock verification, legacy adapters |
| `phases.py` | Derive/override valid approach, contact, attachment, and lift windows per seed |
| `methods.py` | Registry for well-only, legacy local Cartesian, larger Cartesian IK, joint approach, elbow pulse, and opt-in experimental methods |
| `placement.py` | Coupled rack/tip proposals and bounded refinement, including explicit table-height changes |
| `validation.py` | Common final preflight and unchanged physics acceptance with explicit simulator factory |
| `recording.py` | Independent replay, native camera capture, trajectory serialization, full video audit, atomic acceptance |
| `store.py` | Local SQLite task ledger owned by the coordinator, immutable manifests/artifacts, leases/recovery, globally unique capture identity |
| `runner.py` | Parent/method scheduling, budgets, deterministic proposal assignment, isolated worker execution |
| `export.py` | Multi-run catalog, stable episode mapping, v2/v3 dataset publication and audit |

Keep the current simulator and physics implementation. Initially retain process isolation per expensive rollout. Reuse `RecordedSimulation` behavior after moving its factory dependency behind an explicit interface. Renderer utilities consume accepted manifests and never call proposal generation or change success state.

The core flow is:

```text
approved seed + phase windows + deterministic recipe
  -> complete command trajectory
  -> initial scene proposal
  -> shared command/scene preflight
  -> full physics screening
       -> bounded placement retry with identical commands
       -> optional new trajectory revision (e.g. lift trim), with separate identity
  -> independent recording replay
  -> full video + numeric audit
  -> atomically accepted capture
  -> immutable dataset manifest / versioned export
```

Use separate identities for `root_seed`, `proposal`, `trajectory_revision`, `placement_attempt`, `capture`, and `export_episode`. A placement retry is not another perturbation. A lift trim changes commands and is a new revision. All descendants retain the same real-parent split group, including augmentations derived from already augmented captures.

A capture manifest should contain at least:

- Schema version, run ID, source collection/local ID, stable capture ID, root real dataset/episode, immediate parent capture, and complete transformation lineage.
- Snapshot and file hashes, model/mesh fingerprints, joint and hand schema, initial state/reference metadata, dense command clocks, and all command/config/trajectory/video hashes.
- Method and sampler versions, RNG seed, amplitudes/directions, resolved phase windows, placement and lift corrections, and final command bounds.
- Validation policy/version, screen and recording outcomes, contact/lift margins, recording completeness, and actual joint/nozzle deviation.
- Export identity and method labels sufficient for filtering and reporting without access to absolute local source folders.

## Ordered implementation plan

### 1. Shared contracts and replay correctness

Extract the exact loader and validation interfaces; replace monkeypatches; define immutable source/proposal/result schemas and common final preflight. Preserve the 29-radian-motor-target plus two six-register-hand-action contract. Track taped/reconstructed hand provenance independently from simulated finger state.

Acceptance criteria: all archived command channels are restored; changing any source/config/command/validation dependency invalidates reuse; named joint reordering either maps explicitly or fails; the original seed and a fixture from each new family replay with the same verdict and trajectories within a documented numerical tolerance on the pinned environment. No acceptance threshold is relaxed.

### 2. Durable collection and global identity

Implement the coordinator ledger and atomic capture commit. Add deterministic proposal IDs, bounded retries, per-candidate errors, run contracts, and stable export assignment. Import legacy acceptance markers through a read-only adapter rather than rewriting the original folders. Exact duplicate fingerprints include canonical commands, initial/reference state, and physical scene; exclude incidental paths/timestamps. Similar-but-distinct trajectories are tracked separately with diversity metrics rather than silently discarded.

Acceptance criteria: interruption during command writing, screening, video encoding, trajectory saving, video audit, acceptance commit, or coordinator update can be recovered without losing or double-counting a candidate. Concurrent workers cannot own the same attempt. A corrupted accepted artifact is detected and quarantined instead of silently replaced. A stale screen is not reused after a recipe change.

### 3. Integrate proposal families and per-seed timing

Port legacy strategies unchanged first, then larger Cartesian IK and direct joint approach envelopes. Integrate placement refinement and lift revision as explicit transformations. Add elbow pulses behind an opt-in method flag and small budget. Keep persistent joint offsets disabled by default.

Resolve a protected contact window from each baseline's saved replay/contact information, with explicit overrides when automatic phase inference is uncertain. Scale approach envelopes to the available pre-contact interval; scale insertion pulses relative to contact/attachment anchors; scale lift corrections relative to attachment and lift onset. Skip methods that cannot fit their window or command bounds. Apply preflight after the final composition, including initial scene overlaps and footprint checks after placement refinement.

Acceptance criteria: phase-envelope tests cover early attachment (episode 45), medium timing (14), and late attachment (40); approach corrections vanish throughout the protected contact/lift interval; hand channels and untargeted joints remain unchanged; joint and Cartesian methods retain distinct semantics; every revised trajectory passes the same full physics validation before recording.

### 4. Catalog and dataset growth

Build one catalog over the 600 legacy captures and 17 experimental accepted captures. Reconcile provenance, validate integrity, and deduplicate before publishing a merged dataset. Preserve original episode IDs in the existing dataset; a new version can have its own deterministic mapping. Do not just increase the original collector's target after its implementation fingerprint changes.

Make v2 packaging and v3 conversion consume an explicit immutable accepted-capture manifest. Build each export in a staging directory, verify counts/content and native-loader boundaries, then publish it atomically. Rebuild or reject stale v3 output instead of overwriting its provenance. Replace all exact-600 assumptions in `prepare_starvla_motor_mix.py`, make source catalogs explicit, fix chunk routing beyond 1,000 episodes, and persist parent-level split assignments.

Acceptance criteria: an export with more than 600 captures loads; one crossing the 1,000-episode chunk boundary loads correctly; local IDs repeated across runs do not collide; all numeric labels match saved captures/dense commands; v3 camera frames and future-action padding are correct at every episode boundary; no parent or descendant crosses train/validation groups; growing the catalog does not change existing split assignments.

### 5. Multi-parent pilot, then scale

Run a bounded pilot across all 12 currently approved parents rather than producing more descendants only of episode 14. A suggested pilot budget is 12 joint-approach proposals per eligible parent (3 conservative amplitudes x 4 draws), plus 8 Cartesian proposals per eligible parent (two magnitudes x four directions). Short or phase-ambiguous seeds can be skipped with reasons. Add a small elbow-pulse pilot on a few parents only after phase validation. Placement retries have their own per-proposal budget and do not inflate the proposal count.

Report proposed, generation-rejected, physically screened, placement attempts, first-screen passes, post-refinement passes, independently accepted, failed recordings, and exported counts separately. Break results down by parent, method, amplitude, and phase. Measure accepted captures per compute-hour, video/storage cost, actual motion/scene diversity, validation margins, and source-parent balance. Run a small separately labeled neighboring-parameter robustness check for any method promoted beyond the pilot; deterministic replay alone is not robustness evidence.

Promote methods only after demonstrated success on multiple parents and verified artifacts; do not infer generalization from one aggregate percentage. Then set a configurable larger milestone—for example **1,200 total unique accepted captures**—with per-parent caps and method quotas, maintaining fixed validation criteria. This is a proposed milestone, not a launched collection job or acceptance forecast. Broaden the eligible parent set as more real demonstrations obtain verified reconstructions; multiplying the current 12 parents does not add new task coverage.

## Required tests before a large run

- Archive round-trip for all three command arrays, snapshot/initial-state pinning, and mutation rejection before cache reuse.
- Named-joint mapping, finite/range/velocity checks on final composed commands, exact preservation of unaffected joints and hand channels.
- Deterministic generation across worker counts and restarts; separate trajectory revisions versus placement attempts.
- Early/late phase windows and seed-specific protected contact intervals; invalid windows produce explicit generation rejections.
- Rack/lid/tip consistency and footprint/preflight checks repeated after refinements.
- Real filesystem interruption/recovery tests around acceptance and export transactions; no duplicate IDs, stale cache reuse, or dropped successful recordings.
- A small full-physics regression set containing successes and physical failures from the original, Cartesian, joint-approach, and elbow families, plus an opt-in multi-parent smoke run.
- Multi-run export, chunk-boundary behavior, persistent parent splits, native v3 frames/action padding, and exact numeric-label audit.

## Suggested delivery batches

1. Contracts, exact replay, explicit validation interfaces, and regression fixtures.
2. Durable ledger/acceptance recovery, stable IDs, and legacy import adapter.
3. Method registry, per-seed phases, bounded refinement, and configurable CLI.
4. Multi-run export, v3 freshness, training-builder growth, and fixed split manifest.
5. Pilot configuration, acceptance/diversity report, then the larger collection run.

The first concrete implementation should be batches 1–2. They make every later method safe to reuse and resume, while preserving the evidence in the existing collections.
