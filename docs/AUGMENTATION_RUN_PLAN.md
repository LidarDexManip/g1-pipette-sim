# Rack/tip randomization and joint perturbation campaign

Proposed run plan, 2026-09-15. This document does not launch collection.

## Objective and attribution

Generate new independently validated simulator demonstrations, then report yield
by augmentation family **and original real video**. Preserve source datasets and
old runs; publish new immutable outputs. Retain explicit per-episode timing rather
than introducing automatic phase inference for this campaign.

Two primary families:

1. **Placement-only:** keep the source motor and both hand command streams exactly
   unchanged; randomize rack/well selection and small paired rack/tip XY offsets.
   Selecting another active well can move the rack around a nearly stationary tip.
   Include explicit XY-offset proposals to exercise actual tip position changes.
   Begin with offsets inside +/-0.5 mm; consider the previously explored 1.5 mm
   range only after observing success. A previously validated range for IK-assisted
   augmentation is not a guarantee that every placement-only draw succeeds.
2. **Joint approach:** smooth additive right-arm joint perturbations with a planned
   return before contact. Allow bounded fixed placement adjustment to obtain a
   successful replay. Label these as joint-family captures with a separate
   `placement_adjusted` flag; never count the same capture once in each family.

Keep elbow insertion pulses, persistent joint offsets, larger Cartesian IK shifts,
vertical table changes, and lift revisions out of this initial campaign. Elbow
correction can be a separately budgeted family later; combining it with approach
perturbations would obscure their different success rates.

All source camera observations must come from actual recorded simulations.
Placement-swap illustrations, overlays, and slow-motion videos are excluded.

## 1. Freeze the eligible source registry and train/validation assignment

The original 600-capture collection has ready recorded seeds for 12 real parents:
6, 13, 14, 24, 30, 35, 40, 41, 45, 64, 68, and 144.

| Real episode | Existing original-collection captures | Candidate baseline | Attachment (s) |
| ---: | ---: | ---: | ---: |
| 6 | 65 | 4 | 17.7145 |
| 13 | 65 | 5 | 15.6915 |
| 14 | 21 | 0 | 12.8515 |
| 24 | 50 | 6 | 15.0445 |
| 30 | 70 | 7 | 17.6195 |
| 35 | 54 | 20 | 6.0980 |
| 40 | 38 | 9 | 25.8040 |
| 41 | 72 | 10 | 17.1135 |
| 45 | 11 | 47 | 2.8075 |
| 64 | 43 | 25 | 8.8725 |
| 68 | 30 | 50 | 17.9505 |
| 144 | 81 | 3 | 14.1425 |

These are historical counts from augmentation600, not new campaign results.
The layout manifest also lists successful validations for parents 2, 5, 7, and
121. Audit their complete validation/model/source evidence and independently
record approved baseline captures before considering them eligible. Do not assume
that a passing manifest entry already supplies all required captured artifacts.
The remaining originals must remain visible in the eligibility report, with an
explicit unavailable/failed/not-reviewed reason rather than disappearing.

For each selected seed, pin dataset snapshot, original episode ID, source video
paths, baseline capture, all hashes, hand-label provenance, validation version,
and explicit approach/contact windows. Identify parent by snapshot plus real
episode ID, never by collection-local candidate number.

Inspect each source near contact to set a conservative perturbation return window.
Early episode 45 and late episode 40 cannot reuse episode 14 timing. Mark a joint
method ineligible if its available approach is too short for the requested bounds.
Record eligibility separately for each family.

Reuse an existing frozen parent split if one has already been adopted. Otherwise
create it once before selecting training sources. All descendants of a real parent
stay in that parent's partition. Hold-out sources can receive evaluation-only
augmentation quotas, but their captures do not count toward a training-data target.

Reconcile all historical catalogs, including episode64 studies and smoke replays,
with semantic deduplication. Report imported existing captures separately from new
unique additions. Do not add acceptance-marker counts to estimate dataset growth.

## 2. Close the campaign orchestration and reporting gaps

The current runner picks source and recipe with the same `number % count` pattern.
With 12 sources and two methods this gives some parents only one method. Replace
this with an explicit schedule of `(source, family, amplitude, draw)` jobs, or use
one immutable run per parent/family and a campaign coordinator. The latter also
makes per-parent phase windows and bounded resumption straightforward.

Generate deterministic placement recipes: changing only the RNG seed does not
change a fixed `well_only` recipe. Materialize every cell, XY draw, joint direction,
amplitude, and phase window in the immutable proposal manifest. Exclude identity
proposals and duplicates before expensive replay when their semantic fingerprints
already exist.

Persist campaign, original-parent, family, draw, source, and recipe identifiers
for **every** proposal, including generation failures and infrastructure errors.
The current progress summary provides aggregate results only and loses useful
attribution for early failures. Add stage timestamps, explicit failure reasons,
resource usage, first-screen outcome, placement retry outcomes, independent-record
outcome, capture ID, and dedup/export disposition. Assign one proposal ID across
placement attempts and infrastructure retries.

Retain existing run contracts, byte-hash verification, task locks, bounded retry
counts, immutable acceptance receipts, and fresh versioned export directories.
Never reuse changed-code runs as collection caches. Fail on corruption and preserve
artifacts for diagnosis. Distinguish successful budget completion from reaching an
accepted-capture target; the current `target` field is merely the accepted count.

## 3. Run an equally budgeted pilot

For each eligible original video, propose:

- **12 placement-only variants:** four alternate well/layout draws and eight small
  XY-offset draws, with explicit unique cells/offsets. Keep all commands unchanged.
- **12 joint variants:** six fixed random directions, each tested at 1 degree and
  3 degrees, using that parent's phase window. Pair directions across amplitudes
  so amplitude comparisons are meaningful.

For 12 fully eligible sources this is **288 proposals** (144 per family), not
288 guaranteed demonstrations. Missing eligibility reduces the executed budget
and must be reported. Use the same maximum of three fixed-placement attempts per
proposal in either family; at most **864 screening rollouts**, plus one independent
recording replay per chosen successful proposal, excluding bounded infrastructure
retries. Report the actual count and retry cost separately.

Begin with two workers and one small batch to measure CPU/GPU RAM, simulation time,
rendering time, and disk use. Increase to four isolated workers if resources allow.
Use measured cost to set the remaining campaign wall-time/storage budget. Do not
estimate cost from video duration alone.

Each accepted example must pass the unchanged full-physics gates: finite state,
command/scene bounds, initial overlap/contact constraints, attachment timing,
lift/dwell/completion, independent replay, numeric labels, and full video decoding.
A screen pass alone is not a successful augmentation.

Stop/pause on checksum corruption, recurring infrastructure failures, or a resource
budget limit. Do not turn infrastructure errors into physical failures, silently
reroll them as new proposals, or loosen acceptance criteria to meet a quota.

## 4. Produce the pilot report before scaling

Deliver machine-readable JSON and CSV plus a Markdown summary. Report at three
levels: per original video x family, per family totals, and per original video
totals. Include a parent-by-family heatmap of acceptance rate with counts alongside
it; a high percentage from a tiny sample should not look like strong evidence.

Recommended detailed row:

`source_dataset, original_episode, original_video_paths, partition, family,
status, proposed, eligible, duplicates_before_replay, generation_rejected,
distinct_proposals_screened, physics_attempts, first_screen_passes,
passes_after_placement_refinement, recording_attempts, recording_failures,
independently_accepted, duplicate_acceptances, new_unique_accepted,
exported_train, exported_validation, infrastructure_errors,
compute_hours, output_GB`

Compact report table:

| Original video | Placement proposed | Placement accepted | Joint proposed | Joint accepted | New unique additions |
| --- | ---: | ---: | ---: | ---: | ---: |
| One row per original, including eligibility/skipped status | | | | | |
| Total | | | | | |

Define both rates explicitly:

- End-to-end yield = independently accepted / scheduled proposals; additionally
  show skipped/ineligible proposals so denominator changes are visible.
- Conditional physics yield = proposals eventually passing physics / distinct
  proposals physically screened. Placement retry count is not this denominator.

Also report new-unique yield after deduplication, and separately show first-attempt
versus placement-refined success. A perturbation that passes after two adjustments
is **one successful augmentation**, not three. If exact-identical captures arise
in two families, count their proposal successes in both family process metrics,
but assign the unique exported capture once and retain both lineage references.

Break down joint outcomes by amplitude and direction pair. Record commanded and
measured joint deviation, nozzle displacement, rack/tip XY displacement, contact
force/penetration margins, attachment/lift margins, and source balance. Review a
small representative set of successes and failures from every source/family.

## 5. Scale based on yield and coverage, then publish

If the earlier 1,000-simulation-episode objective still applies, use **1,000 unique
accepted episodes in the intended training partition** as a provisional milestone;
explicitly distinguish this from 1,000 new episodes or 1,000 including hold-out.
Compute the additional requirement from the reconciled catalog and frozen split.
The accepted target is not a promise that a fixed proposal budget will reach it.

Give each eligible parent/family a minimum quota and cap contributions from easy
parents. Use pilot yields to allocate extra attempt budgets while preserving
coverage. Prefer adding approved originals over producing hundreds more siblings
from one source. If a source/family has little success, diagnose it or mark it
limited; do not hide it by filling the entire dataset from the easiest sources.

Only after the conservative pilot, consider a separately labeled 6-degree joint
batch on parents that support it. Keep a small neighboring-parameter replay check
separate from accepted examples to assess whether success is fragile.

Publish a new immutable, deduplicated catalog and native dataset export. Verify
labels, video alignment, episode boundaries, action padding, frame totals, source
hashes, and >1,000-episode chunk routing. Preserve the common motor-plus-hand action
schema and exclude privileged diagnostics from policy inputs. Report train and
validation counts separately. Demonstration success in simulation is not evidence
of real-world policy improvement; subsequent VLA evaluation needs held-out real
videos and, ultimately, real task rollouts.

## Deliverables

- Frozen eligible-source registry and unchanged parent split.
- Campaign specification plus deterministic per-parent/family proposal manifests.
- Resumable run folders containing complete attempt logs and accepted captures.
- Per-source/family counts, yield, failure, cost, and diversity report (JSON/CSV/MD).
- New unique accepted catalog, audited export, and provenance-preserving training
  copies if included in the chosen campaign budget.

Implementation changes and collection execution are next steps; neither is
performed by writing this plan.
