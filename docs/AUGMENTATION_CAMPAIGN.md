# Separate placement/joint campaign

Run root: `outputs/hil194/augmentation-campaign-20260915-v1`.
The original `outputs/hil194/augmentation600` is a read-only source. It is not
modified or merged into the campaign's accepted catalog.

## Run and resume

```bash
/home/ytung34/miniconda3/envs/lerobot-act/bin/python -u tools/run_augmentation_campaign.py \
  --output outputs/hil194/augmentation-campaign-20260915-v1 --workers 4
```

Repeating the exact command resumes unfinished work after verifying the immutable
run contract and result receipts. Code, source, model, split, or recipe changes
require a new run directory. The campaign archives the relevant source code and
configuration in `code-snapshot.tar.gz`; external mesh and dataset dependencies
remain pinned by hash and must also be retained for replay.

The schedule contains 12 placement and 12 joint proposals for each of 12 real
parents. Parent and family are explicit fields, independent of modulo scheduling.
Joint directions are paired across 1- and 3-degree amplitudes. Every draw has a
fixed RNG seed, cell/offset, and phase window.

For this campaign, approach windows are frozen from each saved baseline's first
nozzle-to-tip-mouth proximity below 5 mm, with a 0.5-second guard. This is a
conservative geometric proxy rather than a contact detector. Episode 45 has no
usable window and its 12 joint proposals are skipped with a recorded reason.
Its 12 placement proposals still run: 288 scheduled, 276 eligible, 12 skipped.

The existing StarVLA preparation split is imported unchanged. Parents 14 and 68
are validation sources; their synthetic descendants are reported separately and
must not enter the training partition.

## Outputs and meanings

- `campaign.json`, `proposal-schedule.json`, `source-eligibility.json`: frozen
  original-video mapping, eligibility for all 194 originals, and complete recipes.
- `tasks.sqlite3`: durable ownership, outcome, and infrastructure retry counts.
- `runs/proposals/<id>/proposal-spec.json`: provenance even if generation fails.
- `runs/proposals/<id>/execution-*.json`: execution outcome, traceback if needed,
  elapsed worker time and peak RSS; infrastructure retries remain separate.
- `runs/candidates/<id>/`: saved commands/config, physical screens and independent
  recording results. Accepted captures have `accepted.json`, native camera videos,
  full trajectory, camera poses, and checksummed receipts.
- `reports/summary.md`, `.json`, `by-original-and-family.csv`, `by-original.csv`,
  `by-family.csv`, `proposals.csv`, `attempts.csv`: live attribution and counts.
- `accepted-catalog.json`: final campaign-only unique accepted entries, with
  duplicate disposition relative to the existing catalog. No data from the old
  collection is copied into this catalog.

A successful screen is not an accepted demonstration: independent recording and
video/numeric auditing must pass. Placement retries do not increase proposal
counts. Joint captures needing a placement adjustment retain the joint family and
carry a separate placement-adjustment flag.

No duplicated LeRobot export or training-copy videos are created by default.
The accepted catalog can later be passed to the existing immutable export and
training-copy tools. This avoids paying for additional video copies before they
are needed.

## Failure data for later clustering

For every completed physical screen, keep:

- `commands.npz` and `config.json`: exact dense targets and fixed scene.
- `screen.json`: physics outcome, failure reasons, contacts, attachment diagnostics,
  source timing, geometry fingerprints, and acceptance margins.
- `screen-state.npz`: sampled full qpos/qvel plus nozzle and tip-mouth position,
  tip-relative offset, angle, running force/penetration maxima, and attachment state.
  Sampling is 20 Hz, with a final sample at the termination time.
- `screen-alignment.npz`: pre-attachment alignment trace at about 200 Hz.
- `diagnostic-schema.json`: column definitions, units, and sampling caveats.

Successful screens retain the same diagnostic data for comparison. The sparse
states support diagnostic rendering; exact integration replay uses the dense
commands, original initial/reference state, and pinned simulator dependencies.
Failed-screen videos are not rendered automatically. A failed independent
recording may retain partial videos and a numeric failure state; its error is
classified separately from physical proposal rejection.

Final clustering artifacts:

- `reports/clustering-features.npz`: shape `(attempt, 64, 6)`, covering -3 to +2 s
  relative to the baseline attachment time. Channels are tip-local XYZ in mm,
  alignment angle in degrees, running maximum contact force in N, and penetration
  in mm.
- `valid` masks mark truncated/absent samples. Missing values are NaN, not zero.
- `clustering-index.jsonl`: separate outcomes, parent IDs, source-video references,
  phase/placement attempt identities, partitions, and paths to full diagnostics.
- `clustering-schema.json`: feature names and guidance.

Generation/preflight/infrastructure failures are retained in proposal records but
are not mixed into the physical trajectory feature tensor. Start any future
clustering with training-partition failures, normalize feature scales, and compare
with successful neighbors. Inspect representative trajectories and source-video
segments before deciding whether a cluster calls for new demonstrations, revised
sampling, or simulator corrections. No automatic demonstration requests are sent.

## Storage and later cleanup

```bash
/home/ytung34/miniconda3/envs/lerobot-act/bin/python tools/report_augmentation_storage.py \
  --root outputs/hil194/augmentation-campaign-20260915-v1 \
  --output outputs/hil194/augmentation-campaign-20260915-v1/reports/storage.json
```

The report gives logical bytes, allocated bytes, file counts, artifact categories,
and hard-link-aware estimates of reclaimable bytes. Cross-directory hard links
mean deleting one copy may free no space. Filesystem snapshots/reflinks can further
reduce the actual space freed. These estimates never perform deletion.

Preserve accepted camera observations, command archives, configs, failure states,
source references, source code/meshes, and receipts while replay/clustering matters.
Generated illustrations and redundant dataset exports are better cleanup candidates,
after verifying that retained source captures can regenerate them. Old failure
analysis folders need content review: they may contain useful diagnostic evidence.
Deleting receipt-listed artifacts intentionally makes that cache incomplete and
requires an explicit archived/purged workflow; do not silently resume it as intact.

Before-campaign inventories are in `outputs/storage-inventory-before-campaign.*`
and `outputs/storage-augmentation600-before.*`. They do not modify the old collection.

The active run also has a detached `supervisor.py` in its output folder. It updates
storage snapshots every five minutes and monitors the original coordinator. If
that coordinator exits before finalization, it attempts one resume through the
same contract/lock/retry checks. Inspect `supervisor-status.json` and
`supervisor.log` for `complete` or `needs_attention`. Creating a `STOP` file in the
run directory disables this automatic recovery; it does not itself terminate an
already-running coordinator. Disable the monitor before deliberately stopping the
process tree. The supervisor does not delete or upload data.
