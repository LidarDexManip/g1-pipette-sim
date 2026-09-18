# HIL194 failure recordings and clustering

The original all-episode fitted-layout audit remains in `outputs/hil194/pickup-layouts/results.csv`. Its 16 `verified_pickup` episodes are excluded from this failure analysis; all other **178 episodes** are included. Failure means the simulator did not meet the saved attachment and world-lift criteria, not that the real-world demonstration failed.

Outputs are in `outputs/hil194/failure-analysis/`:

- `recordings.json`: resumable recording progress, per-video SHA256, decoded frame count, and discrepancies versus the published replay.
- `episodes/episode-NNN/comparison.mp4`: complete 30 fps H.264 real/simulator three-view comparison. Real views are above simulator views. Columns are ego, physical right wrist, fixed side. Dataset camera key names are retained in the video.
- Each episode directory also retains `workspace.json`, `scene.xml`, `report.json`, `final.png`, `replay.log`, and `recording.json`.
- `input-manifest.json`: frozen original replay validation, fitted config paths, and model/asset fingerprints.
- `failure-features.csv`: all 178 episode IDs, numerical features, outcome categories, precise gate categories, and cluster assignments.
- `clustering-candidates.json`: every tested hyperparameter configuration and its complete labels, including rejected/degenerate partitions.
- `clustering-summary.json`, `cluster-profiles.json`, `preprocessing.json`, `clustering-features.npz`: method scores, scaling and stability results, representatives, and raw/preprocessed feature matrix.
- `index.html`: browser with method/cluster/outcome/episode filters and video player.
- `without-gate-fractions/`: independent clustering run excluding gate failure fractions to assess their effect.

## Run or resume recordings

```bash
cd ~/Projects/g1_pipette_sim
export G1_SIM_PYTHON="$HOME/miniconda3/envs/lerobot-act/bin/python"
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$G1_SIM_PYTHON" tools/record_failed_replays.py --workers 8
```

Eight workers were used after a four-worker pilot. Reduce worker count on machines with less RAM. The renderer uses EGL and FFmpeg must be installed. Completed video checksums and input signatures are checked before reuse. All frames are decoded and duration is checked before marking a recording complete. Each fresh replay's event sequence, final state and contact diagnostics are compared against its matched saved report; discrepancies are recorded rather than hidden. Do not run two recorders against the same output directory concurrently.

The recorder uses normal `run.sh` behavior: the manifest's fitted layout, low-level motor commands, recorded image clock fallback, dynamic contacts and supported pelvis. No policy inference, new object fitting, grasp changes or insertion-threshold changes occur. Camera-view corrections are applied by the standard CLI.

A subset can be recorded into a separate output directory:

```bash
"$G1_SIM_PYTHON" tools/record_failed_replays.py --episodes 46 52 193 \
  --output outputs/hil194/failure-review-subset --workers 3
```

## Compare clustering methods

Dependencies are isolated in `.analysis-deps` to avoid changing simulator packages:

```bash
"$G1_SIM_PYTHON" -m pip install --target .analysis-deps --no-deps \
  scikit-learn==1.7.2 joblib==1.5.2 threadpoolctl==3.6.0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$G1_SIM_PYTHON" tools/cluster_replay_failures.py
```

The existing environment supplies NumPy, SciPy and Matplotlib. Four methods are evaluated: K-means, Ward hierarchical, diagonal Gaussian mixtures, and DBSCAN. K-means/Ward select by silhouette, GMM by BIC, and DBSCAN by silhouette times coverage. Partitions need at least five members per cluster and at least 50% coverage. DBSCAN noise has label `-1`; its silhouette excludes noise and is therefore not directly comparable with full-coverage silhouette alone.

The default feature blocks are insertion geometry, pre-attachment gate-failure fractions, grasp/controller deviations, and tool contacts. Nonnegative magnitudes are log-transformed and standardized. Each block is weighted by the inverse square root of its nonconstant feature count. Episode IDs, recording session, success/failure labels, lift target/error, and fitted object positions are excluded from clustering. Outcomes are used only afterward to interpret clusters.

Every selected method is tested with twenty 80% subsamples; adjusted Rand index (ARI) compares the new partition to the corresponding full-data labels. This is stability of the clustering with fixed preprocessing, not an estimate of causal validity. Raw robust scaling is also tested; the strong dependence on scaling should remain visible in any interpretation.

Repeat with a fixed cluster count:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$G1_SIM_PYTHON" tools/cluster_replay_failures.py --k-min 4 --k-max 4 \
  --output outputs/hil194/failure-analysis/k4
```

DBSCAN does not use `k`; it retains its epsilon/min-samples sweep. Remove potentially exposure-biased gate fractions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$G1_SIM_PYTHON" tools/cluster_replay_failures.py --feature-set without-gate-fractions \
  --output outputs/hil194/failure-analysis/without-gate-fractions
```

## Review

```bash
"$G1_SIM_PYTHON" -m http.server 8000 --bind 127.0.0.1 \
  --directory outputs/hil194/failure-analysis
```

Open http://127.0.0.1:8000. The episode browser updates recording progress every 30 seconds. The alternative feature analysis is at `/without-gate-fractions/`. Files become available as the resumable batch completes.

## What can and cannot be concluded

The exact outcome breakdown is **82 lift shortfalls, 5 lift overshoots, and 91 non-attachments**. The reference lift is a recorded-joint FK estimate with an assumed rigid mount and fixed pelvis, not measured 3D tip motion.

For the 91 non-attachments, full-replay gate counters show:

| Conditions never satisfied | Episodes |
|---|---:|
| Radial and depth | 39 |
| Radial only | 19 |
| Depth only | 16 |
| Radial and angle | 5 |
| Radial, depth and angle | 4 |
| Depth and angle | 1 |
| Each condition passed at some time, but not together for the required dwell | 7 |

These are exact failures of the simulator's necessary attachment conditions: radial <0.7 mm, depth 10–12 mm, angle <5 degrees, continuously for 100 ms. They do not establish whether the underlying problem is camera registration, rack placement, grasp/mount, tracking, or inaccurate insertion physics.

The 14-feature K-means/Ward results mainly divide low versus high gate-failure exposure. Seven-component GMM partitions provide finer candidate groups but have weaker separation and stability. Excluding gate fractions yields a different three-group K-means/Ward partition. No unique, robust causal taxonomy has been established. Review representatives and use controlled interventions before treating any cluster as a common physical cause.

Caution: the saved `closest_approach` minimizes 3D nozzle-to-mouth distance. It is not the best insertion-gate match; being near zero insertion depth there does not prove the nozzle never reached the required insertion depth later. Gate fractions are counted only while the tip remains in the rack, so early attachment changes their observation window. Simulated grasp displacement is not a measurement of real tool slip.

References: [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html), [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html).

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$G1_SIM_PYTHON" -m unittest discover -s tests -p test_failure_analysis.py -v
"$G1_SIM_PYTHON" tools/audit_failure_recordings.py
```

The final audit refuses partial recording sets, checks all video SHA256 values, matches generated versus decoded frame counts, and checks every clustering-to-video link. It writes `completion.json` only after all requested failed episodes are present. Full frame decoding occurs during each recording; the final audit checks those certificates without repeating every decode.

The maximum servo-error feature covers all body servos; it is not a right-arm-specific error. Joint-resolved traces would be needed to localize a tracking cause.

## Corrected ego camera default

The simulator now applies `config/camera-view-overrides.json` by default, including when an explicit `--config` is passed. Episodes 0–45 use the corrected episode-14 ego pose. Episodes 46–193 use the corresponding behind-hand pose fitted from that session’s episode-46 rack landmarks. These are approximate planar fits with an assumed 60-degree vertical field of view, not independently measured camera extrinsics. Both were checked against real images for rack visibility and arm direction. Side and wrist poses are unchanged.

Use `./run.sh --episode 46 --camera-pose config` to preserve the camera poses stored in the selected configuration instead. The default is `--camera-pose corrected`.

All 178 failed-episode comparison videos have been regenerated under the same `failure-analysis/episodes/` paths. The final audit verified all 178 rendered ego poses and 117,943 frames (65.52 minutes), with zero replay discrepancies. The previous recordings and certificates are retained in `outputs/hil194/failure-analysis-old-ego/`. Each new certificate records the correction-file hash and applied ego pose; changing the correction invalidates the recorder’s resume cache. The episode browser uses video hashes to avoid stale browser playback. This regeneration affects the failure-review videos; existing exported training datasets are not regenerated.
