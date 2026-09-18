# Proposed GitHub contents

Target: public `yi-shiuan-tung/g1-pipette-sim`. All `outputs/` are excluded.
See [QUICKSTART.md](QUICKSTART.md) for the tested portable setup.

## Files to include

| Path | Purpose |
| --- | --- |
| `run.sh` | Simulator launcher; invokes `python -m g1_pipette_sim`. |
| `requirements.txt` | Runtime dependencies. Includes Pillow for camera windows and illustration tools. |
| `README.md`, `docs/` | Installation, replay modes, cameras, layout fitting, augmentation, export, and known limitations. |
| `g1_pipette_sim/*.py` | Dataset reader, scene construction, dynamic and kinematic replay, attachment, mesh generation, cameras, recording, calibration. |
| `g1_pipette_sim/augmentation/` | Augmentation methods, reproducible run contracts, validation, failure diagnostics, storage, splits, and export. |
| `assets/robot/g1_inspire.xml`, `assets/robot/meshes/`, `assets/robot/provenance.json` | G1/Inspire model and 59 mesh files required to build the robot; about 26 MiB total. Preserve upstream asset provenance and applicable terms. |
| `prepare_assets.py` | Optional importer for the original robot model; unnecessary when robot assets are included. |
| `config/hil194.json` | Default dataset/session physics and scene configuration. |
| `config/camera-view-overrides.json` | Corrected camera poses. |
| `config/hil194-scene-anchors.json` | Per-episode image-based scene placement anchors. |
| `config/hil194-recorded-reviews.json` | Recorded-demonstration review metadata and default-selection exclusions. |
| `config/hil194-pickup-layouts.json` | Per-episode fitted-layout manifest and validation provenance; about 12.1 MiB; scene inputs are now under `config/pickup/`. |
| Other `config/*.json` | Earlier recording profiles and calibrated physics. Keep for reproducibility. |
| `config/augmentation/*.json` | Recipes controlling augmentation methods and amplitudes; historical recipes retain old seeds; bootstrap creates a portable local recipe. |
| `tools/run_augmentations.py`, `tools/run_augmentation_campaign.py` | Bounded recipe runner and larger collection campaigns. |
| `tools/illustrations/` | Lab scenes, rack/tip visual augmentation, joint perturbation presentations, and real-versus-sim comparisons. |
| Other `tools/*.py` | Layout fitting, audits, contact checks, failure clustering, mesh previews, dataset export, and training-data preparation. Keep helper modules: the augmentation package imports several of these scripts. |
| `tests/` | Existing regression tests. |
| `.gitignore` | Prevent committing outputs, generated meshes, reference media, dependencies, and credentials. |

The currently proposed existing source/config/robot/test/doc files total roughly
46 MiB before Git compression, including portable layout inputs.

## Files to exclude

- `outputs/`: everything, including all historical runs and generated datasets.
- `assets/objects/`: roughly 20 GiB of procedural mesh caches; generated on demand.
- `assets/Media/`: roughly 174 MiB of reference photographs/product pages; runtime
  mesh generation does not load them.
- `.analysis-deps/`, virtual environments, bytecode, test caches, credentials.
- Real datasets, model checkpoints, and simulation training exports: store outside
  Git, with dataset downloads and revisions documented separately.

## Portability provided

- `config/pickup/`: 194 selected scene parameter files (about 6.5 MiB); no recordings or captured trajectories.
- `config/hil194-data-identity.json`: metadata and parquet checksums for the pinned real dataset.
- `tools/setup_portable.py`: verifies data and physics, then binds layouts and camera references under ignored `.local/`.
- `tools/bootstrap_augmentation_seed.py`: generates a seed with physics screening and independent camera recording from the real trajectory; no archived output download needed.
- Lab/rack illustration scripts accept `--collection`; new accepted captures carry their own integrity receipts.
- `tools/index_illustration_collection.py`: creates a presentation index without modifying run progress.

Historical illustration launcher presets and older analysis tools still reference
previous experiments; use the explicit fresh-clone commands in QUICKSTART.md.
Saved pickup status uses historical attachment/lift acceptance; strict rack-contact
validation remains a separate audit and can reject otherwise accepted captures.
