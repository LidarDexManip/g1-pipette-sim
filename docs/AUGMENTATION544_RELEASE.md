# 544-episode augmentation release

Hub destination: `ytung/g1-pipette-augmented` (public).

Local corrected release: `outputs/hil194/augmentation544-ego-repair-v2b/release/hub-upload`.

Camera correction published as `v3.0.1` at commit
`a0f7fafec7971541852011ca37ab7a9b2c1d2772`; the `v3.0` alias also selects it.
The previous release remains available as `v3.0.0`. Re-rendered 513 ego videos
from full saved simulator states using the intended session camera overrides;
31 already-correct ego videos were preserved. Numeric Parquet data and both
other camera streams remain byte-identical in v2.1. Browser previews were also
updated and tested in Chrome. Camera fits remain approximate; physics success
verdicts are inherited from the original accepted demonstrations.

Download into a fresh directory to avoid mixing old and corrected videos.
The full portable bundle is 11,108,975,882 bytes (10.35 GiB).
Camera repair, loader, and remote upload audits are under the corrected release
root and its parent `augmentation544-ego-repair-v2b`.
The 544 episodes contain 389,076 frames: 467 training / 77 validation.
Parent groups stay disjoint. The original 600 captures are not included.

The repository root is LeRobot v3.0. Qwen-OFT compatibility copies live at
`compat/v21/sim_train` and `compat/v21/sim_val`. The root statistics are derived
from training episodes only. The supplied StarVLA adapter assembles 29 motor
angles plus 12 hand registers into 41 action channels, with 16-step windows.

See the release `README.md` for download, checksum verification, installation,
loader testing, and the training launcher. Do not train on the validation copy.
The main 29-D action column is distinct from the 41-D StarVLA adapter output.

Validation artifacts:

- `meta/dataset-audit.json`: all numeric frames, native v3 episode boundaries,
  all three camera views at each boundary, executed-command labels, action padding.
- `starvla-loader-audit.json`: actual v2.1 StarVLA loading for each split.
- `checksums.json`: all portable upload files.
- `release.json`: training-only statistics and sealed local export references.
- Local `upload-result.json`: immutable Hub commit and remote file verification.

The StarVLA smoke test uses temporary metadata copies to avoid cache writes into
sealed exports. No VLA model was fine-tuned during release preparation.
