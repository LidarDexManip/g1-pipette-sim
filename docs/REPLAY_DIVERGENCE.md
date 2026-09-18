# Diagnosing real-to-sim replay discrepancies

The new workflow separates controller tracking, the assumed wrist-to-pipette mount, contact grasp, and insertion mechanics while holding the selected episode layout fixed. It does not infer success from matching a lift height alone.

## Run the diagnostic comparisons

From the project directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
~/miniconda3/envs/lerobot-act/bin/python tools/diagnose_replay_divergence.py \
  --episodes 46 52 14 160 193 --workers 6 \
  --output outputs/hil194/replay-divergence-new

~/miniconda3/envs/lerobot-act/bin/python tools/report_replay_divergence.py \
  --inputs outputs/hil194/replay-divergence-new \
  --output outputs/hil194/replay-divergence-new-report
```

The output includes one frozen scene input, full simulation report, and sampled trajectory per episode and mode. `summary.json` records completion/errors. The report includes CSV comparisons and plots of the arm/tool error decomposition, insertion depth, radial error, and angle.

Modes:

| Mode | What changes | What it answers |
|---|---|---|
| `command` | Nothing; normal motor-target replay | What does the current simulator reproduce? |
| `state_targets` | Measured body positions become servo targets | Does changing targets help? This is **not** exact position playback. |
| `measured_arm` | Impose the 29 measured body joints and their interpolated velocities | Does the tool still diverge when body tracking error is removed? |
| `rigid_tool` | Use a wrist weld under command replay | How sensitive is the result to the contact grasp? |
| `measured_arm_rigid_tool` | Measured body motion plus compliant wrist weld | What remains with both errors reduced? |
| `compliant_tip` | Rack-retention weld rotational scale becomes 0.01 | Does allowing more tip rotation resolve insertion? |
| `effort_bias` | Add shared constant motor-torque offsets from the existing effort regression | Does controller correction improve held-out tracking? |

The torque offsets were fitted on episodes 46, 80 and 120. Episodes 160 and 193 are held out from that regression. Motor targets, torque limits and robot geometry are unchanged by the bias test. These offsets could include effort-sensor bias or unmodeled friction, so better regression error alone does not justify adoption.

Measured-arm playback injects unknown reaction forces. The hand still uses recorded commands because its measured feedback is frozen. A wrist weld is a diagnostic ablation, not a replacement for a calibrated contact grasp. These modes must not be exported as policy-training demonstrations or counted as normal commanded-action successes.

`config.json` files are scene inputs. Runtime interventions are specified by the mode and runner; passing a diagnostic scene input directly to `run.sh` does not reproduce the diagnostic mode.

## Refit placement after the controller hypothesis

Changing a controller can invalidate a layout previously fitted to its tracking errors. Test this confound separately:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
~/miniconda3/envs/lerobot-act/bin/python tools/test_bias_corrected_layouts.py \
  --episodes 46 52 160 193 --workers 4 \
  --output outputs/hil194/bias-corrected-layouts-new
```

This tests up to eight distinct placements per episode with full collision geometry and the existing insertion/contact/world-lift criteria. Contact-free traces are proposal heuristics only. It does not install experimental results or modify the pickup manifest; a promising result still requires independent validation, recorded-video review, and a supported production controller configuration. Use a new output directory for a different experiment.

## What the recorded images establish

[Episode 46 observations](../outputs/hil194/replay-divergence/recorded-event-review.json) record what is actually visible. The final sampled frame has a green tip on the lifted pipette. Near-rack frames do not unambiguously establish the first attachment instant. The [startup frames](../outputs/hil194/replay-divergence/episode046-startup-review.jpg) provide a qualitative grasp reference.

The diagnostic tool measures simulated deviation from the **configured** wrist-to-nozzle transform; it does not measure real 3D slip. Camera intrinsics/extrinsics and that transform remain approximate. The existing `tools/fit_ego_view_from_landmarks.py` can fit camera pose to supplied world/image landmarks, but a rack-only planar fit does not independently establish the robot-to-rack transform or actual nozzle trajectory. No reliable new camera/tool registration was established in this investigation.

The largest joint-FK rise is still only a reference estimate. Replacing it with observed tip motion requires calibrated views, visible tip landmarks and explicit handling of occlusion; success labels alone do not supply those measurements.

[Investigation results and plots](../outputs/hil194/replay-divergence-report/REPORT.md)
