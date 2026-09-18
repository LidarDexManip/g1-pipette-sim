# Why rack placement alone does not reproduce every pickup

The fitted layouts use the recorded low-level targets without changing the grasp or insertion thresholds. A matching lift is a useful numerical check, but it is not sufficient evidence of matching object dynamics. See [the results](PICKUP_RESULTS.md) and the per-episode CSV for the measured outcomes.

## Grasp and tool mounting

The original baseline's median maximum pipette translation relative to the hand was 23.61 mm, and median maximum rotation was 16.29 degrees. A rack translation cannot generally compensate for a changing wrist-to-nozzle transform. The insertion angle tolerance remains 5 degrees.

Diagnostic free-space tests found that preloading the fingers before replay did not resolve the motion: the sampled episode-46 test ended with about 23.2–23.6 mm translation and 11.7–12.9 degrees rotation. The temporary initialization constraint was inactive throughout replay. Increasing tested grip/friction settings also did not establish a stable calibrated grasp.

Changing the initial tool offset by [-8, -4, -21] mm reduced a diagnostic displacement to 2.12 mm, but still left 8.35 degrees of rotation. This demonstrates sensitivity to mounting; it does not identify the real transform. These experimental parameters were **not adopted**. Results are in `outputs/hil194/pickup-layouts/mount-sensitivity.json`, `mount-angle-sensitivity.json`, `grasp-sensitivity.json`, `rolling-sensitivity.json` and `pregrasp-initialization-probe.json`.

The useful next measurements are the wrist-to-nozzle transform in the actual closed grasp, physical finger angles for the recorded registers, and tool motion relative to the palm. Frozen hand feedback cannot supply this information. An independently calibrated multi-camera tool pose or a rigid marker on the pipette would make this discrepancy observable.

## Registration, reference lift and contacts

The lift reference is measured-joint forward kinematics with a fixed pelvis and assumed tool mount. It is not triangulated pipette/tip motion. Sampled IMU-tilt sensitivity changed reference heights by up to about 2.2 mm in the inspected examples; unmeasured base translation and tool motion remain separate uncertainties. Those diagnostic transforms were not adopted.

Camera intrinsics/crops, camera-to-robot transforms, rack pitch, tip bore and insertion depth should be measured in the same setup. Fitting object placement to an executed trajectory can hide registration error. Contact screening rejects large penetration and normal forces, but it does not identify real friction, insertion force or tip retention mechanics.

## Sequence and command discrepancies

Episode 104 visibly contains a human tip removal that is absent from the robot command stream. Episodes 5, 7, 8 and 81 have simulated attachment before 0.2 seconds despite a visibly empty real pipette later. These are documented in `config/hil194-recorded-reviews.json` and excluded from automatic example selection. Video review is sampled, not exhaustive annotation of all episodes.

The current replay consumes `action.low_level`. The VLA training action can instead represent high-level mimic goals, which the real deployment controller converts into motor targets. Preserving both interfaces, controller timing and the real camera assignments is necessary before using simulated demonstrations for policy fine-tuning. The fixed-base replay and approximate grasp do not yet validate that complete pipeline.
