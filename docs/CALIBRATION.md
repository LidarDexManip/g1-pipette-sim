# Recorded-command dynamics calibration

Episode 53 now attaches, releases and disposes the tip with unchanged `action.low_level` commands. The live robot is initialized from the first observed pose, then integrated by MuJoCo. Subsequent observed joint positions are used only for scoring, not feedback, trajectory correction, or pose replacement.

## Run

```bash
cd ~/Projects/g1_pipette_sim
./run.sh --episode 53 --source low_level --config config/episode53-calibrated.json

# Another episode, shared physical parameters, approximate initial layout:
./run.sh --episode 57 --source low_level --config config/calibrated-physics.json --fit-layout
```

Do not combine `--fit-layout` with the frozen episode-53 config if you want its exact saved scene. The default uncalibrated demo/profile remains available. High-level `action` is not the same signal as `action.low_level`; the original high-level controller is not included.

[Episode 53 video](../outputs/calibrated-episode53/simulation.mp4) · [Final state](../outputs/calibrated-episode53/final.png) · [Joint comparison](../outputs/calibration/episode53-joints.png) · [Cross-episode errors](../outputs/calibration/joint-error-comparison.png)

## What changed

1. **Motor gains.** On training episodes **53, 54, 55, 56**, a bounded least-squares fit relates recorded effort to `kp × (low_level_target − observed_position) − kd × observed_velocity + intercept`. Velocity is estimated with a 15-sample Savitzky–Golay filter and median recorded timestep. Candidate command delays are 0–100 ms. Right-arm best fits use zero delay and approximately 38–40 N·m/rad stiffness for shoulder/elbow and 19–20 for wrists. These broadly agree with the local `TWIST2/deploy_real/robot_control/configs/g1.yaml` configuration, but that file alone does not prove which controller was running during capture. Fitted intercepts and candidate delays are diagnostics; neither is injected as a replay correction.
2. **Gravity compensation.** The fitted body motors use PD effort without the old extra model-based gravity compensation. Hand/ejector proxy control retains its previous settings. Motor force limits remain active.
3. **Effective wrist payload.** The original model underestimates free-space right-arm effort. An additional **0.783 kg effective inertial load**, centered approximately **[158, −11, −2] mm** relative to `right_wrist_yaw_link`, is fitted to slow free-space samples in the same four training episodes. This reduces the selected effort residual from **1.56 to 0.50 N·m RMS**. It is an aggregate correction for missing/inaccurate hand, mounting, payload, or model properties—not a measured pipette mass. Its small rotational inertia is a modelling assumption, not independently identified. No visible object mesh was enlarged or given this mass.
4. **Command-replay initialization.** Command replay now starts from the observed initial robot pose instead of assuming that initial command targets were already reached. No observed-state feedback is used after initialization.
5. **Ejection arming.** The calibrated profile arms above 400 raw thumb-register units rather than 600. Training episode 56 holds at about 470, so the old cutoff could never arm there. Release still requires the command to fall below 200 after attachment. This remains a command-triggered mechanism proxy, not a calibrated thumb-contact/ejector mechanism.

The successful final profile retains the photo-derived box, lid, pipette and tip sizes, the existing object masses/friction, and the existing layout fitter. Experimental cell/layout alternatives were not adopted. Rack tape, pipette grip and tip retention remain solver constraints. The pipette mount itself has not been measured or calibrated from images.

## Validation scope

The comparison uses exactly the same low-level commands, timestamps and per-episode layout-initialization method for original and calibrated physics. The seven forward-dynamics evaluation episodes **0, 20, 52, 57, 60, 65, 77** were excluded from motor/payload fitting and span both recording sessions. Episode 53 is a training episode. **These results establish improved joint response; they do not establish matched object dynamics.**

`--fit-layout` uses each evaluated episode's observed tool trajectory to initialize its scene. This is retrospective layout estimation, not a held-out prediction of scene placement. Additional tests froze episode 53's entire scene for episodes 57 and 77: right-arm RMS errors remained about **0.0121 and 0.0128 rad**, respectively, but both missed pickup. Those fixed-scene results demonstrate that improved arm response does not depend on feeding observed joint states to the controller, while also exposing the remaining registration error.

Motor-effort prediction was separately scored on **all 74 episodes excluded from fitting**. Median right-arm effort RMS error was **0.128 N·m**. Median per-joint R² was approximately 0.69–0.99. This is a motor-response check, not 74 forward-dynamics task rollouts. Damping is less identifiable on nearly stationary joints, and frozen state timestamps limit exact timing interpretation.

All 13 regression tests passed, including a test that rejects observed-state sampling after command-replay initialization. Episode 53's rendered rollout has no MuJoCo warnings. Numeric results and task states below come from the final forward-dynamics runs; simulated task-state labels are not verified real-world success annotations.

## What remains unresolved

- Several held-out rollouts miss pickup or do not land the released tip in the lid. Matching arm motion within roughly a degree can still leave millimetre-to-centimetre tool errors, significant compared with a 7 mm well.
- No calibrated camera intrinsics/extrinsics, measured wrist-to-nozzle transform, or per-frame 3D object trajectories are available. Pixel-level/object-pose agreement has not been scored; the displayed camera views remain approximate.
- The actual global root trajectory, changing support conditions, exact hand geometry, contact friction, adhesive compliance, interference fit and ejection impulse are not identified. The supported-base approximation leaves larger lower-body errors.
- The receiving lid has no hinge joint. Contact geometry and tip-bore mechanics remain simplified. Adjusting masses/friction merely until every simulated episode succeeds would not establish that these dynamics match the recordings.

The next useful calibration inputs are a measured wrist-to-nozzle transform, calibrated camera views or measured scene landmarks, and annotated pickup/release frames with object trajectories. Mass/centre-of-mass measurements of the mounted hand/camera/tool assembly would resolve ambiguity in the effective payload fit.

## Reproduce identification and evaluation

Run from the project directory with the existing environment:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.identify_controller --train 53 54 55 56
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.identify_payload --train 53 54 55 56
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.validate_effort

~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.evaluate \
  --episodes 53 0 20 52 57 60 65 77 --source low_level \
  --config config/calibrated-physics.json --fit-layout \
  --output outputs/calibration/final-calibrated --workers 4

~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.evaluate \
  --episodes 0 20 52 57 60 65 77 --source low_level --fit-layout \
  --output outputs/calibration/heldout-baseline --workers 2

~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.evaluate \
  --episodes 53 --source low_level --fit-layout \
  --output outputs/calibration/baseline --workers 1

~/miniconda3/envs/lerobot-act/bin/python tools/calibration_report.py
```

The report script also reads the saved episode-53 original-physics evaluation in `outputs/calibration/baseline`. Each evaluated episode saves its resolved config, report and numerical trace. Training data and source videos are never modified. No hardware commands are sent.

## Final forward-dynamics results

| Episode | Original arm RMS (rad) | Calibrated arm RMS (rad) | Final simulated tip state |
|---|---:|---:|---|
| 53 | 0.0788 | 0.0130 | disposed |
| 0 | 0.0867 | 0.0151 | released |
| 20 | 0.0809 | 0.0128 | released |
| 52 | 0.0854 | 0.0164 | in_rack |
| 57 | 0.0764 | 0.0149 | released |
| 60 | 0.0825 | 0.0131 | in_rack |
| 65 | 0.0817 | 0.0152 | released |
| 77 | 0.0779 | 0.0125 | disposed |

Across the seven held-out episodes, mean right-arm RMS error decreased from **0.0816 to 0.0143 rad (82.5%)**. All eight final rollouts were numerically finite with no MuJoCo warnings. Episodes 53 and 77 reached simulated disposal; the other outcomes remain unresolved.
