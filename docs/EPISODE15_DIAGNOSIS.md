# Episode 15 insertion failure

The full 13.735-second HG-DAgger episode-15 dynamic replay using recorded
`low_level` targets and the automatic episode scene does **not** attach a tip.
This was reproduced on 2026-09-10. The real episode's wrist video shows a tip
on the pipette later in the recording.

| Measurement | Replay result | Attachment requirement |
|---|---:|---:|
| Minimum lateral nozzle-to-tip-mouth error | 47.98 mm | <3 mm |
| Lateral error at closest 3D approach | 48.13 mm | <3 mm |
| Insertion depth at closest 3D approach | -6.73 mm (above mouth) | 2–5 mm |
| Angle at closest 3D approach | 9.58 degrees | <12 degrees |
| Longest continuous valid seating | 0 seconds | 0.1 seconds |
| Maximum tool displacement from intended grasp frame | 24.41 mm | Not an attachment gate |

The measured initial joint pose, evaluated with the estimated rigid pipette
mount, already places the nozzle about 58.5 mm laterally from the configured
target. Thus the mismatch exists independently of dynamic servo tracking or
later grasp motion. This does not identify which individual calibration
parameter is wrong: rack pose/well identification, camera-to-robot alignment,
robot base pose and pipette mounting geometry are still uncertain.

The automatic initializer registers rack image features across episodes using
an approximate camera model. It does not establish millimeter-accurate alignment
between the camera, robot and grasped tool. A visually plausible rack overlay is
not validation that unchanged recorded robot targets will insert the tip.
The contact grasp also moves substantially during this replay and needs separate
validation against the real grasp. There were no MuJoCo numerical warnings.

## Reproduce

From the project directory:

```bash
./run.sh --episode 15 --physics dynamic --source low_level --scene-layout auto
/home/ytung34/miniconda3/envs/lerobot-act/bin/python tools/diagnose_episode15.py
```

The diagnostic writes `outputs/episode15-diagnosis/{report,samples,config}.json`.
Normal dynamic runs now include `attachment_diagnostics` in `report.json` and
print a failure summary if no attachment occurs. Gate failure counts can overlap;
each counts physics steps while the tip is in the rack. Negative insertion depth
means above the mouth. Grasp displacement is relative to the configured mount,
not a measurement of real-world slip.

## Required correction

Calibrate the fixed camera intrinsics and its transform to the robot base using
known 3D landmarks/fiducials, then verify the rack corners and active well in that
same frame. Measure the pipette nozzle pose relative to the right wrist in the
actual grasp. Verify these quantities over multiple robot poses before adjusting
contact parameters or replay gains. Finally validate grasp stability and replay
insertion on held-out episodes with unchanged recorded targets.

A trial fit using four episodes' manually identified nozzle/shaft image points
and the currently guessed mount had roughly 18 pixels RMS reprojection error and
an implausible camera pose; it was rejected. No camera correction was applied
from that fit. Attachment tolerances and replay commands were not changed to
force success. Episode 15 remains unsuccessful with that visual initialization. A subsequent rack translation search produced attachment at 9.7805 s, but its scene had 28 mm initial rack/lid penetration and that placement validation was invalid. Collision-aware correction and replay results are documented separately; see [Replay layout search](REPLAY_LAYOUT_SEARCH.md). That fitted placement does not resolve the real-scene calibration mismatch.
