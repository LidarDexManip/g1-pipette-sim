# Episode 14 ego-camera correction

The old ego pose viewed the rack from in front of the hand, making the hand obscure the tray. Planar rack landmarks admit two plausible camera-pose solutions. The solution behind the hand both fits the annotated well centers better and reproduces the rack visibility in the recorded frame.

`config/camera-view-overrides.json` applies the corrected `rgb` pose after the validated pickup config is selected. This episode-14 correction is now the default for HIL194 episodes 0–45; episodes 46–193 use a corresponding fit from that session’s rack landmarks. Only camera position, orientation and field of view can be overridden. The side and wrist cameras, robot commands, rack placement, grasp, and physical parameters are unchanged. Explicit `--config` runs also receive the correction by default. Pass `--camera-pose config` to retain their own cameras. See [the dataset-wide default and regenerated videos](FAILURE_ANALYSIS.md#corrected-ego-camera-default).

```bash
./run.sh --episode 14 --headless --compare \
  --output outputs/episode14-comparison-ego-fixed
```

The video is `simulation.mp4` in that directory. The correction also applies to the live three-camera window and `--camera rgb` output. The original `outputs/episode14-comparison/` remains intact.

This is an approximate rendering correction, using manually selected well centers, the assumed 10.5 mm rack pitch and a 60-degree vertical field of view. It does not establish calibrated camera intrinsics or a measured camera-to-robot transform. The fitted landmarks and alternative camera poses are in `outputs/episode14-ego-fix/`. To reproduce the pose candidates:

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/fit_ego_view_from_landmarks.py \
  outputs/episode14-ego-fix/ego-landmarks.json \
  --output outputs/episode14-ego-fix/calibration-report.json
```

An A/B replay check produced exactly equal joint positions and velocities for 200 steps. The new full comparison is checked against the original attachment event as well; changing camera pose does not change pickup validation.
