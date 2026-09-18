# Episode-specific visual scene initialization

The default HG-DAgger replay now loads an episode-specific object layout before starting physics:

```bash
cd ~/Projects/g1_pipette_sim
./run.sh --episode 15
```

The initializer uses only the **initial recorded camera images**, reference object dimensions, and configured camera poses. It does not optimize against the commanded trajectory, use recorded success labels, or overwrite live object poses after simulation starts. The robot still initializes from measured joint positions; motor commands are unchanged. Gravity, contacts, and the existing attachment logic determine subsequent motion.

Episode 15 still fails insertion in a full dynamic replay: the minimum lateral nozzle-to-tip error is about 48 mm, versus a 3 mm attachment tolerance. Passing image registration is not task-success validation. See [the diagnosis](EPISODE15_DIAGNOSIS.md) for measurements and remaining calibration work.

The default lid is now physically hinged to the rack. Its opening angle remains a configured estimate; independent planar lid annotations apply only to legacy free-lid configurations. See [the hinge model](BOX_HINGE.md).

## What is estimated

- **Rack:** x/y position and yaw on the configured tabletop, relative to the episode-0 rack image. Features are tracked inside the rack outline in the fixed side view, with forward/backward consistency, robust homography fitting, coverage and reprojection checks.
- **Active tip:** detect the visible green feature inside the rack, compensate approximately for its mouth being 5 mm above the plate, and select the nearest well. Near-boundary selections are flagged as uncertain.
- **Lid:** independent x/y/yaw from manually corrected corner landmarks. Without annotations it is carried with the rack at the reference offset, **explicitly marked assumed**. Transparent-lid features can follow the background, so they are not treated as reliable automatic pose measurements.
- **Table height, robot base, and initial pipette mounting transform:** remain configured assumptions. This is planar scene initialization, not full six-degree-of-freedom reconstruction or camera calibration.

The fixed overview view is tracked separately as an independent check. Its calibration currently disagrees with the side-view reconstruction in some episodes, and its rack is often occluded. The tool records that disagreement rather than averaging inconsistent camera estimates. The wrist view is rendered for comparison but is not used to fit object locations because its hand-eye transform is still approximate.

## Files and review

Files live under `outputs/episode-scenes/<dataset-snapshot-path-hash>/`:

- `episode_015.json`: complete resolved scene configuration, object statuses and provenance.
- `episode_015.overlay.jpg`: initial real frames with **orange** reference object projections, **green** initialized projections, and **cyan** tracked/annotated outlines. The magenta marker is the detected/annotated active tip.
- `episode_015.annotations.json`: optional manually edited image landmarks.
- `episode_015.comparison.jpg`: real/sim initial-frame pairs when generated with `--preview`.

Cache validity checks include the resolved dataset snapshot, episode, complete base configuration, video file sizes/modification times, annotations and initializer version. Changing the mesh/config or annotations causes re-estimation. A resolved scene supplied explicitly must belong to the requested episode unless `--scene-layout fixed` is selected.

Generate or review selected layouts:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.episode_scene \
  --episodes 0 15 30 45 --preview
```

Correct one episode visually:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.episode_scene --edit 15
```

Drag the yellow rack corners onto its top plate boundary, the cyan lid corners onto its top rim, and the magenta marker onto the active tip mouth. Keep the corner identities/order; the labels correspond to the same physical corners throughout. Save to refit. The editor is an approximate pinhole/planar fit: if the outlines cannot be matched consistently, investigate camera calibration or the object's geometry/height rather than forcing the corners to fit.

To prepare all episodes:

```bash
~/miniconda3/envs/lerobot-act/bin/python -m g1_pipette_sim.episode_scene --episodes all
```

Rejected estimates are listed in `outputs/episode-scenes/failures.json`; the command exits nonzero if any requested episode is rejected. Successful scenes are still saved. Subset runs preserve failures for episodes they did not process.

## Overrides and compatibility

```bash
# Use exactly the object layout in the supplied config:
./run.sh --episode 15 --scene-layout fixed

# Recompute the episode layout, retaining any saved manual annotations:
./run.sh --episode 15 --refresh-scene

# Existing trajectory-based layout heuristic (explicitly bypasses visual initialization):
./run.sh --episode 15 --fit-layout
```

`--scene-cache PATH` selects another replay cache directory; use `--cache PATH` with the preparation/editor command for the same directory. Auto initialization is enabled by `episode_scene.enabled` in the base HG-DAgger config. Older configurations and demo mode retain their existing layouts. Camera video export also resolves the episode scene by default; use its `--scene-layout fixed` to disable that behavior.

## Current coverage and limits

The first prepared batch contains **44 of 46** scene estimates. Episode 12 uses an assistant-supplied approximate visual rack-corner annotation after automatic registration failed; its annotation provenance is recorded. Episode 11 is rejected because the detected green feature is not consistent with a rack well. Episode 42 is rejected because its initial rack view does not fit the existing fixed-camera/table assumptions (roughly 63 px fit RMS with the supplied approximate corners). Episode 42 may need camera/reference calibration, not just object translation. Those episodes do not silently fall back to the shared layout; use the editor/calibration or explicitly opt into a fixed layout.

Of the prepared estimates, 20 active-tip assignments are marked near a well boundary and should be checked. All unannotated lid poses remain assumptions. The estimated camera poses and object geometry prevent a claim of millimetre-level physical accuracy. Side-view fitting and a small image residual alone do not establish cross-camera agreement, correct robot-to-rack alignment, or successful task replay. Inspect the overlays before using these scenes for policy training.

Tests cover synthetic pose recovery, image tracking/rejection, manual lid translation/yaw, cache reuse and invalidation, and episode mismatch protection. Real/sim previews are provided for episodes 0, 15, 30 and 45. Replay reports include the scene-initialization provenance alongside the dynamics results.

For a separate, trajectory-conditioned search that tests unchanged motor commands in dynamics, see [Replay layout search](REPLAY_LAYOUT_SEARCH.md). This does not replace visual reconstruction or calibrate the real scene.
