# Rack-connected lid

The September 10 photos `190247532` through `190330131` show the lid closed,
partially open, and fully open, with close-ups of the rear hinge. The user
confirmed that it rotates beyond 180 degrees and rests on the tabletop.

The default HG-DAgger model now has one passive revolute joint (`lid_hinge`)
connecting the lid to the rack along its long rear edge. Two visible hinge
stations represent the photographed connection. The historical body name
`waste` is retained because the open lid also serves as the receiving tray.
It no longer has an independent `waste_free` joint in this configuration.
Gravity, contact and hinge damping determine its motion; there is no lid pose
reset during dynamic replay. The rack's existing tape-to-table constraint remains.

## Geometry and parameters

Parameters live in `config/hgdagger-cameras.json`, under `waste.hinge`:

- `position`: `[0, -0.054, 0.060]` metres in the rack frame. X is along the
  long edge; negative Y is the hinge side. The offset is estimated, not measured.
- `initial_angle_deg`: 198. Zero is closed, 90 upright, 180 flat open.
- `range_deg`: `[0, 230]`. The upper limit allows the lid to reach the tabletop;
  **230 is not a measured hardware stop**. Table contact stops the default lid
  well before this bound.
- `damping`: 0.003 N m s/rad; `frictionloss`: 0.001 N m. Both estimated.
- `enabled`: true for the default HG-DAgger setup. Legacy configurations without
  this setting retain the separate tray for compatibility with old experiments.

The estimated lid shell is 136 x 96 x 32 mm, allowing it to close outside the
128 x 88 mm rack without overlapping its walls. Hinge dimensions and shell
clearance remain approximate. Explicit shell contact pairs preserve rack/lid
collisions despite MuJoCo's usual parent-child contact filtering. Pin/knuckle
mating surfaces are visual approximations; the revolute joint supplies the
mechanical connection. Well meshes are not separately paired with the lid shell.

The default lid settles at about **197.7 degrees**, supported by the table,
without rack/lid penetration in the tested rollout. Tests cover closed, 45, 90,
180 and 198 degree initial poses, gravity-driven rotation, tabletop support,
and the lid moving with the rack.

## Scene initialization and search

Rack translation/yaw now carries the articulated lid. Independent lid XY/yaw
annotations do not move it; reports label its initial angle as assumed. The
landmark editor exposes rack and tip handles for this configuration. Change
`waste.hinge.initial_angle_deg` to adjust the initial lid opening. That angle
has not yet been fitted from each episode's images.

`tools/search_replay_layout.py` supports this attached geometry. Its `carry`
mode preserves the hinge; `fixed` is rejected because an attached lid cannot
remain fixed while translating its rack. Initial overlaps and rack/lid
penetration during replay are checked. Collision screening does not establish
calibrated contact mechanics or task success.

## Episode 15 and running

The formerly shared placement had 28 mm of rack/lid overlap. That free-lid
configuration is archived under `outputs/episode15-layout-search/rejected-overlap`.
The shared `best.json` and `insertion-candidate.json` now use the connected lid.

```bash
./run.sh --episode 15 --physics dynamic --source low_level --scene-layout fixed \
  --config outputs/episode15-layout-search/best.json
```

The corrected model registers attachment, but only about **0.24 mm of attached
lift** in the tested placement. **Episode 15 still does not complete pickup.**
The overlap fix and physical connection are verified separately from task
success; no attachment thresholds, motor targets, hand posture, or tool mounting
transform were relaxed to force a result.

To use the default visually initialized scene with the new lid:

```bash
./run.sh --episode 15
```

To render hinge poses:

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/preview_box_hinge.py
```

The image is `outputs/box-hinge/hinge-preview.jpg`. Full replay and search reports
are under `outputs/box-hinge/`.

The subsequent attached-tip photos changed the nozzle/tip geometry and required seating depth. See [the current mesh and attachment update](TIP_SEATING.md); previous episode-15 attachment results do not apply to the revised gate.
