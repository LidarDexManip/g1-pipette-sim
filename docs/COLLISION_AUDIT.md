# Pipette collision audit

The previous dynamic model filtered tool–tip contacts and used a solid convex tip proxy. A whole-mesh convex hull also bridged changes in the shaft profile. The rack plate contacts were too soft for its thickness.

The tip now uses convex annular sectors preserving its hollow bore. Lower-shaft and nozzle collision meshes use individual tapered sections. Dynamic tool–tip contacts remain enabled during insertion, attachment and release. Rack contact stiffness has increased. The red nozzle reference marker is invisible; its attachment reference point remains available.

Tests compare compiled collision and visual geometry in body coordinates. They also check that a centered nozzle fits the tip bore, lateral displacement produces contact forces, an empty rack well permits entry, and the rack web blocks the nozzle.

The full episode-15 replay with `outputs/episode15-layout-search/best.json`, dynamic physics, low-level targets and fixed layout recorded maximum penetration of 0.0105 mm at the rack and 0.0159 mm at the tip, with nonzero contact forces and no numerical warnings. The report is `outputs/collision-audit/episode15-stiff-rack/report.json`. This replay still produced no attachment event; improved contacts do not establish task success or calibrated real-world dynamics.

Inspect collisions with:

```bash
./run.sh --physics dynamic --show-collisions
```

The option shows collision geometry in the main viewer and exported renders. Kinematic mode prescribes poses and should not be used to assess physical collision response. Tip retention and attachment still use solver welds rather than calibrated press-fit mechanics.
