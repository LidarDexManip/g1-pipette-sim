# Photo-guided nozzle and seated tip

The four photos `20260910_194518156` through `20260910_194533467` show the same
pipette and tip separately and fully assembled. The assembled tip covers almost
all of the black nozzle, leaving a short black collar beside the blue shaft.

The updated default HG-DAgger model uses these **photo-scaled estimates**, based
on the user's previously measured 254 mm pipette length:

| Quantity | Updated value |
|---|---:|
| Bare black nozzle length | 12.7 mm |
| Tip length | 36 mm |
| Tip maximum outside diameter | 7 mm |
| Fully seated insertion | 11 mm |
| Black collar left exposed | 1.7 mm |

The bare pipette's overall 254 mm length remains unchanged: the blue/black
transition moved to lengthen the black nozzle. The tip has a revised external
taper, flared mouth, open outlet and explicit internal profile. Its unobserved
bore is an approximation fitted to the nozzle, with roughly 0.05–0.10 mm radial
clearance at the nominal coaxial seated pose. The photos do not measure that bore
or its material stiffness. Surface roughness is not reconstructed.

## Files

- `outputs/tip-fit/nozzle.obj`: standalone black nozzle, pipette-local coordinates.
- `outputs/tip-fit/tip.obj`: hollow tip, centered at its body origin.
- `outputs/pipette-photo-refinement/pipette.obj` and `.mtl`: updated bare pipette.
- `outputs/tip-fit/assembled/pipette.obj` and `.mtl`: complete pipette with fully
  seated tip, in pipette-local coordinates, metres.
- `outputs/tip-fit/seated-comparison.jpg`: previous shallow pose, revised seated
  pose, and a transparent tip with the nozzle highlighted for visibility.

These are reproducible parametric meshes. Simulator assets are regenerated under
`assets/objects/` with a new geometry cache key. Default settings and the shared
`outputs/episode15-layout-search/best.json`/`insertion-candidate.json` use the new
profiles. Other older saved configurations retain their own geometry parameters.

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/export_pipette_mesh.py
~/miniconda3/envs/lerobot-act/bin/python tools/export_pipette_mesh.py \
  --with-tip --output outputs/tip-fit/assembled
~/miniconda3/envs/lerobot-act/bin/python tools/preview_tip_fit.py
```

The assembled export and preview prescribe a fully seated reference pose. They
are **not evidence of successful episode replay**.

## Attachment and the observed hover

The old replay began attachment at about 2.48 mm insertion, then developed up to
4.50 mm axial separation while still reporting attached. A mesh-only change
would conceal this constraint error rather than fix it.

The revised gate requires 10–12 mm insertion, lateral error below 0.7 mm, angle
below 5 degrees, and the existing continuous 0.1 s dwell. Lateral/angular limits
are approximate capture tolerances, not a calibrated elastic interference-fit
model. The attachment weld uses a 2 ms time constant and stronger impedance;
it still applies forces through physics rather than overwriting poses. A focused
load test applies 0.1 N lateral and 0.2 N axial force and verifies that seating
remains within 0.3 mm over 0.3 seconds. This does not validate real press-fit or
pull-off forces.

The tip collision geometry uses convex annular sectors that preserve the hollow
bore and follow the visual wall profiles. In dynamic mode, tool/tip contacts
remain enabled during insertion, attachment and release. The tapered shaft is
split into convex sections to match its visual profile. These rigid contacts
and the attachment weld are not a calibrated elastic press-fit model.
Use `--show-collisions` to inspect collision geometry in the main viewer.

With the updated meshes and gate, the shared episode-15 placement produces no
attachment event. Its previous search result is explicitly invalidated. Robot
commands, initial rack placement, hand mapping, tool mount and the connected
lid were not retuned to force success. Episode-15 pickup remains unresolved.

Validation covers mesh length, bore/nozzle clearance, unchanged bare-pipette
bounds, shallow-fit rejection, continuous dwell, and seating under load.
