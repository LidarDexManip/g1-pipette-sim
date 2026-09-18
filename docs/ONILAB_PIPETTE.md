# ONiLAB 0.5–10 µL pipette and contact grasp

The default run is now:

```bash
cd ~/Projects/g1_pipette_sim
./run.sh
```

This loads the downloaded September 4 HG-DAgger dataset, episode 0, `config/hgdagger-cameras.json`, dynamic physics, `low_level` motor commands, and the free overview camera. Camera export uses the same dataset/config/episode defaults. The previous August dataset and generic scene remain available by explicitly passing `--dataset ~/Datasets/g1-pipette-3view-20260819-plus21 --config config/workspace.json --episode 53`. Python library helpers retain their legacy defaults for older calibration scripts; the executable defaults above are explicit.

## Product evidence and reconstruction

Reference: the saved Amazon product page under `assets/Media`, for [ONiLAB Toppette, ASIN B09L6BW5BN](https://www.amazon.com/dp/B09L6BW5BN), selected **0.5–10 µL**. Live Amazon retrieval failed; the locally saved listing and its images were inspected.

- Package dimensions: **10.87 × 3.43 × 1.65 inches**, converted to **276.098 × 87.122 × 41.910 mm**. These describe the package, not the bare pipette. They provide a bounding sanity check, not an exact scale calibration.
- Product image `61pbsSljBvL._SL1500_.jpg` says **80 g / 2.8 ounces**. The listing table separately says 3 ounces (about 85 g). The simulator uses the requested **80 g for the complete assembly**, including the moving ejector, without counting its mass twice.
- Product photos show a white slim handle, blue shaft/plunger/ejector, curved finger rest and vertical volume window. The later unobstructed photos refine these shapes and show that the bare lower nozzle is black; see the photo-refinement update below. The display is illustrative, not a simulated volume setting.
- The user measured the pipette as **10 × 2.5 × 1 inches**, converted to **254 × 63.5 × 25.4 mm**. The assembled geometry now matches these overall bounds. The 63.5 mm dimension is interpreted as including the finger rest and side ejector; 25.4 mm is the handle thickness. This assumes the bare pipette with the plunger at rest. Individual component proportions remain photo-derived approximations.

The hand-to-tool initial transform remains unchanged. The nozzle endpoint is now approximately **−156.72 mm** in the tool frame, rather than −145 mm; replay and tip attachment use this new position. Detail lengths and shaft/handle profiles were scaled consistently to the measured 254 mm overall length. The revised handle tapers, lower shaft, thinner plunger stem/button, curved rest, sleeve, and display improve visible proportions. The tip remains a separate object. Liquid volume, piston metering and aspiration are not simulated. Mass distribution and inertia are estimated from the collision shapes; no measured center of mass is available.

Preview: [reconstructed pipette](../outputs/onilab-pipette.png). Run `python tools/preview_onilab_pipette.py` from the project directory with the `lerobot-act` environment to regenerate it. Geometry lives in `g1_pipette_sim/object_meshes.py` and `photo_objects.py`; scene parameters live in `config/hgdagger-cameras.json`.

## Physical grasp

`physics.grasp_mode` is now `contact` in HG-DAgger. The wrist weld starts **inactive** and is not activated by normal playback. The pipette is a free rigid body; normal and friction forces from the existing finger/palm collision meshes support it. Robot–tool collisions are enabled when compiling the scene. No position resets, suction, grasp springs, or artificial forces retain it during playback. The legacy `weld` mode remains available in older configurations. Tip–rack retention and tip attachment still use their separate constraints.

Recorded hand channels are frozen. The explicit feedback-register pose approximation remains, with `finger_closed_rad` increased from 1.25 to **1.40 radians** to close the proxy fingers against the slimmer handle. This is a simulation grasp adjustment, not recovered real finger motion or calibrated hand kinematics. Sliding friction is estimated (`grasp_friction=1.2`); the existing torque-limited finger servos supply the squeeze.

Press **G** to command the right fingers open. Contact is still physical: the curved rest may hang on the palm even after opening, so G does not forcibly delete contacts or eject the pipette. **R** restores the initial grasp. With kinematic physics the tool remains prescribed with the wrist; use dynamic mode to evaluate the contact grasp.

Validation includes an elevated free-space hold, positive finger contact forces, no active wrist or tip attachment constraint during that hold, correct 80 g total mass, and gravity-driven fall when moved clear of supporting surfaces. Eight-second checks of episodes 0, 15, and 45 assess retention, not task completion or matching the real insertion dynamics. Some slip/settling inside the hand is expected and changes the nozzle trajectory. Opening alone can leave the tool resting on its finger hook; this was observed and is not reported as a successful drop test.

`tools/validate_contact_grasp.py` writes `outputs/contact-grasp-validation.json`. The report also exposes `grasp_mode`, `grip_constraint_active`, and `grip_position_error_m`; in contact mode that error measures displacement from the initial mount, not a weld solver error. These checks do not establish sim-to-real grasp fidelity or successful attachment across all episodes. The assumed DFQ hand geometry, real grasp posture, friction and insertion forces still need measurement.

## Unobstructed-photo refinement (September 10)

Inspected all twelve new `20260910_155954011_iOS.heic` through `20260910_160035812_iOS.heic` images in `assets/Media`. These provide unobstructed front/back, oblique top, and shaft views; they supersede the generic product illustrations for visible shape. No new intrinsic calibration or photogrammetric scan was performed.

The active `photo_revision` now generates a handle with flat faces and diagonal bevels, a longer constant-width white body, a stepped blue shoulder with a long tapered shaft, a short black nozzle, a broad curved moulded finger rest, and a blue side-ejector paddle. The plunger has a white stem and a collar near the head. The earlier silver/red rings near the nozzle were removed because these photos show a blue-to-black transition there. The display remains illustrative; branding, fine print, surface seams and small internal mechanism details are not reconstructed.

Overall model bounds remain **254 × 63.5 × 25.4 mm**, total assembly mass **80 g**. Internal proportions are inferred from perspective photos; local dimensions, friction and center of mass remain estimates. The finger-rest collision shape uses convex sections following the curved visual ribbon instead of treating the whole concave rest as a single convex hull. The handle and shaft contact surfaces follow their revised meshes. No wrist weld was added.

- [Before/after preview](../outputs/pipette-photo-refinement/before-after.jpg)
- [Reference contact sheet](../outputs/pipette-photo-refinement/references.jpg)
- [Assembled visual OBJ](../outputs/pipette-photo-refinement/pipette.obj) and [materials](../outputs/pipette-photo-refinement/pipette.mtl): keep together. Coordinates are metres in the pipette frame, assembled at rest. This is a visual asset; use the simulator's separate components for physical articulation and collision.

Regenerate with `python tools/export_pipette_mesh.py` from the project directory using the `lerobot-act` environment. The simulator automatically generates component meshes from `g1_pipette_sim/pipette_mesh.py`; the exported combined OBJ is not a replacement for its articulated scene.

Most useful remaining photographs:

1. Straight-on front, back and side silhouettes with a ruler in the same plane as the pipette. Keep the camera perpendicular and stand farther back to reduce perspective distortion. Existing images already cover general appearance well.
2. Macro side views of the bare black nozzle and a fully seated tip, with a millimetre scale; nozzle diameter and seated insertion depth are particularly useful for contact modelling.
3. The underside and edge of the curved finger rest, plus a straight top-end view of the plunger and blue ejector. Thickness/width measurements with calipers would resolve ambiguities that more oblique photos cannot.
4. For matching the robot grasp, two opposite close-ups of the actual fingers touching the handle, showing the pipette's position relative to the palm. These improve mounting/contact placement rather than the standalone silhouette.

## Additional ruler and close-up photos

The fourteen additional images `20260910_163752101_iOS.heic` through `20260910_164036640_iOS.heic` are incorporated in `photo_revision=20260910-ruler-and-closeups-v2`. The component source list now records all 26 September 10 photos.

The bare-nozzle close-ups show a short black stem with small diameter steps rather than a continuous pointed cone; these steps and its collar are now represented. The blue shaft has a gentler taper with a more distinct shoulder. The head views show a broad ejector thumb pad above a narrower sliding stem, and shallow ridges around the plunger cap; both now use separate mesh geometry. Small feature dimensions are visual estimates from perspective images, not caliper measurements. Nozzle internal bore and press-fit deformation remain unmeasured.

The ruler images support the overall proportions, but perspective and ruler/object depth offsets prevent treating them as calibrated orthographic measurements. The user's direct 254 × 63.5 × 25.4 mm dimensions remain the scale constraint, with 80 g total assembly mass. The taped robot views constrain visible mounting direction but do not expose the actual finger-to-handle surfaces, so no new grasp transform or hand feedback was inferred from them. The default remains contact-based grasp without the wrist weld.

[Additional reference contact sheet](../outputs/pipette-photo-refinement/additional-references.jpg). The existing OBJ/MTL export and standalone preview have been regenerated. A millimetre-scale side close-up with a **fully seated tip** would be the most useful next reference for insertion depth; these new images already cover the bare nozzle and head much better.

The subsequent attached-tip photos changed the nozzle/tip geometry and required seating depth. See [the current mesh and attachment update](TIP_SEATING.md); previous episode-15 attachment results do not apply to the revised gate.
