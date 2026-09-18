# Episode 46 rack and tip placement

The default image-based layout misses the tip. A separately saved, trajectory-fitted placement attaches at **11.30 s** using unchanged HIL194 episode-46 `low_level` commands. An independent full replay reproduced the event.

```bash
cd ~/Projects/g1_pipette_sim
./run.sh --episode 46 --scene-layout fixed \
  --config outputs/hil194/episode46-layout-search/attached-12mm.json
```

Use the explicit config above: plain `./run.sh --episode 46` still selects the image-based estimate. The fitted placement is not substituted for camera-derived scene estimates in other episodes or in the original 194-episode audit.

## What was wrong

The image detector selects the green tip in zero-based rack cell `[2, 8]` (row 3, column 9 in model coordinates). Reviewing the side-view overlay supports this well selection. The default lateral nozzle miss never becomes smaller than **13.956 mm**, and is approximately **22 mm** through much of the real insertion interval. There is no sample satisfying all insertion gates.

Episode 46 is itself the new-session camera reference. Its small rack reprojection error is therefore not independent evidence of the rack's position relative to the robot: camera fitting assumed that world placement. Table height, camera intrinsics, robot/world registration and tool mounting remain approximate.

## Verified placement

| Parameter | Image-based estimate | Attaching placement |
|---|---:|---:|
| Rack center X (m) | 0.288864864 | 0.303178032 |
| Rack center Y (m) | 0.083277097 | 0.104732554 |
| Active well (zero-based) | `[2, 8]` | `[2, 8]` |

The rack translation is **+14.313 mm X, +21.455 mm Y** in simulator world coordinates. The tip stays in the same well, and the hinged lid moves with the rack. Configured tip-mouth position is `[0.329428033, 0.088982555, 0.713074678]` m; the actual dynamic pose can settle slightly. Rack yaw, height, robot targets, hand commands, mount, friction, collisions and attachment thresholds are unchanged.

Eleven complete placement trials were evaluated. Further refinement was stopped after independently verifying an attaching layout; one partial trial was discarded. This is a bounded search, not an exhaustive optimum or recovered real-world pose. Search artifacts are in `outputs/hil194/episode46-layout-search/`.

At the successful 100 ms dwell interval (11.205–11.300 s, sampled every 5 ms):

- Maximum lateral error: **0.154 mm**, versus the 0.7 mm limit.
- Insertion depth: **10.020–10.430 mm**, within the required 10–12 mm.
- Maximum angle: **1.911 degrees**, versus the 5-degree limit.
- Maximum attached lift: **12.620 mm**. The search's additional **20 mm lift criterion was not met**.
- No initial rack overlap, rack/lid penetration or numerical warnings.
- Maximum pipette/rack penetration: **0.273 mm**, peak normal force **31.06 N**.
- Maximum pipette/tip penetration: **0.144 mm**, peak normal force **4.32 N**.

The contact forces and press-fit mechanism remain uncalibrated. Pipette grasp drift still reaches about **22.9 mm and 22.0 degrees**. The placement compensates for the current simulator's trajectory error; it does not establish matching real grasp dynamics or independently verify real attachment timing. Do not count it as held-out policy success.

## Evidence

- `outputs/hil194/episode46-placement-diagnosis/real-sequence.jpg`: real video sequence.
- `outputs/episode-scenes/0a4ac42e06f7/episode_046.overlay.jpg`: detected rack and tip.
- `outputs/hil194/episode46-placement-diagnosis/baseline-gates.png`: default insertion errors.
- `outputs/hil194/episode46-placement-diagnosis/attached/diagnosis.json`: independent attaching replay and gate measurements.
- `outputs/hil194/episode46-placement-replay/`: replay through the normal run script.
