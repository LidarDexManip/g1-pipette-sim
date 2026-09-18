# Search initial rack/tip placement for command replay

Current collision-model result: episode 15 attaches and lifts at a newly searched placement, but initial grasp overlap and post-attachment tip penetration remain unresolved. See [the updated diagnosis and run command](EPISODE15_CONTACT_DIAGNOSIS.md).

`tools/search_replay_layout.py` searches for an initial rack placement that allows
a selected episode's unchanged motor commands to attach and lift the tip in the
current simulator. This is **trajectory-conditioned fitting**, not recovery of
the real scene from photographs. It is useful for constructing a replay-compatible
scene or investigating whether position alone can explain a failure.

From the project directory:

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/search_replay_layout.py \
  --episode 15 --max-evaluations 24 --output outputs/episode15-layout-search
```

Then replay the best configuration:

```bash
./run.sh --episode 15 --mode replay --physics dynamic --source low_level \
  --scene-layout fixed --config outputs/episode15-layout-search/best.json
```

Inspect `search.json` before treating the result as successful. `best.json` is
saved even when no successful placement was found. The script prints its outcome;
completion alone does not imply success. Do not add `--fit-layout` when replaying
the result, because that would replace the searched placement.

## Search and success criteria

1. Load the episode's visually initialized scene (or use `--scene-layout fixed`).
2. Run a baseline full dynamic replay and sample the nozzle/tip-mouth geometry.
3. Propose rack XY translations from nozzle samples near insertion height.
4. Test each candidate in a fresh full dynamic replay, then refine the best
   translation with progressively smaller coordinate steps (8 mm down to 0.5 mm).
5. Save the best scene, report and complete evaluated-candidate summary.

The tip stays in its configured well and translates with the rack. Rack yaw,
active well, table height, tool mounting pose, hand posture, cameras, gains,
friction, gravity and attachment rules remain unchanged. The default HG-DAgger lid is now connected to the rack by a passive hinge; see
[the hinge model](BOX_HINGE.md). With this model, carry preserves the connection
and fixed is rejected. For legacy free lids, carry moves the lid with a conservative
clearance allowance for hinges and ribs. `--lid-placement fixed` preserves its world pose
and rejects overlapping candidates. Both footprints must remain on the table, with 5 mm edge margin. Initial rack
penetrations into the lid, robot or tool exceeding 0.1 mm reject a candidate;
rack/lid penetration exceeding 0.1 mm anywhere in the rollout invalidates its
result. Table support and retained-tip contacts are excluded from the startup
screen. This screen is not a full physical-fidelity certification.
Only initial configuration is changed; no live object or robot poses are reset
during replay. Full physics therefore includes any new contacts caused by moving
the rack.

The continuous search score considers radial, depth and angular errors over a
full attachment dwell window. It is only a search heuristic. `attached` requires
an actual simulator attachment event. `success` additionally requires the tip
mouth to rise at least 20 mm above the rack mouth while attached, a finite rollout,
and no MuJoCo numerical warnings. This does not certify ejection/disposal, sustained
grip robustness, calibrated insertion forces, or real-world performance. Tip
attachment still uses the simulator's existing geometric gate and weld model.

## Options and outputs

- `--episode`: episode index, default 15.
- `--dataset`: LeRobot root, snapshot, or HF cache; default HG-DAgger dataset.
- `--config`: initial configuration, default `config/hgdagger-cameras.json`.
- `--scene-layout auto|fixed`: whether to initialize visually before searching.
- `--lid-placement carry|fixed`: carry with clearance (default), or keep lid world pose.
- `--source low_level|action`: unchanged recorded targets; default `low_level`.
- `--max-evaluations`: full replay budget, default 24, including baseline.
- `--max-shift`: maximum translation norm in metres, default 0.12.
- `--lift-height`: required attached lift in metres, default 0.02.
- `--output`: output directory, default `outputs/layout-search`.

`baseline.json` preserves the starting scene. `best.json` includes the exact
placement and fitting provenance. `best-report.json` contains its dynamic replay
report. `search.json` records scores, shifts, attachment and lift results for every
evaluated placement. Search stops early at a successful candidate. Runs can take
minutes because every candidate simulates the whole episode at the normal physics
timestep.

This is a bounded local search, not a proof that no successful layout exists when
it fails. It does not optimize yaw, active well, height, robot base, or grasp.
Those require additional evidence and search dimensions. Validate any selected
placement in a separate replay, inspect the camera views, and test small placement
perturbations before using it to generate demonstrations. Success on the episode
used for fitting is not held-out policy or sim-to-real validation.

## Correction to the original episode-15 result

The first search's reported attachment was not a valid placement validation:
it kept the lid fixed while moving the rack, producing **28 mm initial rack/lid
penetration**. Replaying the exact original scene still produces attachment at
9.7805 s, but the overlap invalidates that result. The original validation did
not screen collisions. Do not use the archived overlapping configuration.

The corrected search carries the lid with the rack and adds hinge clearance;
it also checks actual MuJoCo contacts initially and during replay. The corrected
same-rack-position candidate starts with no rack contacts and attaches at 3.083 s.
A complete diagnostic rollout found no rack/lid contacts and a maximum pipette/rack
penetration of about 0.172 mm during intentional interaction. Table height, robot
commands, pipette mount, hand settings and attachment thresholds remain unchanged.

The default and shared saved episode-15 config now use the connected, tabletop-supported lid. Its attachment event does not yield a successful pickup (about 0.24 mm lift); see [the current result](BOX_HINGE.md). The intermediate free-lid clearance result above is retained as diagnostic history.
