# Episode 15 placement and grasp diagnosis

The updated collision model admits an episode-15 insertion with unchanged low-level commands. A 24-evaluation search stopped at candidate 16 after attachment and 21.77 mm lift. An independent replay through `run.sh` reproduced attachment at 6.686 seconds. There was no ejection/disposal event.

The selected rack center is **[0.3359743882, 0.0631376897] m**, a translation of **[+43.06, -33.87] mm** from the visually initialized baseline. The active tip stays in well [1, 1]; the connected lid moves with the rack. Yaw, table height, robot commands, hand settings, collision settings and attachment gate were unchanged. This is a fitted simulation placement, not a measured real-world location.

Run the candidate from the project directory:

```bash
./run.sh --episode 15 --mode replay --physics dynamic --source low_level \
  --scene-layout fixed --config outputs/episode15-contact-layout-search/best.json
```

Allow the complete 13.735-second episode to run. Add `--show-collisions` to inspect collision geometry. Do not add `--fit-layout` or switch to automatic scene placement when reproducing this result.

## What succeeds, and what remains wrong

Immediately before attachment, insertion is 10.65 mm, lateral error is 0.090 mm and axis error is 1.13 degrees. The complete 100 ms dwell satisfies the current gate. Maximum tool–tip penetration through attachment is 0.0102 mm, so the initial insertion does not rely on the later large penetration.

The model still has two significant problems:

- At initialization, the index, middle and ring finger colliders penetrate the handle by approximately 11.2–11.5 mm. Within 0.1 seconds the tool moves about 6 mm and rotates about 15 degrees relative to the hand in the separately diagnosed nearby seed. At attachment in the successful candidate, drift from the first 5 ms sample is 13.45 mm and 16.91 degrees. Placement optimization cannot repair this grasp initialization.
- After attachment, tool–tip penetration first exceeds 0.1 mm at 7.8275 seconds and reaches 0.6334 mm at 8.264 seconds, with a recorded peak normal force of 41.34 N. The attachment constraint remains active. The constraint/contact response under subsequent motion needs investigation before calling this physically reliable.

The search's `valid` field screens initial rack overlap, rack/lid penetration, finite state and numerical warnings. It does not screen initial finger/handle overlap or post-attachment tool/tip penetration. Its success flag is therefore a task-event result, not physical validation. `audit.json` explicitly records `physically_validated: false`.

Next, fit a nonintersecting wrist-to-tool pose and finger closure that maintains the measured real grasp under gravity. Validate translation and rotation, not just whether the tool falls. Then investigate the attachment constraint/contact compliance under load and repeat this placement search, since correcting the grasp changes the nozzle trajectory. Preserve the required insertion depth and contact-enabled geometry during that work.

## Reproduce the diagnosis

```bash
~/miniconda3/envs/lerobot-act/bin/python tools/diagnose_replay_insertion.py \
  --episode 15 --config outputs/episode15-contact-layout-search/best.json \
  --output outputs/episode15-contact-diagnosis/recheck
```

The diagnostic saves exact config and dataset identity, insertion gate durations, grasp drift, initial tool contacts, the full simulator report, and a 5 ms trace CSV. Drift is relative to the first sampled pose at 5 ms. Gate-duration summaries are sampled; the attachment event itself is evaluated at the simulator's 0.5 ms timestep.

Artifacts:

- `outputs/episode15-contact-layout-search/search.json`: all 16 evaluated placements.
- `outputs/episode15-contact-layout-search/verification/report.json`: independent normal-entry-point replay.
- `outputs/episode15-contact-layout-search/audit.json`: physical-validation caveats.
- `outputs/episode15-contact-diagnosis/best/diagnosis.json`: selected candidate diagnosis.
- `outputs/episode15-contact-diagnosis/best-contact-timing.json`: contact threshold crossings before/after attachment.
- `outputs/episode15-contact-diagnosis/initial-grasp-contacts.json`: startup finger/handle contacts.
- `outputs/episode15-contact-diagnosis/real-episode15.jpg`: three recorded camera views at 0, 3, 6, 9 and 12 seconds.
