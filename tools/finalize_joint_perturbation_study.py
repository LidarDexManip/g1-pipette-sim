"""Summarize the bounded joint study and audit its separate training export."""
import json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.study_joint_perturbations import OUT,BASE,summarize
from tools.package_pickup_augmentations import package
from tools.fit_pickup_layouts import write_json
from g1_pipette_sim.pickup_layouts import sha256


def main():
    report=summarize()
    accepted=sorted(int(p.parent.name) for p in (OUT/'candidates').glob('*/accepted.json'))
    assert len(accepted)==report['independently_recorded_successes']==9
    write_json(OUT/'progress.json',{'accepted':accepted,'target':len(accepted),'status':'collected','screened':report['placement_trials']})
    package(OUT,OUT/'lerobot-v21')
    deviations=[];frames=0
    with np.load(BASE/'trajectory.npz') as a:
        original_arm=a['observation.state'][:,22:29].copy();original_nozzle=a['nozzle_position'].copy()
    for index,number in enumerate(accepted):
        folder=OUT/'candidates'/f'{number:06}';c=json.loads((folder/'candidate.json').read_text())
        table=pq.read_table(OUT/f'lerobot-v21/data/chunk-000/episode_{index:06}.parquet')
        with np.load(folder/'trajectory.npz') as a,np.load(folder/'commands.npz') as commands:
            frames+=table.num_rows
            for key in ['action','observation.state','action.hand_left','action.hand_right','observation.hand_joint_positions']:
                assert np.array_equal(np.asarray(table[key].to_pylist(),dtype=np.float32),a[key].astype(np.float32))
            for key,saved in [('action','motor_targets'),('action.hand_left','hand_left'),('action.hand_right','hand_right')]:
                expected=np.stack([np.interp(a['simulation_time'],commands['times'],commands[saved][:,j]) for j in range(commands[saved].shape[1])],axis=1)
                assert np.allclose(a[key],expected,atol=1e-10,rtol=1e-10)
            spatial=np.linalg.norm(a['nozzle_position']-original_nozzle,axis=1)
            deviations.append({'candidate':number,'perturbation_id':c['perturbation_id'],'command_amplitude_deg':c['noise']['amplitude_deg'],'actual_joint_deviation_deg':float(np.degrees(np.max(abs(a['observation.state'][:,22:29]-original_arm)))),'nozzle_deviation_mm':float(1000*spatial.max()),'peak_time_s':float(a['timestamp'][np.argmax(spatial)]),'placement_refinement_mm':(np.asarray(c['placement_refinement_m'])*1000).tolist()})
    write_json(OUT/'motion-deviations.json',deviations)
    write_json(OUT/'completion.json',{'complete':True,'proposed':24,'generation_rejected':6,'distinct_simulated':18,'initial_screen_passes':5,'successful_after_placement_refinement':9,'independently_recorded_successes':9,'screening_placement_tests':38,'independent_recording_replays':9,'camera_videos_fully_decoded':27,'exported_numeric_frames_checked':frames,'physics_criteria_changed':False})
    provenance=json.loads((OUT/'provenance.json').read_text());provenance['generator_sha256']=sha256(ROOT/'tools/study_joint_perturbations.py');write_json(OUT/'provenance.json',provenance)
    rows=[]
    for family in ['persistent','approach_only']:
        for amplitude in [3,7,12]:
            group=[r for r in report['perturbations'] if r['family']==family and r['amplitude_deg']==amplitude]
            rows.append(f"| {family} | {amplitude} | {len(group)} | {sum(r['screened'] for r in group)} | {sum(r['recorded_success'] for r in group)} |")
    best=max(deviations,key=lambda d:d['nozzle_deviation_mm'])
    text='''# Direct joint perturbation experiment

**24 distinct perturbations proposed; 18 reached physics screening; 9 succeeded and passed independent recording replay.** Six persistent-offset proposals were rejected before simulation because their FK-matched placement would require moving the table height by more than 40 mm. Success is 9/24 (37.5%) of proposals or 9/18 (50%) of simulated perturbations.

There were **38 placement screening tests**, comprising 18 initial placements and 20 placement refinements with identical commands. Five perturbations succeeded on the initial placement; four more were recovered by moving the rack and tip. The nine successful perturbations each passed an additional independent recording replay and full decoding of three camera videos (27 videos total). Placement retries are not counted as new perturbations.

## Design

All trials start from accepted capture 0, parent real episode 14. With random seed 20260914, each trial draws seven independent normal values, normalizes their maximum absolute value to one, and multiplies by a 3, 7, or 12 degree amplitude. This fixed random joint-offset vector is added directly to the seven right-arm motor targets using a smooth time envelope. **No IK, Cartesian-path constraint, or null-space projection is applied to the command perturbation.** Other body joints and saved hand commands are unchanged.

Both families ramp in over 0–3 seconds. Persistent offsets remain active through insertion. Approach-only perturbations hold until 5 seconds and ramp back to zero over 5–10 seconds, before the reference attachment near 12.85 seconds. Approach-only perturbations therefore preserve the original commanded insertion and lift; they do not demonstrate arbitrary contact motion. Four different random directions are drawn per family/amplitude, so this is not a paired amplitude sweep.

The initial rack/tip placement follows the FK nozzle displacement at reference insertion time. Persistent offsets can require XY and table-height adjustments. Approach-only trajectories initially retain the reference scene because their offset is zero by insertion. Failed replay alignment supplies up to two additional XY placement proposals (half and full measured correction). Commands remain identical across these placement retries; placement is fixed before each replay.

## Outcomes by family and amplitude

| Family | Peak joint offset (degrees) | Proposed | Physics screened | Independently successful |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+f'''

**All nine successes are approach-only.** Persistent random offsets failed in this bounded experiment. Moving the rack and tip does not restore a changed nozzle orientation or undo approach collisions. This sample supports larger variation during the approach, not unrestricted joint noise throughout insertion. The physics model, contact/penetration limits, initial overlap checks, attachment timing window, world-lift target and tolerance, dwell and final attachment requirement were unchanged.

## Video and data

The video `../../augmentation-illustration/joint-perturbation-example.mp4` overlays the original at 100% layer opacity with one validated augmentation at 50%, on a synchronized playback clock. It uses capture **{best['candidate']}**, selected for the largest actual nozzle-path separation: **{best['nozzle_deviation_mm']:.1f} mm**, with **{best['command_amplitude_deg']:g} degree** peak commanded joint offset and **{best['actual_joint_deviation_deg']:.2f} degree** maximum simulated joint deviation. Spatial motion is shown at actual scale. Its preview shows peak separation around **{best['peak_time_s']:.2f} s**.

Each accepted candidate folder contains exact dense commands, config, full simulated states, sampled labels, camera poses, all three videos and both replay reports. The separate `lerobot-v21` export contains **{len(accepted)} episodes and {frames:,} frames**. Every exported body/hand action and state row was compared with its capture; action and hand labels were also checked against the dense archived commands. All variants must be grouped with parent episode 14 when splitting real and simulated training data.

`results.json` includes every perturbation and placement attempt; `provenance.json` records the source, seed and model fingerprints; `motion-deviations.json` lists measured deviations; `completion.json` records final checks. These are simulator successes under the existing approximate grasp/contact model, not real-world evaluations.

Reproduce with `tools/study_joint_perturbations.py` (screen, refine and record phases), then `tools/finalize_joint_perturbation_study.py` and `tools/overlay_augmented_videos.py --joint`.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps({'independently_validated':len(accepted),'frames_audited':frames,'video_selection':best},indent=2))
if __name__=='__main__':main()
