"""Separate command tracking, measured-arm playback, grasp and insertion errors.

All modes use one frozen per-episode layout. Counterfactual modes are diagnostics,
never policy demonstrations or published successful pickups.
"""
import argparse, copy, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.dynamics import DynamicSimulation
from g1_pipette_sim.simulation import set_robot_pose
from g1_pipette_sim.pickup_layouts import ROOT, MANIFEST, model_fingerprints, sha256
from tools.fit_pickup_layouts import write_json

MODES = ['command', 'state_targets', 'measured_arm', 'rigid_tool', 'compliant_tip', 'effort_bias', 'measured_arm_rigid_tool']

class MeasuredArmSimulation(DynamicSimulation):
    """Impose only the 29 measured body joints; hand and objects stay dynamic.

This injects unmeasured constraint forces. It must not be used for RL or counted
as success under commanded actions. State-as-servo-target is a separate mode.
"""
    def _impose(self, t):
        q, _, _ = self.episode.sample(t, 'state')
        h = .001
        lo, hi = max(0., t-h), min(self.episode.duration, t+h)
        velocity = (self.episode.sample(hi, 'state')[0]-self.episode.sample(lo, 'state')[0])/max(hi-lo, 1e-9)
        for name, pos, vel in zip(self.episode.names, q, velocity):
            joint = self.model.joint(name).id
            self.data.qpos[self.model.jnt_qposadr[joint]] = pos
            self.data.qvel[self.model.jnt_dofadr[joint]] = vel
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        self._impose(self.time)
        super().step()
        self._impose(self.time)


class EffortBiasSimulation(DynamicSimulation):
    """Test the existing shared motor-effort regression's constant torque terms.

These offsets may include sensor bias or unmodeled friction, so they are a
hypothesis until held-out dynamic tracking improves, not calibrated ground truth.
"""
    def _apply_controller(self):
        super()._apply_controller()
        if not hasattr(self, '_effort_bias'):
            fit=json.loads((ROOT/'outputs/hil194/pickup-layouts/servo-gain-check.json').read_text())
            offsets={r['joint']:r['bias'] for r in fit['fits']}
            self._effort_bias=np.array([offsets.get(self.model.joint(int(j)).name,0.) for j in self.joints])
        self.data.ctrl[:]+=self._effort_bias/self.kp


def decompose(actual, arm_fk, measured_fk):
    """Exact vector identity; component norms generally do not add."""
    return actual-measured_fk, arm_fk-measured_fk, actual-arm_fk


def gate_summary(times, radial, depth, angle, settings):
    masks = {
        'radial': radial < settings['radial_tolerance'],
        'depth': (depth >= settings['min_insertion_depth']) & (depth <= settings['max_insertion_depth']),
        'angle': angle < settings['angle_tolerance_deg'],
    }
    masks['all'] = masks['radial'] & masks['depth'] & masks['angle']
    result = {}
    for key, mask in masks.items():
        start = None; longest = 0.; first = None
        for t, ok in zip(times, mask):
            if ok:
                if start is None: start = t
                if first is None: first = t
                longest = max(longest, t-start)
            else: start = None
        result[key] = {'fraction': float(np.mean(mask)), 'longest_sampled_span_s': float(longest), 'first_sample_s': None if first is None else float(first)}
    return result


def evaluate(job):
    index, mode, cfg, output, fingerprints = job
    ep = Episode(index=index); cfg = copy.deepcopy(cfg)
    if mode in {'rigid_tool', 'measured_arm_rigid_tool'}: cfg['physics']['grasp_mode'] = 'weld'
    cls = MeasuredArmSimulation if mode in {'measured_arm', 'measured_arm_rigid_tool'} else EffortBiasSimulation if mode == 'effort_bias' else DynamicSimulation
    sim = cls(cfg, ep, source='state' if mode == 'state_targets' else 'low_level')
    if mode == 'compliant_tip':
        # Hypothesis probe: 100x smaller rotational weld scale, not a measured fit.
        sim.model.eq_data[sim.retention_id, 10] = .01
    m, d = sim.model, sim.data
    arm_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in ep.names])
    measured = mujoco.MjData(sim.reference_model); arm = mujoco.MjData(sim.reference_model)
    arrays = {k: [] for k in ['time', 'q', 'observed', 'actual_nozzle', 'arm_fk_nozzle', 'measured_fk_nozzle', 'wrist_error', 'tip_mouth', 'radial', 'depth', 'angle', 'in_rack', 'grip_translation', 'grip_rotation', 'measured_axis', 'actual_axis']}
    stride = max(1, round(.01/m.opt.timestep)); step = 0; anchor = None; max_lift = 0.; held = run = 0.; initial_tip = d.body('tip').xpos.copy()
    first_contacts = {}; first_divergence = {}; thresholds = [1., 2., 5., 10.]
    while sim.time < ep.duration-1e-9:
        sim.step(); step += 1
        tip = d.body('tip'); R = tip.xmat.reshape(3,3); mouth = tip.xpos+R[:,2]*cfg['pipette']['tip_length']/2
        if sim.state == 'attached':
            if anchor is None: anchor = float(mouth[2])
            height = float(mouth[2]-anchor); max_lift = max(max_lift, height)
            lifted = height >= cfg['pickup_fit_criteria']['required_lift_m'] and mouth[2]-d.site('rack_mouth').xpos[2] >= .002
            run = run+m.opt.timestep if lifted else 0.; held = max(held, run)
        for pair in sim.contact_diagnostics:
            first_contacts.setdefault(pair, sim.time)
        if step % stride: continue
        q, left, right = ep.sample(sim.time, 'state')
        set_robot_pose(sim.reference_model, measured, ep.names, q, left, right, cfg)
        set_robot_pose(sim.reference_model, arm, ep.names, d.qpos[arm_q], left, right, cfg)
        mujoco.mj_forward(sim.reference_model, measured); mujoco.mj_forward(sim.reference_model, arm)
        nozzle = d.site('nozzle'); error = R.T@(nozzle.xpos-mouth)
        grip = d.site('grip_frame'); tool = d.body('pipette')
        rotation = grip.xmat.reshape(3,3).T@tool.xmat.reshape(3,3)
        values = [sim.time, d.qpos[arm_q].copy(), q, nozzle.xpos.copy(), arm.site('nozzle').xpos.copy(), measured.site('nozzle').xpos.copy(), d.body('right_wrist_yaw_link').xpos-measured.body('right_wrist_yaw_link').xpos, mouth.copy(), np.linalg.norm(error[:2]), -error[2], np.degrees(np.arccos(np.clip(R[:,2]@nozzle.xmat.reshape(3,3)[:,2],-1,1))), sim.state=='in_rack', np.linalg.norm(tool.xpos-grip.xpos), np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))), measured.site('nozzle').xmat.reshape(3,3)[:,2].copy(), nozzle.xmat.reshape(3,3)[:,2].copy()]
        for key, value in zip(arrays, values): arrays[key].append(value)
        _, control_error, grasp_error = decompose(values[3], values[4], values[5])
        for label, e in [('arm_tracking', control_error), ('tool_relative_to_mount', grasp_error)]:
            for threshold in thresholds:
                if np.linalg.norm(e)*1000 > threshold: first_divergence.setdefault(f'{label}_over_{threshold:g}_mm', sim.time)
    arrays = {k: np.asarray(v) for k,v in arrays.items()}
    total, control, grasp = decompose(arrays['actual_nozzle'], arrays['arm_fk_nozzle'], arrays['measured_fk_nozzle'])
    def rms(x): return float(np.sqrt(np.mean(np.sum(x*x, axis=1)))*1000)
    mask = arrays['in_rack'].astype(bool)
    report = sim.report(); target = cfg['pickup_fit_criteria']
    contacts_ok = all(s['maximum_penetration_mm'] <= .5 and s['maximum_normal_force_N'] <= 100 for s in report['contact_diagnostics'].values())
    result = {'episode': index, 'mode': mode, 'diagnostic_only': mode!='command', 'source': sim.source,
        'dataset': str(ep.root), 'model_fingerprints': fingerprints,
        'diagnostic_script_sha256': sha256(Path(__file__)),
        'effort_regression_sha256': sha256(ROOT/'outputs/hil194/pickup-layouts/servo-gain-check.json') if mode=='effort_bias' else None,
        'scope': 'Fixed saved layout. FK assumes configured mount; no measured 3D tool ground truth. Counterfactuals are not policy demonstrations.',
        'right_arm_rmse_rad': float(np.sqrt(np.mean((arrays['q'][:,22:29]-arrays['observed'][:,22:29])**2))),
        'wrist_rmse_mm': rms(arrays['wrist_error']), 'nozzle_rmse_vs_measured_fk_mm': rms(total),
        'arm_tracking_nozzle_rmse_mm': rms(control), 'grasp_nozzle_rmse_mm': rms(grasp),
        'maximum_grip_translation_mm': float(arrays['grip_translation'].max()*1000),
        'maximum_grip_rotation_deg': float(arrays['grip_rotation'].max()),
        'first_divergence_s': first_divergence, 'first_tool_contact_s': first_contacts,
        'pre_attachment_gates': gate_summary(arrays['time'][mask], arrays['radial'][mask], arrays['depth'][mask], arrays['angle'][mask], cfg['attachment']) if mask.any() else {},
        'world_lift_mm': max_lift*1000, 'recorded_fk_lift_mm': target['target_m']*1000,
        'lift_dwell_s': held, 'tool_contact_screen_passed': contacts_ok,
        'height_and_dwell_match_only': bool(held >= .1 and max_lift <= target['target_m']+target['tolerance_m'] and sim.state=='attached'),
        'published_success': False, 'report': report}
    out = Path(output)/f'episode-{index:03}'/mode; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'trace.npz', **arrays); write_json(out/'config.json', cfg); write_json(out/'diagnosis.json', result)
    return {k:v for k,v in result.items() if k not in ['report','model_fingerprints','pre_attachment_gates','scope']} | {'events': report['events'], 'warnings': report['warnings']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, nargs='+', default=[46,52,14,160,193])
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/hil194/replay-divergence')
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    manifest=json.loads(MANIFEST.read_text()); fp=model_fingerprints()
    if manifest['model_fingerprints'] != fp: raise ValueError('Stale pickup manifest')
    jobs=[]
    for index in a.episodes:
        entry=manifest['episodes'][str(index)]; path=ROOT/entry['config']
        if sha256(path)!=entry['sha256']: raise ValueError('Modified saved layout')
        cfg=json.loads(path.read_text())
        for mode in a.modes: jobs.append((index,mode,cfg,str(a.output),fp))
    write_json(a.output/'input-manifest.json',manifest)
    write_json(a.output/'run.json',{'episodes':a.episodes,'modes':a.modes,'script_sha256':sha256(Path(__file__)),'model_fingerprints':fp,'workers':a.workers})
    summary={'results':[], 'errors':[]}
    with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as pool:
        pending={pool.submit(evaluate,j):(j[0],j[1]) for j in jobs}
        for f in as_completed(pending):
            try:
                result=f.result(); summary['results'].append(result)
                print(json.dumps(result),flush=True)
            except Exception as error:
                summary['errors'].append({'job':pending[f], 'error':repr(error)}); print('ERROR',pending[f],repr(error),flush=True)
            write_json(a.output/'summary.json',summary)
    if summary['errors']: raise SystemExit(1)
if __name__=='__main__': main()
