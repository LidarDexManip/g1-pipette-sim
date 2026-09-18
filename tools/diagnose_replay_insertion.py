"""Record insertion gates and grasp drift during an unchanged dynamic replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode, DEFAULT_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.dynamics import DynamicSimulation


def longest(mask, dt):
    run = best = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best * dt


def diagnose(config, episode, source):
    sim = DynamicSimulation(config, episode, source=source)
    initial_tool_contacts = []
    for contact in sim.data.contact:
        bodies = [sim.model.body(int(sim.model.geom_bodyid[g])).name for g in [contact.geom1, contact.geom2]]
        if 'pipette' in bodies and contact.dist < 0:
            initial_tool_contacts.append({'bodies': bodies, 'penetration_mm': float(-contact.dist * 1000)})
    a = config['attachment']
    stride = max(1, round(.005 / sim.model.opt.timestep))
    dt = stride * sim.model.opt.timestep
    rows = []
    initial = None
    count = 0
    while sim.time < episode.duration:
        sim.step()
        count += 1
        if count % stride:
            continue
        tip = sim.data.body('tip')
        rotation = tip.xmat.reshape(3, 3)
        mouth = tip.xpos + rotation[:, 2] * config['pipette']['tip_length'] / 2
        nozzle = sim.data.site('nozzle')
        error = rotation.T @ (nozzle.xpos - mouth)
        angle = np.degrees(np.arccos(np.clip(rotation[:, 2] @ nozzle.xmat.reshape(3, 3)[:, 2], -1, 1)))
        grip = sim.data.site('grip_frame')
        tool = sim.data.body('pipette')
        relative_rotation = grip.xmat.reshape(3, 3).T @ tool.xmat.reshape(3, 3)
        relative_position = grip.xmat.reshape(3, 3).T @ (tool.xpos - grip.xpos)
        if initial is None:
            initial = (relative_position.copy(), relative_rotation.copy())
        drift_angle = np.degrees(np.arccos(np.clip((np.trace(initial[1].T @ relative_rotation) - 1) / 2, -1, 1)))
        rows.append([sim.time, *error, angle, np.linalg.norm(relative_position - initial[0]), drift_angle, float(sim.state == 'in_rack')])
    trace = np.asarray(rows)
    t = trace[trace[:, 7] == 1]
    radial = np.linalg.norm(t[:, 1:3], axis=1)
    depth = -t[:, 3]
    radial_ok = radial < a['radial_tolerance']
    depth_ok = (depth >= a['min_insertion_depth']) & (depth <= a['max_insertion_depth'])
    angle_ok = t[:, 4] < a['angle_tolerance_deg']
    violation = np.maximum.reduce([radial/a['radial_tolerance'], t[:, 4]/a['angle_tolerance_deg'],
        1+np.maximum(a['min_insertion_depth']-depth,depth-a['max_insertion_depth'])/max(a['max_insertion_depth']-a['min_insertion_depth'],.001)])
    window = int(np.ceil(a['dwell_seconds']/dt))
    start = int(np.argmin(np.max(np.lib.stride_tricks.sliding_window_view(violation, window), axis=1)))
    def summary(mask):
        return {'total_s': float(np.sum(mask)*dt), 'longest_continuous_s': longest(mask,dt)}
    def sample(i):
        return {'time_s': float(t[i,0]), 'radial_mm': float(radial[i]*1000), 'depth_mm': float(depth[i]*1000),
                'angle_deg': float(t[i,4]), 'grasp_translation_drift_mm': float(t[i,5]*1000), 'grasp_rotation_drift_deg': float(t[i,6])}
    result = {'sample_period_s': float(dt), 'required_dwell_s': a['dwell_seconds'],
        'initial_tool_contacts': initial_tool_contacts, 'grasp_drift_reference_time_s': float(trace[0,0]),
        'gates': {name:summary(mask) for name,mask in [('radial',radial_ok),('depth',depth_ok),('angle',angle_ok),('radial_and_depth',radial_ok&depth_ok),('all',radial_ok&depth_ok&angle_ok)]},
        'closest_radial': sample(int(np.argmin(radial))), 'best_instantaneous_gate_match': sample(int(np.argmin(violation))),
        'best_dwell_window': {'start_s': float(t[start,0]), 'end_s': float(t[start+window-1,0]),
            'max_radial_mm': float(radial[start:start+window].max()*1000), 'depth_range_mm': (np.array([depth[start:start+window].min(),depth[start:start+window].max()])*1000).tolist(),
            'max_angle_deg': float(t[start:start+window,4].max())},
        'maximum_grasp_translation_drift_mm': float(trace[:,5].max()*1000), 'maximum_grasp_rotation_drift_deg': float(trace[:,6].max()),
        'report':sim.report()}
    return result, trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--episode',type=int,default=15)
    parser.add_argument('--source',choices=['low_level','action'],default='low_level')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    episode=Episode(args.dataset,args.episode)
    result,trace=diagnose(config,episode,args.source)
    result['input']={'config':str(args.config.resolve()),'dataset':str(episode.root),'episode':args.episode,'source':args.source,'scene_layout':'fixed'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    (args.output/'diagnosis.json').write_text(json.dumps(result,indent=2))
    np.savetxt(args.output/'trace.csv',trace,delimiter=',',header='time_s,error_x_m,error_y_m,error_z_m,angle_deg,grasp_translation_drift_m,grasp_rotation_drift_deg,in_rack',comments='')
    print(json.dumps({k:v for k,v in result.items() if k!='report'},indent=2))


if __name__=='__main__':
    main()
