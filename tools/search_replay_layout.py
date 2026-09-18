"""Search rack/tip initial XY placement for unchanged dynamic command replay.

This is trajectory-conditioned simulation fitting, not real-scene reconstruction.
Run from the project root. Every candidate starts a fresh full dynamic replay.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode, DEFAULT_DATASET
from g1_pipette_sim.scene import load_config
from g1_pipette_sim.episode_scene import resolve_scene
from g1_pipette_sim.dynamics import DynamicSimulation


def footprint(cfg, name, padding=0.):
    obj = cfg[name]
    if name=='waste' and obj.get('hinge',{}).get('enabled'):
        from g1_pipette_sim.box_lid import pose
        from scipy.spatial import ConvexHull
        origin, rotation=pose(cfg);x,y=np.asarray(obj['size_xy'])/2+padding
        vertices=np.array([[a,b,z] for a in [-x,x] for b in [-y,y] for z in [0,obj['height']]])@rotation.T+origin
        points=vertices[:,:2];return points[ConvexHull(points).vertices]
    angle = math.radians(obj.get('yaw_deg', 0))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    center = np.asarray(cfg['rack']['center_xy'])
    if name == 'waste':
        center = center + obj['offset_from_rack']
    half = np.asarray(obj['size_xy']) / 2 + padding
    return (np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * half) @ rotation.T + center


def separated(a, b):
    for polygon in [a, b]:
        for edge in np.diff(np.vstack([polygon, polygon[0]]), axis=0):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return True
    return False


def shifted_config(base, delta, lid_placement='carry'):
    """Move rack/tip; carry lid with clearance or explicitly preserve its world pose."""
    cfg = copy.deepcopy(base)
    cfg['rack']['center_xy'] = (np.asarray(base['rack']['center_xy']) + delta).tolist()
    offset = np.asarray(base['waste']['offset_from_rack'], dtype=float)
    if cfg['waste'].get('hinge',{}).get('enabled'):
        if lid_placement=='fixed':raise ValueError('An attached lid cannot remain fixed while moving its rack; use --lid-placement carry')
    elif lid_placement == 'fixed':
        offset = offset - delta
    else:
        # Include a conservative allowance for protruding lid hinges and rack ribs.
        direction = offset / np.linalg.norm(offset) if np.linalg.norm(offset) > 1e-9 else np.array([0., -1.])
        for _ in range(500):
            cfg['waste']['offset_from_rack'] = offset.tolist()
            if separated(footprint(cfg, 'rack', .006), footprint(cfg, 'waste', .006)):
                break
            offset = offset + .001 * direction
        else:
            raise ValueError('Could not separate rack and lid footprints')
    cfg['waste']['offset_from_rack'] = offset.tolist()
    cfg.pop('scene_initialization', None)
    cfg.setdefault('episode_scene', {})['enabled'] = False
    return cfg


def on_table(cfg):
    return all(np.all(np.abs(footprint(cfg, name) - cfg['table']['center_xy']) <=
                      np.asarray(cfg['table']['size_xy']) / 2 - .005)
               for name in ['rack', 'waste'])


def rack_penetrations(sim, tolerance_m=.0001):
    """Initial placement screen; table support and retained tip are expected contacts."""
    bad = []
    for contact in sim.data.contact:
        pair = sorted(sim.model.body(int(sim.model.geom_bodyid[g])).name
                      for g in [contact.geom1, contact.geom2])
        if 'rack' not in pair or 'table_body' in pair or 'tip' in pair:
            continue
        if contact.dist < -tolerance_m:
            bad.append({'bodies': pair, 'depth_mm': float(-contact.dist * 1000)})
    return bad


def evaluate(cfg, episode, source, lift_height):
    sim = DynamicSimulation(cfg, episode, source=source)
    initial_overlaps = rack_penetrations(sim)
    if initial_overlaps:
        return {'score': 1e6, 'attached': False, 'success': False, 'valid': False,
                'initial_overlaps': initial_overlaps, 'lift_m': 0., 'report': sim.report()}, np.empty((0, 8))
    maximum_lid_penetration = 0.
    settings = cfg['attachment']
    lower = settings.get('min_insertion_depth', -settings.get('axial_tolerance', .005))
    upper = settings.get('max_insertion_depth', settings.get('axial_tolerance', .005))
    trace, violations = [], []
    lift = 0.
    step_stride = max(1, round(.005 / sim.model.opt.timestep))
    step_count = 0
    while sim.time < episode.duration:
        sim.step()
        step_count += 1
        for contact in sim.data.contact:
            pair = {sim.model.body(int(sim.model.geom_bodyid[g])).name for g in [contact.geom1, contact.geom2]}
            if pair == {'rack', 'waste'}:
                maximum_lid_penetration = max(maximum_lid_penetration, -float(contact.dist))
        if sim.state == 'attached':
            # Lift relative to the moving rack, while attachment remains active.
            mouth = sim.data.body('tip').xpos + sim.data.body('tip').xmat.reshape(3, 3)[:, 2] * cfg['pipette']['tip_length'] / 2
            lift = max(lift, float(mouth[2] - sim.data.site('rack_mouth').xpos[2]))
        if step_count % step_stride or sim.state != 'in_rack':
            continue
        tip = sim.data.body('tip')
        rotation = tip.xmat.reshape(3, 3)
        mouth = tip.xpos + rotation[:, 2] * cfg['pipette']['tip_length'] / 2
        nozzle = sim.data.site('nozzle')
        error = rotation.T @ (nozzle.xpos - mouth)
        radial, depth = np.linalg.norm(error[:2]), -error[2]
        angle = np.degrees(np.arccos(np.clip(rotation[:, 2] @ nozzle.xmat.reshape(3, 3)[:, 2], -1, 1)))
        # Smooth, dimensionless proxy. Actual success uses simulator events.
        violations.append(float(max(radial / settings['radial_tolerance'],
            1 + max(lower - depth, depth - upper) / max(upper - lower, .001),
            angle / settings['angle_tolerance_deg'])))
        trace.append([sim.time, *nozzle.xpos, *mouth, angle])
    report = sim.report()
    attached = any(e['state'] == 'attached' for e in sim.events)
    valid = report['finite'] and not report['warnings'] and maximum_lid_penetration <= .0001
    success = bool(valid and attached and lift >= lift_height)
    window = max(1, math.ceil(settings['dwell_seconds'] / (step_stride * sim.model.opt.timestep)))
    # Minimum worst violation over a full dwell window, excluding padded edges.
    if len(violations) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(np.asarray(violations), window)
        score = float(np.min(np.max(windows, axis=1)))
    else:
        score = float(max(violations, default=1e6))
    if attached:
        score = -1 - min(lift / lift_height, 1.)
    if not valid:
        score = 1e6
    result = {'score': score, 'attached': attached, 'success': success,
              'lift_m': lift, 'valid': bool(valid), 'initial_overlaps': initial_overlaps,
              'maximum_lid_penetration_mm': maximum_lid_penetration * 1000, 'report': report}
    return result, np.asarray(trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, default=15)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--config', type=Path, default=Path('config/hil194.json'))
    parser.add_argument('--scene-layout', choices=['auto', 'fixed'], default='auto')
    parser.add_argument('--lid-placement', choices=['carry', 'fixed'], default='carry',
                        help='carry: move lid with rack and add clearance; fixed: preserve lid world pose and reject collisions.')
    parser.add_argument('--source', choices=['low_level', 'action'], default='low_level')
    parser.add_argument('--max-evaluations', type=int, default=24)
    parser.add_argument('--max-shift', type=float, default=.12, help='Maximum rack translation norm in metres.')
    parser.add_argument('--lift-height', type=float, default=.02, help='Required attached lift above rack mouth, metres.')
    parser.add_argument('--output', type=Path, default=Path('outputs/layout-search'))
    args = parser.parse_args()
    if args.max_evaluations < 2 or args.max_shift <= 0 or args.lift_height <= 0:
        parser.error('Require at least two evaluations and positive shift/lift limits.')
    episode = Episode(args.dataset, args.episode)
    base, _ = resolve_scene(load_config(args.config), episode, args.scene_layout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'baseline.json').write_text(json.dumps(base, indent=2))
    history, seen = [], set()
    best = None

    def trial(delta):
        nonlocal best
        delta = np.asarray(delta, dtype=float)
        key = tuple(np.round(delta, 6))
        if key in seen or np.linalg.norm(delta) > args.max_shift + 1e-9:
            return None
        cfg = shifted_config(base, delta, args.lid_placement)
        if not on_table(cfg):
            return None
        seen.add(key)
        print(f'Evaluating candidate {len(history) + 1}/{args.max_evaluations}: shift={delta.tolist()} m', flush=True)
        result, trace = evaluate(cfg, episode, args.source, args.lift_height)
        summary = {k: v for k, v in result.items() if k != 'report'}
        summary.update(evaluation=len(history), shift_xy_m=delta.tolist(), rack_center_xy=cfg['rack']['center_xy'])
        history.append(summary)
        if result.get('valid', False) and (best is None or result['score'] < best['score']):
            best = summary
            cfg['layout_search'] = {'method': 'dynamic_replay_xy_search', 'episode': args.episode,
                'dataset': str(episode.root) if hasattr(episode, 'root') else str(args.dataset),
                'source': args.source, 'clock': 'recorded', 'result': summary,
                'baseline_config': str((args.output / 'baseline.json').resolve()),
                'lid_placement': args.lid_placement, 'max_shift_m': args.max_shift, 'required_lift_m': args.lift_height,
                'max_evaluations': args.max_evaluations,
                'meaning': 'Trajectory-conditioned simulation fit; not recovered real-world placement.'}
            (args.output / 'best.json').write_text(json.dumps(cfg, indent=2))
            (args.output / 'best-report.json').write_text(json.dumps(result['report'], indent=2))
        (args.output / 'search.json').write_text(json.dumps({'best': best, 'trials': history}, indent=2))
        print(json.dumps(summary), flush=True)
        return trace

    trace = trial([0., 0.])
    if trace is None or not len(trace):
        parser.error('Baseline is outside table bounds or has initial collisions; correct the supplied scene first.')
    # Propose XY shifts near the observed path, prioritizing samples at insertion height.
    depth = trace[:, 6] - trace[:, 3]
    a = base['attachment']
    desired_depth = (a.get('min_insertion_depth', 0.) + a.get('max_insertion_depth', 0.)) / 2
    ranked = np.argsort(np.abs(depth - desired_depth) + .001 * np.maximum(trace[:, 7] - a['angle_tolerance_deg'], 0))
    seeds = []
    for i in ranked:
        delta = trace[i, 1:3] - trace[i, 4:6]
        if all(np.linalg.norm(delta - seed) > .004 for seed in seeds):
            seeds.append(delta)
        if len(seeds) >= 6:
            break
    for seed in seeds:
        if len(history) >= args.max_evaluations or best['success']:
            break
        trial(seed)
    # Coarse-to-fine coordinate search after dynamic re-evaluation of path seeds.
    for spacing in [.008, .004, .002, .001, .0005]:
        improved = True
        while improved and len(history) < args.max_evaluations and not best['success']:
            origin, old_score = np.asarray(best['shift_xy_m']), best['score']
            for offset in [[spacing, 0], [-spacing, 0], [0, spacing], [0, -spacing]]:
                if len(history) >= args.max_evaluations or best['success']:
                    break
                trial(origin + offset)
            improved = best['score'] < old_score
    print('SUCCESS: attachment and requested lift verified.' if best['success'] else
          f"Lift goal not met within budget. Best attached={best['attached']}, lift={best['lift_m'] * 1000:.2f} mm (required {args.lift_height * 1000:.2f} mm).", flush=True)
    print(f"Best config: {args.output / 'best.json'}; replay with --scene-layout fixed --episode {args.episode} --source {args.source}", flush=True)


if __name__ == '__main__':
    main()
