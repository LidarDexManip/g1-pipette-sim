"""Bounded deterministic collection. Receipts, not partial files, authorize reuse."""
import copy
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
from .contracts import atomic_json, bind_run, dependencies, digest, file_hash, lock, seal, verify
from .sources import import_seed, load_episode, model_assets
from .methods import generate, lift_revision
from .validation import preflight
from g1_pipette_sim.camera_views import apply_camera_views
from .store import Ledger
from tools.collect_pickup_augmentations import validate, placed_config, SIZES


def candidate_inputs(c):
    return {'candidate': c, 'run_id': c['run_id']}


def stage(c, record=False):
    folder = Path(c['folder']); name = 'record' if record else 'screen'
    inputs = candidate_inputs(c)
    # Validate source/config/archive BEFORE inspecting any result cache.
    load_episode(c)
    if verify(folder, name, inputs):
        return json.loads((folder / ('accepted.json' if record else 'screen.json')).read_text())['validation' if record else 'result']
    if record and verify(folder, 'record-failed', inputs):
        return json.loads((folder/'verification.json').read_text())['result']
    if (folder / 'accepted.json').exists():
        # Interruption after atomic acceptance but before receipt: preserve evidence,
        # audit and seal it, never rerender over a potentially corrupt acceptance.
        if not record:
            raise ValueError('Unexpected uncommitted acceptance')
        from tools.collect_pickup_augmentations import audit_videos
        accepted = json.loads((folder / 'accepted.json').read_text())
        from .contracts import file_hash
        if not accepted.get('artifact_hashes'):
            raise ValueError('Uncommitted legacy acceptance cannot be adopted')
        for artifact_name, expected in accepted['artifact_hashes'].items():
            if file_hash(folder/artifact_name) != expected:
                raise ValueError('Corrupt uncommitted acceptance')
        if accepted['candidate'] != c:
            raise ValueError('Uncommitted acceptance candidate differs')
        with np.load(folder / 'trajectory.npz') as rows:
            n = len(rows['timestamp'])
            if n != accepted['validation']['frame_count']:
                raise ValueError('Incomplete accepted trajectory')
        audit_videos(folder, n)
        result = accepted['validation']
    else:
        result = validate(c, record=record)
    paths = [folder / ('verification.json' if record else 'screen.json')]
    if record and result['pickup']:
        paths += [folder / n for n in ['accepted.json', 'trajectory.npz', 'camera-poses.json', *[v+'.mp4' for v in SIZES]]]
    paths += [p for p in [folder/(name+'-alignment.npz'),folder/'diagnostic-schema.json',
                           folder/('failed-recording-state.npz' if record else 'screen-state.npz')] if p.exists()]
    # Failed record verdicts get a separate receipt; no accepted cache to reuse.
    if record and not result['pickup']:
        seal(folder, 'record-failed', inputs, paths)
    else:
        seal(folder, name, inputs, paths)
    return result


def task(spec):
    with lock(Path(spec['output'])/'task-locks'/spec['id']):
        return execute_task(spec)


def execute_task(spec):
    source, ep, record = import_seed(spec['source'])
    if source != spec['source_contract']:
        raise ValueError('Source changed since run contract was bound')
    original = ep.arrays['action.low_level'].copy()
    base = apply_camera_views(json.loads(Path(source['config']).read_text()), ep)
    attachment = next(e['time'] for e in record['validation']['events'] if e['state'] == 'attached')
    try:
        motor, cfg, method = generate(ep, base, spec['recipe'], spec['seed'], attachment)
    except ValueError as error:
        return {'state': 'rejected', 'generation_rejected': True, 'reason': str(error), 'screened': 0, 'placement_attempts': 0}
    root = Path(spec['output']); trials = []; revisions = [(motor, cfg, method, 0.)]
    # Placement attempts are deterministic children; lift trim is a trajectory revision.
    for revision in range(2 if spec['recipe'].get('lift_refinement', False) else 1):
        motor, cfg, method, trim = revisions[revision]
        offsets = [[0., 0.]]
        result = {}
        for attempt in range(spec['placement_attempts']):
            if attempt >= len(offsets):
                break
            number = spec['number'] * 100 + revision * 10 + attempt
            folder = root / 'candidates' / f'{number:06}'; folder.mkdir(parents=True, exist_ok=True)
            scene = placed_config(cfg, cfg['rack']['active_cell'], offsets[attempt])
            try:
                bounds = preflight(ep, original, motor, scene)
            except ValueError as error:
                atomic_json(folder/'preflight-rejection.json',{'number':number,'proposal_id':spec['id'],'parent_episode':ep.index,
                    'family':spec.get('family',spec['recipe']['method']),'reason':str(error),'scene':scene,'placement_attempt':attempt})
                from .failure_data import atomic_npz
                atomic_npz(folder/'rejected-commands.npz',times=ep.action_times,motor_targets=motor)
                trials.append({'number': number, 'preflight_rejection': str(error)}); continue
            identity = {'task': spec['id'], 'revision': revision, 'placement_attempt': attempt,
                        'offset_m': offsets[attempt], 'lift_trim_m': trim}
            path = folder / 'candidate.json'
            if verify(folder, 'proposal', identity):
                c = json.loads(path.read_text())
                load_episode(c)
            else:
                if (folder/'accepted.json').exists():
                    raise ValueError('Acceptance without proposal receipt')
                commands = folder / 'commands.npz'
                np.savez_compressed(commands, times=ep.action_times, motor_targets=motor,
                                    hand_left=ep.arrays['action.hand_left'], hand_right=ep.arrays['action.hand_right'],
                                    initial_state=ep.arrays['observation.state'][0])
                atomic_json(folder / 'config.json', scene)
                c = {'number': number, 'capture_id': digest(identity), 'proposal_id': spec['id'], 'run_id': spec['run_id'],
                     'parent_episode': ep.index, 'dataset_snapshot': str(ep.root), 'source_files': source['source_files'],
                     'family':spec.get('family',spec['recipe']['method']),'partition':spec.get('partition','unspecified'),
                     'campaign_id':spec.get('campaign_id'), 'draw':spec.get('draw'),
                     'model_assets': model_assets(scene), 'joint_names': ep.names, 'source_capture': spec['source'], 'kind': spec['recipe']['method'],
                     'lineage': [method, identity], 'hand_provenance': ep.hand_action_provenance,
                     'cell': scene['rack']['active_cell'], 'target_shift_m': (rack_target_xy(scene)-rack_target_xy(base)).tolist(),
                     'noise': bounds, 'seed_attachment_s': attachment, 'folder': str(folder),
                     'config': str(folder/'config.json'), 'commands': str(commands),
                     'config_sha256': file_hash(folder/'config.json'), 'commands_sha256': file_hash(commands)}
                # Uncommitted proposals are deterministically regenerated, never reused.
                atomic_json(path, c)
                seal(folder, 'proposal', identity, [path, folder/'config.json', commands])
            result = stage(c)
            trials.append({'number': number, 'revision': revision, 'placement_attempt': attempt, 'pickup': result['pickup']})
            if result['pickup']:
                if spec['record']:
                    result = stage(c, True)
                    trials[-1]['recorded'] = result['pickup']
                return {'state': 'accepted' if spec['record'] and result['pickup'] else ('screened' if result['pickup'] else 'rejected'),
                        'candidate': number, 'capture_id': c['capture_id'], 'screened': len(trials), 'trials': trials,
                        'deviation': deviation(folder, Path(spec['source'])) if spec['record'] and result['pickup'] else None}
            if attempt == 0:
                screen = json.loads((folder/'screen.json').read_text())
                reference = json.loads((Path(spec['source'])/'verification.json').read_text())
                closest = screen['report'].get('attachment_diagnostics', {}).get('closest_approach')
                ref = reference['report'].get('attachment_diagnostics', {}).get('closest_approach')
                if closest and ref:
                    delta = (np.array(closest['nozzle_world_m'])-np.array(closest['tip_mouth_world_m'])
                             -np.array(ref['nozzle_world_m'])+np.array(ref['tip_mouth_world_m']))[:2]
                    if np.linalg.norm(delta) <= .02:
                        offsets += [(delta*.5).tolist(), delta.tolist()]
        if revision == 0 and spec['recipe'].get('lift_refinement', False):
            extra = result.get('lift_m', 0)-result.get('lift_target_m', 0)
            if not result.get('attached') or not 0 < extra <= .01:
                break
            try:
                revised = lift_revision(ep, cfg, motor, extra)
                preflight(ep, original, revised, cfg)
            except ValueError:
                break
            revisions.append((revised, copy.deepcopy(cfg), method, extra))
    return {'state': 'rejected', 'screened': len(trials), 'trials': trials}


def rack_target_xy(cfg):
    from g1_pipette_sim.scene import rack_target
    return rack_target(cfg)[:2]


def run(config, output, workers=1):
    output = Path(output).resolve()
    if config.get('proposals', 0) < 1 or not config.get('sources') or not config.get('recipes') or workers < 1:
        raise ValueError('Positive proposal/worker counts and nonempty sources/recipes required')
    if config.get('infrastructure_attempts', 2) < 1:
        raise ValueError('At least one infrastructure attempt required')
    if not 1 <= config.get('placement_attempts', 3) <= 3:
        raise ValueError('placement_attempts must be 1..3')
    with lock(output):
        seeds = []
        for folder in config['sources']:
            candidate, ep, record = import_seed(folder)
            # Other parents require explicit, episode-specific phase windows.
            # Lift IK still has its original absolute schedule and stays on 14.
            if ep.index != 14:
                for recipe in config['recipes']:
                    required = {'joint_approach': 'approach_s', 'joint_persistent': 'approach_s', 'elbow_pulse': 'pulse_s'}.get(recipe['method'])
                    if required and required not in recipe:
                        raise ValueError(f'Parent {ep.index} requires explicit {required}')
                    if recipe.get('lift_refinement'):
                        raise ValueError('Fixed lift refinement currently requires parent 14')
            paths = [Path(folder)/n for n in ['accepted.json', 'verification.json', 'commands.npz', 'config.json']]
            seeds.append({'candidate': candidate, 'files': {str(p.resolve()): file_hash(p) for p in paths}})
        contract = {'schema': 1, 'config': config, 'sources': seeds, 'dependencies': dependencies()}
        bind_run(output, contract)
        run_id = digest(contract)
        ledger = Ledger(output)
        specs = []
        for number in range(config['proposals']):
            spec = {'number': number, 'run_id': run_id, 'source': str(Path(config['sources'][number % len(seeds)]).resolve()),
                    'recipe': config['recipes'][number % len(config['recipes'])],
                    'source_contract': seeds[number % len(seeds)]['candidate'],
                    'seed': int(config['seed']) + number, 'output': str(output), 'record': config.get('record', True),
                    'placement_attempts': config.get('placement_attempts', 3)}
            spec['id'] = digest(spec)
            specs.append(spec); ledger.add(spec['id'])
        # Verify every receipt before trusting completed ledger state. A failed
        # checksum aborts without overwriting the damaged artifacts.
        for path in (output/'candidates').glob('*/candidate.json'):
            c = json.loads(path.read_text()); load_episode(c)
            if not verify(path.parent, 'proposal', c['lineage'][-1]):
                if (path.parent/'accepted.json').exists():raise ValueError('Accepted capture lacks proposal receipt')
            for name in ['screen', 'record', 'record-failed']:
                verify(path.parent, name, candidate_inputs(c))
            if (path.parent/'accepted.json').exists():
                stage(c, True)  # Recover commit-to-receipt interruption even for terminal ledger tasks.
        retries = config.get('infrastructure_attempts', 2)
        for _ in range(retries):
            pending = [s for s in specs if ledger.claim(s['id'], retries)]
            if not pending:
                break
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(task, s): s for s in pending}
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        result = future.result(); ledger.finish(spec['id'], result['state'], result)
                    except Exception as error:
                        result = {'error': repr(error)}; ledger.finish(spec['id'], 'error', result)
                    print(json.dumps({'proposal': spec['number'], **result}), flush=True)
                    report(output, ledger, config)
        result = report(output, ledger, config)
        ledger.close()
        if dependencies() != contract['dependencies']:
            raise ValueError('Dependencies changed during collection; captures are not reusable under this contract')
        if result['status'] != 'collected':
            raise RuntimeError('Collection incomplete; see progress.json and tasks.sqlite3')
        return result


def report(output, ledger, config):
    rows = ledger.rows()
    results = [json.loads(r['result']) for r in rows if r['result']]
    accepted = sorted(r['candidate'] for r in results if r.get('state') == 'accepted')
    summary = {'status': 'collected' if all(r['state'] not in ['pending','running','error'] for r in rows) else 'incomplete',
               'target': len(accepted), 'accepted': accepted, 'proposed': len(rows),
               'generation_rejected': sum(r.get('generation_rejected', False) for r in results),
               'physics_attempts': sum(sum('pickup' in t for t in r.get('trials', [])) for r in results),
               'screen_successes': sum(any(t.get('pickup') for t in r.get('trials', [])) for r in results),
               'errors': sum(r['state'] == 'error' for r in rows), 'results': results}
    atomic_json(Path(output)/'progress.json', summary)
    return summary


def deviation(folder, source):
    with np.load(folder/'trajectory.npz') as current, np.load(source/'trajectory.npz') as original:
        times=current['simulation_time']
        reference=np.stack([np.interp(times,original['simulation_time'],original['observation.state'][:,j]) for j in range(29)],axis=1)
        nozzle=np.stack([np.interp(times,original['simulation_time'],original['nozzle_position'][:,j]) for j in range(3)],axis=1)
        return {'max_measured_joint_delta_deg':float(np.degrees(np.max(abs(current['observation.state']-reference)))),
                'max_nozzle_deviation_mm':float(np.max(np.linalg.norm(current['nozzle_position']-nozzle,axis=1))*1000)}
