"""Compare failure clusters using matched published replay diagnostics, not outcome labels.

K-means, Ward hierarchical clustering, diagonal Gaussian mixtures, and DBSCAN.
All candidate partitions, preprocessing, feature definitions, stability checks,
representative episodes and browser review links are saved for reproducibility.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis-deps'))
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances

BLOCKS = {
    'insertion_geometry': ['closest_radial_mm', 'minimum_radial_mm', 'closest_depth_shortfall_mm',
                           'closest_depth_excess_mm', 'closest_angle_deg'],
    'gate_exposure': ['radial_failure_fraction', 'depth_failure_fraction', 'angle_failure_fraction'],
    'grasp_and_controller': ['maximum_grasp_drift_mm', 'maximum_servo_error_rad'],
    'contacts': ['rack_force_N', 'tip_force_N', 'rack_penetration_mm', 'tip_penetration_mm'],
}
FEATURES = sum(BLOCKS.values(), [])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def outcome(entry):
    if entry.get('status') == 'verified_pickup':
        return 'verified_pickup'
    v = entry['validation']
    if not v['attached']:
        return 'no_attachment'
    if v['lift_m'] < v['lift_target_m'] - v['lift_tolerance_m']:
        return 'lift_shortfall'
    if v['lift_m'] > v['lift_target_m'] + v['lift_tolerance_m']:
        return 'lift_overshoot'
    return 'dwell_or_retention_failure'


def gate_category(diag):
    """Exact necessary-condition failures from all pre-attachment physics steps."""
    if diag['samples'] <= 0:
        return 'no_gate_observations'
    never = [g for g in ('radial', 'depth', 'angle') if diag['gate_failures'][g] == diag['samples']]
    return 'never_' + '_and_'.join(never) if never else 'gates_pass_individually_but_not_sustained_together'


def extract(manifest, include_verified=False):
    from g1_pipette_sim.pickup_layouts import sha256
    records = []
    for index, entry in sorted(manifest['episodes'].items(), key=lambda item: int(item[0])):
        if entry['status'] == 'verified_pickup' and not include_verified:
            continue
        if sha256(ROOT / entry['config']) != entry['sha256']:
            raise ValueError(f'Config changed for episode {index}')
        config = json.loads((ROOT / entry['config']).read_text())
        report_path = ROOT / entry['report']
        r = json.loads(report_path.read_text())
        v, a = entry['validation'], config['attachment']
        diag = r['attachment_diagnostics']
        closest = diag['closest_approach'] or {}
        samples = diag['samples']
        depth = closest.get('insertion_depth_mm', np.nan)
        contacts = r['contact_diagnostics']
        events = [e['time'] for e in r['events'] if e['state'] == 'attached']
        record = {
            'episode': int(index), 'published_status': entry['status'], 'outcome': outcome(entry),
            'gate_category': gate_category(diag) if not v['attached'] else 'attached',
            'recording_group': 'inherited_HG_DAgger' if int(index) < 46 else 'new_HIL194',
            'simulated_lift_mm': v['lift_m'] * 1000, 'target_lift_mm': v['lift_target_m'] * 1000,
            'lift_tolerance_mm': v['lift_tolerance_m'] * 1000,
            'lift_error_mm': (v['lift_m'] - v['lift_target_m']) * 1000,
            'lift_dwell_s': v['lift_dwell_s'], 'attachment_time_s': events[0] if events else None,
            'early_attachment': bool(events and events[0] < 1),
            'duration_s': r['time'], 'config': entry['config'], 'report': entry['report'],
            'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'video': f'episodes/episode-{int(index):03d}/comparison.mp4',
            'closest_time_s': closest.get('time_s'), 'closest_depth_mm': depth,
            'closest_radial_mm': closest.get('radial_error_mm', np.nan),
            'minimum_radial_mm': diag.get('minimum_radial_error_mm', np.nan),
            'closest_depth_shortfall_mm': max(0., a['min_insertion_depth'] * 1000 - depth),
            'closest_depth_excess_mm': max(0., depth - a['max_insertion_depth'] * 1000),
            'closest_angle_deg': closest.get('angle_deg', np.nan),
            'maximum_grasp_drift_mm': diag.get('maximum_grip_displacement_mm', np.nan),
            'maximum_servo_error_rad': r.get('max_servo_tracking_error', np.nan),
            'maximum_insertion_dwell_s': diag.get('maximum_valid_dwell_s'),
        }
        for gate in ('radial', 'depth', 'angle'):
            record[gate + '_failure_fraction'] = diag['gate_failures'][gate] / samples if samples else np.nan
        for part in ('rack', 'tip'):
            c = contacts.get('pipette/' + part, {})
            record[part + '_force_N'] = c.get('maximum_normal_force_N', 0.)
            record[part + '_penetration_mm'] = c.get('maximum_penetration_mm', 0.)
        records.append(record)
    return records


def preprocess(raw, variant):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    filled = imputer.fit_transform(raw)
    if variant == 'log_standard':
        filled = np.log1p(np.maximum(filled, 0))
        scaler = StandardScaler()
    else:
        scaler = RobustScaler(quantile_range=(10, 90))
    X = scaler.fit_transform(filled)
    constant = np.nanstd(filled, axis=0) < 1e-10
    X[:, constant] = 0
    weights = []
    for columns in BLOCKS.values():
        active = sum(not constant[FEATURES.index(name)] for name in columns)
        weights.extend([1 / np.sqrt(max(active, 1))] * len(columns))
    X *= np.asarray(weights)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite preprocessed features')
    return X, {'variant': variant, 'features': FEATURES, 'blocks': BLOCKS,
               'imputation_medians': imputer.statistics_.tolist(),
               'scale': scaler.scale_.tolist(),
               'center': (scaler.mean_ if variant == 'log_standard' else scaler.center_).tolist(),
               'weights': weights, 'constant_features': [f for f, c in zip(FEATURES, constant) if c]}


def fit(method, parameters, X, seed):
    if method == 'kmeans':
        model = KMeans(n_clusters=parameters['k'], random_state=seed, n_init=20)
    elif method == 'ward':
        model = AgglomerativeClustering(n_clusters=parameters['k'], linkage='ward')
    elif method == 'gmm':
        model = GaussianMixture(n_components=parameters['k'], covariance_type='diag',
                                n_init=5, reg_covar=1e-4, random_state=seed)
    elif method == 'dbscan':
        model = DBSCAN(**parameters)
    else:
        raise ValueError(method)
    labels = model.fit_predict(X)
    return labels, model


def score(X, labels):
    kept = labels >= 0
    counts = Counter(labels[kept].tolist())
    valid = 2 <= len(counts) < kept.sum()
    return {'clusters': len(counts), 'coverage': float(kept.mean()),
            'noise': int((~kept).sum()), 'smallest_cluster': min(counts.values(), default=0),
            'silhouette': float(silhouette_score(X[kept], labels[kept])) if valid else None}


def sweep(X, seed, k_min=2, k_max=10):
    candidates, best = [], {}
    distances = np.sort(pairwise_distances(X), axis=1)
    for method in ('kmeans', 'ward', 'gmm', 'dbscan'):
        params = [{'k': k} for k in range(k_min, k_max + 1)]
        if method == 'dbscan':
            params = [{'eps': float(eps), 'min_samples': ms} for ms in (4, 6, 10)
                      for eps in np.unique(np.quantile(distances[:, ms - 1], [.4, .55, .7, .85, .95])) if eps > 0]
        trials = []
        for param in params:
            labels, model = fit(method, param, X, seed)
            row = {'method': method, 'parameters': param, **score(X, labels), 'labels': labels.tolist()}
            if method == 'gmm':
                row['bic'] = float(model.bic(X))
                row['converged'] = bool(model.converged_)
            trials.append(row)
        candidates.extend(trials)
        eligible = [r for r in trials if r['silhouette'] is not None and r['smallest_cluster'] >= 5
                    and r['coverage'] >= .5 and r.get('converged', True)]
        if eligible:
            chosen = min(eligible, key=lambda r: r['bic']) if method == 'gmm' else max(
                eligible, key=lambda r: r['silhouette'] * r['coverage'])
            best[method] = chosen
        else:
            best[method] = {'method': method, 'unresolved': True,
                            'reason': 'No nondegenerate partition with >=50% coverage and >=5 members per cluster'}
    return candidates, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/hil194/failure-analysis')
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=10)
    parser.add_argument('--feature-set', choices=['all', 'without-gate-fractions'], default='all')
    parser.add_argument('--recordings-root', type=Path, default=ROOT / 'outputs/hil194/failure-analysis')
    parser.add_argument('--stability-repeats', type=int, default=20)
    args = parser.parse_args()
    global BLOCKS, FEATURES
    if args.feature_set == 'without-gate-fractions':
        BLOCKS = {k:v for k,v in BLOCKS.items() if k != 'gate_exposure'}
        FEATURES = sum(BLOCKS.values(), [])
    if not 2 <= args.k_min <= args.k_max <= 20:
        parser.error('Require 2 <= k-min <= k-max <= 20')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.recordings_root / 'input-manifest.json'
    manifest = json.loads(source.read_text())
    if source.resolve() != (args.output / 'input-manifest.json').resolve():
        (args.output / 'input-manifest.json').write_bytes(source.read_bytes())
    records = extract(manifest)
    import os
    for r in records:
        r['video'] = os.path.relpath(args.recordings_root / r['video'], args.output)
    controls = [r for r in extract(manifest, include_verified=True) if r['published_status']=='verified_pickup']
    late_controls = [r for r in controls if not r['early_attachment']]
    comparisons = {}
    for feature in FEATURES:
        failure_values = np.asarray([r[feature] for r in records], dtype=float)
        success_values = np.asarray([r[feature] for r in late_controls], dtype=float)
        comparisons[feature] = {'failure_median': float(np.nanmedian(failure_values)),
            'later_attachment_success_median': float(np.nanmedian(success_values)),
            'cliffs_delta_fail_minus_success': float(np.mean(np.sign(failure_values[:,None]-success_values[None,:])))}
    write(args.output / 'success-reference.json', {'all_verified_episodes':[r['episode'] for r in controls],
        'later_attachment_reference_episodes':[r['episode'] for r in late_controls],
        'features':comparisons,
        'meaning':'Descriptive reference only, excluded from clustering. Excludes four <1s attachments; remaining references are not all manually verified real/sim timing matches.'})
    raw = np.array([[r[f] for f in FEATURES] for r in records], dtype=float)
    missing = np.isnan(raw).sum(0)
    X, preprocessing = preprocess(raw, 'log_standard')
    candidates, best = sweep(X, args.seed, args.k_min, args.k_max)
    write(args.output / 'clustering-candidates.json', candidates)
    write(args.output / 'preprocessing.json', preprocessing)
    np.savez_compressed(args.output / 'clustering-features.npz', raw=raw, scaled=X,
                        episode=np.array([r['episode'] for r in records]), features=np.array(FEATURES))
    sensitivity_X, sensitivity_preprocessing = preprocess(raw, 'raw_robust')
    _, sensitivity_best = sweep(sensitivity_X, args.seed, args.k_min, args.k_max)
    write(args.output / 'scaling-sensitivity.json', {'preprocessing': sensitivity_preprocessing, 'best': sensitivity_best})
    rng = np.random.default_rng(args.seed)
    profiles = {}
    for method, result in best.items():
        if result.get('unresolved'):
            continue
        labels = np.array(result['labels'])
        stability = []
        for repeat in range(args.stability_repeats):
            subset = np.sort(rng.choice(len(X), size=int(.8 * len(X)), replace=False))
            new, _ = fit(method, result['parameters'], X[subset], args.seed + repeat + 1)
            stability.append(float(adjusted_rand_score(labels[subset], new)))
        result['subsample_stability_ari'] = {'mean': float(np.mean(stability)), 'min': float(min(stability)),
                                             'values': stability, 'fraction': .8}
        group = [r['recording_group'] for r in records]
        result['recording_group_ari'] = float(adjusted_rand_score(group, labels))
        result['outcome_ari'] = float(adjusted_rand_score([r['outcome'] for r in records], labels))
        other = sensitivity_best[method]
        result['scaling_sensitivity_ari'] = (float(adjusted_rand_score(labels, other['labels']))
                                            if 'labels' in other else None)
        profiles[method] = []
        for label in sorted(set(labels)):
            indices = np.flatnonzero(labels == label)
            center = X[indices].mean(0)
            nearest = indices[np.argsort(np.linalg.norm(X[indices] - center, axis=1))[:3]]
            features = sorted(zip(FEATURES, center.tolist()), key=lambda item: abs(item[1]), reverse=True)
            profiles[method].append({'cluster': int(label), 'size': len(indices),
                'episodes': [records[i]['episode'] for i in indices],
                'representatives': [records[i]['episode'] for i in nearest],
                'outcomes': dict(Counter(records[i]['outcome'] for i in indices)),
                'recording_groups': dict(Counter(records[i]['recording_group'] for i in indices)),
                'median_features': {f: float(np.nanmedian(raw[indices, j])) for j, f in enumerate(FEATURES)},
                'distinctive_scaled_features': features[:5]})
        for record, label in zip(records, labels):
            record[method + '_cluster'] = int(label)
    methods = [m for m, r in best.items() if 'labels' in r]
    agreement = {a: {b: float(adjusted_rand_score(best[a]['labels'], best[b]['labels'])) for b in methods} for a in methods}
    summary = {'episodes': len(records), 'seed': args.seed, 'feature_set': args.feature_set,
               'recordings_url': os.path.relpath(args.recordings_root / 'recordings.json', args.output),
               'alternative_review_url': os.path.relpath(args.recordings_root / ('index.html' if args.feature_set=='without-gate-fractions' else 'without-gate-fractions/index.html'), args.output),
               'nonattachment_gate_categories': dict(Counter(r['gate_category'] for r in records if r['outcome']=='no_attachment')), 'outcomes': dict(Counter(r['outcome'] for r in records)),
               'early_attachments': [r['episode'] for r in records if r['early_attachment']],
               'missing_feature_counts': dict(zip(FEATURES, missing.tolist())),
               'methods': best, 'method_agreement_ari': agreement,
               'selection': 'Kmeans/Ward: maximum silhouette; GMM: minimum diagonal BIC; DBSCAN: silhouette times coverage. Require >=5 members/cluster and >=50% coverage.',
               'limitations': ['Clusters are descriptive associations, not identified physical causes.',
                 'Closest approach minimizes 3D nozzle-mouth distance, not insertion gate violation; near-zero depth at closest approach is not proof of never inserting.',
                 'Gate fractions are measured only while in_rack; their observation windows differ for early attachments.',
                 'Grasp drift is simulated displacement from the configured mount, not observed real slip.',
                 'Recorded lift targets are joint FK estimates, not measured tip trajectories.',
                 'No episode ID, session label, outcome, lift target/error, fitted position or video pixels enter clustering.',
                 'Recorded/sim camera poses and image clocks are approximate; videos require human review.']}
    write(args.output / 'clustering-summary.json', summary)
    write(args.output / 'cluster-profiles.json', profiles)
    write(args.output / 'episodes.json', records)
    with (args.output / 'failure-features.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    plot(X, best, args.output)
    make_report(records, summary, profiles, args.output)
    make_dashboard(records, summary, profiles, args.output)
    print(json.dumps({'outcomes': summary['outcomes'], 'methods': {m: {k:v for k,v in r.items() if k != 'labels'} for m,r in best.items()}}, indent=2))


def plot(X, best, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    points = pca.fit_transform(X)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (method, r) in zip(axes.flat, best.items()):
        if 'labels' not in r:
            ax.set_title(method + ': no valid clustering'); continue
        labels = np.asarray(r['labels'])
        for label in sorted(set(labels)):
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], s=24, alpha=.75,
                       label=f'{label} ({mask.sum()})', color='gray' if label < 0 else None)
        ax.set_title(f"{method}: {r['clusters']} clusters, silhouette {r['silhouette']:.3f}")
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=7)
    fig.suptitle(f'Failure diagnostics: PCA view only ({sum(pca.explained_variance_ratio_):.0%} variance); clustering uses all features')
    fig.savefig(output / 'cluster-comparison.png', dpi=160); plt.close(fig)
    write(output / 'pca.json', {'points': points.tolist(), 'explained_variance_ratio': pca.explained_variance_ratio_.tolist()})


def make_report(records, summary, profiles, output):
    lines = ['# HIL194 failed replay review', '', f"Analyzed {len(records)} failed episodes from the saved fitted-layout validation.", '',
             'These are simulator replay failures, not labels that the real demonstrations failed.', '',
             '## Outcome categories', '', '| Category | Episodes |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in summary['outcomes'].items()]
    lines += ['', '## Non-attachment gate categories', '', 'Exact necessary-condition failures across the replay, not causal attribution to real hardware. Gates require radial error <0.7 mm, insertion depth 10–12 mm, angle <5 degrees, held together for 100 ms.', '', '| Gate category | Episodes |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in summary['nonattachment_gate_categories'].items()]
    lines += ['', '## Clustering comparison', '', '| Method | Clusters | Coverage | Silhouette | Stability ARI | Scaling ARI |', '|---|---:|---:|---:|---:|---:|']
    for m,r in summary['methods'].items():
        if 'labels' not in r:
            lines.append(f'| {m} | no valid partition | | | | |'); continue
        scale=r['scaling_sensitivity_ari']
        lines.append(f"| {m} | {r['clusters']} | {r['coverage']:.1%} | {r['silhouette']:.3f} | {r['subsample_stability_ari']['mean']:.3f} | {scale if scale is not None else 'unresolved'} |")
    lines += ['', summary['selection'], '', 'Stability compares each fitted partition with 20 refits on 80% subsets (same preprocessing). Scaling sensitivity refits/selects with raw robust scaling instead of log standard scaling. ARI=1 means identical partitions; near 0 means chance-level agreement.', '', '## Representative episodes']
    for method, clusters in profiles.items():
        lines += ['', f'### {method}', '']
        for c in clusters:
            links=', '.join(f"[episode {i}]({next(r['video'] for r in records if r['episode']==i)})" for i in c['representatives'])
            lines.append(f"- Cluster {c['cluster']}: **{c['size']} episodes**, outcomes {c['outcomes']}; representatives: {links}.")
    lines += ['', '## Successful reference episodes', '', 'The 12 verified pickups with attachment after the first second are retained as a descriptive reference, excluded from clustering. [Feature medians and effect sizes](success-reference.json) help distinguish failure-specific signals from simulator behavior also present in passing replays. These references do not establish faithful real/sim timing.', '']
    lines += ['', '## Interpretation limits', ''] + ['- '+v for v in summary['limitations']]
    lines += ['', '## Review and reproduce', '', '[Interactive episode browser](index.html) · [Features/assignments](failure-features.csv) · [All candidate partitions](clustering-candidates.json) · [Cluster profiles](cluster-profiles.json) · [Recording progress](recordings.json)', '',
              'Videos show real cameras on the top row and their corresponding simulator views below. Labels use source camera keys: rgb=ego, wrist_left=physical right wrist, wrist_right=fixed side. Playback is 30 fps, full episode, paired-image clock.', '',
              'Run from the project root:', '```bash', 'python tools/record_failed_replays.py --workers 4', 'python tools/cluster_replay_failures.py',
              'python -m http.server 8000 --bind 127.0.0.1 --directory outputs/hil194/failure-analysis', '```', '',
              'Open http://127.0.0.1:8000. The recorder resumes verified completed videos. The clustering script requires scikit-learn, scipy, numpy and matplotlib (see docs/FAILURE_ANALYSIS.md).', '',
              'Method documentation: https://scikit-learn.org/stable/modules/clustering.html ; https://scikit-learn.org/stable/modules/mixture.html']
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')


def make_dashboard(records, summary, profiles, output):
    payload=json.dumps({'records':records,'summary':summary,'profiles':profiles}).replace('</','<\\/')
    page='''<!doctype html><html><head><meta charset="utf-8"><title>HIL194 failure review</title>
<style>body{font:15px system-ui;margin:24px;background:#111923;color:#e3eaf3}a{color:#8acfff}select,input,button{padding:7px;background:#243347;color:white;border:1px solid #567}video{width:min(960px,100%);background:#000}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #34435a;text-align:left}tr:hover{background:#26354a}#detail{margin:15px 0;white-space:pre-wrap}.muted{color:#a9bbce}#list{max-height:460px;overflow:auto}</style></head><body>
<h1>HIL194 failed replay review</h1><p>178 failed fitted-layout replays. Clusters describe similarities; they do not establish physical causes.</p><p id="progress">Checking recording progress…</p>
<label>Method <select id="method"></select></label> <label>Cluster <select id="cluster"></select></label> <label>Outcome <select id="outcome"><option value="">All outcomes</option></select></label> <label>Episode <input id="search" size="6" placeholder="ID"></label>
<p id="methodInfo" class="muted"></p><div id="list"><table><thead><tr><th>Episode</th><th>Cluster</th><th>Outcome</th><th>Lift / target mm</th><th>Attachment s</th><th>Grasp drift mm</th><th>Recording</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2 id="title">Select an episode</h2><video id="video" controls preload="none"></video><p>Top: real cameras. Bottom: matching simulator views. Left=ego; middle=right wrist; right=fixed side.</p><div id="detail"></div>
<p><a id="alternative">Alternative feature-set review</a> · <a href="REPORT.md">Report</a> · <a href="failure-features.csv">Features and assignments CSV</a> · <a href="cluster-profiles.json">Cluster profiles</a> · <a href="clustering-candidates.json">All clustering trials</a></p><img src="cluster-comparison.png" style="max-width:100%">
<script>const DATA=__DATA__;let done=new Set();let videoVersions=new Map();const $=id=>document.getElementById(id);const methods=Object.keys(DATA.summary.methods).filter(m=>DATA.summary.methods[m].labels);
$('alternative').href=DATA.summary.alternative_review_url;methods.forEach(m=>$('method').add(new Option(m,m)));Object.keys(DATA.summary.outcomes).forEach(o=>$('outcome').add(new Option(o,o)));
function clusters(){let m=$('method').value;$('cluster').innerHTML='';$('cluster').add(new Option('All clusters',''));DATA.profiles[m].forEach(c=>$('cluster').add(new Option(c.cluster+' ('+c.size+' episodes)',c.cluster)));let r=DATA.summary.methods[m];$('methodInfo').textContent=`${r.clusters} clusters; silhouette ${r.silhouette.toFixed(3)}; coverage ${(100*r.coverage).toFixed(0)}%; stability ARI ${r.subsample_stability_ari.mean.toFixed(3)}. Gray/noise=-1.`;render()}
function render(){let m=$('method').value,c=$('cluster').value,o=$('outcome').value,q=$('search').value.trim();$('rows').innerHTML='';DATA.records.filter(r=>(c===''||r[m+'_cluster']==c)&&(!o||r.outcome===o)&&(!q||String(r.episode).includes(q))).forEach(r=>{let tr=document.createElement('tr');[r.episode,r[m+'_cluster'],r.outcome,`${r.simulated_lift_mm.toFixed(2)} / ${r.target_lift_mm.toFixed(2)}`,r.attachment_time_s===null?'none':r.attachment_time_s.toFixed(3),r.maximum_grasp_drift_mm.toFixed(2),done.has(r.episode)?'ready':'pending'].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});tr.onclick=()=>select(r);$('rows').appendChild(tr)})}
function select(r){$('title').textContent='Episode '+r.episode+' — '+r.outcome;$('video').src=r.video+'?v='+(videoVersions.get(r.episode)||'pending');$('detail').textContent=`Cluster memberships: ${methods.map(m=>m+'='+r[m+'_cluster']).join(', ')}\nLift error: ${r.lift_error_mm.toFixed(2)} mm; tolerance ±${r.lift_tolerance_mm.toFixed(2)} mm\nClosest approach: radial ${r.closest_radial_mm.toFixed(2)} mm, depth ${r.closest_depth_mm.toFixed(2)} mm, angle ${r.closest_angle_deg.toFixed(2)}°\n${r.early_attachment?'Flag: attachment before 1 s; inspect real timing.':''}\nGate category: ${r.gate_category}\n${r.recording_group}`}
async function progress(){try{let r=await(await fetch(DATA.summary.recordings_url+'?'+Date.now())).json();done=new Set(r.recordings.map(x=>x.episode));videoVersions=new Map(r.recordings.map(x=>[x.episode,x.video_sha256]));$('progress').textContent=`Videos verified: ${r.completed}/${r.requested.length}; recording errors: ${r.failed}.`;render()}catch(e){$('progress').textContent='Serve this folder with python -m http.server to load live recording status.'}}
$('method').onchange=clusters;['cluster','outcome','search'].forEach(id=>$(id).oninput=render);clusters();progress();setInterval(progress,30000);</script></body></html>'''
    (output/'index.html').write_text(page.replace('__DATA__',payload))


if __name__ == '__main__':
    main()
