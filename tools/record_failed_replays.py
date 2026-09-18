"""Resume full real/sim comparison videos for every failed published HIL194 replay.

Uses the normal headless CLI, saved fitted layouts, recorded low-level commands,
and full physics. Videos are H.264 for browser playback; every frame is decoded
before an episode is marked complete. Never changes the published layouts.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def decode_video(path, duration):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (size[1], size[0]):
                raise RuntimeError('Inconsistent decoded frame shape')
            count += 1
    finally:
        cap.release()
    if abs(fps - 30) > .01 or abs(count / fps - duration) > .1:
        raise RuntimeError(f'Incomplete recording: {count} frames at {fps} fps for {duration}s')
    return {'frames': count, 'fps': fps, 'resolution': size, 'duration_s': count / fps}


def record(job):
    index, entry, output, contract = job
    folder = Path(output) / 'episodes' / f'episode-{index:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'recording.json'
    video = folder / 'comparison.mp4'
    if done.exists():
        previous = json.loads(done.read_text())
        if (previous.get('contract') == contract and video.exists()
                and previous.get('video_sha256') == sha(video)):
            return previous
    start = time.monotonic()
    env = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cmd = [sys.executable, '-m', 'g1_pipette_sim', '--headless', '--episode', str(index),
           '--compare', '--output', str(folder)]
    with (folder / 'replay.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=3600)
    report = json.loads((folder / 'report.json').read_text())
    if not report['finite']:
        raise RuntimeError('Nonfinite replay')
    workspace=json.loads((folder/'workspace.json').read_text())
    correction=workspace.get('camera_view_correction',{})
    if correction.get('source_sha256')!=contract['camera_overrides_sha256'] or 'rgb' not in correction.get('views',[]):
        raise RuntimeError('Corrected ego camera was not applied to this recording')
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(folder / 'simulation.mp4'), '-an', '-c:v', 'libx264',
                    '-threads', '1', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(video)], check=True, timeout=600)
    stats = decode_video(video, report['time'])
    expected_report = json.loads((ROOT / entry['report']).read_text())
    # Independently flag drift from the published replay, not just video completion.
    discrepancies = []
    if report['events'] != expected_report['events']:
        discrepancies.append('event sequence differs from published replay')
    if report['state'] != expected_report['state']:
        discrepancies.append('final state differs from published replay')
    if report['contact_diagnostics'] != expected_report['contact_diagnostics']:
        discrepancies.append('contact diagnostics differ from published replay')
    if report.get('warnings'):
        discrepancies.append('simulator warnings present')
    result = {'episode': index, 'complete': True, 'published_status': entry['status'],
              'contract': contract, 'video': str(video.relative_to(output)),
              'video_sha256': sha(video), 'wall_seconds': time.monotonic() - start,
              'replay_discrepancies': discrepancies, 'camera_view_correction': correction,
              'ego_camera': workspace['cameras']['rgb'], **stats}
    write(done, result)
    (folder / 'simulation.mp4').unlink()  # Retain verified browser-compatible encode.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/hil194/failure-analysis')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', nargs='+', type=int, help='Optional subset of failed episode IDs')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from g1_pipette_sim.pickup_layouts import MANIFEST, model_fingerprints
    manifest = json.loads(MANIFEST.read_text())
    if manifest['model_fingerprints'] != model_fingerprints():
        raise ValueError('Published physics/model fingerprint mismatch; revalidate before recording')
    requested = sorted(int(i) for i, e in manifest['episodes'].items() if e['status'] != 'verified_pickup')
    if args.episodes:
        if not set(args.episodes) <= set(requested):
            raise ValueError('--episodes must contain failed episodes only')
        requested = sorted(set(args.episodes))
    contract = {'manifest_sha256': sha(MANIFEST), 'recorder_sha256': sha(__file__),
                'cli_sha256': sha(ROOT / 'g1_pipette_sim/__main__.py'),
                'camera_views_sha256': sha(ROOT / 'g1_pipette_sim/camera_views.py'),
                'camera_overrides_sha256': sha(ROOT / 'config/camera-view-overrides.json'),
                'fps': 30, 'source': 'low_level', 'clock': 'recorded', 'physics': 'dynamic'}
    frozen = args.output / 'input-manifest.json'
    if frozen.exists() and sha(frozen) != sha(MANIFEST):
        raise ValueError('Output directory belongs to a different published manifest')
    frozen.write_bytes(MANIFEST.read_bytes())
    results, errors = {}, {}
    started = time.time()
    def progress():
        write(args.output / 'recordings.json', {'complete': len(results) == len(requested) and not errors,
              'requested': requested, 'completed': len(results), 'failed': len(errors),
              'started_unix': started, 'elapsed_s': time.time() - started, 'contract': contract,
              'recordings': sorted(results.values(), key=lambda r: r['episode']), 'errors': errors})
    progress()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(record, (i, manifest['episodes'][str(i)], args.output, contract)): i for i in requested}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                results[index] = result
                print(f"{len(results)}/{len(requested)} episode {index}: video verified; discrepancies={result['replay_discrepancies']}", flush=True)
            except Exception as error:
                errors[str(index)] = repr(error)
                print(f'ERROR episode {index}: {error}', flush=True)
            progress()
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
