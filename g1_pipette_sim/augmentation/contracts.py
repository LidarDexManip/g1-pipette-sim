"""Fail-closed content receipts. No mtime shortcuts or unverified cache adoption."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.augmentation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def dependencies():
    # Includes render/validation/export code and meshes, not only the physics list.
    paths = [p for directory in ['g1_pipette_sim', 'tools', 'assets/robot', 'config']
             for p in (ROOT / directory).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts
             and (directory in ['assets/robot', 'config'] or p.suffix == '.py')]
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'mujoco', 'scipy', 'opencv-python', 'pyarrow']}
    return {'files': {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}, 'versions': versions,
            'runtime': {k: os.environ.get(k) for k in ['MUJOCO_GL', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']}}


def bind_run(root, contract):
    path = Path(root) / 'run-contract.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Run dependencies/recipe changed; choose a new output directory')
    else:
        existing = [p for p in Path(root).iterdir() if p.name != '.augmentation.lock']
        if existing:
            raise ValueError('Unversioned output cannot be adopted as a cache; use a new directory')
        atomic_json(path, contract)


def seal(root, name, inputs, paths):
    root = Path(root)
    receipt = {'schema': 1, 'inputs': digest(inputs), 'artifacts': {
        str(Path(p).relative_to(root)): file_hash(p) for p in paths}}
    atomic_json(root / (name + '.receipt.json'), receipt)
    return receipt


def verify(root, name, inputs):
    root = Path(root)
    path = root / (name + '.receipt.json')
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt['inputs'] != digest(inputs):
        raise ValueError(f'Stale {name} inputs in {root}; use a new output')
    for relative, expected in receipt['artifacts'].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f'Corrupt {name} artifact: {root / relative}')
    return True


def tree_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file()
                  and p.name not in ['.augmentation.lock', 'publication.receipt.json'])


def legacy_write_disabled():
    raise RuntimeError('This experimental writer is retired: its unversioned caches are read-only. Use tools/run_augmentations.py with a new output directory.')
