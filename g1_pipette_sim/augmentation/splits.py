"""Frozen parent membership, independent of how many synthetic children exist."""
import json
from pathlib import Path
from .contracts import atomic_json, digest, lock


def parent_splits(path, snapshot, count, seed, validation_fraction=.2):
    path = Path(path).resolve()
    contract = {'snapshot': str(Path(snapshot).resolve()), 'seed': seed, 'validation_fraction': validation_fraction}
    with lock(path.parent / ('.'+path.name+'-lock')):
        if path.exists():
            value = json.loads(path.read_text())
            if value['contract'] != contract:
                raise ValueError('Split contract differs; reuse the original split settings')
        else:
            value = {'schema': 1, 'contract': contract, 'parents': {}}
        for index in range(count):
            # Hash threshold keeps prior assignments stable when real data grows.
            value['parents'].setdefault(str(index), 'val' if int(digest([seed, str(Path(snapshot).resolve()), index])[:16],16)/2**64 < validation_fraction else 'train')
        atomic_json(path, value)
    return value


def episode_chunk(index):
    return index // 1000
