"""Apply reviewed scene presence, without suppressing contacts of present props."""
import hashlib
import json
import weakref
from pathlib import Path

import numpy as np

from . import ROOT

SCENE_REVIEW = ROOT / 'config/environments/lidar-episode-scenes.json'
_original = weakref.WeakKeyDictionary()


def apply_episode_scene(model, dataset, episode):
    """Restore the imported props, then apply only fingerprint-matched reviews.

    Static prop bodies remain in the model for stable IDs. An absent prop has no
    visible or contact geometry. Baselines are per model, so switching episodes
    restores both visual and collision properties before applying a new review.
    """
    config = json.loads(SCENE_REVIEW.read_text())
    baseline = _original.setdefault(model, {})
    for name in {name for row in config['episodes'].values() for name in row['absent_objects']}:
        try:
            body = model.body(name).id
        except KeyError:
            continue
        if name not in baseline:
            ids = np.flatnonzero(model.geom_bodyid == body)
            baseline[name] = (ids, model.geom_rgba[ids].copy(),
                              model.geom_contype[ids].copy(), model.geom_conaffinity[ids].copy())
    for ids, rgba, contype, affinity in baseline.values():
        model.geom_rgba[ids] = rgba
        model.geom_contype[ids] = contype
        model.geom_conaffinity[ids] = affinity

    provenance = Path(dataset) / 'meta/source_provenance.jsonl'
    matches = (provenance.exists() and hashlib.sha256(provenance.read_bytes()).hexdigest()
               == config['source_provenance_sha256'])
    review = config['episodes'].get(str(episode)) if matches else None
    result = {'reviewed': review is not None, 'absent_objects': [],
              'dataset_fingerprint_matches': matches}
    if review is not None:
        for name in review['absent_objects']:
            if name not in baseline:
                raise ValueError(f'Reviewed object is missing from model: {name}')
            ids = baseline[name][0]
            model.geom_rgba[ids, 3] = 0
            model.geom_contype[ids] = 0
            model.geom_conaffinity[ids] = 0
        result.update(review)
    return result
