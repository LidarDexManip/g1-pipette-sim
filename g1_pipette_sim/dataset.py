from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

HGDAGGER_DATASET = Path.home() / '.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hgdagger-20260904'

LEGACY_DATASET = Path.home() / 'Datasets/g1-pipette-3view-20260819-plus21'
HIL194_DATASET = Path.home() / '.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hil194'
DEFAULT_DATASET = HIL194_DATASET

def resolve_dataset_root(root):
    """Accept a LeRobot directory, HF snapshot, or downloaded HF cache repository."""
    root = Path(root).expanduser()
    if (root / 'meta/info.json').is_file():
        return root
    ref = root / 'refs/main'
    if ref.is_file():
        snapshot = root / 'snapshots' / ref.read_text().strip()
        if (snapshot / 'meta/info.json').is_file():
            return snapshot
        raise FileNotFoundError(f'HF main snapshot lacks meta/info.json: {snapshot}')
    snapshots = sorted(p for p in (root / 'snapshots').glob('*') if (p / 'meta/info.json').is_file())
    if len(snapshots) == 1:
        return snapshots[0]
    raise FileNotFoundError(f'No unambiguous LeRobot dataset at {root}; pass a snapshot directory')

class Episode:
    """Load only replay channels; never use frozen hand feedback or tactile as forces."""
    def __init__(self, root=DEFAULT_DATASET, index=0, clock='recorded'):
        self.root, self.index = resolve_dataset_root(root), index
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if not 0 <= index < self.info['total_episodes']:
            raise ValueError(f'Episode must be in [0, {self.info["total_episodes"]-1}]')
        self.names = self.info['features']['observation.state']['names'][:29]
        self.fps = self.info['fps']
        path = self.root / self.info['data_path'].format(episode_chunk=index // self.info['chunks_size'], episode_index=index)
        columns = ['observation.state', 'action', 'action.low_level', 'action.hand_left', 'action.hand_right', 'observation.t_img_ms', 'observation.t_action_ms']
        table = pq.read_table(path, columns=columns)
        self.arrays = {k: np.asarray(table[k].to_pylist(), dtype=float) for k in columns}
        if any(not np.isfinite(a).all() for a in self.arrays.values()):
            raise ValueError('Nonfinite values in replay channels')
        image_ms = self.arrays['observation.t_img_ms'].reshape(-1)
        self.times = (image_ms-image_ms[0])/1000 if clock == 'recorded' else np.arange(len(image_ms))/self.fps
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError('Replay requires strictly increasing image times')
        # Both recorder clocks use the same epoch. Align commands to their own clocks,
        # while preserving frame pairing if the timestamps are non-monotonic.
        action_times = (self.arrays['observation.t_action_ms'].reshape(-1)-image_ms[0])/1000
        self.action_times = action_times if clock == 'recorded' and np.all(np.diff(action_times)>0) else self.times
        self.clock = clock
        self.timing = {'clock': clock, 'command_alignment': 'action_timestamp' if clock == 'recorded' and np.all(np.diff(action_times)>0) else 'paired_image_rows',
                       'action_timestamps_frozen': bool(np.ptp(action_times)==0),
                       'action_timestamps_strictly_increasing': bool(np.all(np.diff(action_times)>0)),
                       'image_dt_max_s': float(np.diff(self.times).max()),
                       'latency_recovered': bool(clock == 'recorded' and np.all(np.diff(action_times)>0))}
        notes=self.root/'meta/source_notes.json'
        self.provenance=json.loads(notes.read_text()) if notes.exists() else {}
        self.hand_action_provenance=('reconstructed: four closed-finger targets and imputed thumb medians' if self.index<46 else 'recorded hand commands') if 'episodes_46_193' in self.provenance else 'not classified'
        for key in ['action','action.low_level']:
            if self.info['features'][key]['names'][:29] != self.names:
                raise ValueError(f'{key} joint order differs from observation.state; explicit remapping required')

    @property
    def duration(self):
        return float(self.times[-1])

    def interpolate(self, key, t):
        a = self.arrays[key]
        ts = self.action_times if key.startswith('action') else self.times
        if t <= ts[0]: return a[0].copy()
        if t >= ts[-1]: return a[-1].copy()
        i = int(np.searchsorted(ts, t)-1)
        u = (t-ts[i])/(ts[i+1]-ts[i])
        return a[i]*(1-u)+a[i+1]*u

    def sample(self, t, source='state'):
        key = {'state':'observation.state', 'action':'action', 'low_level':'action.low_level'}[source]
        return self.interpolate(key,t)[:29], self.interpolate('action.hand_left',t), self.interpolate('action.hand_right',t)

    def video_path(self, view):
        return self.root / self.info['video_path'].format(episode_chunk=self.index//self.info['chunks_size'], episode_index=self.index, video_key=f'observation.images.{view}')

    def frame_at(self, t):
        return int(np.clip(np.searchsorted(self.times,t,side="right")-1,0,len(self.times)-1))
