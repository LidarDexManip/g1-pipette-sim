"""Small replay diagnostics for success/failure comparison and future clustering."""
from pathlib import Path
import numpy as np
from g1_pipette_sim.dynamics import DynamicSimulation
from .contracts import atomic_json

SAMPLE_HZ = 20
TRACE_COLUMNS = ['time_s','nozzle_x_m','nozzle_y_m','nozzle_z_m','mouth_x_m','mouth_y_m','mouth_z_m','angle_deg']


class DiagnosticSimulation(DynamicSimulation):
    """Observe the unchanged simulation; do not change control or collision rules."""
    def __init__(self, cfg, ep):
        self._diagnostic_tip_length=cfg['pipette']['tip_length']
        super().__init__(cfg, ep, source='low_level')
        self.diagnostic_rows = []
        self.next_diagnostic = 0.
        self.sample_diagnostics()

    def sample_diagnostics(self):
        tip = self.data.body('tip')
        rotation = tip.xmat.reshape(3, 3)
        mouth = tip.xpos + rotation[:, 2]*self._diagnostic_tip_length/2
        nozzle = self.data.site('nozzle')
        relative = rotation.T @ (nozzle.xpos-mouth)
        angle = np.degrees(np.arccos(np.clip(rotation[:,2] @ nozzle.xmat.reshape(3,3)[:,2], -1, 1)))
        stats = list(self.contact_diagnostics.values())
        force = max((s.get('maximum_normal_force_N',0.) for s in stats),default=0.)
        penetration = max((s.get('maximum_penetration_mm',0.) for s in stats),default=0.)
        self.diagnostic_rows.append((float(self.time), self.data.qpos.copy(), self.data.qvel.copy(),
                                     nozzle.xpos.copy(), mouth.copy(), relative.copy(), float(angle),
                                     float(force), float(penetration), self.state == 'attached'))
        self.next_diagnostic = self.time + 1/SAMPLE_HZ

    def step(self):
        super().step()
        if self.time >= self.next_diagnostic-1e-10:
            self.sample_diagnostics()

    def save_diagnostics(self, path):
        if not self.diagnostic_rows or self.diagnostic_rows[-1][0] != self.time:
            self.sample_diagnostics()
        keys=['time_s','qpos','qvel','nozzle_world_m','tip_mouth_world_m','nozzle_relative_tip_m',
              'angle_deg','maximum_contact_force_N','maximum_penetration_mm','attached']
        arrays = {key:np.asarray([row[i] for row in self.diagnostic_rows]) for i,key in enumerate(keys)}
        atomic_npz(path, **arrays)


def make_diagnostic_simulation(cfg, ep):
    return DiagnosticSimulation(cfg,ep)


def atomic_npz(path, **arrays):
    import os, tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.npz',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_trace(folder, stage, trace):
    atomic_npz(Path(folder)/(stage+'-alignment.npz'),trace=np.asarray(trace,dtype=np.float64).reshape(-1,8))
    atomic_json(Path(folder)/'diagnostic-schema.json',{
        'schema':1,'sample_hz':SAMPLE_HZ,'alignment_columns':TRACE_COLUMNS,
        'state_precision':'float64; sparse states support diagnostic rendering, not exact integration replay',
        'force_and_penetration':'running maxima; not instantaneous force measurements',
        'dense_replay':'commands.npz plus pinned source initial state and physics dependencies',
        'future_clustering':'Use tip-relative positions/angles and phase-relative time; preserve missing masks. Keep labels and parent IDs out of distance features.'})
