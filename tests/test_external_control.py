"""External control ownership regressions without a dataset dependency."""
from types import SimpleNamespace
import numpy as np
import pytest
from g1_pipette_sim.dynamics import DynamicSimulation


def test_target_mapping_and_validation():
    sim=DynamicSimulation.__new__(DynamicSimulation)
    sim.control_mode='external';sim.desired=np.zeros(4)
    mapping={'servo_elbow':2,'servo_waist':0}
    sim.model=SimpleNamespace(actuator=lambda n:SimpleNamespace(id=mapping[n]))
    sim.set_motor_targets(['elbow','waist'],[.3,.1])
    np.testing.assert_array_equal(sim.desired,[.1,0,.3,0])
    with pytest.raises(ValueError):sim.set_motor_targets(['elbow'],[np.nan])
    with pytest.raises(ValueError):sim.set_motor_targets(['elbow'],[1,2])
    sim.control_mode='replay'
    with pytest.raises(RuntimeError):sim.set_motor_targets(['elbow'],[1])


def test_external_targets_survive_physics_steps():
    from pathlib import Path
    import json
    from g1_pipette_sim.dataset import Episode
    root=Path(__file__).resolve().parents[1]
    cache=Path.home()/'.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hil293'
    config_path=root/'outputs/hil293/fixed-height-sweep/episode-011/validated-config.json'
    if not cache.exists() or not config_path.is_file():
        pytest.skip('local HIL293 dataset and validated workspace required')
    ep=Episode(cache,11)
    config=json.loads(config_path.read_text())
    config['physics']['gravity_compensation']=False
    config['physics']['joint_servos']={}
    sim=DynamicSimulation(config,ep,control_mode='external')
    q_before=sim.data.qpos.copy()
    target=ep.sample(0)[0].copy();target[25]+=.05
    sim.set_motor_targets(ep.names,target)
    np.testing.assert_array_equal(sim.data.qpos,q_before)
    expected=sim.desired.copy()
    def forbidden(t):raise AssertionError('Replay overwrote external targets')
    sim._reference=forbidden
    for _ in range(40):sim.step()
    np.testing.assert_array_equal(sim.desired,expected)
    np.testing.assert_array_equal(sim.data.ctrl,expected)
