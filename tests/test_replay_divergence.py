import numpy as np
from tools.diagnose_replay_divergence import decompose, gate_summary


def test_decomposition_catches_opposing_tracking_and_grasp_errors():
    actual=np.array([[1.,0.,0.]])
    arm=np.array([[3.,0.,0.]])
    measured=np.zeros((1,3))
    total,tracking,grasp=decompose(actual,arm,measured)
    np.testing.assert_allclose(total,tracking+grasp)
    np.testing.assert_allclose(tracking,[[3.,0.,0.]])
    np.testing.assert_allclose(grasp,[[-2.,0.,0.]])


def test_dwell_uses_contiguous_elapsed_span_not_count_times_sample_period():
    settings={'radial_tolerance':.0007,'min_insertion_depth':.010,'max_insertion_depth':.012,'angle_tolerance_deg':5.}
    times=np.array([0.,.01,.02,.03,.04,.05])
    radial=np.array([0.,0.,.001,0.,0.,0.])
    result=gate_summary(times,radial,np.full(6,.011),np.zeros(6),settings)
    assert abs(result['all']['longest_sampled_span_s']-.02)<1e-10
    assert result['all']['fraction']==5/6
