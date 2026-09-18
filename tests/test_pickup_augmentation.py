import copy
import numpy as np
from tools.collect_pickup_augmentations import placed_config,smoothstep
from g1_pipette_sim.scene import rack_target


def test_well_change_preserves_tip_world_position_and_moves_attached_rack():
    base={'rack':{'rows':8,'cols':12,'active_cell':[3,4],'pitch':.0105,'yaw_deg':23.,'center_xy':[.3,.04],'height':.06},'table':{'top_z':.6},'waste':{'hinge':{'enabled':True},'offset_from_rack':[.1,0]}}
    saved=copy.deepcopy(base)
    changed=placed_config(base,[5,8],[0,0])
    np.testing.assert_allclose(rack_target(changed),rack_target(base))
    assert changed['waste']==base['waste'] and base==saved
    assert changed['rack']['active_cell']==[5,8]


def test_action_shift_is_applied_to_tip_target_in_world_frame():
    base={'rack':{'rows':8,'cols':12,'active_cell':[3,4],'pitch':.0105,'yaw_deg':23.,'center_xy':[.3,.04],'height':.06},'table':{'top_z':.6},'waste':{}}
    changed=placed_config(base,[0,11],[.001,-.002])
    np.testing.assert_allclose(rack_target(changed)-rack_target(base),[.001,-.002,0])
    assert changed['table']==base['table']


def test_action_blend_keeps_initial_pose_and_constant_contact_offset():
    np.testing.assert_allclose(smoothstep(np.array([-1.,0.,.5,1.,2.])),[0,0,.5,1,1])
    h=1e-6
    assert smoothstep(h)/h<1e-4
    assert (1-smoothstep(1-h))/h<1e-4
