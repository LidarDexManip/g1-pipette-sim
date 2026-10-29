import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lidar_ppo.actions import demonstration_action, command_action, masked_bc, PROPRIO_DIM
from lidar_ppo.models import Teacher, Student, sample_action
from lidar_ppo.returns import gae
from lidar_ppo.data import SequenceData, letterbox
from lidar_ppo.controller import TwistController


def test_command_delta_and_action_masks():
    command=np.zeros((2,35));previous=command.copy();command[:,22]=[.02,.2]
    action,mask=demonstration_action(command,previous,np.ones((2,6))*500,.04)
    assert action[0,0]==.5 and mask[0,0]==1 and mask[1,0]==0
    ref,hand=command_action(torch.tensor(action),torch.zeros(2,35),torch.zeros(2,6),
                            torch.tensor([[-1.,1.]]*7),.04,.08)
    assert ref[0,22]==.02 and torch.allclose(hand,torch.full((2,6),.08))
    pred=torch.zeros(2,13,requires_grad=True)
    mask=torch.ones_like(pred);mask[:,7:]=0
    masked_bc(pred,torch.ones_like(pred),mask).backward()
    assert torch.count_nonzero(pred.grad[:,7:])==0


def test_gae_timeout_bootstraps_but_never_crosses_reset():
    reward=torch.tensor([[1.],[100.]])
    values=torch.zeros_like(reward);next_values=torch.tensor([[5.],[0.]])
    terminated=torch.tensor([[False],[True]]);truncated=torch.tensor([[True],[False]])
    adv,ret=gae(reward,values,next_values,terminated,truncated,.9,.95)
    assert adv[0,0]==5.5 and ret[1,0]==100
    terminated[0]=True
    assert gae(reward,values,next_values,terminated,truncated,.9,.95)[0][0,0]==1


def test_squashed_policy_ratios_and_real_optimizer_update():
    torch.manual_seed(1)
    model=Teacher(12);optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    state=torch.randn(16,12);mean,value=model(state)
    action,latent,old,_=sample_action(mean,model.log_std)
    assert bool((action.abs()<=1).all())
    assert torch.allclose(old,sample_action(mean,model.log_std,latent)[2])
    old=old.detach();before=model.actor[0].weight.detach().clone()
    mean,value=model(state);logp=sample_action(mean,model.log_std,latent)[2]
    loss=-(logp-old).exp().mean()+value.square().mean()
    optim.zero_grad();loss.backward();optim.step()
    assert not torch.equal(before,model.actor[0].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_recurrent_reset_prevents_episode_leakage():
    model=Student(24,32)
    features=torch.randn(2,24);proprio=torch.randn(2,PROPRIO_DIM)
    starts=torch.tensor([True,True])
    a,_=model(features,proprio,torch.randn(2,32)*100,starts)
    b,_=model(features,proprio,torch.zeros(2,32),starts)
    assert torch.equal(a,b)


def test_bc_sequences_never_cross_episode_and_preserve_masks(tmp_path):
    records=[]
    for i in range(2):
        np.savez(tmp_path/f"{i}.npz",images=np.full((20,3,8,8),i,dtype=np.uint8),
                 proprio=np.full((20,PROPRIO_DIM),i,dtype=np.float32),actions=np.zeros((20,13),dtype=np.float32),
                 masks=np.ones((20,13),dtype=np.float32))
        records.append(dict(file=f"{i}.npz",split="train",frames=20))
    (tmp_path/"manifest.json").write_text(json.dumps(dict(records=records)))
    sample=SequenceData(tmp_path,"train",8).sample(32,"cpu")
    assert sample["images"].shape==(8,32,3,8,8)
    assert torch.equal(sample["proprio"][0],sample["proprio"][-1])


def test_letterbox_preserves_aspect_and_full_frame():
    result=letterbox(torch.full((1,3,40,80),255,dtype=torch.uint8),80)
    assert result.shape==(1,3,80,80)
    assert result[:,:,:20].sum()==0 and (result[:,:,20:60]==255).all()


def test_batched_twist_observation_matches_recorded_contract(tmp_path):
    class Record(torch.nn.Module):
        def forward(self,x):return x[:,40:69]*.01
    path=tmp_path/"twist.pt"
    torch.jit.trace(Record(),torch.zeros(1,1432)).save(str(path))
    cfg=dict(default_angles=[0.]*29,action_scale=.5)
    core=TwistController(path,cfg,2,"cpu")
    reference=torch.arange(35.).repeat(2,1);q=torch.ones(2,29);dq=torch.ones(2,29)
    quat=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
    core.step(reference,q,dq,quat)
    current=core.history[:,-1]
    assert torch.equal(current[:,:6],reference[:,29:])
    assert torch.equal(current[:,6:35],reference[:,:29])
    assert current.shape==(2,127)
    assert torch.count_nonzero(current[:,69:98][:,[4,5,10,11]])==0


def test_cpu_contact_force_and_penetration_reduction():
    import mujoco
    from lidar_ppo.physics import CPUPhysics
    xml='''<mujoco><option timestep="0.001"/><worldbody>
      <geom type="plane" size="1 1 .1"/><body name="env_01_pipette" pos="0 0 .009">
      <freejoint/><geom type="sphere" size=".01" mass=".08"/></body>
      <body name="right_wrist_yaw_link" pos="1 0 1"><geom size=".01"/></body>
      </worldbody></mujoco>'''
    physics=CPUPhysics(mujoco.MjModel.from_xml_string(xml),2)
    physics.forward()
    assert bool((physics.metrics[:,0]>.0009).all())
    assert bool((physics.metrics[:,2]>0).all())
    assert physics.metrics[:,1].sum()==0


def test_warp_contact_reductions_and_overflow_on_cpu(tmp_path):
    wp=pytest.importorskip("warp")
    from lidar_ppo.warp_metrics import reduce_contacts,update_peak,track_overflow
    wp.config.kernel_cache_dir=str(tmp_path/"warp-cache")
    wp.init()
    with wp.ScopedDevice("cpu"):
        count=wp.array([2],dtype=int);world=wp.array([0,1],dtype=int)
        geoms=wp.array([[0,1],[0,2]],dtype=wp.vec2i)
        distance=wp.array([-.002,-.001],dtype=float)
        bodies=wp.array([1,2,0],dtype=int);category=wp.array([2,3,1],dtype=int)
        force=wp.array([[3.,0,0,0,0,0],[2.,0,0,0,0,0]],dtype=wp.spatial_vector)
        stats=wp.zeros((2,4),dtype=float);peaks=wp.zeros((2,4),dtype=float)
        wp.launch(reduce_contacts,dim=2,inputs=[count,world,geoms,distance,bodies,category,1,force],outputs=[stats])
        wp.launch(update_peak,dim=(2,4),inputs=[stats],outputs=[peaks])
        np.testing.assert_allclose(peaks.numpy(),[[.002,3,0,0],[.001,0,2,0]],atol=1e-6)
        flag=wp.zeros(1,dtype=int);nefc=wp.array([1,1],dtype=int)
        wp.launch(track_overflow,dim=2,inputs=[count,nefc,1,10],outputs=[flag])
        assert flag.numpy()[0]==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA GPU required")
def test_warp_cpu_contact_parity():
    import mujoco
    from lidar_ppo.physics import CPUPhysics,WarpPhysics
    xml='''<mujoco><option timestep="0.001" cone="elliptic" integrator="implicitfast"/><worldbody>
      <geom type="plane" size="1 1 .1"/><body name="env_01_pipette" pos="0 0 .009">
      <freejoint/><geom type="sphere" size=".01" mass=".08"/></body>
      <body name="right_wrist_yaw_link" pos="1 0 1"><geom size=".01"/></body>
      </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    cpu=CPUPhysics(model,2);gpu=WarpPhysics(model,2,"cuda:0",nconmax=16,njmax=64,cuda_graph=True)
    cpu.forward();gpu.forward()
    cpu.step(torch.empty(2,0),20);gpu.step(torch.empty(2,0,device="cuda"),20)
    torch.testing.assert_close(gpu.get("qpos").cpu(),cpu.get("qpos"),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(gpu.metrics.cpu(),cpu.metrics,atol=.1,rtol=.05)
