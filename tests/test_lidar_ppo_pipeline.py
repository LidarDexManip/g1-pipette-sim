"""End-to-end learner plumbing with actual MuJoCo and a tiny synthetic scene.

This is deliberately NOT a claim of task learning or DINO/GPU validation.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from lidar_ppo.config import write_json,digest
from lidar_ppo.actions import PROPRIO_DIM


def make_fixture(root):
    import mujoco
    root.mkdir()
    xml=ET.Element("mujoco")
    ET.SubElement(xml,"option",timestep=".001",integrator="implicitfast")
    world=ET.SubElement(xml,"worldbody")
    ET.SubElement(world,"geom",type="plane",size="2 2 .1")
    pelvis=ET.SubElement(world,"body",name="pelvis",pos="0 0 1")
    names=[f"joint_{i}" for i in range(29)]
    hands=["R_"+n for n in ["pinky_proximal_joint","ring_proximal_joint","middle_proximal_joint","index_proximal_joint",
                            "thumb_proximal_yaw_joint","thumb_proximal_pitch_joint"]]
    actuator=ET.SubElement(xml,"actuator")
    for i,name in enumerate(names+hands):
        body=ET.SubElement(pelvis,"body",name="right_wrist_yaw_link" if i==28 else f"link_{i}",pos=f"{i*.03} 0 0")
        ET.SubElement(body,"joint",name=name,range="-3 3",type="hinge")
        ET.SubElement(body,"geom",type="sphere",size=".01",mass=".1",contype="0",conaffinity="0")
        ET.SubElement(actuator,"position",name="servo_"+name,joint=name,kp="1",kv=".1")
    body=ET.SubElement(world,"body",name="env_01_pipette",pos="0 0 .02")
    ET.SubElement(body,"freejoint");ET.SubElement(body,"geom",type="sphere",size=".02",mass=".08")
    ET.SubElement(world,"camera",name="egoview",pos="0 -1 1",xyaxes="1 0 0 0 1 1")
    ET.ElementTree(xml).write(root/"scene.xml",encoding="unicode")
    model=mujoco.MjModel.from_xml_path(str(root/"scene.xml"));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    class ZeroController(torch.nn.Module):
        def forward(self,x):return x[:,:29]*0
    torch.jit.trace(ZeroController(),torch.zeros(1,1432)).save(str(root/"twist2.pt"))
    write_json(root/"controller.json",dict(default_angles=[0.]*29,action_scale=.5))
    write_json(root/"contract.json",dict(version=1,joint_names=names,fixed_pelvis=True,policy_hz=25,
        hand_mapping=dict(finger_closed_rad=1.,thumb_bend_closed_rad=.5,thumb_rotation_max_rad=1.)))
    fields={k:np.stack([getattr(data,k)]*2) for k in ["qpos","qvel","ctrl","act"]}
    fields.update(warmstart=np.zeros((2,model.nv)),history=np.zeros((2,10,127)),previous=np.zeros((2,29)),
                  reference=np.zeros((2,35)),hand=np.ones((2,6)),start_height=np.full(2,.02),
                  episode=np.array([1,2]),time=np.zeros(2),natural=np.ones(2,dtype=bool))
    np.savez(root/"resets.npz",**fields)
    write_json(root/"checksums.json",{p.name:digest(p) for p in root.iterdir()})
    write_json(root/"review.json",dict(accepted_reset_indices=[0,1],validation_episode_ids=[2]))
    return names


class TinyEncoder(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.output_dim=12
        self.backbone=SimpleNamespace(config=SimpleNamespace(_commit_hash="test-only"))
    def forward(self,x):return F.adaptive_avg_pool2d(x.float()/255,2).flatten(1)


def test_training_rejects_unqualified_ordinary_resets(tmp_path):
    from lidar_ppo.environment import PickupEnv
    cfg=json.loads((Path(__file__).parents[1]/"config/lidar_ppo/b200.json").read_text())
    bundle=tmp_path/"bundle";make_fixture(bundle)
    write_json(bundle/"review.json",dict(accepted_reset_indices=[],validation_episode_ids=[2]))
    with pytest.raises(ValueError,match="No qualified resets"):
        PickupEnv(bundle,cfg,1,backend="cpu",device="cpu")


def test_complete_pipeline_checkpoint_resume_and_export(tmp_path,monkeypatch):
    import lidar_ppo.train as train
    from lidar_ppo.environment import PickupEnv
    from lidar_ppo.evaluate import evaluate
    from lidar_ppo.policy import export_policy
    cfg=json.loads((Path(__file__).parents[1]/"config/lidar_ppo/b200.json").read_text())
    cfg["episode_seconds"]=.08
    cfg["ppo"].update(rollout_steps=4,sequence_length=2,minibatch_size=4,epochs=1,save_every=1,real_bc_batch=2)
    cfg["bc"].update(batch_size=2,sequence_length=2)
    cfg["hidden_dim"]=32;cfg["image_size"]=14
    cfg["randomization"].update(arm_target_noise_rad=0,friction_scale=[1,1],servo_scale=[1,1])
    bundle=tmp_path/"bundle";names=make_fixture(bundle)
    data=tmp_path/"data";data.mkdir();records=[]
    for i,split in enumerate(["train","validation"]):
        np.savez(data/f"{i}.npz",images=np.zeros((10,3,14,14),dtype=np.uint8),
                 proprio=np.zeros((10,PROPRIO_DIM),dtype=np.float32),actions=np.zeros((10,13),dtype=np.float32),
                 masks=np.ones((10,13),dtype=np.float32))
        records.append(dict(file=f"{i}.npz",split=split,frames=10))
    write_json(data/"manifest.json",dict(records=records,policy_hz=25,arm_step_rad=.04,image_size=14,joint_names=names))
    monkeypatch.setattr(train,"VisionEncoder",TinyEncoder)
    monkeypatch.setattr(PickupEnv,"images",lambda self:torch.zeros(self.nworld,3,14,14,dtype=torch.uint8,device=self.device))
    train.pretrain_bc(cfg,data,tmp_path/"bc",1,"cpu")
    teacher=tmp_path/"teacher"
    train.train_ppo(cfg,bundle,teacher,1,"cpu","cpu",2)
    before=train.load_checkpoint(teacher/"latest.pt","cpu")
    train.train_ppo(cfg,bundle,teacher,1,"cpu","cpu",2,resume=teacher/"latest.pt")
    after=train.load_checkpoint(teacher/"latest.pt","cpu")
    assert after["step"]==2
    assert any(not torch.equal(before["model"][k],after["model"][k]) for k in before["model"])
    train.distill(cfg,bundle,data,teacher/"latest.pt",tmp_path/"bc/best.pt",tmp_path/"distill",1,"cpu","cpu",2)
    train.train_ppo(cfg,bundle,tmp_path/"visual",1,"cpu","cpu",2,student=tmp_path/"distill/latest.pt",data_root=data)
    report=evaluate(cfg,bundle,tmp_path/"visual/latest.pt",tmp_path/"evaluation.json",2,"cpu","cpu",1)
    assert report["episodes"]==2 and report["ordinary_starts_only"]
    export_policy(tmp_path/"visual/latest.pt",tmp_path/"export")
    actor=torch.load(tmp_path/"export/actor.pt",weights_only=True)
    assert not any(k.startswith("critic.") for k in actor)
