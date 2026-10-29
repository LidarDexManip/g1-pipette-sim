"""Deployment inference interface; no robot transport or hardware side effects."""
import json
from pathlib import Path

import torch

from .config import write_json, digest
from .data import letterbox
from .models import Student, VisionEncoder
from .train import load_checkpoint


def export_policy(checkpoint,destination):
    saved=load_checkpoint(checkpoint,"cpu")
    if saved["kind"]=="teacher":raise ValueError("A privileged teacher cannot be deployed to the real robot")
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    weights={k:v for k,v in saved["model"].items() if not k.startswith("critic.")}
    torch.save(weights,destination/"actor.pt")
    write_json(destination/"policy.json",dict(version=1,feature_dim=saved["feature_dim"],
        config=saved["config"],encoder_revision=saved["encoder_revision"],
        source_checkpoint_sha256=digest(checkpoint),
        proprio="measured q29, measured dq29*0.05, previous commanded body29+root6, hand6/1000, previous normalized action13",
        action="7 right-arm increments times arm_step_rad, then 6 absolute hand fractions=(action+1)/2; apply hand slew limit",
        camera="RGB ego full frame, letterboxed; no wrist cameras",
        controller="TWIST2 50 Hz, fixed pelvis in training; no hardware transport included"))


class PickupPolicy:
    def __init__(self,directory,device="cuda:0"):
        directory=Path(directory)
        self.meta=json.loads((directory/"policy.json").read_text());self.cfg=self.meta["config"]
        self.device=torch.device(device)
        self.encoder=VisionEncoder(self.cfg["encoder"],self.meta["encoder_revision"]).to(device).eval()
        self.actor=Student(self.meta["feature_dim"],self.cfg["hidden_dim"]).to(device).eval()
        self.actor.load_state_dict(torch.load(directory/"actor.pt",map_location=device,weights_only=True))
        self.hidden=None

    @torch.no_grad()
    def __call__(self,rgb_nchw,proprio,episode_start):
        rgb_nchw=letterbox(rgb_nchw.to(self.device),self.cfg["image_size"])
        if self.hidden is None:
            self.hidden=torch.zeros(len(rgb_nchw),self.actor.hidden_dim,device=self.device)
        mean,self.hidden=self.actor(self.encoder(rgb_nchw),proprio.to(self.device),self.hidden,episode_start.to(self.device))
        return mean.tanh()
