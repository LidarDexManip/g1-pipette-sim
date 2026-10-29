import time
from pathlib import Path

import torch

from .config import digest, write_json
from .environment import PickupEnv
from .models import Teacher
from .train import load_checkpoint, setup_student, encode, seed_all


@torch.no_grad()
def evaluate(cfg,bundle,checkpoint,output,episodes,backend,device,nworld):
    seed_all(cfg["seed"]+10000)
    saved=load_checkpoint(checkpoint,device)
    env=PickupEnv(bundle,cfg,nworld,backend,device,evaluation=True)
    encoder=None
    if saved["kind"] == "teacher":
        if saved["bundle_sha256"] != digest(Path(bundle)/"checksums.json"):
            raise ValueError("Teacher bundle differs")
        model=Teacher(env.state_dim).to(device);model.load_state_dict(saved["model"])
    else:model,encoder,_=setup_student(cfg,device,checkpoint=checkpoint)
    model.eval();hidden=torch.zeros(nworld,cfg["hidden_dim"],device=device)
    records=[];peak_force=torch.zeros(nworld,device=device);peak_pen=torch.zeros_like(peak_force)
    try:
        while len(records)<episodes:
            if encoder:
                mean,hidden=model(encode(encoder,env.images(),cfg["encoder_batch"]),env.proprio(),hidden,env.episode_start)
            else:mean,_=model(env.state())
            _,term,trunc,info=env.step(mean.tanh())
            peak_force=torch.maximum(peak_force,info["peak_hand_force"])
            peak_pen=torch.maximum(peak_pen,info["peak_penetration"])
            done=term|trunc
            for i in torch.where(done)[0].tolist():
                if len(records)<episodes:
                    records.append(dict(success=bool(info["success"][i]),failure=bool(info["failure"][i]),
                        timeout=bool(trunc[i]),duration_s=float(info["elapsed"][i]),
                        peak_hand_force_n=float(peak_force[i]),peak_penetration_m=float(peak_pen[i])))
            peak_force[done]=0;peak_pen[done]=0
            if bool(done.any()):env.reset(done)
    finally:env.close()
    report=dict(checkpoint_sha256=digest(checkpoint),bundle_sha256=digest(Path(bundle)/"checksums.json"),
        ordinary_starts_only=True,held_out_episodes_only=True,episodes=len(records),
        successes=sum(r["success"] for r in records),records=records)
    write_json(output,report)
    return {k:v for k,v in report.items() if k!="records"}


@torch.no_grad()
def benchmark(cfg,bundle,backend,device,nworld,steps):
    env=PickupEnv(bundle,cfg,nworld,backend,device,diagnostic=True)
    action=torch.zeros(nworld,13,device=device)
    try:
        for _ in range(2):
            _,t,u,_=env.step(action)
            if bool((t|u).any()):env.reset(t|u)
        if str(device).startswith("cuda"):torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(steps):
            _,t,u,_=env.step(action)
            if bool((t|u).any()):env.reset(t|u)
        if str(device).startswith("cuda"):torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        return dict(backend=backend,worlds=nworld,steps=steps,seconds=seconds,
            policy_transitions_per_second=nworld*steps/seconds,physics_steps_per_second=nworld*steps*40/seconds,
            diagnostic_only=True,includes_rendering=False,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None)
    finally:env.close()
