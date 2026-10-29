"""PPO teacher/visual refinement and real-BC/DAgger student training."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .actions import masked_bc
from .config import digest, write_json
from .data import SequenceData
from .environment import PickupEnv
from .models import Teacher, Student, VisionEncoder, sample_action
from .returns import gae


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def load_checkpoint(path, device):
    return torch.load(path,map_location=device,weights_only=True)


def save_checkpoint(path,model,optimizer,cfg,step,kind,**extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    torch.save(dict(version=1,kind=kind,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    config=cfg,step=step,torch_rng=torch.get_rng_state(),**extra),tmp)
    tmp.replace(path)


def open_run(output,cfg,resume):
    import platform
    output=Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f"Use a fresh output or --resume: {output}")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"config.json",cfg)
    write_json(output/"runtime.json",dict(python=platform.python_version(),torch=str(torch.__version__),
        cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    return output


def log(root,record):
    with (root/"metrics.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
    print(json.dumps(record),flush=True)


@torch.no_grad()
def encode(encoder,images,batch):
    shape=images.shape[:-3]
    flat=images.reshape(-1,*images.shape[-3:])
    return torch.cat([encoder(x) for x in flat.split(batch)]).reshape(*shape,-1)


def setup_student(cfg,device,state_dim=None,checkpoint=None):
    saved=load_checkpoint(checkpoint,device) if checkpoint else None
    if saved:
        if saved["config"]["arm_step_rad"] != cfg["arm_step_rad"] or saved["config"]["policy_hz"] != cfg["policy_hz"]:
            raise ValueError("Student action contract differs")
        if saved["config"]["encoder"] != cfg["encoder"]:
            raise ValueError("Student encoder differs")
        revision=saved.get("encoder_revision",cfg["encoder_revision"])
    else:revision=cfg["encoder_revision"]
    encoder=VisionEncoder(cfg["encoder"],revision).to(device).eval()
    cfg["encoder_revision"]=getattr(encoder.backbone.config,"_commit_hash",None) or revision
    student=Student(encoder.output_dim,cfg["hidden_dim"],state_dim).to(device)
    if saved:
        incompatible=student.load_state_dict(saved["model"],strict=False)
        if any(not k.startswith("critic.") for k in incompatible.missing_keys+incompatible.unexpected_keys):
            raise ValueError(f"Incompatible student checkpoint: {incompatible}")
    return student,encoder,saved


def bc_batch_loss(model,encoder,data,cfg,batch,device):
    sample=data.sample(batch,device)
    features=sample["features"].float() if "features" in sample else encode(encoder,sample["images"],cfg["encoder_batch"])
    length,batch=features.shape[:2]
    hidden=torch.zeros(batch,model.hidden_dim,device=device)
    starts=torch.zeros(length,batch,device=device,dtype=torch.bool);starts[0]=True
    # First half reconstructs recurrent context; supervise only the second half.
    burn=length//2
    with torch.no_grad():
        _,hidden=model.sequence(features[:burn],sample["proprio"][:burn],hidden,starts[:burn])
    means,_=model.sequence(features[burn:],sample["proprio"][burn:],hidden.detach(),starts[burn:])
    return masked_bc(means.tanh(),sample["actions"][burn:],sample["masks"][burn:])


def check_data(data,cfg,joint_names=None):
    for key in ("policy_hz","arm_step_rad","image_size"):
        if data.manifest[key] != cfg[key]:raise ValueError(f"Prepared data {key} differs from config")
    if joint_names and data.manifest["joint_names"] != joint_names:
        raise ValueError("Prepared data body joint ordering differs from simulator")
    if "feature_encoder" in data.manifest:
        if data.manifest["feature_encoder"]!=cfg["encoder"] or data.manifest["encoder_revision"]!=cfg["encoder_revision"]:
            raise ValueError("Cached feature encoder/revision differs; recompute features")


def pretrain_bc(cfg,data_root,output,updates,device,resume=None):
    seed_all(cfg["seed"])
    model,encoder,saved=setup_student(cfg,device,checkpoint=resume)
    root=open_run(output,cfg,resume)
    data=SequenceData(data_root,"train",cfg["bc"]["sequence_length"]*2,cfg["seed"])
    validation=SequenceData(data_root,"validation",cfg["bc"]["sequence_length"]*2,123)
    check_data(data,cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["bc"]["learning_rate"])
    start=0;best=float("inf")
    if saved:
        if saved["kind"]!="bc":raise ValueError("BC resume requires a BC checkpoint")
        if saved["data_manifest_sha256"]!=digest(Path(data_root)/"manifest.json") or saved["config"]!=cfg:
            raise ValueError("BC resume data/config differs")
        optimizer.load_state_dict(saved["optimizer"]);start=saved["step"];best=saved.get("best_validation",best)
    for update in range(start,start+updates):
        loss=bc_batch_loss(model,encoder,data,cfg,cfg["bc"]["batch_size"],device)
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (update+1)%cfg["ppo"]["save_every"]==0 or update+1==start+updates:
            with torch.no_grad():
                # Reset validation sampler so checkpoint selection uses the same sequences.
                validation.rng=np.random.default_rng(123)
                val=float(torch.stack([bc_batch_loss(model,encoder,validation,cfg,cfg["bc"]["batch_size"],device) for _ in range(4)]).mean())
            improved=val<best;best=min(best,val)
            extra=dict(feature_dim=encoder.output_dim,encoder_revision=cfg["encoder_revision"],
                       data_manifest_sha256=digest(Path(data_root)/"manifest.json"),best_validation=best)
            save_checkpoint(root/"latest.pt",model,optimizer,cfg,update+1,"bc",**extra)
            if improved:save_checkpoint(root/"best.pt",model,optimizer,cfg,update+1,"bc",**extra)
            log(root,dict(update=update+1,bc_loss=float(loss),validation_bc=val,best_validation=best))


def train_ppo(cfg,bundle,output,updates,backend,device,nworld,resume=None,student=None,data_root=None):
    seed_all(cfg["seed"])
    visual=student is not None or (resume and load_checkpoint(resume,"cpu")["kind"]=="visual_ppo")
    if visual and not data_root:raise ValueError("Visual PPO requires --data for real BC rehearsal")
    env=PickupEnv(bundle,cfg,nworld,backend,device)
    encoder=None;real=None
    if visual:
        model,encoder,saved=setup_student(cfg,device,env.state_dim,resume or student)
        real=SequenceData(data_root,"train",cfg["bc"]["sequence_length"]*2,cfg["seed"])
        check_data(real,cfg,env.contract["joint_names"])
    else:
        model=Teacher(env.state_dim).to(device);saved=load_checkpoint(resume,device) if resume else None
        if saved:model.load_state_dict(saved["model"])
    root=open_run(output,cfg,resume)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["ppo"]["learning_rate"])
    bundle_hash=digest(Path(bundle)/"checksums.json")
    start=0
    if resume:
        if saved["bundle_sha256"]!=bundle_hash:raise ValueError("Resume bundle differs")
        if saved["config"]!=cfg:raise ValueError("Resume configuration differs; use an explicit new run")
        if visual and saved["data_manifest_sha256"]!=digest(Path(data_root)/"manifest.json"):
            raise ValueError("Resume real-data manifest differs")
        optimizer.load_state_dict(saved["optimizer"]);start=saved["step"]
        torch.set_rng_state(saved["torch_rng"].cpu())
    spec=cfg["ppo"];length=spec["rollout_steps"]
    hidden=torch.zeros(nworld,cfg["hidden_dim"],device=device)
    started=time.monotonic()
    try:
        for update in range(start,start+updates):
            env.update=update
            rows=[];successes=failures=episodes=0
            with torch.no_grad():
                for t in range(length):
                    state=env.state();starts=env.episode_start.clone()
                    row=dict(state=state,starts=starts)
                    if visual:
                        features=encode(encoder,env.images(),cfg["encoder_batch"]);proprio=env.proprio()
                        row.update(features=features,proprio=proprio,hidden=hidden.clone())
                        mean,hidden=model(features,proprio,hidden,starts)
                        value=model.critic(state).squeeze(-1)
                    else:mean,value=model(state)
                    action,latent,logp,_=sample_action(mean,model.log_std)
                    reward,term,trunc,info=env.step(action)
                    next_value=(model.critic(env.state()).squeeze(-1) if visual else model(env.state())[1])
                    row.update(latent=latent,logp=logp,value=value,reward=reward,next_value=next_value,term=term,trunc=trunc)
                    rows.append(row)
                    done=term|trunc
                    successes+=int(info["success"].sum());failures+=int(info["failure"].sum());episodes+=int(done.sum())
                    if bool(done.any()):env.reset(done)
            roll={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
            advantage,returns=gae(roll["reward"],roll["value"],roll["next_value"],roll["term"],roll["trunc"],spec["gamma"],spec["gae_lambda"])
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            kl=0.;bc_value=0.;stop=False
            seq=spec["sequence_length"] if visual else 1
            # Flatten time blocks x worlds; each recurrent minibatch keeps its temporal order.
            def chunks(x):return x.reshape(length//seq,seq,nworld,*x.shape[2:]).transpose(1,2).reshape(-1,seq,*x.shape[2:])
            chunks_roll={k:chunks(v) for k,v in roll.items()}
            adv,ret=chunks(advantage),chunks(returns)
            total=len(adv);batch=max(1,spec["minibatch_size"]//seq)
            for epoch in range(spec["epochs"]):
                for ids in torch.randperm(total,device=device).split(batch):
                    b={k:v[ids].transpose(0,1) for k,v in chunks_roll.items()}
                    if visual:
                        mean,_=model.sequence(b["features"],b["proprio"],b["hidden"][0],b["starts"])
                        value=model.critic(b["state"]).squeeze(-1)
                    else:mean,value=model(b["state"])
                    _,_,logp,entropy=sample_action(mean,model.log_std,b["latent"])
                    logratio=logp-b["logp"];ratio=logratio.exp()
                    with torch.no_grad():kl=float(((ratio-1)-logratio).mean())
                    if kl>spec["target_kl"]*1.5:stop=True;break
                    advantages=adv[ids].transpose(0,1)
                    policy_loss=-torch.minimum(ratio*advantages,ratio.clamp(1-spec["clip"],1+spec["clip"])*advantages).mean()
                    value_loss=(value-ret[ids].transpose(0,1)).square().mean()
                    loss=policy_loss+spec["value_coef"]*value_loss-spec["entropy_coef"]*entropy.mean()
                    if real:
                        bc=bc_batch_loss(model,encoder,real,cfg,spec["real_bc_batch"],device)
                        loss=loss+spec["real_bc_weight"]*bc;bc_value=float(bc.detach())
                    optimizer.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),spec["max_grad_norm"],error_if_nonfinite=True)
                    optimizer.step()
                if stop:break
            log(root,dict(update=update+1,transitions=(update+1)*length*nworld,episodes=episodes,successes=successes,
                failures=failures,mean_reward=float(roll["reward"].mean()),approx_kl=kl,real_bc=bc_value,
                transitions_per_second=(update-start+1)*length*nworld/(time.monotonic()-started)))
            if (update+1)%spec["save_every"]==0 or update+1==start+updates:
                extra=dict(state_dim=env.state_dim,bundle_sha256=bundle_hash)
                if visual:extra.update(feature_dim=encoder.output_dim,encoder_revision=cfg["encoder_revision"],
                                      data_manifest_sha256=digest(Path(data_root)/"manifest.json"))
                save_checkpoint(root/"latest.pt",model,optimizer,cfg,update+1,"visual_ppo" if visual else "teacher",**extra)
    finally:env.close()


def distill(cfg,bundle,data_root,teacher_path,student_path,output,updates,backend,device,nworld,resume=None):
    seed_all(cfg["seed"])
    env=PickupEnv(bundle,cfg,nworld,backend,device)
    teacher_saved=load_checkpoint(teacher_path,device)
    if teacher_saved["kind"]!="teacher" or teacher_saved["bundle_sha256"]!=digest(Path(bundle)/"checksums.json"):
        raise ValueError("Teacher kind/bundle differs")
    teacher=Teacher(env.state_dim).to(device);teacher.load_state_dict(teacher_saved["model"]);teacher.eval()
    model,encoder,saved=setup_student(cfg,device,checkpoint=resume or student_path)
    root=open_run(output,cfg,resume)
    real=SequenceData(data_root,"train",cfg["bc"]["sequence_length"]*2,cfg["seed"])
    check_data(real,cfg,env.contract["joint_names"])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["bc"]["learning_rate"])
    start=0
    if resume:
        if saved["kind"]!="distill":raise ValueError("Distillation resume requires distill checkpoint")
        if (saved["config"]!=cfg or saved["teacher_sha256"]!=digest(teacher_path)
            or saved["bundle_sha256"]!=digest(Path(bundle)/"checksums.json")
            or saved["data_manifest_sha256"]!=digest(Path(data_root)/"manifest.json")):
            raise ValueError("Distillation resume provenance differs")
        optimizer.load_state_dict(saved["optimizer"]);start=saved["step"]
    hidden=torch.zeros(nworld,model.hidden_dim,device=device)
    try:
        for update in range(start,start+updates):
            env.update=update
            initial=hidden.detach().clone();features=[];props=[];starts=[];targets=[]
            with torch.no_grad():
                for _ in range(cfg["bc"]["sequence_length"]):
                    f=encode(encoder,env.images(),cfg["encoder_batch"]);p=env.proprio();s=env.episode_start.clone()
                    mean,hidden=model(f,p,hidden,s)
                    target=teacher(env.state())[0].tanh()
                    features.append(f);props.append(p);starts.append(s);targets.append(target)
                    # DAgger: collect states visited by the student, not only by teacher.
                    _,term,trunc,_=env.step(mean.tanh())
                    if bool((term|trunc).any()):env.reset(term|trunc)
            means,_=model.sequence(torch.stack(features),torch.stack(props),initial,torch.stack(starts))
            sim_loss=(means.tanh()-torch.stack(targets)).square().mean()
            real_loss=bc_batch_loss(model,encoder,real,cfg,cfg["bc"]["batch_size"],device)
            fraction=cfg["bc"]["real_fraction"]
            loss=(1-fraction)*sim_loss+fraction*real_loss
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            log(root,dict(update=update+1,teacher_bc=float(sim_loss.detach()),real_bc=float(real_loss.detach())))
            if (update+1)%cfg["ppo"]["save_every"]==0 or update+1==start+updates:
                save_checkpoint(root/"latest.pt",model,optimizer,cfg,update+1,"distill",feature_dim=encoder.output_dim,
                    encoder_revision=cfg["encoder_revision"],teacher_sha256=digest(teacher_path),
                    bundle_sha256=digest(Path(bundle)/"checksums.json"),data_manifest_sha256=digest(Path(data_root)/"manifest.json"))
    finally:env.close()
