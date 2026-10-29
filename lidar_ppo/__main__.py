import argparse
import json
from pathlib import Path

from .config import read_config


def main():
    parser=argparse.ArgumentParser(description="LIDAR pickup: BC, PPO teacher, DAgger, visual PPO")
    parser.add_argument("--config",default="config/lidar_ppo/b200.json")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("export-bundle")
    p.add_argument("--source-repo",required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--controller-root",required=True);p.add_argument("--episodes",nargs="+",type=int,required=True)
    p.add_argument("--output",required=True)
    p=sub.add_parser("audit-bundle");p.add_argument("--bundle",required=True)
    p=sub.add_parser("prepare-data");p.add_argument("--manifest",required=True);p.add_argument("--bundle",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("cache-features");p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p=sub.add_parser("preflight");p.add_argument("--bundle");p.add_argument("--device",default="cuda:0");p.add_argument("--vision",action="store_true")
    for name in ("bc","teacher","distill","visual-ppo","evaluate","benchmark"):
        p=sub.add_parser(name);p.add_argument("--device",default="cuda:0")
        if name!="bc":
            p.add_argument("--bundle",required=True);p.add_argument("--backend",choices=["cpu","warp"],default="warp")
            p.add_argument("--num-envs",type=int,default=None)
        if name in ("bc","distill","visual-ppo"):p.add_argument("--data",required=True)
        if name in ("distill","visual-ppo"):p.add_argument("--student",required=True)
        if name=="distill":p.add_argument("--teacher",required=True)
        if name=="evaluate":
            p.add_argument("--checkpoint",required=True);p.add_argument("--episodes",type=int,default=100)
        elif name=="benchmark":p.add_argument("--steps",type=int,default=100)
        else:
            p.add_argument("--updates",type=int,default=1000);p.add_argument("--resume")
        if name!="benchmark":p.add_argument("--output",required=True)
    p=sub.add_parser("export-policy");p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    args=parser.parse_args();cfg=read_config(args.config)
    if args.command=="export-bundle":
        from .bundle import export_bundle
        result=export_bundle(args.source_repo,args.dataset,args.episodes,args.controller_root,args.output,cfg)
    elif args.command=="audit-bundle":
        from .environment import audit_bundle
        result=audit_bundle(args.bundle,cfg)
    elif args.command=="prepare-data":
        from .data import prepare
        contract=json.loads((Path(args.bundle)/"contract.json").read_text())
        result=prepare(args.manifest,args.output,cfg,contract["joint_names"])
    elif args.command=="preflight":
        import importlib.metadata
        import torch
        result={p:importlib.metadata.version(p) for p in ["torch","mujoco","mujoco-warp","warp-lang","transformers"]}
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
            result.update(gpu=torch.cuda.get_device_name(args.device),capability=torch.cuda.get_device_capability(args.device))
            # Execute kernels, not just device discovery. B200 requires sm_100-capable wheels.
            a=torch.randn(64,64,device=args.device);result["cuda_matmul_finite"]=bool(torch.isfinite(a@a).all())
            import warp as wp
            wp.init();array=wp.zeros(16,device=args.device);array.fill_(1);wp.synchronize()
            result["warp_kernel_sum"]=float(array.numpy().sum())
        if args.bundle:
            from .environment import load_bundle
            model,_=load_bundle(args.bundle)
            result.update(nq=model.nq,nv=model.nv,ngeom=model.ngeom,scene_timestep=model.opt.timestep)
        if args.vision:
            from .models import VisionEncoder
            encoder=VisionEncoder(cfg["encoder"],cfg["encoder_revision"]).to(args.device).eval()
            feature=encoder(torch.zeros(1,3,cfg["image_size"],cfg["image_size"],dtype=torch.uint8,device=args.device))
            result.update(encoder_feature_shape=list(feature.shape),encoder_features_finite=bool(torch.isfinite(feature).all()),
                          encoder_revision=getattr(encoder.backbone.config,"_commit_hash",None))
    elif args.command=="cache-features":
        from .data import cache_features
        result=cache_features(args.data,args.output,cfg,args.device)
    elif args.command=="export-policy":
        from .policy import export_policy
        result=export_policy(args.checkpoint,args.output)
    else:
        from .train import pretrain_bc,train_ppo,distill
        n=getattr(args,"num_envs",None) or (cfg["ppo"]["num_envs"] if args.command=="teacher" else 128)
        if args.command=="bc":result=pretrain_bc(cfg,args.data,args.output,args.updates,args.device,args.resume)
        elif args.command in ("teacher","visual-ppo"):
            result=train_ppo(cfg,args.bundle,args.output,args.updates,args.backend,args.device,n,args.resume,
                             getattr(args,"student",None),getattr(args,"data",None))
        elif args.command=="distill":
            result=distill(cfg,args.bundle,args.data,args.teacher,args.student,args.output,args.updates,args.backend,args.device,n,args.resume)
        elif args.command=="evaluate":
            from .evaluate import evaluate
            result=evaluate(cfg,args.bundle,args.checkpoint,args.output,args.episodes,args.backend,args.device,n)
        else:
            from .evaluate import benchmark
            result=benchmark(cfg,args.bundle,args.backend,args.device,n,args.steps)
    if result is not None:print(json.dumps(result,indent=2))


if __name__=="__main__":main()
