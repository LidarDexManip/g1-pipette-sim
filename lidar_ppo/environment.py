"""Contact-qualified pickup task. No grasp weld, pose forcing, or demo action clock."""
import json
from pathlib import Path

import numpy as np
import torch

from .actions import command_action, hand_joint_targets
from .config import digest, write_json
from .controller import TwistController
from .data import letterbox
from .physics import CPUPhysics, WarpPhysics


def load_bundle(path):
    path = Path(path)
    checksums = json.loads((path/"checksums.json").read_text())
    for name, expected in checksums.items():
        if digest(path/name) != expected:
            raise ValueError(f"Bundle changed: {name}; re-export after changing scene/controller")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path/"scene.xml"))
    contract = json.loads((path/"contract.json").read_text())
    return model, contract


class PickupEnv:
    def __init__(self,bundle,cfg,nworld,backend="warp",device="cuda:0",evaluation=False,diagnostic=False):
        self.bundle, self.cfg, self.nworld = Path(bundle), cfg, nworld
        self.device = torch.device(device)
        self.model,self.contract = load_bundle(bundle)
        m = self.model
        if not np.isclose(m.opt.timestep,cfg["physics_dt"]):
            raise ValueError("Scene timestep differs from training contract")
        if self.contract["policy_hz"] != cfg["policy_hz"] or not self.contract["fixed_pelvis"]:
            raise ValueError("Only validated fixed-pelvis 25 Hz action contract is supported")
        self.physics = (WarpPhysics if backend == "warp" else CPUPhysics)(m,nworld,str(device),**cfg["warp"])
        control_cfg = json.loads((self.bundle/"controller.json").read_text())
        self.controller = TwistController(self.bundle/"twist2.pt",control_cfg,nworld,self.device)
        names = self.contract["joint_names"]
        self.qa = torch.tensor([int(m.joint(n).qposadr[0]) for n in names],device=self.device)
        self.va = torch.tensor([int(m.joint(n).dofadr[0]) for n in names],device=self.device)
        self.body_act = torch.tensor([m.actuator("servo_"+n).id for n in names],device=self.device)
        self.hand_act = torch.tensor([m.actuator("servo_R_"+n).id for n in
            ["pinky_proximal_joint","ring_proximal_joint","middle_proximal_joint","index_proximal_joint",
             "thumb_proximal_yaw_joint","thumb_proximal_pitch_joint"]],device=self.device)
        self.arm_limits = torch.tensor(np.stack([m.joint(n).range for n in names[22:29]]),device=self.device,dtype=torch.float32)
        self.pipette, self.pelvis, self.wrist = [m.body(n).id for n in ("env_01_pipette","pelvis","right_wrist_yaw_link")]
        with np.load(self.bundle/"resets.npz",allow_pickle=False) as z:
            self.bank = {k:torch.tensor(z[k],device=self.device) for k in z.files}
        for k in self.bank:
            if self.bank[k].is_floating_point():self.bank[k]=self.bank[k].float()
        review=json.loads((self.bundle/"review.json").read_text())
        accepted=review["accepted_reset_indices"]
        if diagnostic:
            accepted=list(range(len(self.bank["qpos"])))
        if not accepted:
            raise ValueError("No qualified resets. Run audit-bundle and correct invalid scene/start states.")
        ids=torch.tensor(accepted,device=self.device,dtype=torch.long)
        heldout=torch.tensor(review["validation_episode_ids"],device=self.device,dtype=torch.long)
        if not diagnostic and not len(heldout):
            raise ValueError("Reserve validation_episode_ids in review.json before training")
        is_eval=torch.isin(self.bank["episode"][ids],heldout)
        self.pool=ids[is_eval if evaluation else ~is_eval]
        if diagnostic:self.pool=ids
        self.natural_pool=self.pool[self.bank["natural"][self.pool].bool()]
        if not len(self.pool) or not len(self.natural_pool):
            raise ValueError("Need qualified ordinary-start states in each split; demo-only resets are insufficient")
        self.evaluation=evaluation
        self.reference=torch.zeros(nworld,35,device=self.device)
        self.hand=torch.zeros(nworld,6,device=self.device)
        self.last_action=torch.zeros(nworld,13,device=self.device)
        self.elapsed=torch.zeros(nworld,device=self.device)
        self.hold=torch.zeros(nworld,device=self.device)
        self.start_height=torch.zeros(nworld,device=self.device)
        self.dt=1/cfg["policy_hz"]
        self.update=0
        self.episode_start=torch.ones(nworld,device=self.device,dtype=torch.bool)
        self.reset(self.episode_start)
        self.state_dim=self.state().shape[-1]

    def reset(self,mask):
        cfg=self.cfg["reset"]
        fraction=cfg["demo_fraction_start"]+(cfg["demo_fraction_end"]-cfg["demo_fraction_start"])*min(1,self.update/cfg["anneal_updates"])
        use_demo=(torch.rand(self.nworld,device=self.device)<fraction) & (not self.evaluation)
        demo=self.pool[torch.randint(len(self.pool),(self.nworld,),device=self.device)]
        natural=self.natural_pool[torch.randint(len(self.natural_pool),(self.nworld,),device=self.device)]
        ids=torch.where(use_demo,demo,natural)
        values={k:self.bank[key][ids] for k,key in [("qpos","qpos"),("qvel","qvel"),("ctrl","ctrl"),("act","act"),("qacc_warmstart","warmstart")]}
        factors=[]
        for key in ("friction_scale","servo_scale"):
            lo,hi=self.cfg["randomization"].get(key,[1.,1.])
            factors.append(torch.ones(self.nworld,device=self.device) if self.evaluation else
                           lo+(hi-lo)*torch.rand(self.nworld,device=self.device))
        self.physics.randomize(mask,*factors)
        self.physics.reset(mask,values)
        for target,key in [(self.reference,"reference"),(self.hand,"hand"),(self.start_height,"start_height"),
                           (self.controller.history,"history"),(self.controller.previous,"previous")]:
            target[mask]=self.bank[key][ids][mask]
        if not self.evaluation:
            noise=self.cfg["randomization"]["arm_target_noise_rad"]
            self.reference[mask,22:29]+=torch.randn_like(self.reference[mask,22:29])*noise
            self.reference[:,22:29]=torch.maximum(torch.minimum(self.reference[:,22:29],self.arm_limits[:,1]),self.arm_limits[:,0])
        self.last_action[mask]=0
        self.elapsed[mask]=0;self.hold[mask]=0
        self.physics.forward()
        self.potential=self.phi()
        self.episode_start=mask.clone()

    def proprio(self):
        q=self.physics.get("qpos")[:,self.qa]
        dq=self.physics.get("qvel")[:,self.va]*.05
        noise=0 if self.evaluation else self.cfg["randomization"]["observation_noise_std"]
        return torch.cat([q+torch.randn_like(q)*noise,dq+torch.randn_like(dq)*noise,
                          self.reference,self.hand,self.last_action],-1)

    def state(self):
        return torch.cat([self.physics.get("qpos"),self.physics.get("qvel")*.05,
            self.physics.get("ctrl"),self.reference,self.hand,self.last_action,
            self.controller.history.flatten(1),self.controller.previous,
            self.start_height[:,None],self.hold[:,None],self.elapsed[:,None]/self.cfg["episode_seconds"]],-1).clone()

    def phi(self):
        positions=self.physics.get("xipos")
        distance=(positions[:,self.pipette]-positions[:,self.wrist]).norm(dim=-1)
        lift=(positions[:,self.pipette,2]-self.start_height).clamp(0,.05)/.05
        return .5*torch.exp(-distance/.08)+.5*lift

    def images(self):
        image=self.physics.render(self.cfg["camera"],self.cfg["camera_width"],self.cfg["camera_height"])
        image=letterbox(image,self.cfg["image_size"])
        if not self.evaluation:
            amount=self.cfg["randomization"]["image_brightness"]
            gain=1+(torch.rand(self.nworld,1,1,1,device=self.device)*2-1)*amount
            image=(image.float()*gain).clamp(0,255).to(torch.uint8)
        return image

    @torch.no_grad()
    def step(self,action):
        reward_cfg=self.cfg["reward"]
        previous_action=self.last_action.clone()
        self.reference,self.hand=command_action(action,self.reference,self.hand,self.arm_limits,
                                               self.cfg["arm_step_rad"],self.cfg["hand_step_fraction"])
        ctrl=self.physics.get("ctrl").clone()
        peaks=torch.zeros_like(self.physics.peak)
        qualified=torch.ones(self.nworld,device=self.device,dtype=torch.bool)
        for _ in range(2):
            ctrl[:,self.body_act]=self.controller.step(self.reference,self.physics.get("qpos")[:,self.qa],
                self.physics.get("qvel")[:,self.va],self.physics.get("xquat")[:,self.pelvis])
            ctrl[:,self.hand_act]=hand_joint_targets(self.hand,self.contract["hand_mapping"])
            self.physics.step(ctrl,20)
            peaks=torch.maximum(peaks,self.physics.peak)
            positions=self.physics.get("xipos")
            height=positions[:,self.pipette,2]-self.start_height
            metrics=self.physics.metrics
            speed=self.physics.get("cvel")[:,self.pipette,3:].norm(dim=-1)
            qualified &= ((height>=reward_cfg["lift_m"]) & (metrics[:,1]>reward_cfg["contact_force_min_n"])
                & (metrics[:,2]<=reward_cfg["contact_force_min_n"]) & (metrics[:,3]<=reward_cfg["contact_force_min_n"])
                & (speed<=reward_cfg["max_object_speed_m_s"]))
        self.hold=torch.where(qualified,self.hold+self.dt,torch.zeros_like(self.hold))
        self.elapsed+=self.dt;self.last_action=action.clone()
        invalid=(peaks[:,0]>reward_cfg["max_penetration_m"]) | (peaks[:,1]>reward_cfg["max_hand_force_n"])
        dropped=height < -reward_cfg["drop_m"]
        finite=torch.isfinite(self.physics.get("qpos")).all(-1)&torch.isfinite(self.physics.get("qvel")).all(-1)
        if not bool(finite.all()):raise RuntimeError("Nonfinite physics: stop training instead of learning from corrupt state")
        failure=invalid|dropped
        success=(self.hold>=reward_cfg["hold_seconds"]-1e-6)&~failure
        terminated=success|failure
        truncated=(self.elapsed>=self.cfg["episode_seconds"]-1e-6)&~terminated
        potential=self.phi()
        shaped=self.cfg["ppo"]["gamma"]*torch.where(terminated,torch.zeros_like(potential),potential)-self.potential
        reward=(reward_cfg["success"]*success+reward_cfg["failure"]*failure+shaped
            -reward_cfg["time_per_second"]*self.dt
            -reward_cfg["action_change"]*(action-previous_action).square().mean(-1))
        self.potential=potential
        info=dict(success=success,failure=failure,elapsed=self.elapsed.clone(),peak_penetration=peaks[:,0],
                  peak_hand_force=peaks[:,1],height=height)
        # Caller obtains terminal observations BEFORE calling reset; no autoreset leakage.
        self.episode_start=torch.zeros_like(terminated)
        return reward,terminated,truncated,info

    def close(self):self.physics.close()


def audit_bundle(bundle,cfg):
    """Check every stored state on CPU, retaining per-state evidence and split policy."""
    model,contract=load_bundle(bundle)
    physics=CPUPhysics(model,1)
    path=Path(bundle)
    records=[]
    with np.load(path/"resets.npz",allow_pickle=False) as bank:
        for i in range(len(bank["qpos"])):
            values={k:torch.tensor(bank[v][i:i+1],dtype=torch.float32) for k,v in
                    [("qpos","qpos"),("qvel","qvel"),("ctrl","ctrl"),("act","act"),("qacc_warmstart","warmstart")]}
            physics.reset(torch.tensor([True]),values);physics.forward()
            metrics=physics.metrics[0]
            valid=bool(metrics[0]<=cfg["reward"]["max_penetration_m"] and metrics[1]<=cfg["reward"]["max_hand_force_n"])
            records.append(dict(index=i,episode=int(bank["episode"][i]),time=float(bank["time"][i]),
                natural=bool(bank["natural"][i]),valid=valid,penetration_m=float(metrics[0]),hand_force_n=float(metrics[1])))
    report=dict(records=records,accepted_reset_indices=[r["index"] for r in records if r["valid"]],
                valid_natural_starts=[r for r in records if r["valid"] and r["natural"]])
    write_json(path/"reset-audit.json",report)
    review=json.loads((path/"review.json").read_text())
    review["accepted_reset_indices"]=report["accepted_reset_indices"]
    write_json(path/"review.json",review)
    return {k:v for k,v in report.items() if k!="records"}
