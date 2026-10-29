"""Export a portable scene/controller/reset bundle from the existing replay checkout."""
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .config import digest, write_json


class BatchReshape(torch.nn.Module):
    """Keep batch dynamic when the ONNX converter constructs static shape tensors."""
    def __init__(self,tail,numerator=1,denominator=1):
        super().__init__();self.tail=tuple(tail);self.numerator=numerator;self.denominator=denominator

    def forward(self,value,shape):
        return value.reshape(value.shape[0]*self.numerator//self.denominator,*self.tail)


def convert_controller(onnx_path, target):
    import onnx
    import onnxruntime as ort
    from onnx2torch import convert
    model = convert(onnx.load(str(onnx_path))).eval()
    example = torch.randn(2, 1432)
    # This TWIST2 graph reshapes only within each independent batch row. Verify
    # that assumption, then replace converter reshapes that otherwise fix batch=2.
    shapes={};hooks=[]
    for name,module in model.named_modules():
        if type(module).__name__ == "OnnxReshape":
            def record(mod,args,result,name=name):
                from fractions import Fraction
                shape=args[1].tolist()
                factor=Fraction(result.shape[0],args[0].shape[0])
                shapes[name]=(shape[1:],factor.numerator,factor.denominator)
            hooks.append(module.register_forward_hook(record))
    with torch.no_grad():model(example)
    for hook in hooks:hook.remove()
    for name,shape in shapes.items():
        parent,_,child=name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model,child,BatchReshape(*shape))
    converted = torch.jit.trace(model, example, check_trace=False)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    errors = []
    for batch in (1, 2, 17):
        x = np.random.default_rng(batch).normal(0, .2, (batch,1432)).astype(np.float32)
        # Some source ONNX files fix batch=1. Compare each row independently.
        expected = np.concatenate([session.run(None, {session.get_inputs()[0].name:row[None]})[0] for row in x])
        with torch.no_grad():
            actual = converted(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
        errors.append(float(np.max(np.abs(expected-actual))))
    converted.save(str(target))
    return errors


def export_bundle(source_repo, dataset, episodes, controller_root, output, cfg):
    """Archive replay state at 25 Hz. Training rejects unqualified start states.

    No silent scene correction or claim of physical validation is made here.
    The explicit review JSON is filled after inspecting replay/contact/camera tests.
    """
    import mujoco
    source_repo = Path(source_repo).resolve()
    sys.path.insert(0, str(source_repo))
    from g1_pipette_sim.dataset import Episode
    from g1_pipette_sim.environments.physical import build_physical_scene
    from g1_pipette_sim.environments.twist2 import LidarControllerReplay, apply_body_gains
    from g1_pipette_sim.scene import load_config
    import yaml
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    controller_root = Path(controller_root).resolve()
    control_cfg = yaml.safe_load((controller_root/"deploy_real/robot_control/configs/g1.yaml").read_text())
    checkpoint = controller_root/"assets/ckpts/twist2_1017_20k.onnx"
    xml = ET.fromstring(build_physical_scene("lidar"))
    # Build first with the original paths, then serialize gain changes into MJCF.
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    names = Episode(dataset, episodes[0]).names
    apply_body_gains(model, names, control_cfg)
    mujoco.mj_saveLastXML(str(output/"scene.xml"), model)
    xml = ET.parse(output/"scene.xml").getroot()
    compiler=xml.find("compiler")
    assets = output/"assets"; assets.mkdir()
    for element in xml.iter():
        if "file" not in element.attrib:
            continue
        src = Path(element.get("file"))
        if not src.is_absolute():
            directory=compiler.get("texturedir" if element.tag=="texture" else "meshdir","")
            src=Path(directory)/src
        if not src.is_absolute():raise ValueError(f"Unresolved asset path: {src}")
        name = digest(src)[:16] + src.suffix
        shutil.copy2(src, assets/name)
        element.set("file", "assets/"+name)
    for key in ("meshdir","texturedir","assetdir"):
        compiler.attrib.pop(key,None)
    ET.ElementTree(xml).write(output/"scene.xml", encoding="unicode")
    errors = convert_controller(checkpoint, output/"twist2.pt")
    snapshots = []
    for episode_id in episodes:
        episode = Episode(dataset, episode_id)
        m = mujoco.MjModel.from_xml_path(str(output/"scene.xml")); d = mujoco.MjData(m)
        replay = LidarControllerReplay(m,d,episode,source="twist2",body_gains="twist2",checkpoint=checkpoint)
        for t in np.arange(0, episode.duration, 1/cfg["policy_hz"]):
            replay.advance(float(t))
            # State is sampled BEFORE the next controller tick. Save all control memory.
            snapshots.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(),
                act=d.act.copy(), warmstart=d.qacc_warmstart.copy(),
                history=np.asarray(replay.core.history).copy(), previous=replay.core.previous_action.copy(),
                reference=episode.interpolate("action", max(0.,t-1/cfg["policy_hz"])),
                hand=episode.interpolate("action.hand_right", max(0.,t-1/cfg["policy_hz"]))/load_config()["hand_mapping"]["register_max"],
                episode=episode_id, time=t, natural=t == 0,
                start_height=replay.start_com["env_01_pipette"][2]))
    np.savez_compressed(output/"resets.npz", **{k:np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
    write_json(output/"controller.json", control_cfg)
    write_json(output/"contract.json", dict(version=1, joint_names=names,
        hand_mapping=load_config()["hand_mapping"], episodes=episodes,
        policy_hz=cfg["policy_hz"], physics_dt=cfg["physics_dt"],
        fixed_pelvis=True, original_onnx_sha256=digest(checkpoint), controller_parity_max_abs=errors,
        source_repo=str(source_repo), dataset=str(Path(dataset).resolve())))
    write_json(output/"review.json", dict(physics_reviewed=False, camera_reviewed=False,
        accepted_reset_indices=[], validation_episode_ids=cfg["reset"].get("validation_episode_ids",[]),
        note="Review contact audit, actual camera registration, and measured platform/controller setup. "
             "Set accepted_reset_indices to physically valid snapshots, including ordinary starts. "
             "Reserve whole episodes for validation. Flags are human assertions, not automatic validation."))
    files = {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file() and p.name != "review.json"}
    write_json(output/"checksums.json", files)
    return dict(output=str(output), snapshots=len(snapshots), controller_parity=errors)
