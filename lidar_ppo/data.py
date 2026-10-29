"""Explicit LeRobot-v2 adapters and episode-balanced, boundary-safe BC sampling."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .actions import demonstration_action, PROPRIO_DIM
from .config import digest, write_json


def letterbox(images, size):
    """NCHW uint8; preserve full ego frame, identical real/sim preprocessing."""
    h, w = images.shape[-2:]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = F.interpolate(images.float(), size=(nh, nw), mode="bilinear", align_corners=False)
    return F.pad(resized, ((size-nw)//2, size-nw-(size-nw)//2,
                          (size-nh)//2, size-nh-(size-nh)//2), value=0).round().to(torch.uint8)


def _interp(times, values, query):
    return np.stack([np.interp(query, times, values[:, j]) for j in range(values.shape[1])], -1)


def prepare(manifest_path, destination, cfg, joint_names):
    """No simulation-success filtering. Explicit episode/session splits are mandatory.

    LeRobot v2 parquet + per-episode MP4 is supported. Other schemas must first be
    converted; reject them instead of guessing ordering, units, task, or timing.
    """
    import cv2
    import pyarrow.parquet as pq
    manifest = json.loads(Path(manifest_path).read_text())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records, seen_sessions, seen_episodes = [], {}, set()
    for source in manifest["datasets"]:
        root = Path(source["root"]).expanduser().resolve()
        info = json.loads((root / "meta/info.json").read_text())
        if "chunks_size" not in info:
            raise ValueError("Convert LeRobot v3 to the documented v2/shard contract first")
        for key in ("observation.state", "action"):
            if info["features"][key]["names"][:29] != joint_names:
                raise ValueError(f"{root}: {key} ordering differs; explicit remapping required")
        if source.get("action_contract") != "g1_joint29_root6_hand_registers":
            raise ValueError("Explicit action_contract required; EE-relative data needs an adapter")
        for split in ("train", "validation"):
            for entry in source[split]:
                episode = int(entry["episode"])
                session = str(entry["session"])
                if session in seen_sessions and seen_sessions[session] != split:
                    raise ValueError(f"Session {session} crosses train/validation")
                seen_sessions[session] = split
                identity = (str(root), episode)
                if identity in seen_episodes:
                    raise ValueError(f"Duplicate episode: {identity}")
                seen_episodes.add(identity)
                if entry.get("task") != "pipette_pickup":
                    raise ValueError("Unconditioned policy accepts only reviewed pipette_pickup segments")
                path = root / info["data_path"].format(episode_chunk=episode//info["chunks_size"], episode_index=episode)
                keys = ["observation.state", "action", "action.hand_right", "observation.t_img_ms", "observation.t_action_ms"]
                table = pq.read_table(path, columns=keys)
                raw = {key: np.asarray(table[key].to_pylist(), dtype=np.float64) for key in keys}
                if any(not np.isfinite(x).all() for x in raw.values()):
                    raise ValueError(f"Nonfinite episode {episode}")
                times = raw["observation.t_img_ms"].reshape(-1)
                act_times = (raw["observation.t_action_ms"].reshape(-1) - times[0]) / 1000
                times = (times - times[0]) / 1000
                if np.any(np.diff(times) <= 0):
                    raise ValueError("Image timestamps must be strictly increasing")
                alignment = source["command_alignment"]
                if alignment == "paired_image_rows":
                    act_times = times
                elif alignment != "action_timestamp" or np.any(np.diff(act_times) <= 0):
                    raise ValueError("Invalid action clock; explicitly review paired_image_rows fallback")
                dt = 1 / cfg["policy_hz"]
                start = max(float(entry.get("start_s", 0)), times[0] + 2*dt, act_times[0] + 2*dt)
                end = min(float(entry.get("end_s", times[-1])), times[-1], act_times[-1])
                ts = np.arange(start, end, dt)
                if len(ts) < cfg["bc"]["sequence_length"]:
                    raise ValueError(f"Episode {episode} segment too short")
                measured = _interp(times, raw["observation.state"][:, :29], ts)
                velocity = (measured - _interp(times, raw["observation.state"][:, :29], ts-dt))/dt
                command = _interp(act_times, raw["action"], ts)
                previous = _interp(act_times, raw["action"], ts-dt)
                older = _interp(act_times, raw["action"], ts-2*dt)
                hand = _interp(act_times, raw["action.hand_right"], ts)
                prev_hand = _interp(act_times, raw["action.hand_right"], ts-dt)
                register_max = float(source["register_max"])
                actions, masks = demonstration_action(command, previous, hand, cfg["arm_step_rad"], register_max)
                prev_action, _ = demonstration_action(previous, older, prev_hand, cfg["arm_step_rad"], register_max)
                proprio = np.concatenate([measured, velocity*.05, previous, prev_hand/register_max, prev_action], -1).astype(np.float32)
                if proprio.shape[1] != PROPRIO_DIM:
                    raise ValueError("Expected 35-value body/root command")
                for exclusion in entry.get("exclude_bc", []):
                    rows = (ts >= exclusion["start_s"]) & (ts < exclusion["end_s"])
                    channels = exclusion.get("channels", list(range(13)))
                    masks[np.ix_(rows, channels)] = 0
                idle = (np.abs(actions[:, :7]).max(-1) < .02) & (np.abs(hand-prev_hand).max(-1)/register_max < .01)
                masks[idle] *= float(source.get("idle_weight", .2))
                key = source.get("video_key", "observation.images.rgb")
                video = root / info["video_path"].format(episode_chunk=episode//info["chunks_size"], episode_index=episode, video_key=key)
                cap = cv2.VideoCapture(str(video))
                if not cap.isOpened():
                    raise FileNotFoundError(video)
                # Dataset rows identify synchronized frames; do not assume nominal video FPS.
                frame_ids = np.searchsorted(times, ts, side="right") - 1
                images, decoded, frame = [], -1, None
                try:
                    for frame_id in frame_ids:
                        while decoded < frame_id:
                            ok, frame = cap.read(); decoded += 1
                            if not ok:
                                raise ValueError(f"Missing video frame {decoded} in {video}")
                        rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2,0,1)[None]
                        images.append(letterbox(rgb, cfg["image_size"])[0].numpy())
                finally:
                    cap.release()
                name = f"{len(records):06d}.npz"
                np.savez_compressed(destination/name, images=np.stack(images), proprio=proprio,
                                    actions=actions, masks=masks, times=ts)
                records.append(dict(file=name, split=split, session=session, root=str(root), episode=episode,
                                    parquet_sha256=digest(path), video_sha256=digest(video), frames=len(ts),
                                    excluded_action_fraction=float((masks==0).mean())))
                print(json.dumps(dict(prepared_episode=episode,split=split,frames=len(ts),
                                      excluded_action_fraction=records[-1]["excluded_action_fraction"])),flush=True)
    if not all(any(r["split"] == split for r in records) for split in ("train", "validation")):
        raise ValueError("Both training and session-held-out validation data are required")
    write_json(destination/"manifest.json", dict(version=1, input_sha256=digest(manifest_path),
        policy_hz=cfg["policy_hz"], arm_step_rad=cfg["arm_step_rad"], image_size=cfg["image_size"],
        joint_names=joint_names, records=records))


class SequenceData:
    def __init__(self, root, split, length, seed=42):
        self.root, self.length = Path(root), length
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.records = [r for r in self.manifest["records"] if r["split"] == split and r["frames"] >= length]
        if not self.records:
            raise ValueError(f"No usable {split} sequences")
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def sample(self, batch, device):
        result = []
        for _ in range(batch):
            r = self.records[int(self.rng.integers(len(self.records)))]
            path = r["file"]
            if path not in self.cache:
                with np.load(self.root/path, allow_pickle=False) as z:
                    visual="features" if "features" in z.files else "images"
                    self.cache[path] = {k:z[k] for k in (visual, "proprio", "actions", "masks")}
                # Bound decoded RAM; each shard is an episode, not the entire dataset.
                if len(self.cache) > 8:
                    self.cache.pop(next(iter(self.cache)))
            arrays = self.cache[path]
            start = int(self.rng.integers(r["frames"] - self.length + 1))
            result.append({k:v[start:start+self.length] for k,v in arrays.items()})
        return {k:torch.from_numpy(np.stack([r[k] for r in result], 1)).to(device) for k in result[0]}


def cache_features(source,destination,cfg,device):
    """Avoid repeated DINO inference/MP4-sized image decompression for frozen real BC."""
    from .models import VisionEncoder
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    encoder=VisionEncoder(cfg["encoder"],cfg["encoder_revision"]).to(device).eval()
    manifest=json.loads((source/"manifest.json").read_text())
    if "feature_encoder" in manifest:raise ValueError("Dataset already contains encoder features")
    for record in manifest["records"]:
        with np.load(source/record["file"],allow_pickle=False) as z:
            images=torch.from_numpy(z["images"])
            features=[]
            for chunk in images.split(cfg["encoder_batch"]):
                features.append(encoder(chunk.to(device)).cpu().to(torch.float16).numpy())
            np.savez_compressed(destination/record["file"],features=np.concatenate(features),
                                proprio=z["proprio"],actions=z["actions"],masks=z["masks"])
        print(json.dumps(dict(cached_episode=record["episode"],frames=record["frames"])),flush=True)
    manifest.update(feature_encoder=cfg["encoder"],encoder_revision=getattr(encoder.backbone.config,"_commit_hash",None) or cfg["encoder_revision"],
                    source_manifest_sha256=digest(source/"manifest.json"))
    write_json(destination/"manifest.json",manifest)
