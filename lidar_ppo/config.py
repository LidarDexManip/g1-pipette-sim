"""Explicit, versioned training contracts. All frequencies are simulated time."""
import hashlib
import json
from pathlib import Path


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("version") != 1:
        raise ValueError("Expected configuration version 1")
    if cfg["policy_hz"] != 25 or cfg["controller_hz"] != 50:
        raise ValueError("This action/data contract is 25 Hz policy / 50 Hz TWIST2")
    if cfg["physics_dt"] != .001:
        raise ValueError("Validate contact parity before changing the 1 ms timestep")
    if cfg["ppo"]["rollout_steps"] % cfg["ppo"]["sequence_length"]:
        raise ValueError("Rollout length must be divisible by sequence length")
    return cfg


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
