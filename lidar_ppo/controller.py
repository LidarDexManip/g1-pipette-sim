"""Batched implementation of the recorded TWIST2 observation contract."""
import torch


class TwistController:
    def __init__(self, model_path, config, worlds, device):
        self.policy = torch.jit.load(str(model_path), map_location=device).eval()
        self.default = torch.tensor(config["default_angles"], device=device, dtype=torch.float32)
        self.scale = float(config["action_scale"])
        self.history = torch.zeros(worlds, 10, 127, device=device)
        self.previous = torch.zeros(worlds, 29, device=device)

    @torch.no_grad()
    def step(self, reference, q, dq, quat_wxyz):
        w, x, y, z = quat_wxyz.unbind(-1)
        roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = torch.asin((2*(w*y-z*x)).clamp(-1,1))
        velocity = dq.clone()
        velocity[:, [4,5,10,11]] = 0
        # This scene has a fixed pelvis: local angular velocity is exactly zero.
        angular = torch.zeros_like(q[:, :3])
        proprio = torch.cat([angular, roll[:, None], pitch[:, None], q-self.default,
                             velocity*.05, self.previous], -1)
        root_first = torch.cat([reference[:,29:35], reference[:,:29]], -1)
        current = torch.cat([root_first, proprio], -1)
        obs = torch.cat([current, self.history.flatten(1), root_first], -1)
        raw = self.policy(obs)
        self.history.copy_(torch.cat([self.history[:, 1:], current[:, None]], 1))
        self.previous.copy_(raw)  # UNCLIPPED output is fed back, matching hardware.
        return self.default + raw.clamp(-10,10)*self.scale
