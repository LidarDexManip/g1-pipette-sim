import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from .actions import ACTION_DIM, PROPRIO_DIM


def mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Teacher(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.actor = mlp([state_dim, 256, 256, 128, ACTION_DIM])
        self.critic = mlp([state_dim, 256, 256, 128, 1])
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.5))

    def forward(self, state):
        return self.actor(state), self.critic(state).squeeze(-1)


def sample_action(mean, log_std, latent=None):
    """Store pre-tanh actions; PPO ratios include the same invertible transform."""
    dist = Normal(mean, log_std.clamp(-5, 1).exp())
    latent = dist.sample() if latent is None else latent
    action = latent.tanh()
    log_prob = (dist.log_prob(latent) - 2 * (torch.log(torch.tensor(2., device=mean.device))
                - latent - F.softplus(-2 * latent))).sum(-1)
    # Base Gaussian entropy is an exploration surrogate, not squashed entropy.
    return action, latent, log_prob, dist.entropy().sum(-1)


class VisionEncoder(nn.Module):
    """Frozen DINOv2 patch features, pooled to a spatial 2x2 grid."""
    def __init__(self, model_id, revision="main"):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.backbone.requires_grad_(False)
        self.output_dim = self.backbone.config.hidden_size * 4
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        self.backbone.eval()
        image = (images.float() / 255 - self.mean) / self.std
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16,
                            enabled=image.is_cuda):
            patches = self.backbone(pixel_values=image).last_hidden_state[:, 1:]
        side = int(patches.shape[1] ** .5)
        if side * side != patches.shape[1]:
            raise ValueError("Expected square DINOv2 patch grid, without register tokens")
        grid = patches.transpose(1, 2).reshape(len(image), -1, side, side).float()
        return F.adaptive_avg_pool2d(grid, 2).flatten(1)


class Student(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, state_dim=None):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 256), nn.ELU())
        self.proprio = mlp([PROPRIO_DIM, 128, 128])
        self.rnn = nn.GRUCell(384, hidden_dim)
        self.head = nn.Linear(hidden_dim, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.5))
        self.hidden_dim = hidden_dim
        self.critic = mlp([state_dim, 256, 256, 1]) if state_dim else None

    def forward(self, features, proprio, hidden, episode_start=None):
        if episode_start is not None:
            hidden = hidden * (~episode_start).float()[:, None]
        hidden = self.rnn(torch.cat([self.visual(features), self.proprio(proprio)], -1), hidden)
        return self.head(hidden), hidden

    def sequence(self, features, proprio, hidden, starts):
        means = []
        for t in range(features.shape[0]):
            mean, hidden = self(features[t], proprio[t], hidden, starts[t])
            means.append(mean)
        return torch.stack(means), hidden
