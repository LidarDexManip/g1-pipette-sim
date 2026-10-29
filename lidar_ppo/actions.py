"""One contract for recorded actions, simulator commands, and deployment.

Arm actions increment the PREVIOUS COMMANDED high-level reference, never measured
qpos. Hands are six absolute register fractions mapped to [-1, 1], then slew limited.
"""
import numpy as np
import torch

ARM = slice(22, 29)
ACTION_DIM = 13
# Measured body q, body dq, commanded reference35, commanded hand6, previous action13.
PROPRIO_DIM = 29 + 29 + 35 + 6 + 13


def demonstration_action(command, previous_command, hand, arm_step, register_max=1000.):
    action = np.concatenate([(command[..., ARM] - previous_command[..., ARM]) / arm_step,
                             2 * hand / register_max - 1], axis=-1).astype(np.float32)
    # Never silently clip a demonstration target outside the execution envelope.
    valid = np.isfinite(action) & (np.abs(action) <= 1.00001)
    return np.clip(action, -1, 1), valid.astype(np.float32)


def command_action(action, reference, hand, limits, arm_step, hand_step):
    action = action.clamp(-1, 1)
    updated = reference.clone()
    updated[:, ARM] = torch.maximum(torch.minimum(reference[:, ARM] + arm_step * action[:, :7],
                                                  limits[:, 1]), limits[:, 0])
    desired_hand = (action[:, 7:] + 1) * .5
    updated_hand = hand + (desired_hand - hand).clamp(-hand_step, hand_step)
    return updated, updated_hand


def masked_bc(prediction, target, mask):
    # Each action channel can be excluded independently (e.g. an unwanted opening).
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)


def hand_joint_targets(hand, mapping):
    values = (1 - hand[:, :4]) * mapping["finger_closed_rad"]
    bend = (1 - hand[:, 4:5]) * mapping["thumb_bend_closed_rad"]
    yaw = hand[:, 5:6] * mapping["thumb_rotation_max_rad"]
    return torch.cat([values, yaw, bend], dim=-1)
