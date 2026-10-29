import torch


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Timeouts bootstrap terminal observations, but neither terminal kind leaks GAE."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        advantage[t] = carry
    return advantage, advantage + values
