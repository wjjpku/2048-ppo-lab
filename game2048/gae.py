import torch
from .model import require_finite


@torch.no_grad()
def compute_gae(rewards, values, bootstrap_values, terminated, truncated, gamma=.999, gae_lambda=.95):
    if not (rewards.shape == values.shape == bootstrap_values.shape == terminated.shape == truncated.shape):
        raise ValueError('GAE tensors must have identical [T,N] shapes')
    if rewards.ndim != 2 or terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError('GAE expects [T,N] and boolean boundaries')
    for name, x in [('rewards', rewards), ('values', values), ('bootstrap_values', bootstrap_values)]:
        require_finite(x, name)
    bootstrap_mask = (~terminated).to(values.dtype)
    trace_mask = (~(terminated | truncated)).to(values.dtype)
    delta = rewards + gamma * bootstrap_mask * bootstrap_values - values
    advantages = torch.empty_like(values)
    carry = torch.zeros_like(values[0])
    for t in range(len(values) - 1, -1, -1):
        carry = delta[t] + gamma * gae_lambda * trace_mask[t] * carry
        advantages[t] = carry
    returns = advantages + values
    require_finite(returns, 'returns')
    return advantages.detach(), returns.detach()


def normalize_advantages(advantages):
    """Exactly one whole-rollout normalization; value targets stay unnormalized."""
    require_finite(advantages, 'advantages')
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    normalized = (advantages - mean) / (std + 1e-8)
    return normalized.detach(), {'advantage_mean': float(mean), 'advantage_std': float(std),
                                'advantage_near_zero_std': bool(std < 1e-8)}
