"""FP32 network; masked categorical arithmetic excludes invalid -inf entries."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def require_finite(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f'{name} contains NaN/Inf')


def encode_boards(boards):
    if boards.ndim < 2 or tuple(boards.shape[-2:]) != (4, 4):
        raise ValueError('boards must have shape [...,4,4]')
    if boards.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError('board exponents must be integers')
    if torch.any(boards < 0) or torch.any(boards > 20):
        raise ValueError('board exponent outside 0..20; no clipping permitted')
    return F.one_hot(boards.to(torch.int64), num_classes=21).flatten(-3).to(torch.float32)


class ActorCritic(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('hidden_size must be a positive integer')
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(nn.Linear(336, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.policy = nn.Linear(hidden_size, 4)
        self.value = nn.Linear(hidden_size, 1)
        for layer in (self.trunk[0], self.trunk[2]):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy.weight, .01)
        nn.init.orthogonal_(self.value.weight, 1)
        nn.init.zeros_(self.policy.bias)
        nn.init.zeros_(self.value.bias)
        assert sum(p.numel() for p in self.parameters()) == hidden_size**2 + 343*hidden_size + 5

    def forward(self, onehot):
        h = self.trunk(onehot)
        return self.policy(h), self.value(h).squeeze(-1)


def masked_log_probs(logits, masks):
    if logits.shape != masks.shape or logits.shape[-1] != 4 or masks.dtype != torch.bool:
        raise ValueError('expected equally shaped [...,4] logits and boolean masks')
    require_finite(logits, 'raw logits')
    if not masks.any(-1).all():
        raise ValueError('all-illegal rows must follow terminal path before softmax')
    result = F.log_softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)
    require_finite(result[masks], 'legal action log probabilities')
    return result


def categorical_entropy(log_probs, masks):
    # Replace invalid entries *before* multiplication; 0 * -inf is NaN.
    safe = log_probs.masked_fill(~masks, 0.)
    return -(log_probs.exp() * safe).sum(-1)


def exact_kl(old_log_probs, new_log_probs, masks):
    old_safe = old_log_probs.masked_fill(~masks, 0.)
    new_safe = new_log_probs.masked_fill(~masks, 0.)
    return (old_log_probs.exp() * (old_safe - new_safe)).sum(-1)


def sample_actions(log_probs, generator):
    probs = log_probs.exp()
    require_finite(probs, 'action probabilities')
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
