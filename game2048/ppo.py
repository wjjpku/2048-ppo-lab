import time
import torch
from .model import masked_log_probs, categorical_entropy, exact_kl, require_finite
from .rollout import synchronize


def ppo_losses(new_logprob, old_logprob, advantages, value, returns, entropy, config):
    ratio = (new_logprob - old_logprob).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1-config.clip_epsilon, 1+config.clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = ((value - returns) ** 2).mean()  # no extra half, no value clipping
    entropy_mean = entropy.mean()
    total = policy_loss + config.value_coefficient*value_loss - config.entropy_coefficient*entropy_mean
    require_finite(total, 'PPO loss')
    return total, {'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy_mean,
                   'clip_fraction': ((ratio-1).abs()>config.clip_epsilon).float().mean()}


def make_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(.9,.999),
                            eps=config.adam_eps, weight_decay=0)


@torch.no_grad()
def rollout_kl(model, data, minibatch_size):
    total = torch.zeros((), device=data['obs_onehot'].device)
    count = len(data['action'])
    for start in range(0, count, minibatch_size):
        sl = slice(start, start+minibatch_size)
        logits, _ = model(data['obs_onehot'][sl])
        new = masked_log_probs(logits, data['legal_mask'][sl])
        total += exact_kl(data['old_log_probs'][sl], new, data['legal_mask'][sl]).sum()
    require_finite(total, 'exact KL')
    return float(total / count)


def ppo_update(model, optimizer, rollout, config, shuffle_rng, budget=None):
    d = rollout.flatten()
    if any(v.requires_grad for v in d.values()):
        raise ValueError('rollout targets must be detached')
    model.train()
    B = len(d['action'])
    device = d['obs_onehot'].device
    sums = {k: 0. for k in ['policy_loss','value_loss','entropy','normalized_entropy','clip_fraction',
                            'grad_norm_before','grad_norm_after']}
    steps, epochs = 0, 0
    kl_history = []
    start = time.monotonic()
    backward_adam = 0.
    for epoch in range(config.epochs):
        permutation = torch.randperm(B, generator=shuffle_rng, device='cpu')
        for cpu_ix in permutation.split(config.minibatch_size):
            if budget: budget.check()
            ix = cpu_ix.to(device)
            logits, value = model(d['obs_onehot'][ix])
            lp = masked_log_probs(logits, d['legal_mask'][ix])
            chosen = lp.gather(1,d['action'][ix,None]).squeeze(1)
            entropy = categorical_entropy(lp, d['legal_mask'][ix])
            loss, metrics = ppo_losses(chosen, d['old_logprob'][ix], d['policy_advantage'][ix],
                                       value, d['return'][ix], entropy, config)
            k = d['legal_mask'][ix].sum(-1)
            normalized_entropy = torch.where(k>1, entropy / k.clamp(min=2).float().log(), 0.).mean()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            before = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            after = torch.sqrt(sum((p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))
            optimizer.step()
            synchronize(device); backward_adam += time.monotonic()-tick
            for name, val in metrics.items(): sums[name] += float(val.detach())
            sums['normalized_entropy'] += float(normalized_entropy.detach())
            sums['grad_norm_before'] += float(before)
            sums['grad_norm_after'] += float(after)
            steps += 1
        epochs += 1
        # Guard after the complete epoch, on the entire immutable rollout.
        kl = rollout_kl(model, d, config.minibatch_size)
        kl_history.append(kl)
        if kl > config.target_kl: break
    target_var = float(d['return'].var(unbiased=False))
    explained = None if target_var == 0 else 1 - float((d['return']-d['old_value']).var(unbiased=False))/target_var
    synchronize(device)
    return {**{k:v/steps for k,v in sums.items()}, 'exact_kl': kl_history[-1], 'kl_by_epoch': kl_history,
            'epochs': epochs, 'optimizer_steps': steps, 'kl_early_stop': epochs < config.epochs,
            'explained_variance_old': explained, 'update_seconds': time.monotonic()-start,
            'backward_adam_seconds': backward_adam}
