from dataclasses import dataclass
import time
import numpy as np
import torch
from .model import encode_boards, masked_log_probs, sample_actions, require_finite
from .gae import compute_gae, normalize_advantages
from .rewards import (merge_output_counts, merge_reward_weights, raw_rewards_from_merge_counts,
                      rewards_from_merge_counts, validate_reward_base)


@dataclass
class Rollout:
    data: dict
    policy_version: int
    episodes: list
    timings: dict
    advantage_stats: dict

    def flatten(self):
        # Time-major i=t*N+n, identical reshape for every aligned field.
        return {k: v.reshape((-1,) + v.shape[2:]) for k, v in self.data.items()}

    def merge_reward_diagnostics(self, reward_base=2, *, scheduled_step=None):
        """Lazy, RNG-free diagnostics; no fields are added to training tensors."""
        base = validate_reward_base(reward_base, scheduled_step=scheduled_step)
        boards = self.data['board_exp'].reshape(-1, 4, 4).cpu().numpy()
        actions = self.data['action'].reshape(-1).cpu().numpy()
        counts = merge_output_counts(boards, actions)
        raw = raw_rewards_from_merge_counts(counts)
        if not np.array_equal(raw, self.data['raw_reward'].reshape(-1).cpu().numpy()):
            raise RuntimeError('merge-output counts disagree with stored raw rewards')
        rewards = rewards_from_merge_counts(counts, base, scheduled_step=scheduled_step)
        totals = counts.sum(axis=0, dtype=np.int64)
        merged = counts.any(axis=1)
        return dict(reward_base=base, merge_counts_by_exponent=totals.tolist(),
                    weighted_reward_by_exponent=(totals * merge_reward_weights(base, scheduled_step=scheduled_step)).tolist(),
                    total_merges=int(totals.sum()), merge_transitions=int(merged.sum()),
                    no_merge_transitions=int((~merged).sum()), raw_reward_sum=int(raw.sum()),
                    reward_sum=float(rewards.sum()), reward_mean=float(rewards.mean()) if len(rewards) else 0.,
                    reward_max=float(rewards.max()) if len(rewards) else 0.)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def collect_rollout(model, env, config, device, sample_rng, policy_version, budget=None, reward_base=2, *, scheduled_step=None):
    reward_base = validate_reward_base(reward_base, scheduled_step=scheduled_step)
    if reward_base != 2 and config.reward_scale != 1024:
        raise ValueError('per-merge reward candidates require reward_scale=1024')
    T, N = config.rollout_steps, config.num_envs
    d = {}
    shapes = {'obs_onehot': ((336,), torch.float32), 'board_exp': ((4,4), torch.uint8),
              'actual_next_board_exp': ((4,4), torch.uint8), 'legal_mask': ((4,), torch.bool),
              'action': ((), torch.int64), 'raw_reward': ((), torch.int64),
              'reward_scaled': ((), torch.float32), 'old_logprob': ((), torch.float32),
              'old_value': ((), torch.float32), 'bootstrap_value': ((), torch.float32),
              'old_log_probs': ((4,), torch.float32), 'terminated': ((), torch.bool),
              'truncated': ((), torch.bool), 'episode_id': ((), torch.int64),
              'episode_step': ((), torch.int32)}
    for key, (shape, dtype) in shapes.items():
        d[key] = torch.empty((T,N) + shape, dtype=dtype, device=device)
    timings = dict(encode=0., h2d=0., policy_forward=0., d2h=0., env_step=0., bootstrap=0., gae=0.)
    episodes = []
    start = time.monotonic()
    model.eval()
    for t in range(T):
        if budget: budget.check()
        tick = time.monotonic()
        board = torch.as_tensor(env.boards.copy(), device=device)
        mask = torch.as_tensor(env.masks.copy(), device=device)
        d['board_exp'][t].copy_(board)
        d['legal_mask'][t].copy_(mask)
        d['episode_id'][t].copy_(torch.as_tensor(env.episode_ids.copy(), device=device))
        d['episode_step'][t].copy_(torch.as_tensor(env.lengths.copy(), device=device))
        synchronize(device); timings['h2d'] += time.monotonic()-tick
        tick = time.monotonic()
        obs = encode_boards(board)
        d['obs_onehot'][t].copy_(obs)
        synchronize(device); timings['encode'] += time.monotonic()-tick
        tick = time.monotonic()
        logits, value = model(obs)
        require_finite(value, 'behavior value')
        lp = masked_log_probs(logits, mask)
        action = sample_actions(lp, sample_rng)
        if not mask.gather(1, action[:,None]).all():
            raise RuntimeError('sampled an illegal action')
        d['old_log_probs'][t].copy_(lp)
        d['old_logprob'][t].copy_(lp.gather(1, action[:,None]).squeeze(1))
        d['old_value'][t].copy_(value)
        d['action'][t].copy_(action)
        synchronize(device); timings['policy_forward'] += time.monotonic()-tick
        tick = time.monotonic()
        actions = action.cpu().numpy()
        synchronize(device); timings['d2h'] += time.monotonic()-tick
        tick = time.monotonic()
        if reward_base != 2:
            # Use the pre-action board, before env.step consumes any spawn RNG.
            merge_counts = merge_output_counts(env.boards, actions)
            candidate_rewards = rewards_from_merge_counts(merge_counts, reward_base, scheduled_step=scheduled_step)
            expected_raw_rewards = raw_rewards_from_merge_counts(merge_counts)
        step = env.step(actions)
        if 'changed' in step and not np.all(step['changed']):
            raise RuntimeError('rollout contains a non-player-action transition')
        timings['env_step'] += time.monotonic()-tick
        tick = time.monotonic()
        # Copy actual post-action boards BEFORE reset. Never store a mutable view.
        d['actual_next_board_exp'][t].copy_(torch.as_tensor(step['boards'], device=device))
        reward = torch.as_tensor(step['rewards'], device=device)
        d['raw_reward'][t].copy_(reward)
        if reward_base == 2:
            # Preserve the control's exact prior arithmetic and FP32 conversion.
            d['reward_scaled'][t].copy_(reward / config.reward_scale)
        else:
            if not np.array_equal(expected_raw_rewards, step['rewards']):
                raise RuntimeError('merge-output counts disagree with environment raw rewards')
            # Weights and their sums stay float64 until this FP32 buffer copy.
            d['reward_scaled'][t].copy_(torch.as_tensor(candidate_rewards, device=device))
        require_finite(d['reward_scaled'][t], 'scaled rewards')
        d['terminated'][t].copy_(torch.as_tensor(step['terminated'], device=device))
        d['truncated'][t].copy_(torch.as_tensor(step['truncated'], device=device))
        synchronize(device); timings['h2d'] += time.monotonic()-tick
        episodes.extend(step['episodes'])
        done = step['terminated'] | step['truncated']
        tick = time.monotonic()
        if np.any(done): env.reset_done(done)
        timings['env_step'] += time.monotonic()-tick
    tick = time.monotonic()
    # Same frozen behavior network; terminal rows bypass masked policy entirely.
    flat_next = d['actual_next_board_exp'].reshape(-1,4,4)
    nonterminal = ~d['terminated'].flatten()
    next_values = d['bootstrap_value'].flatten()
    next_values.zero_()
    indices = nonterminal.nonzero(as_tuple=False).squeeze(-1)
    for ix in indices.split(config.minibatch_size):
        if budget: budget.check()
        _, v = model(encode_boards(flat_next[ix]))
        require_finite(v, 'bootstrap value')
        next_values[ix] = v
    synchronize(device); timings['bootstrap'] = time.monotonic()-tick
    tick = time.monotonic()
    advantages, returns = compute_gae(d['reward_scaled'], d['old_value'], d['bootstrap_value'],
                                     d['terminated'], d['truncated'], config.gamma, config.gae_lambda)
    require_finite(advantages, 'GAE advantages')
    require_finite(returns, 'GAE returns')
    d['advantage'], d['return'] = advantages, returns
    d['policy_advantage'], stats = normalize_advantages(advantages)
    require_finite(d['policy_advantage'], 'normalized policy advantages')
    gathered = d['old_log_probs'].gather(-1, d['action'].unsqueeze(-1)).squeeze(-1)
    if not torch.equal(gathered, d['old_logprob']):
        raise RuntimeError('chosen logprob and full old distribution disagree')
    synchronize(device); timings['gae'] = time.monotonic()-tick
    timings['collect_total'] = time.monotonic()-start
    return Rollout(d, policy_version, episodes, timings, stats)
