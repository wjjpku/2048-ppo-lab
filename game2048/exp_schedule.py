"""Authorized zero-indexed reward/cosine schedules; independent of Adam steps."""
import math

START, TARGET, UPDATES, PEAK = 5840, 7840, 2000, 4e-5
KIND = 'power_reward_exp48_cosine2000'
PROTOCOL = 'base4_5840_exp4to8_cosine2000_v1'
SCHEDULE = {'type': 'exponential_reward_and_cosine_lr', 'unit': 'ppo_update',
    'updates': UPDATES, 'first_index': 0, 'last_index': 1999,
    'reward_start': 4.0, 'reward_end': 8.0,
    'reward_formula': '4*2**(j/1999)',
    'lr_formula': '0.5*4e-5*(1+cos(pi*j/1999))',
    'peak_learning_rate': PEAK, 'final_learning_rate': 0.0, 'warmup_updates': 0,
    'reward_anchor_exponent': 10, 'scope': 'same_base_for_entire_rollout_and_ppo_update'}


def values(j):
    if type(j) is not int or not 0 <= j < UPDATES:
        raise ValueError('schedule index must be an integer in 0..1999')
    return 4.0 * 2.0 ** (j / 1999), 0.5 * PEAK * (1.0 + math.cos(math.pi*j/1999))


def set_update(optimizer, j):
    base, lr = values(j)
    if not optimizer.param_groups:
        raise ValueError('missing Adam parameter groups')
    for group in optimizer.param_groups:
        group['lr'] = lr
    return base, lr
