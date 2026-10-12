"""Per-merge training rewards, independent of spawning and the game's score.

An output tile with exponent k contributes ``base ** (k - 10)``. Contributions
are summed separately for every merge, so a 1024 output always contributes one.
These helpers never move the live environment, mutate inputs, or consume RNG.
"""
from functools import lru_cache
from numbers import Real

import numpy as np

from .env import MAX_EXPONENT, _actions_array, _boards_array, _orient_left


SUPPORTED_REWARD_BASES = (2, 2.1, 2.4, 2.6, 2.8, 3, 3.3, 3.5, 4)
REWARD_ANCHOR_EXPONENT = 10


def validate_reward_base(reward_base, *, scheduled_step=None):
    """Return a canonical approved base; booleans and approximate matches fail."""
    if isinstance(reward_base, (bool, np.bool_)) or not isinstance(reward_base, Real):
        raise TypeError('reward_base must be a real number, not a boolean')
    value = float(reward_base)
    if scheduled_step is not None:
        from .exp_schedule import values
        expected, _ = values(scheduled_step)
        if not np.isfinite(value) or value != expected:
            raise ValueError('reward base differs from declared exponential update')
        return value
    if not np.isfinite(value) or value not in SUPPORTED_REWARD_BASES:
        raise ValueError(f'reward_base must be one of {SUPPORTED_REWARD_BASES}')
    return SUPPORTED_REWARD_BASES[SUPPORTED_REWARD_BASES.index(value)]


def _merge_row_counts(rows):
    """Count each output once after stable compression; do not chain merges."""
    count = len(rows)
    # Moving empty cells to the right preserves the order of every real tile.
    order = np.argsort(rows == 0, axis=1, kind='stable')
    packed = np.take_along_axis(rows, order, axis=1)
    result = np.zeros((count, MAX_EXPONENT + 1), dtype=np.uint8)
    indices = np.arange(count)
    cursor = np.zeros(count, dtype=np.int64)
    for _ in range(4):
        exponent = packed[indices, np.minimum(cursor, 3)]
        following = packed[indices, np.minimum(cursor + 1, 3)]
        merged = (cursor < 3) & (exponent != 0) & (exponent == following)
        if np.any(merged & (exponent == MAX_EXPONENT)):
            raise OverflowError('merge would produce exponent 21 outside the supported [0, 20] range')
        selected = indices[merged]
        result[selected, exponent[merged] + 1] += 1
        cursor += 1 + merged
    return result


@lru_cache(maxsize=1)
def _row_merge_count_table():
    """All four-nibble rows, including their exponent-16 merge outputs."""
    keys = np.arange(65536, dtype=np.uint32)
    rows = ((keys[:, None] >> (4 * np.arange(4))) & 15).astype(np.uint8)
    result = _merge_row_counts(rows)
    result.flags.writeable = False
    return result


def merge_output_counts(boards, actions):
    """Return uint8[N,21] histograms of actual deterministic merge outputs.

    Column k counts outputs with exponent k, not input tiles of that exponent.
    Inputs are pre-action boards; spawning must not be included. A scalar action
    applies to all boards. Legal slides and no-op moves both yield zero if no
    tiles merge. Invalid actions and exponent-21 merges fail like pure moves.
    """
    source = _boards_array(boards)
    action_array = _actions_array(actions, len(source))
    result = np.zeros((len(source), MAX_EXPONENT + 1), dtype=np.uint8)
    for action in range(4):
        indices = np.flatnonzero(action_array == action)
        if not len(indices):
            continue
        rows = _orient_left(source[indices], action).reshape(-1, 4)
        counts = np.empty((len(rows), MAX_EXPONENT + 1), dtype=np.uint8)
        low = np.all(rows <= 15, axis=1)
        if np.any(low):
            tiles = rows[low].astype(np.uint32)
            keys = tiles[:, 0] | (tiles[:, 1] << 4) | (tiles[:, 2] << 8) | (tiles[:, 3] << 12)
            counts[low] = _row_merge_count_table()[keys]
        if np.any(~low):
            counts[~low] = _merge_row_counts(rows[~low])
        result[indices] = counts.reshape(-1, 4, MAX_EXPONENT + 1).sum(axis=1, dtype=np.uint8)
    return result


def _validated_counts(counts):
    array = np.asarray(counts)
    if array.ndim < 1 or array.shape[-1] != MAX_EXPONENT + 1:
        raise ValueError('merge counts must have final dimension 21')
    if array.dtype.kind not in 'iu':
        raise TypeError('merge counts must be integers')
    if np.any(array < 0) or np.any(array[..., :2]):
        raise ValueError('merge counts must be nonnegative with no exponent-0/1 outputs')
    return array


def merge_reward_weights(reward_base=2, *, scheduled_step=None):
    """Float64 weights indexed by output exponent, anchored at 1024 -> 1."""
    base = validate_reward_base(reward_base, scheduled_step=scheduled_step)
    weights = np.power(np.float64(base), np.arange(MAX_EXPONENT + 1) - REWARD_ANCHOR_EXPONENT)
    weights[:2] = 0.  # Empty cells and unmerged new tiles never earn a reward.
    if not np.isfinite(weights).all():
        raise FloatingPointError('merge reward weights contain NaN/Inf')
    return weights


def rewards_from_merge_counts(counts, reward_base=2, *, scheduled_step=None):
    """Sum per-output contributions in float64; never exponentiate a score sum."""
    array = _validated_counts(counts)
    rewards = np.sum(array.astype(np.float64) * merge_reward_weights(reward_base, scheduled_step=scheduled_step), axis=-1, dtype=np.float64)
    if not np.isfinite(rewards).all():
        raise FloatingPointError('merge rewards contain NaN/Inf')
    return rewards


def merge_rewards(boards, actions, reward_base=2):
    """Compute float64 per-merge rewards directly from pre-action boards."""
    validate_reward_base(reward_base)
    return rewards_from_merge_counts(merge_output_counts(boards, actions), reward_base)


def raw_rewards_from_merge_counts(counts):
    """Reconstruct exact standard score increments for transition validation."""
    array = _validated_counts(counts)
    # This helper is for actual transitions, each containing at most eight merges.
    if np.any(array > 8) or np.any(array.sum(axis=-1, dtype=np.uint64) > 8):
        raise ValueError('a board transition cannot contain more than eight merges')
    values = np.left_shift(np.int64(1), np.arange(MAX_EXPONENT + 1))
    return np.sum(array.astype(np.int64) * values, axis=-1, dtype=np.int64)
