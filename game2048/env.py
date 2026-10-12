"""Standard 2048 rules with explicit resets and reproducible, isolated RNGs.

Boards hold tile *exponents* (0 is empty), not tile values. The action order is
UP, RIGHT, DOWN, LEFT. Reaching 2048 does not end an episode. A step returns the
actual post-spawn observation; ``reset_done`` must be called separately.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)
ACTION_NAMES = ("up", "right", "down", "left")
MAX_EXPONENT = 20


def _boards_array(boards: Any, *, single: bool = False) -> np.ndarray:
    value = np.asarray(boards)
    if (single and value.shape != (4, 4)) or (
        not single and (value.ndim != 3 or value.shape[1:] != (4, 4))
    ):
        raise ValueError("board shape must be (4, 4)" if single else "boards shape must be (N, 4, 4)")
    if value.dtype.kind not in "iu":
        raise TypeError("board exponents must be integers")
    if value.size and (np.any(value < 0) or np.any(value > MAX_EXPONENT)):
        raise ValueError(f"board exponents must be in [0, {MAX_EXPONENT}]")
    return value.astype(np.uint8, copy=False)


def _actions_array(actions: Any, count: int, *, allow_noop: bool = False) -> np.ndarray:
    value = np.asarray(actions)
    if count == 0 and value.shape == (0,):
        return np.empty(0, dtype=np.int64)
    if value.dtype.kind not in "iu":
        raise TypeError("actions must be integers")
    if value.ndim == 0:
        value = np.full(count, value.item(), dtype=value.dtype)
    if value.shape != (count,):
        raise ValueError(f"actions shape must be ({count},)")
    if np.any(value < (-1 if allow_noop else 0)) or np.any(value > 3):
        raise ValueError("actions must be 0=up, 1=right, 2=down, or 3=left")
    return value.astype(np.int64, copy=False)


def _merge_line(line: np.ndarray) -> tuple[np.ndarray, int]:
    """Compress and merge once toward the left; use Python ints for rewards."""
    tiles = [int(x) for x in line if x]
    result = np.zeros(4, dtype=np.uint8)
    read = write = reward = 0
    while read < len(tiles):
        exponent = tiles[read]
        if read + 1 < len(tiles) and exponent == tiles[read + 1]:
            if exponent == MAX_EXPONENT:
                raise OverflowError("merge would produce exponent 21 outside the supported [0, 20] range")
            exponent += 1
            reward += 1 << exponent
            read += 2
        else:
            read += 1
        result[write] = exponent
        write += 1
    return result, reward


@lru_cache(maxsize=1)
def _row_tables() -> tuple[np.ndarray, np.ndarray]:
    """All 65,536 input rows with exponents 0..15, including output exponent 16."""
    rows = np.empty((65536, 4), dtype=np.uint8)
    rewards = np.empty(65536, dtype=np.int64)
    for key in range(65536):
        row = np.array([(key >> (4 * i)) & 15 for i in range(4)], dtype=np.uint8)
        rows[key], rewards[key] = _merge_line(row)
    rows.flags.writeable = False
    rewards.flags.writeable = False
    return rows, rewards


def _move_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = rows.reshape(-1, 4)
    result = np.empty_like(rows)
    rewards = np.empty(len(rows), dtype=np.int64)
    low = np.all(rows <= 15, axis=1)
    if np.any(low):
        source = rows[low].astype(np.uint32)
        keys = source[:, 0] | (source[:, 1] << 4) | (source[:, 2] << 8) | (source[:, 3] << 12)
        table, reward_table = _row_tables()
        result[low], rewards[low] = table[keys], reward_table[keys]
    for index in np.flatnonzero(~low):
        result[index], rewards[index] = _merge_line(rows[index])
    return result, rewards


def _orient_left(boards: np.ndarray, action: int) -> np.ndarray:
    if action == UP:
        return boards.transpose(0, 2, 1)
    if action == RIGHT:
        return boards[:, :, ::-1]
    if action == DOWN:
        return boards.transpose(0, 2, 1)[:, :, ::-1]
    return boards


def _restore_orientation(boards: np.ndarray, action: int) -> np.ndarray:
    if action == UP:
        return boards.transpose(0, 2, 1)
    if action == RIGHT:
        return boards[:, :, ::-1]
    if action == DOWN:
        return boards[:, :, ::-1].transpose(0, 2, 1)
    return boards


def move_batch(boards: Any, actions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure batched moves, without random spawning.

    Returns new uint8 boards, int64 merge rewards, and bool changed flags. Inputs
    are never modified. An attempted exponent-21 merge raises before returning.
    A scalar action applies the same direction to every board.
    """
    source = _boards_array(boards)
    action_array = _actions_array(actions, len(source))
    result = np.empty_like(source)
    rewards = np.zeros(len(source), dtype=np.int64)
    for action in range(4):
        indices = np.flatnonzero(action_array == action)
        if not len(indices):
            continue
        oriented = _orient_left(source[indices], action)
        moved, row_rewards = _move_rows(oriented.reshape(-1, 4))
        result[indices] = _restore_orientation(moved.reshape(-1, 4, 4), action)
        rewards[indices] = row_rewards.reshape(-1, 4).sum(axis=1, dtype=np.int64)
    changed = np.any(result != source, axis=(1, 2))
    return result, rewards, changed


def deterministic_move(board: Any, action: int) -> tuple[np.ndarray, np.int64, bool]:
    """Pure single-board version of :func:`move_batch`."""
    source = _boards_array(board, single=True)
    result, rewards, changed = move_batch(source[None], action)
    return result[0], rewards[0], bool(changed[0])


def reference_move(board: Any, action: int) -> tuple[np.ndarray, np.int64, bool]:
    """Slow independent cell-by-cell reference, deliberately not using row LUTs."""
    source = _boards_array(board, single=True)
    action = int(_actions_array(action, 1)[0])
    result = source.copy()
    merged = np.zeros((4, 4), dtype=bool)
    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
    row_order = range(3, -1, -1) if action == DOWN else range(4)
    col_order = range(3, -1, -1) if action == RIGHT else range(4)
    reward = 0
    for row in row_order:
        for col in col_order:
            exponent = int(result[row, col])
            if not exponent:
                continue
            r, c = row, col
            while 0 <= r + dr < 4 and 0 <= c + dc < 4 and result[r + dr, c + dc] == 0:
                r += dr
                c += dc
            nr, nc = r + dr, c + dc
            if (0 <= nr < 4 and 0 <= nc < 4 and int(result[nr, nc]) == exponent
                    and not merged[nr, nc]):
                if exponent == MAX_EXPONENT:
                    raise OverflowError("merge would produce exponent 21 outside the supported [0, 20] range")
                result[row, col] = 0
                result[nr, nc] = exponent + 1
                merged[nr, nc] = True
                reward += 1 << (exponent + 1)
            elif (r, c) != (row, col):
                result[row, col] = 0
                result[r, c] = exponent
    return result, np.int64(reward), bool(np.any(result != source))


def legal_mask(boards: Any) -> np.ndarray:
    """Legal directions for one board or a batch, without simulating merges.

    A supported exponent-20 board may have a legal merge that exceeds the model
    range. Such a move is legal under 2048 rules and raises when actually made.
    """
    array = np.asarray(boards)
    single = array.ndim == 2
    array = _boards_array(array, single=single)
    batch = array[None] if single else array
    horizontal_equal = np.any((batch[:, :, :-1] == batch[:, :, 1:]) & (batch[:, :, :-1] != 0), axis=(1, 2))
    vertical_equal = np.any((batch[:, :-1, :] == batch[:, 1:, :]) & (batch[:, :-1, :] != 0), axis=(1, 2))
    mask = np.empty((len(batch), 4), dtype=bool)
    mask[:, UP] = vertical_equal | np.any((batch[:, :-1, :] == 0) & (batch[:, 1:, :] != 0), axis=(1, 2))
    mask[:, RIGHT] = horizontal_equal | np.any((batch[:, :, :-1] != 0) & (batch[:, :, 1:] == 0), axis=(1, 2))
    mask[:, DOWN] = vertical_equal | np.any((batch[:, :-1, :] != 0) & (batch[:, 1:, :] == 0), axis=(1, 2))
    mask[:, LEFT] = horizontal_equal | np.any((batch[:, :, :-1] == 0) & (batch[:, :, 1:] != 0), axis=(1, 2))
    return mask[0] if single else mask


class SpawnOutcome(NamedTuple):
    probability: float
    board: np.ndarray
    position: int
    exponent: int


def spawn_outcomes(board: Any) -> list[SpawnOutcome]:
    """Enumerate the exact post-spawn distribution (flat positions 0..15).

    A full board has no spawn outcomes. Each empty cell has probability 0.9/E
    of receiving exponent 1 and probability 0.1/E of receiving exponent 2.
    """
    source = _boards_array(board, single=True)
    empty = np.flatnonzero(source.reshape(-1) == 0)
    outcomes = []
    for exponent, probability in ((1, 0.9), (2, 0.1)):
        for position in empty:
            result = source.copy()
            result.flat[position] = exponent
            outcomes.append(SpawnOutcome(probability / len(empty), result, int(position), exponent))
    return outcomes


def spawn(board: Any, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Spawn one tile on a copied board, or return (-1, 0) on a full board."""
    result = _boards_array(board, single=True).copy()
    position, exponent = _spawn_inplace(result, rng)
    return result, position, exponent


def _spawn_inplace(board: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    empty = np.flatnonzero(board.reshape(-1) == 0)
    if not len(empty):
        return -1, 0
    position = int(empty[rng.integers(len(empty))])
    exponent = 1 if rng.random() < 0.9 else 2
    board.flat[position] = exponent
    return position, exponent


def _nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return int(value)


class VectorEnv:
    """Single-process NumPy vector environment with one RNG per episode.

    Episode RNGs use SeedSequence([root_seed, split_id, env_id, episode_id]).
    Episodes start at ID zero. Illegal moves consume no RNG and do not increment
    the episode length. Finished environments remain frozen until reset_done.
    """

    def __init__(self, n: int, seed: int = 1, split_id: int = 0,
                 max_episode_steps: int | None = None):
        self.n = _nonnegative_int(n, "n", positive=True)
        self.root_seed = _nonnegative_int(seed, "seed")
        self.split_id = _nonnegative_int(split_id, "split_id")
        self.max_episode_steps = (None if max_episode_steps is None else
                                  _nonnegative_int(max_episode_steps, "max_episode_steps", positive=True))
        self.boards = np.zeros((self.n, 4, 4), dtype=np.uint8)
        self.scores = np.zeros(self.n, dtype=np.int64)
        self.lengths = np.zeros(self.n, dtype=np.int64)
        self.episode_ids = np.zeros(self.n, dtype=np.int64)
        self.max_exponents = np.zeros(self.n, dtype=np.uint8)
        self._terminated = np.zeros(self.n, dtype=bool)
        self._truncated = np.zeros(self.n, dtype=bool)
        self.rngs: list[np.random.Generator] = []
        for env_id in range(self.n):
            self.rngs.append(self._episode_rng(env_id))
            self._initialize_board(env_id)

    @property
    def masks(self) -> np.ndarray:
        return legal_mask(self.boards)

    @property
    def max_tiles(self) -> np.ndarray:
        return np.where(self.max_exponents > 0, np.left_shift(np.int64(1), self.max_exponents.astype(np.int64)), 0)

    def _episode_rng(self, env_id: int) -> np.random.Generator:
        seed = np.random.SeedSequence([self.root_seed, self.split_id, env_id, int(self.episode_ids[env_id])])
        return np.random.default_rng(seed)

    def _initialize_board(self, env_id: int) -> None:
        self.boards[env_id].fill(0)
        self.scores[env_id] = self.lengths[env_id] = 0
        self._terminated[env_id] = self._truncated[env_id] = False
        _spawn_inplace(self.boards[env_id], self.rngs[env_id])
        _spawn_inplace(self.boards[env_id], self.rngs[env_id])
        self.max_exponents[env_id] = self.boards[env_id].max()

    def step(self, actions: Any) -> dict[str, Any]:
        old_done = self._terminated | self._truncated
        actions = _actions_array(actions, self.n, allow_noop=True)
        if np.any((actions == -1) & ~old_done):
            raise ValueError("action -1 is only allowed for already-done environments")
        masks = self.masks
        active = ~old_done & masks[np.arange(self.n), np.maximum(actions, 0)]
        indices = np.flatnonzero(active)
        rewards = np.zeros(self.n, dtype=np.int64)
        changed = np.zeros(self.n, dtype=bool)
        positions = np.full(self.n, -1, dtype=np.int64)
        exponents = np.zeros(self.n, dtype=np.uint8)
        # Compute every deterministic move before any mutation/RNG draw. An
        # out-of-range merge therefore leaves the entire vector state intact.
        moved, merge_rewards, did_change = move_batch(self.boards[indices], actions[indices])
        if not np.all(did_change):
            raise RuntimeError("legal_mask disagrees with deterministic move")
        for local_id, env_id in enumerate(indices):
            positions[env_id], exponents[env_id] = _spawn_inplace(moved[local_id], self.rngs[env_id])
        self.boards[indices] = moved
        rewards[indices] = merge_rewards
        changed[indices] = True
        self.scores[indices] += merge_rewards
        self.lengths[indices] += 1
        self.max_exponents[indices] = np.maximum(self.max_exponents[indices], moved.max(axis=(1, 2)))
        next_masks = self.masks
        self._terminated |= ~next_masks.any(axis=1)
        if self.max_episode_steps is not None:
            self._truncated |= self.lengths >= self.max_episode_steps
        newly_done = (self._terminated | self._truncated) & ~old_done
        episodes = [dict(score=int(self.scores[i]), length=int(self.lengths[i]),
                         max_tile=int(self.max_tiles[i]), env_id=int(i), episode_id=int(self.episode_ids[i]),
                         terminated=bool(self._terminated[i]), truncated=bool(self._truncated[i]),
                         final_board=self.boards[i].copy()) for i in np.flatnonzero(newly_done)]
        return dict(boards=self.boards.copy(), rewards=rewards, terminated=self._terminated.copy(),
                    truncated=self._truncated.copy(), masks=next_masks.copy(), episodes=episodes,
                    spawn_positions=positions, spawn_exponents=exponents, changed=changed,
                    episode_ids=self.episode_ids.copy(), episode_steps=self.lengths.copy())

    def reset_done(self, done: Any) -> dict[str, np.ndarray]:
        """Reset only selected finished environments; returned arrays are copies."""
        selected = np.asarray(done)
        if selected.shape != (self.n,) or selected.dtype != np.bool_:
            raise ValueError(f"done must be a boolean array of shape ({self.n},)")
        if np.any(selected & ~(self._terminated | self._truncated)):
            raise ValueError("reset_done can only reset terminated or truncated environments")
        if np.any(self.episode_ids[selected] == np.iinfo(np.int64).max):
            raise OverflowError("episode counter exceeds int64")
        for env_id in np.flatnonzero(selected):
            self.episode_ids[env_id] += 1
            self.rngs[env_id] = self._episode_rng(int(env_id))
            self._initialize_board(int(env_id))
        return dict(boards=self.boards.copy(), masks=self.masks, episode_ids=self.episode_ids.copy())

    def state_dict(self) -> dict[str, Any]:
        """Complete independent snapshot, including generator states and flags."""
        return dict(version=1, n=self.n, root_seed=self.root_seed, split_id=self.split_id,
                    max_episode_steps=self.max_episode_steps, boards=self.boards.copy(),
                    scores=self.scores.copy(), lengths=self.lengths.copy(),
                    episode_ids=self.episode_ids.copy(), max_exponents=self.max_exponents.copy(),
                    terminated=self._terminated.copy(), truncated=self._truncated.copy(), masks=self.masks,
                    rng_states=[deepcopy(rng.bit_generator.state) for rng in self.rngs])

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate a snapshot in full before replacing the live environment."""
        if state.get("version") != 1 or state.get("n") != self.n:
            raise ValueError("unsupported state version or mismatched vector size")
        root_seed = _nonnegative_int(state["root_seed"], "root_seed")
        split_id = _nonnegative_int(state["split_id"], "split_id")
        limit = state["max_episode_steps"]
        if limit is not None:
            limit = _nonnegative_int(limit, "max_episode_steps", positive=True)
        boards = _boards_array(state["boards"]).copy()
        if boards.shape != (self.n, 4, 4):
            raise ValueError("snapshot board count differs from n")
        arrays = {}
        for name in ("scores", "lengths", "episode_ids"):
            value = np.asarray(state[name])
            if value.shape != (self.n,) or value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value > np.iinfo(np.int64).max):
                raise ValueError(f"invalid snapshot {name}")
            arrays[name] = value.astype(np.int64, copy=True)
        for name in ("terminated", "truncated"):
            value = np.asarray(state[name])
            if value.shape != (self.n,) or value.dtype != np.bool_:
                raise ValueError(f"invalid snapshot {name}")
            arrays[name] = value.copy()
        maximum = np.asarray(state["max_exponents"])
        if (maximum.shape != (self.n,) or maximum.dtype.kind not in "iu" or
                np.any(maximum < boards.max(axis=(1, 2))) or np.any(maximum > MAX_EXPONENT)):
            raise ValueError("invalid snapshot max_exponents")
        mask = legal_mask(boards)
        if not np.array_equal(np.asarray(state["masks"]), mask):
            raise ValueError("snapshot masks do not match boards")
        if not np.array_equal(arrays["terminated"], ~mask.any(axis=1)):
            raise ValueError("snapshot termination flags do not match boards")
        expected_truncated = np.zeros(self.n, dtype=bool) if limit is None else arrays["lengths"] >= limit
        if not np.array_equal(arrays["truncated"], expected_truncated):
            raise ValueError("snapshot truncation flags do not match the time limit")
        rng_states = state["rng_states"]
        if len(rng_states) != self.n:
            raise ValueError("snapshot RNG count differs from n")
        rngs = []
        for rng_state in rng_states:
            rng = np.random.default_rng()
            rng.bit_generator.state = deepcopy(rng_state)
            rngs.append(rng)
        self.root_seed, self.split_id, self.max_episode_steps = root_seed, split_id, limit
        self.boards = boards
        self.scores, self.lengths, self.episode_ids = arrays["scores"], arrays["lengths"], arrays["episode_ids"]
        self.max_exponents = maximum.astype(np.uint8, copy=True)
        self._terminated, self._truncated = arrays["terminated"], arrays["truncated"]
        self.rngs = rngs
