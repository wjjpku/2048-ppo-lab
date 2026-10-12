"""Rule, distribution, reproducibility, and vector-boundary regression tests."""
import pickle
import unittest

import numpy as np

from game2048.env import (
    DOWN, LEFT, RIGHT, UP, VectorEnv, deterministic_move, legal_mask,
    move_batch, reference_move, spawn, spawn_outcomes,
)


def _line_oracle(row):
    """Independent merge oracle used to exhaust the entire four-nibble input."""
    packed = [int(x) for x in row if x != 0]
    output = []
    reward = 0
    while packed:
        tile = packed.pop(0)
        if packed and packed[0] == tile:
            packed.pop(0)
            tile += 1
            reward += 2 ** tile
        output.append(tile)
    return output + [0] * (4 - len(output)), reward


def _tile_sums(boards):
    return np.where(boards == 0, 0, np.left_shift(np.int64(1), boards.astype(np.int64))).sum(axis=(-1, -2))


def _state_bytes(env):
    return pickle.dumps(env.state_dict(), protocol=5)


class TestRules(unittest.TestCase):
    def test_merge_once_and_65536(self):
        for source, expected, reward in [
            ([1, 1, 1, 1], [2, 2, 0, 0], 8),
            ([1, 1, 2, 0], [2, 2, 0, 0], 4),
            ([2, 2, 2, 0], [3, 2, 0, 0], 8),
            ([15, 15, 0, 0], [16, 0, 0, 0], 65536),
            ([19, 19, 0, 0], [20, 0, 0, 0], 1048576),
        ]:
            board = np.zeros((4, 4), dtype=np.uint8)
            board[0] = source
            moved, actual_reward, changed = deterministic_move(board, LEFT)
            np.testing.assert_array_equal(moved[0], expected)
            self.assertEqual(actual_reward, reward)
            self.assertEqual(actual_reward.dtype, np.dtype(np.int64))
            self.assertTrue(changed)
            np.testing.assert_array_equal(board[0], source)

    def test_exhaustive_65536_rows(self):
        keys = np.arange(65536, dtype=np.uint32)
        rows = ((keys[:, None] >> (4 * np.arange(4))) & 15).astype(np.uint8)
        boards = np.zeros((65536, 4, 4), dtype=np.uint8)
        boards[:, 0] = rows
        result, rewards, changed = move_batch(boards, LEFT)
        expected = np.zeros_like(rows)
        expected_rewards = np.zeros(65536, dtype=np.int64)
        for index, row in enumerate(rows):
            expected[index], expected_rewards[index] = _line_oracle(row)
        np.testing.assert_array_equal(result[:, 0], expected)
        np.testing.assert_array_equal(result[:, 1:], 0)
        np.testing.assert_array_equal(rewards, expected_rewards)
        np.testing.assert_array_equal(changed, np.any(rows != expected, axis=1))
        np.testing.assert_array_equal(_tile_sums(result), _tile_sums(boards))

    def test_10000_boards_times_four_reference_differential(self):
        rng = np.random.default_rng(7128)
        boards = rng.integers(0, 20, size=(10000, 4, 4), dtype=np.uint8)
        boards[rng.random(boards.shape) < 0.35] = 0
        masks = legal_mask(boards)
        for action in range(4):
            actual, rewards, changes = move_batch(boards, action)
            expected = np.empty_like(boards)
            expected_rewards = np.empty(10000, dtype=np.int64)
            expected_changes = np.empty(10000, dtype=bool)
            for index, board in enumerate(boards):
                expected[index], expected_rewards[index], expected_changes[index] = reference_move(board, action)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(rewards, expected_rewards)
            np.testing.assert_array_equal(changes, expected_changes)
            np.testing.assert_array_equal(masks[:, action], expected_changes)
            np.testing.assert_array_equal(_tile_sums(actual), _tile_sums(boards))
        mixed_actions = rng.integers(0, 4, size=len(boards))
        actual, rewards, changes = move_batch(boards, mixed_actions)
        for index in range(100):
            expected, reward, changed = reference_move(boards[index], mixed_actions[index])
            np.testing.assert_array_equal(actual[index], expected)
            self.assertEqual(rewards[index], reward)
            self.assertEqual(changes[index], changed)

    def test_d4_equivariance(self):
        rng = np.random.default_rng(8841)
        boards = rng.integers(0, 17, size=(256, 4, 4), dtype=np.uint8)
        boards[rng.random(boards.shape) < 0.3] = 0
        for action in range(4):
            moved, rewards, changed = move_batch(boards, action)
            for mirror in (False, True):
                for rotation in range(4):
                    def transform(value):
                        return np.rot90(value[:, :, ::-1] if mirror else value, rotation, axes=(1, 2))
                    transformed_action = ((-action if mirror else action) - rotation) % 4
                    result, actual_rewards, actual_changed = move_batch(transform(boards), transformed_action)
                    np.testing.assert_array_equal(result, transform(moved))
                    np.testing.assert_array_equal(actual_rewards, rewards)
                    np.testing.assert_array_equal(actual_changed, changed)

    def test_all_illegal_terminal_and_empty_board(self):
        terminal = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
        self.assertFalse(legal_mask(terminal).any())
        for action in range(4):
            moved, reward, changed = deterministic_move(terminal, action)
            np.testing.assert_array_equal(moved, terminal)
            self.assertEqual(reward, 0)
            self.assertFalse(changed)
        self.assertFalse(legal_mask(np.zeros((4, 4), dtype=np.uint8)).any())

    def test_bounds_hard_fail_without_clamp(self):
        board = np.zeros((4, 4), dtype=np.uint8)
        board[0, :2] = 20
        self.assertTrue(legal_mask(board)[LEFT])
        for mover in (deterministic_move, reference_move):
            before = board.copy()
            with self.assertRaises(OverflowError):
                mover(board, LEFT)
            np.testing.assert_array_equal(board, before)
        for bad in (np.full((4, 4), 21), np.full((4, 4), -1)):
            with self.assertRaises(ValueError):
                deterministic_move(bad, LEFT)
            with self.assertRaises(ValueError):
                legal_mask(bad)
        with self.assertRaises(TypeError):
            deterministic_move(np.zeros((4, 4), dtype=float), LEFT)
        for action in (-1, 4):
            with self.assertRaises(ValueError):
                deterministic_move(board, action)
        with self.assertRaises(TypeError):
            deterministic_move(board, 1.5)

    def test_batch_empty_and_shapes(self):
        result, rewards, changed = move_batch(np.empty((0, 4, 4), dtype=np.uint8), [])
        self.assertEqual(result.shape, (0, 4, 4))
        self.assertEqual(rewards.shape, (0,))
        self.assertEqual(changed.shape, (0,))
        with self.assertRaises(ValueError):
            move_batch(np.zeros((1, 4, 4), dtype=np.uint8), [0, 1])
        with self.assertRaises(ValueError):
            deterministic_move(np.zeros((3, 4), dtype=np.uint8), UP)


class TestSpawning(unittest.TestCase):
    def test_exact_two_empty_distribution(self):
        board = np.ones((4, 4), dtype=np.uint8)
        board[0, 0] = board[3, 3] = 0
        outcomes = spawn_outcomes(board)
        self.assertEqual(len(outcomes), 4)
        np.testing.assert_allclose([x.probability for x in outcomes], [0.45, 0.45, 0.05, 0.05])
        self.assertAlmostEqual(sum(x.probability for x in outcomes), 1.0)
        self.assertEqual({(x.position, x.exponent) for x in outcomes}, {(0, 1), (15, 1), (0, 2), (15, 2)})
        for outcome in outcomes:
            self.assertEqual(np.count_nonzero(outcome.board != board), 1)
            self.assertEqual(int(_tile_sums(outcome.board) - _tile_sums(board)), 2 ** outcome.exponent)
        self.assertEqual(len(spawn_outcomes(np.ones((4, 4), dtype=np.uint8))), 0)

    def test_empirical_spawn_uniform_empty_and_ninety_ten(self):
        board = np.ones((4, 4), dtype=np.uint8)
        positions = np.array([1, 5, 9, 12])
        board.flat[positions] = 0
        rng = np.random.default_rng(827)
        counts = np.zeros(16, dtype=np.int64)
        twos = 0
        for _ in range(20000):
            result, position, exponent = spawn(board, rng)
            counts[position] += 1
            twos += exponent == 2
            self.assertEqual(np.count_nonzero(result != board), 1)
            self.assertEqual(int(_tile_sums(result) - _tile_sums(board)), 2 ** exponent)
        np.testing.assert_allclose(counts[positions] / 20000, 0.25, atol=0.012)
        self.assertAlmostEqual(twos / 20000, 0.1, delta=0.012)
        self.assertEqual(counts.sum(), counts[positions].sum())
        self.assertEqual(np.count_nonzero(board == 0), 4)

    def test_full_board_spawn_preserves_rng(self):
        board = np.ones((4, 4), dtype=np.uint8)
        rng = np.random.default_rng(44)
        before = pickle.dumps(rng.bit_generator.state)
        result, position, exponent = spawn(board, rng)
        self.assertEqual(pickle.dumps(rng.bit_generator.state), before)
        np.testing.assert_array_equal(board, result)
        self.assertEqual((position, exponent), (-1, 0))


class TestVectorEnv(unittest.TestCase):
    def test_initial_boards_and_seed_sequence_contract(self):
        env = VectorEnv(8, seed=935, split_id=4)
        np.testing.assert_array_equal(np.count_nonzero(env.boards, axis=(1, 2)), 2)
        self.assertTrue(np.all((env.boards >= 0) & (env.boards <= 2)))
        np.testing.assert_array_equal(env.scores, 0)
        np.testing.assert_array_equal(env.lengths, 0)
        np.testing.assert_array_equal(env.episode_ids, 0)
        for i in range(8):
            rng = np.random.default_rng(np.random.SeedSequence([935, 4, i, 0]))
            board, _, _ = spawn(np.zeros((4, 4), dtype=np.uint8), rng)
            board, _, _ = spawn(board, rng)
            np.testing.assert_array_equal(env.boards[i], board)

    def test_illegal_move_has_no_side_effects(self):
        env = VectorEnv(2, seed=142, max_episode_steps=1)
        env.boards.fill(0)
        env.boards[:, 0, 0] = 1
        before = _state_bytes(env)
        transition = env.step([UP, LEFT])
        self.assertEqual(_state_bytes(env), before)
        np.testing.assert_array_equal(transition["changed"], False)
        np.testing.assert_array_equal(transition["rewards"], 0)
        np.testing.assert_array_equal(transition["spawn_positions"], -1)
        np.testing.assert_array_equal(transition["spawn_exponents"], 0)
        np.testing.assert_array_equal(transition["truncated"], False)
        self.assertEqual(transition["episodes"], [])

    def test_overflow_is_atomic_across_batch(self):
        env = VectorEnv(2, seed=341)
        env.boards.fill(0)
        env.boards[0, 0, :2] = 1
        env.boards[1, 0, :2] = 20
        env.max_exponents[:] = [1, 20]
        before = _state_bytes(env)
        with self.assertRaises(OverflowError):
            env.step([LEFT, LEFT])
        self.assertEqual(_state_bytes(env), before)

    def test_scores_int64_and_continue_past_2048(self):
        env = VectorEnv(2)
        env.boards.fill(0)
        env.boards[0, 0, :2] = 10
        env.boards[1, 0, :2] = 15
        env.scores[:] = 2 ** 40
        step = env.step([LEFT, LEFT])
        np.testing.assert_array_equal(step["rewards"], [2048, 65536])
        self.assertEqual(step["rewards"].dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(env.scores, np.array([2048, 65536], dtype=np.int64) + 2 ** 40)
        np.testing.assert_array_equal(step["terminated"], False)
        np.testing.assert_array_equal(env.max_tiles, [2048, 65536])

    def test_terminal_post_spawn_board_and_one_summary(self):
        env = VectorEnv(1)
        env.boards[0] = [[1, 1, 3, 4], [3, 4, 1, 3], [4, 1, 2, 4], [1, 2, 3, 1]]
        step = env.step([LEFT])
        self.assertTrue(step["terminated"][0])
        self.assertFalse(step["truncated"][0])
        self.assertFalse(step["masks"].any())
        self.assertEqual(step["episodes"][0]["score"], 4)
        self.assertEqual(step["episodes"][0]["length"], 1)
        self.assertEqual(step["episodes"][0]["max_tile"], 16)
        self.assertEqual(np.count_nonzero(step["boards"][0]), 16)
        before = _state_bytes(env)
        second = env.step([-1])
        self.assertEqual(second["episodes"], [])
        self.assertEqual(_state_bytes(env), before)
        self.assertEqual(second["spawn_positions"][0], -1)

    def test_time_limit_reset_and_final_observation_copy(self):
        env = VectorEnv(2, seed=62, max_episode_steps=1)
        step = env.step(np.argmax(env.masks, axis=1))
        self.assertTrue(step["truncated"].all())
        self.assertFalse(step["terminated"].any())
        self.assertEqual(len(step["episodes"]), 2)
        final_boards = step["boards"].copy()
        final_summaries = [episode["final_board"].copy() for episode in step["episodes"]]
        untouched_rng = pickle.dumps(env.rngs[1].bit_generator.state)
        env.reset_done(np.array([True, False]))
        np.testing.assert_array_equal(step["boards"], final_boards)
        np.testing.assert_array_equal(env.boards[1], final_boards[1])
        self.assertEqual(pickle.dumps(env.rngs[1].bit_generator.state), untouched_rng)
        np.testing.assert_array_equal(env.episode_ids, [1, 0])
        np.testing.assert_array_equal(env.lengths, [0, 1])
        for actual, expected in zip(step["episodes"], final_summaries):
            np.testing.assert_array_equal(actual["final_board"], expected)
        rng = np.random.default_rng(np.random.SeedSequence([62, 0, 0, 1]))
        expected, _, _ = spawn(np.zeros((4, 4), dtype=np.uint8), rng)
        expected, _, _ = spawn(expected, rng)
        np.testing.assert_array_equal(env.boards[0], expected)
        with self.assertRaises(ValueError):
            env.reset_done(np.array([True, False]))
        with self.assertRaises(ValueError):
            env.reset_done([1, 0])
        with self.assertRaises(ValueError):
            env.step([-1, -1])
        # -1 can freeze the finished peer while the reset environment continues.
        last = env.step([int(np.flatnonzero(env.masks[0])[0]), -1])
        np.testing.assert_array_equal(last["boards"][1], final_boards[1])
        self.assertFalse(last["changed"][1])

    def test_natural_termination_and_time_limit_can_both_be_true(self):
        env = VectorEnv(1, max_episode_steps=1)
        env.boards[0] = [[1, 1, 3, 4], [3, 4, 1, 3], [4, 1, 2, 4], [1, 2, 3, 1]]
        step = env.step([LEFT])
        self.assertTrue(step["terminated"][0])
        self.assertTrue(step["truncated"][0])

    def test_seed_isolation_from_batch_size_other_resets_and_splits(self):
        small = VectorEnv(1, seed=129, split_id=7, max_episode_steps=11)
        large = VectorEnv(8, seed=129, split_id=7, max_episode_steps=11)
        other_split = VectorEnv(8, seed=129, split_id=8)
        self.assertFalse(np.array_equal(large.boards, other_split.boards))
        rng = np.random.default_rng(516)
        for _ in range(150):
            np.testing.assert_array_equal(small.boards[0], large.boards[0])
            self.assertEqual(small.scores[0], large.scores[0])
            self.assertEqual(small.episode_ids[0], large.episode_ids[0])
            small_action = int(np.flatnonzero(small.masks[0])[0])
            actions = np.array([int(rng.choice(np.flatnonzero(mask))) for mask in large.masks])
            actions[0] = small_action
            # Illegal actions for unrelated environments alter their reset timing.
            for i in range(1, large.n):
                illegal = np.flatnonzero(~large.masks[i])
                if len(illegal) and rng.random() < 0.5:
                    actions[i] = int(illegal[0])
            one = small.step([small_action])
            many = large.step(actions)
            np.testing.assert_array_equal(one["boards"][0], many["boards"][0])
            self.assertEqual(one["spawn_positions"][0], many["spawn_positions"][0])
            self.assertEqual(one["spawn_exponents"][0], many["spawn_exponents"][0])
            small.reset_done(one["terminated"] | one["truncated"])
            large.reset_done(many["terminated"] | many["truncated"])
        self.assertGreater(small.episode_ids[0], 0)

    def test_state_roundtrip_continuation_and_snapshot_independence(self):
        original = VectorEnv(4, seed=55, split_id=9, max_episode_steps=7)
        for _ in range(5):
            step = original.step(np.argmax(original.masks, axis=1))
            original.reset_done(step["terminated"] | step["truncated"])
        snapshot = original.state_dict()
        clone = VectorEnv(4, seed=987)
        clone.load_state_dict(snapshot)
        self.assertEqual(_state_bytes(original), _state_bytes(clone))
        snapshot["boards"].fill(0)
        snapshot["rng_states"][0]["state"]["state"] = 0
        self.assertEqual(_state_bytes(original), _state_bytes(clone))
        for _ in range(30):
            actions = np.argmax(original.masks, axis=1)
            first, second = original.step(actions), clone.step(actions)
            for key in ("boards", "rewards", "terminated", "truncated", "masks", "spawn_positions", "spawn_exponents"):
                np.testing.assert_array_equal(first[key], second[key])
            original.reset_done(first["terminated"] | first["truncated"])
            clone.reset_done(second["terminated"] | second["truncated"])
            self.assertEqual(_state_bytes(original), _state_bytes(clone))
        invalid = original.state_dict()
        invalid["masks"][:] = False
        before = _state_bytes(clone)
        with self.assertRaises(ValueError):
            clone.load_state_dict(invalid)
        self.assertEqual(_state_bytes(clone), before)


if __name__ == "__main__":
    unittest.main()
