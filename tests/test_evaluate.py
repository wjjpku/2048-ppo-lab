"""Evaluation must be reproducible, isolated, honest about partials, and replayable."""
from copy import deepcopy
import importlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from game2048.env import VectorEnv, deterministic_move
from game2048.evaluate import evaluate, verify_replay
from game2048.model import ActorCritic


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    # Tiny batched policy calls are much faster without huge default pools.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class UniformPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.child = nn.Identity()
        self.forward_calls = []

    def forward(self, encoded):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.forward_calls.append(len(encoded))
        return encoded.new_zeros((len(encoded), 4)) + self.anchor, encoded.new_full((len(encoded),), 1.25)


def test_random_baseline_completes_reproducibly_and_reports_statistics():
    first = evaluate(policy="random", episodes=4, seed=19)
    second = evaluate(policy="random", episodes=4, seed=19)
    assert first["complete"] and first["completed_episodes"] == 4
    assert not first["interrupted"] and first["incomplete_episodes"] == 0
    assert first["episodes"] == second["episodes"]
    assert first["max_episode_steps"] is None
    assert first["mean_score"] == np.mean(first["scores"])
    assert first["median_score"] == np.median(first["scores"])
    assert first["std_score"] == np.std(first["scores"])
    assert sum(first["max_tile_counts"].values()) == 4
    assert first["total_steps"] == sum(row["length"] for row in first["episodes"])
    assert all(row["terminated"] and not row["truncated"] and row["episode_id"] == 0 for row in first["episodes"])
    assert all(not any(deterministic_move(np.array(row["final_board"], dtype=np.uint8), action)[2]
                       for action in range(4)) for row in first["episodes"])
    assert first["win_rate_2048"] == sum(row["max_tile"] >= 2048 for row in first["episodes"]) / 4
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("policy", ["random", "merge", "model"])
def test_replay_contains_every_event_and_self_contained_viewer(tmp_path, policy):
    model = UniformPolicy() if policy == "model" else None
    result = evaluate(model, policy=policy, episodes=3, seed=31, replay_path=tmp_path / f"{policy}.json")
    path = tmp_path / f"{policy}.json"
    replay = json.loads(path.read_text())
    assert verify_replay(path) == {"valid": True, "steps": replay["result"]["length"],
                                   "score": replay["result"]["score"], "complete": True}
    assert replay["result"] == result["episodes"][0]
    assert replay["seed_components"] == [31, 1, 0, 0]
    html = (tmp_path / f"{policy}.html").read_text()
    assert '<script id="replay-data" type="application/json">' in html
    assert 'src="http' not in html and 'href="http' not in html
    assert 'id="seek"' in html and "Incomplete episode" in html
    # Recreate the exact episode from its seed, not only from recorded boards.
    env = VectorEnv(1, seed=31, split_id=1)
    assert np.array_equal(env.boards[0], replay["initial_board"])
    for step in replay["steps"]:
        transition = env.step([step["action"]])
        assert int(transition["rewards"][0]) == step["raw_reward"]
        assert int(transition["spawn_positions"][0]) == step["spawn"]["position"]
        assert int(transition["spawn_exponents"][0]) == step["spawn"]["exponent"]
        assert np.array_equal(transition["boards"][0], step["board_after"])
        assert int(env.scores[0]) == step["score"]
        if policy == "merge":
            board = np.array(step["board_before"], dtype=np.uint8)
            legal = np.flatnonzero(step["legal_mask"])
            expected = max(legal, key=lambda action: deterministic_move(board, action)[1])
            assert step["action"] == expected
            assert step["chosen_probability"] == 1
            assert step["value"] is None
        elif policy == "random":
            assert step["chosen_probability"] == 1 / sum(step["legal_mask"])
        else:
            assert step["action"] == int(np.flatnonzero(step["legal_mask"])[0])
            assert step["value"] == 1.25
    if policy == "model":
        assert max(model.forward_calls) == 3  # Batched inference, not one call per game.
        assert result["total_steps"] > len(model.forward_calls)


def test_model_mode_parameters_and_all_global_rngs_are_preserved():
    model = ActorCritic()
    model.train()
    model.trunk[0].eval()  # Preserve mixed child flags too.
    model.policy.weight.grad = torch.ones_like(model.policy.weight)
    before_parameters = {name: value.clone() for name, value in model.state_dict().items()}
    before_gradient = model.policy.weight.grad.clone()
    before_flags = [module.training for module in model.modules()]
    random.seed(711)
    np.random.seed(712)
    torch.manual_seed(713)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    first = evaluate(model, episodes=3, mode="sample", seed=714)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert [module.training for module in model.modules()] == before_flags
    assert torch.equal(model.policy.weight.grad, before_gradient)
    assert all(torch.equal(value, before_parameters[name]) for name, value in model.state_dict().items())
    random.seed(999)
    np.random.seed(998)
    torch.manual_seed(997)
    second = evaluate(model, episodes=3, mode="sample", seed=714)
    assert first["episodes"] == second["episodes"]


def test_rngs_and_mode_restore_even_when_forward_raises():
    class BrokenModel(UniformPolicy):
        def forward(self, encoded):
            random.random()
            np.random.random()
            torch.rand(10)
            raise RuntimeError("intentional failure")

    model = BrokenModel().train()
    py_state = random.getstate()
    np_state = deepcopy(np.random.get_state())
    torch_state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="intentional"):
        evaluate(model, episodes=1)
    assert model.training and model.child.training
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_zero_deadline_is_explicitly_partial_with_null_statistics(tmp_path):
    result = evaluate(policy="random", episodes=100, max_seconds=0, replay_path=tmp_path / "partial.html")
    assert not result["complete"] and result["interrupted"]
    assert result["interruption_reason"] == "max_seconds"
    assert result["requested_episodes"] == result["incomplete_episodes"] == 100
    assert result["completed_episodes"] == result["total_steps"] == 0
    assert result["scores"] == [] and result["mean_score"] is None
    assert result["win_rate_2048"] is None
    assert all(row["status"] == "incomplete" for row in result["episodes"])
    replay = json.loads((tmp_path / "partial.json").read_text())
    assert not replay["complete"] and not replay["steps"]
    assert verify_replay(replay)["valid"]
    json.dumps(result, allow_nan=False)


def test_stop_callback_excludes_partial_scores_and_returns_coherent_replay(tmp_path):
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return calls > 3
    result = evaluate(policy="merge", episodes=2, should_stop=stop, replay_path=tmp_path / "stop.json")
    assert result["interruption_reason"] == "stop_requested"
    assert result["completed_episodes"] == 0 and result["total_steps"] == 6
    assert all(row["length"] == 3 for row in result["episodes"])
    assert result["mean_score"] is None
    assert verify_replay(result["replay_json"])["steps"] == 3


def test_validation_and_heldout_test_have_disjoint_seed_banks():
    validation = evaluate(policy="random", episodes=5, seed=47, split_id=1, max_seconds=0)
    test = evaluate(policy="random", episodes=5, seed=47, split_id=2, max_seconds=0)
    assert validation["split"] == "validation" and test["split"] == "test"
    assert [row["final_board"] for row in validation["episodes"]] != [row["final_board"] for row in test["episodes"]]
    with pytest.raises(ValueError, match="reserved for training"):
        evaluate(policy="random", split_id=0)


def test_greedy_and_sample_use_same_initial_environment_seed_bank(tmp_path):
    model = UniformPolicy()
    for mode in ("greedy", "sample"):
        evaluate(model, episodes=2, seed=98, mode=mode, max_seconds=0, replay_path=tmp_path / mode)
    greedy = json.loads((tmp_path / "greedy.json").read_text())
    sample = json.loads((tmp_path / "sample.json").read_text())
    assert greedy["initial_board"] == sample["initial_board"]
    assert greedy["seed_components"] == sample["seed_components"]


def test_replay_rejects_tampered_event_and_summary(tmp_path):
    result = evaluate(policy="random", episodes=1, seed=12, replay_path=tmp_path / "valid.json")
    replay = json.loads((tmp_path / "valid.json").read_text())
    bad_score = deepcopy(replay)
    bad_score["steps"][0]["score"] += 2
    with pytest.raises(ValueError, match="board or score"):
        verify_replay(bad_score)
    bad_spawn = deepcopy(replay)
    bad_spawn["steps"][0]["spawn"]["exponent"] = 7
    with pytest.raises(ValueError, match="invalid spawn"):
        verify_replay(bad_spawn)
    bad_summary = deepcopy(replay)
    bad_summary["result"]["terminated"] = False
    bad_summary["complete"] = False
    with pytest.raises(ValueError, match="final flags"):
        verify_replay(bad_summary)
    assert result["complete"]


@pytest.mark.parametrize("kwargs", [
    {"episodes": 0}, {"episodes": 1.5}, {"episodes": True}, {"seed": -1},
    {"split_id": -1}, {"mode": "invalid"}, {"policy": "invalid"},
    {"max_seconds": -1}, {"max_seconds": float("nan")}, {"max_seconds": float("inf")},
])
def test_invalid_evaluation_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        evaluate(**({"policy": "random"} | kwargs))


def test_model_is_required_for_model_policy():
    with pytest.raises(ValueError, match="model is required"):
        evaluate()


def test_completed_episodes_are_never_reset(monkeypatch):
    module = importlib.import_module("game2048.evaluate")
    class NoResetEnv(VectorEnv):
        def reset_done(self, done):
            raise AssertionError("Evaluation must not reset finished episodes")
    monkeypatch.setattr(module, "VectorEnv", NoResetEnv)
    result = evaluate(policy="random", episodes=6, seed=3)
    assert result["complete"] and result["completed_episodes"] == 6
    assert all(row["episode_id"] == 0 for row in result["episodes"])


def test_partial_report_uses_only_finished_games(monkeypatch):
    module = importlib.import_module("game2048.evaluate")
    instances = []
    class ObservedEnv(VectorEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)
    monkeypatch.setattr(module, "VectorEnv", ObservedEnv)
    result = evaluate(policy="random", episodes=10, seed=44,
                      should_stop=lambda: bool(instances[0]._terminated.any()))
    finished = [row for row in result["episodes"] if row["complete"]]
    partial = [row for row in result["episodes"] if not row["complete"]]
    assert finished and partial
    assert result["completed_episodes"] == len(finished)
    assert result["scores"] == [row["score"] for row in finished]
    assert result["mean_score"] == np.mean([row["score"] for row in finished])
    assert result["mean_length"] == np.mean([row["length"] for row in finished])
    assert sum(result["max_tile_counts"].values()) == len(finished)
    assert result["interrupted"] and not result["complete"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_state_is_preserved():
    model = ActorCritic().to("cuda")
    torch.cuda.manual_seed_all(19)
    states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    result = evaluate(model, device="cuda", episodes=2, seed=12, mode="sample")
    assert result["complete"]
    assert all(torch.equal(before, after) for before, after in zip(states, torch.cuda.get_rng_state_all()))
