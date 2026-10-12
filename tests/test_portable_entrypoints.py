from dataclasses import replace
import json
from unittest.mock import patch
import torch
import pytest
from game2048.budget import BudgetExpired
from game2048.config import Config
from game2048.inference import load_model
from game2048.train import train

def test_deadline_keeps_previous_complete_export(tmp_path):
    c = replace(Config(), hidden_size=8, num_envs=2, rollout_steps=2, minibatch_size=2, total_iterations=1)
    def expire(*args, **kwargs):
        # Model state could be partially changed by an interrupted update.
        with torch.no_grad():
            next(args[0].parameters()).fill_(123)
        raise BudgetExpired()
    with patch("game2048.train.collect_rollout", side_effect=expire):
        result = train(c, tmp_path / "run", max_seconds=20)
    assert result["completed_updates"] == 0 and result["stopped"]
    model, info = load_model(weights=tmp_path / "run/model.pt")
    assert info["iteration"] == 0
    assert not torch.all(next(model.parameters()) == 123)
    with pytest.raises(FileExistsError):
        train(c, tmp_path / "run", max_seconds=20)

def test_cli_training_routes_only_after_explicit_subcommand(tmp_path):
    from game2048.cli import main
    with patch("game2048.train.train", return_value={"mock_test": True}) as called:
        main(["models"])
        called.assert_not_called()
        main(["train", "--updates", "1", "--max-seconds", "1", "--output", str(tmp_path / "fresh")])
        assert called.call_count == 1
