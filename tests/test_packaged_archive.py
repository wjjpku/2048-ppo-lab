import json
from pathlib import Path
from game2048.evaluate import verify_replay
from game2048.rewards import merge_output_counts, rewards_from_merge_counts
import numpy as np

def test_full_historical_replays_keep_their_model_and_terminal_data():
    root = Path(__file__).resolve().parents[1] / "web/archive-base3"
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["model"] == "base3@4890" and meta["bank"] == 13
    assert [c["rank"] for c in meta["cases"]] == [1, 10, 50, 90, 100]
    steps = 0
    for case in meta["cases"]:
        replay = json.loads((root / case["replay_url"]).read_text())
        result = verify_replay(replay)
        assert result["valid"] and result["score"] == case["score"]
        assert replay["steps"][-1]["board_after"] == case["final_board"]
        steps += result["steps"]
    assert steps == 7511

def test_reward_anchor_and_raw_score_are_distinct():
    board = np.array([[[9,9,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]], dtype=np.uint8)
    counts = merge_output_counts(board, np.array([3]))
    for base in (2, 3, 4):
        assert rewards_from_merge_counts(counts, base).item() == 1
