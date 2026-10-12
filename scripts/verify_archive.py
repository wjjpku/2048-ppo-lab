"""Recompute every saved game step, RNG spawn, score, and terminal state."""
import hashlib
import json
from pathlib import Path
from game2048.evaluate import verify_replay

def main():
    root = Path(__file__).resolve().parents[1] / "web" / "archive-base3"
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["model"] == "base3@4890" and meta["bank"] == 13
    paths = [root / "replay.json"] + [root / c["replay_url"] for c in meta["cases"]]
    reports = []
    for path in paths:
        replay = json.loads(path.read_text())
        check = verify_replay(replay)
        assert check["valid"] and replay["complete"]
        reports.append({"file": str(path.relative_to(root)), **check})
    for case, audit in zip(meta["cases"], meta["reconstruction"]["cases"]):
        path = root / case["replay_url"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["sha256"]
        replay = json.loads(path.read_text())
        assert replay["result"]["score"] == case["score"]
        assert replay["steps"][-1]["board_after"] == case["final_board"]
    assert sum(r["steps"] for r in reports[1:]) == 7511
    print(json.dumps({"model": meta["model"], "bank": meta["bank"],
                      "five_case_steps": 7511, "all_steps": sum(r["steps"] for r in reports),
                      "all_passed": True, "replays": reports}, indent=2))

if __name__ == "__main__":
    main()
