import hashlib
import json
from pathlib import Path
import pytest
import torch
from game2048.inference import MODEL_DIR, load_model, model_manifest, predict
from game2048.model import encode_boards

@pytest.mark.parametrize("model_id", ["base4-5840", "exp4to8-7640"])
def test_frozen_export_hash_contents_and_reference_forward(model_id):
    torch.set_num_threads(1)
    record = next(r for r in model_manifest()["models"] if r["id"] == model_id)
    path = MODEL_DIR / record["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert not ({"optimizer", "env", "rng", "sample_rng", "shuffle_rng"} & set(blob))
    model, info = load_model(model_id)
    assert sum(p.numel() for p in model.parameters()) == 977605
    fixture = json.loads((MODEL_DIR / "parity-fixtures.json").read_text())[model_id]
    with torch.inference_mode():
        logits, values = model(encode_boards(torch.tensor(fixture["boards"], dtype=torch.uint8)))
    torch.testing.assert_close(logits, torch.tensor(fixture["logits"]), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(values, torch.tensor(fixture["values"]), atol=2e-5, rtol=2e-4)
    assert info["iteration"] == record["iteration"]

def test_predict_handles_terminal_and_rejects_non_exponents():
    model, _ = load_model("base4-5840")
    board = [[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]]
    assert predict(model, board)["terminal"]
    with pytest.raises((ValueError, TypeError)):
        predict(model, [[2048] * 4] * 4)
    with pytest.raises(TypeError):
        predict(model, [[1.5] * 4] * 4)
