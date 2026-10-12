"""Load packaged, hash-checked inference exports without pickle globals."""
import hashlib
import json
from pathlib import Path
import torch
from .model import ActorCritic, encode_boards, masked_log_probs
from .env import legal_mask

MODEL_DIR = Path(__file__).parent / "assets" / "models"

def model_manifest():
    return json.loads((MODEL_DIR / "manifest.json").read_text())

def load_model(model_id=None, *, weights=None, device="cpu"):
    catalog = model_manifest()
    if weights is None:
        model_id = model_id or catalog["default"]
        record = next((r for r in catalog["models"] if r["id"] == model_id), None)
        if record is None:
            raise ValueError(f"unknown model: {model_id}")
        path = MODEL_DIR / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("model SHA256 differs from the packaged manifest")
    else:
        path = Path(weights)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format_version") != 1 or set(state) != {
        "format_version", "model_id", "iteration", "hidden_size", "reward_base", "state_dict"
    }:
        raise ValueError("expected a tensor-only inference export, not a training checkpoint")
    model = ActorCritic(state["hidden_size"])
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    info = {k: v for k, v in state.items() if k != "state_dict"}
    return model, info

@torch.inference_mode()
def predict(model, board):
    """Return probabilities and value; board entries are exponents, not tile values."""
    device = next(model.parameters()).device
    b = torch.as_tensor(board, device=device)
    if b.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("board exponents must be integers")
    if tuple(b.shape) != (4, 4):
        raise ValueError("board must have shape 4x4")
    observation = encode_boards(b[None])  # validate range before uint8 conversion
    mask = torch.as_tensor(legal_mask(b.cpu().numpy()), device=device)
    if not mask.any():
        return {"terminal": True, "action": None, "probabilities": [0.0] * 4, "value": None}
    logits, value = model(observation)
    lp = masked_log_probs(logits, mask[None])
    return {"terminal": False, "action": int(lp.argmax(-1)),
            "probabilities": lp.exp()[0].cpu().tolist(), "value": float(value[0])}
