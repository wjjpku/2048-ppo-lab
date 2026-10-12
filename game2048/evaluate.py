"""Frozen, RNG-isolated evaluation and portable, verifiable 2048 replays.

Split 0 is reserved for training; split 1 is the fixed validation bank and
split 2 is the held-out test bank.  Each evaluation starts one fresh episode
per environment, never resets a finished episode, and never counts a partial
episode as a completed score.  A wall-clock limit is a *soft* deadline checked
between vector steps; standard evaluation has no episode-length limit.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch

from .env import VectorEnv, deterministic_move
from .model import encode_boards, masked_log_probs, sample_actions

VALIDATION_SPLIT = 1
TEST_SPLIT = 2
ACTION_NAMES = ("up", "right", "down", "left")
_ACTION_STREAM = 0x4556414C


@contextmanager
def _isolated_inference(model: torch.nn.Module | None) -> Iterator[None]:
    """Restore global RNGs and every module's original train/eval flag on exit."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    training_flags = [(module, module.training) for module in model.modules()] if model is not None else []
    # Do not initialize CUDA (or touch its generators) for a CPU-only run.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            if model is not None:
                model.eval()
            yield
    finally:
        for module, training in training_flags:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _action_seed(seed: int, split_id: int) -> int:
    state = np.random.SeedSequence([seed, split_id, _ACTION_STREAM]).generate_state(1, dtype=np.uint64)
    return int(state[0]) & ((1 << 63) - 1)


def _spawn_record(position: Any, exponent: Any) -> dict[str, int] | None:
    exponent = int(exponent)
    if exponent <= 0:
        return None
    coordinates = np.asarray(position)
    if coordinates.ndim == 0:
        index = int(coordinates)
        row, column = divmod(index, 4)
    elif coordinates.shape == (2,):
        row, column = (int(x) for x in coordinates)
        index = 4 * row + column
    else:
        raise ValueError(f"Unexpected spawn position shape: {coordinates.shape}")
    if not 0 <= row < 4 or not 0 <= column < 4 or exponent not in (1, 2):
        raise ValueError("Invalid spawn returned by environment")
    return {"position": index, "row": row, "column": column, "exponent": exponent, "tile": 1 << exponent}


def _episode_row(env: VectorEnv, index: int, terminated: bool, truncated: bool) -> dict[str, Any]:
    maximum = int(np.asarray(env.boards[index]).max())
    return {
        "env_id": index,
        "episode_id": int(env.episode_ids[index]),
        "score": int(env.scores[index]),
        "length": int(env.lengths[index]),
        "max_exponent": maximum,
        "max_tile": (1 << maximum) if maximum else 0,
        "reached_2048": maximum >= 11,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "complete": bool(terminated),
        "status": "complete" if terminated else ("truncated" if truncated else "incomplete"),
        "final_board": np.asarray(env.boards[index]).tolist(),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["complete"]]
    scores = np.asarray([row["score"] for row in completed], dtype=np.int64)
    lengths = np.asarray([row["length"] for row in completed], dtype=np.int64)
    count = len(completed)
    maximum_tiles = Counter(str(row["max_tile"]) for row in completed)
    thresholds = [1 << exponent for exponent in range(7, 21)]
    rates = {
        str(tile): sum(row["max_tile"] >= tile for row in completed) / count if count else None
        for tile in thresholds
    }
    return {
        "requested_episodes": len(rows),
        "completed_episodes": count,
        "incomplete_episodes": len(rows) - count,
        "complete": count == len(rows),
        "scores": scores.tolist(),
        "mean_score": float(scores.mean()) if count else None,
        "median_score": float(np.median(scores)) if count else None,
        "std_score": float(scores.std(ddof=0)) if count else None,
        "min_score": int(scores.min()) if count else None,
        "max_score": int(scores.max()) if count else None,
        "score_quantiles": {
            str(percentile): float(np.percentile(scores, percentile)) if count else None
            for percentile in (0, 10, 25, 50, 75, 90, 100)
        },
        "max_tile_counts": dict(sorted(maximum_tiles.items(), key=lambda pair: int(pair[0]))),
        "reached_tile_rates": rates,
        "win_rate_2048": rates["2048"],
        "mean_length": float(lengths.mean()) if count else None,
        "median_length": float(np.median(lengths)) if count else None,
        "total_steps": sum(row["length"] for row in rows),
        "episodes": rows,
    }


def evaluate(
    model: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    episodes: int = 50,
    seed: int = 1,
    split_id: int = VALIDATION_SPLIT,
    mode: str = "greedy",
    policy: str = "model",
    max_seconds: float | None = None,
    replay_path: str | os.PathLike[str] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate independent fresh games and optionally save environment 0's replay.

    ``policy`` is ``model``, uniform-legal ``random``, or immediate-reward
    ``merge``.  Model mode is ``greedy`` (ties use up/right/down/left order) or
    ``sample``.  Baselines use their own natural action selection irrespective
    of ``mode``.  Sampling owns a separate generator and all global Python,
    NumPy, and initialized PyTorch RNG states are restored even on failure.

    The caller places the model on ``device``; evaluation does not move or
    update parameters, optimizer state, training environments, or their RNGs.
    ``should_stop`` may expose a trainer signal flag; it is polled between
    vector steps. Returned score statistics use only naturally terminated episodes.  All
    requested episodes, including partial ones, have an explicit raw row.
    ``split_id=2`` is intended only for final held-out reporting, never model
    selection.  The same seed/split pair shares environment seed identities
    between modes, but different actions need not see the same later boards.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    for name, value in (("seed", seed), ("split_id", split_id)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if split_id == 0:
        raise ValueError("split_id=0 is reserved for training; use 1 for validation or 2 for test")
    if mode in ("sampling", "stochastic"):
        mode = "sample"
    if mode not in ("greedy", "sample"):
        raise ValueError("mode must be 'greedy' or 'sample'")
    if policy not in ("model", "random", "merge"):
        raise ValueError("policy must be 'model', 'random', or 'merge'")
    if policy == "model" and model is None:
        raise ValueError("model is required for policy='model'")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("max_seconds must be finite and nonnegative, or None")

    started = time.perf_counter()
    target_device = torch.device(device)
    replay: dict[str, Any] | None = None
    interruption: str | None = None
    with _isolated_inference(model if policy == "model" else None):
        env = VectorEnv(int(episodes), seed=int(seed), split_id=int(split_id), max_episode_steps=None)
        active = np.ones(episodes, dtype=bool)
        terminated = np.zeros(episodes, dtype=bool)
        truncated = np.zeros(episodes, dtype=bool)
        sample_generator = torch.Generator(device=target_device).manual_seed(_action_seed(int(seed), int(split_id))) if policy == "model" else None
        random_generators = [
            np.random.default_rng(np.random.SeedSequence([int(seed), int(split_id), i, _ACTION_STREAM]))
            for i in range(episodes)
        ] if policy == "random" else []
        metadata = {
            "seed": int(seed), "split_id": int(split_id),
            "split": "validation" if int(split_id)==VALIDATION_SPLIT else "test",
            "test_bank": int(split_id) if int(split_id)>=2 else None,
            "mode": mode, "policy": policy,
            "action_selection": mode if policy == "model" else ("uniform_legal" if policy == "random" else "immediate_merge_reward"),
            "max_episode_steps": None,
        }
        if replay_path is not None:
            replay = {
                "schema_version": 1, "game": "2048", "board_representation": "exponents",
                "action_names": list(ACTION_NAMES), "evaluation": metadata.copy(),
                "env_id": 0, "episode_id": int(env.episode_ids[0]),
                "seed_components": [int(seed), int(split_id), 0, int(env.episode_ids[0])],
                "seed_scheme": "numpy.random.SeedSequence([root_seed, split_id, env_id, episode_id])",
                "initial_board": np.asarray(env.boards[0]).copy().tolist(),
                "initial_mask": np.asarray(env.masks[0], dtype=bool).tolist(),
                "initial_score": int(env.scores[0]), "steps": [],
            }
        try:
            while active.any():
                if should_stop is not None and should_stop():
                    interruption = "stop_requested"
                    break
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    interruption = "max_seconds"
                    break
                ids = np.flatnonzero(active)
                boards = np.asarray(env.boards)[ids].copy()
                masks = np.asarray(env.masks, dtype=bool)[ids].copy()
                if not masks.any(axis=1).all():
                    raise RuntimeError("Active evaluation environment has no legal action")
                values: np.ndarray | None = None
                if policy == "model":
                    observations = encode_boards(torch.as_tensor(boards, device=target_device))
                    logits, tensor_values = model(observations)
                    if not torch.isfinite(logits).all() or not torch.isfinite(tensor_values).all():
                        raise FloatingPointError("Non-finite evaluation policy logits or values")
                    log_probs = masked_log_probs(logits, torch.as_tensor(masks, device=target_device))
                    selected = torch.argmax(log_probs, dim=-1) if mode == "greedy" else sample_actions(log_probs, sample_generator)
                    selected_actions = selected.detach().cpu().numpy().astype(np.int64)
                    probabilities = log_probs.exp().detach().cpu().numpy()
                    values = tensor_values.detach().cpu().numpy().reshape(-1)
                elif policy == "random":
                    probabilities = masks.astype(np.float64) / masks.sum(axis=1, keepdims=True)
                    selected_actions = np.asarray([
                        random_generators[int(env_id)].choice(np.flatnonzero(masks[row]))
                        for row, env_id in enumerate(ids)
                    ], dtype=np.int64)
                else:
                    selected_actions = np.asarray([
                        max(np.flatnonzero(mask), key=lambda action: deterministic_move(board, int(action))[1])
                        for board, mask in zip(boards, masks)
                    ], dtype=np.int64)
                    probabilities = np.zeros((len(ids), 4), dtype=np.float64)
                    probabilities[np.arange(len(ids)), selected_actions] = 1.0
                if selected_actions.shape != (len(ids),) or not masks[np.arange(len(ids)), selected_actions].all():
                    raise RuntimeError("Evaluation policy selected an illegal action")
                actions = np.full(episodes, -1, dtype=np.int64)
                actions[ids] = selected_actions
                result = env.step(actions)
                just_terminated = np.asarray(result["terminated"], dtype=bool)
                just_truncated = np.asarray(result["truncated"], dtype=bool)
                if replay is not None and active[0]:
                    # Active indices are sorted, so env 0 is the first row until done.
                    action = int(selected_actions[0])
                    replay["steps"].append({
                        "step": int(env.lengths[0]), "board_before": boards[0].tolist(),
                        "legal_mask": masks[0].tolist(), "action": action,
                        "action_name": ACTION_NAMES[action], "probabilities": probabilities[0].tolist(),
                        "chosen_probability": float(probabilities[0, action]),
                        "value": float(values[0]) if values is not None else None,
                        "raw_reward": int(result["rewards"][0]), "score": int(env.scores[0]),
                        "spawn": _spawn_record(result["spawn_positions"][0], result["spawn_exponents"][0]),
                        "board_after": np.asarray(result["boards"][0]).copy().tolist(),
                        "terminated": bool(just_terminated[0]), "truncated": bool(just_truncated[0]),
                    })
                terminated |= active & just_terminated
                truncated |= active & just_truncated
                active &= ~(just_terminated | just_truncated)
        except KeyboardInterrupt:
            interruption = "keyboard_interrupt"
        rows = [_episode_row(env, i, bool(terminated[i]), bool(truncated[i])) for i in range(episodes)]
        summary = _summarize(rows)
        if not summary["complete"] and interruption is None:
            interruption = "environment_truncated"
        summary.update(metadata)
        summary.update({
            "interrupted": interruption is not None,
            "interruption_reason": interruption,
            "max_seconds": max_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "replay_json": None, "replay_html": None,
        })
        if replay is not None:
            replay.update({"result": rows[0], "complete": rows[0]["complete"],
                           "interrupted": not rows[0]["complete"],
                           "interruption_reason": interruption if not rows[0]["complete"] else None})
            json_path, html_path = write_replay(replay, replay_path)
            summary["replay_json"] = str(json_path)
            summary["replay_html"] = str(html_path)
    # Include replay verification and serialization in the evaluation budget.
    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary


def verify_replay(replay: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    """Recompute every move/spawn and fail on any board, score, or flag mismatch.

    Uses only recorded events, so it verifies partial replays and is independent
    of a model, PyTorch device, and RNG implementation/version.
    """
    if isinstance(replay, (str, os.PathLike)):
        replay = json.loads(Path(replay).read_text(encoding="utf-8"))
    if replay.get("schema_version") != 1 or replay.get("board_representation") != "exponents":
        raise ValueError("Unsupported replay schema")
    board = np.asarray(replay["initial_board"])
    if board.shape != (4, 4) or not np.issubdtype(board.dtype, np.integer) or np.any(board < 0) or np.any(board > 20):
        raise ValueError("Invalid initial board")
    board = board.astype(np.uint8)
    score = int(replay.get("initial_score", 0))
    finished = False
    for number, step in enumerate(replay["steps"], start=1):
        prefix = f"Replay step {number}: "
        if finished:
            raise ValueError(prefix + "transition after episode ended")
        if step["step"] != number or not np.array_equal(board, np.asarray(step["board_before"])):
            raise ValueError(prefix + "previous board or step number mismatch")
        mask = np.asarray([deterministic_move(board, action)[2] for action in range(4)], dtype=bool)
        if not np.array_equal(mask, np.asarray(step["legal_mask"], dtype=bool)):
            raise ValueError(prefix + "legal mask mismatch")
        action = int(step["action"])
        if action not in range(4) or not mask[action]:
            raise ValueError(prefix + "illegal action")
        probabilities = np.asarray(step["probabilities"], dtype=np.float64)
        if (probabilities.shape != (4,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6)
                or np.any(probabilities[~mask] != 0)
                or not np.isclose(probabilities[action], step["chosen_probability"], atol=1e-6)):
            raise ValueError(prefix + "invalid action probabilities")
        moved, reward, changed = deterministic_move(board, action)
        if not changed or int(reward) != int(step["raw_reward"]):
            raise ValueError(prefix + "merge reward mismatch")
        spawn = step["spawn"]
        if spawn is None:
            raise ValueError(prefix + "valid move is missing its spawn")
        row, column, exponent = int(spawn["row"]), int(spawn["column"]), int(spawn["exponent"])
        if (not 0 <= row < 4 or not 0 <= column < 4 or exponent not in (1, 2)
                or moved[row, column] != 0 or spawn["position"] != row * 4 + column
                or spawn["tile"] != 1 << exponent):
            raise ValueError(prefix + "invalid spawn")
        board = np.asarray(moved).copy()
        board[row, column] = exponent
        score += int(reward)
        actual_terminal = not any(deterministic_move(board, action)[2] for action in range(4))
        if not np.array_equal(board, np.asarray(step["board_after"])) or score != step["score"]:
            raise ValueError(prefix + "resulting board or score mismatch")
        if actual_terminal != bool(step["terminated"]):
            raise ValueError(prefix + "termination mismatch")
        finished = actual_terminal or bool(step["truncated"])
    result = replay["result"]
    final_terminal = not any(deterministic_move(board, action)[2] for action in range(4))
    final_truncated = bool(replay["steps"][-1]["truncated"]) if replay["steps"] else False
    if (bool(result["terminated"]) != final_terminal
            or bool(result["truncated"]) != final_truncated
            or bool(result["complete"]) != final_terminal
            or int(result["max_exponent"]) != int(board.max())
            or int(result["max_tile"]) != ((1 << int(board.max())) if board.max() else 0)):
        raise ValueError("Replay final flags or maximum tile mismatch")
    if (score != result["score"] or len(replay["steps"]) != result["length"]
            or not np.array_equal(board, np.asarray(result["final_board"]))
            or bool(replay["complete"]) != bool(result["terminated"])):
        raise ValueError("Replay final summary mismatch")
    return {"valid": True, "steps": len(replay["steps"]), "score": score, "complete": bool(replay["complete"])}


def write_replay(replay: Mapping[str, Any], replay_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Validate and write UTF-8 JSON plus a dependency-free HTML replay viewer."""
    verify_replay(replay)
    path = Path(replay_path)
    if path.suffix.lower() in (".html", ".htm"):
        path = path.with_suffix(".json")
    elif path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    html_path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(replay, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for destination, content in ((path, payload + "\n"), (html_path, _replay_html(payload))):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    return path, html_path


def _replay_html(payload: str) -> str:
    # Escaping '<' prevents a metadata string from closing the JSON script tag.
    safe_payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    return _HTML_TEMPLATE.replace("__REPLAY_JSON__", safe_payload)


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>2048 policy replay</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;background:#faf8ef;color:#544c44}
body{max-width:780px;margin:2rem auto;padding:0 1rem}h1{margin-bottom:.3rem;font-size:2.4rem}
.sub{color:#756d65;margin:.4rem 0 1.2rem}.layout{display:flex;gap:1.5rem;align-items:start;flex-wrap:wrap}
#board{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;padding:9px;background:#b9aa9d;border-radius:10px;width:340px;max-width:calc(100vw - 50px)}
.tile{aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#cfc2b5;border-radius:6px;font-size:30px;font-weight:750;color:#665b50}
.tile.spawn{outline:3px solid #356fba;outline-offset:-3px}.stats{min-width:230px;flex:1}.score{font-size:1.5rem;font-weight:700}
.controls{display:flex;gap:.6rem;align-items:center;margin:1.2rem 0;flex-wrap:wrap}button,select{background:#eee4d7;border:1px solid #b5a495;border-radius:5px;padding:.55rem .8rem;color:#544c44;font:inherit;cursor:pointer}button:focus-visible{outline:3px solid #356fba}
#seek{width:100%;accent-color:#8f7965}.prob{display:grid;grid-template-columns:50px 1fr 52px;gap:8px;align-items:center;margin:.5rem 0}.track{background:#e6ded3;height:12px;border-radius:6px;overflow:hidden}.fill{height:100%;background:#aa9782}.chosen .fill{background:#356fba}.chosen{font-weight:700}.muted{color:#837a72;font-size:.9rem}.badge{display:inline-block;padding:.3rem .6rem;border-radius:5px;background:#e5eddf}.partial{background:#fff0c6}#value{margin-top:1rem}footer{border-top:1px solid #d7cfc5;margin-top:1.8rem;padding-top:.8rem;line-height:1.5;font-size:.85rem;color:#837a72}
</style>
<h1>2048 policy replay</h1><p class="sub" id="meta"></p>
<div class="layout"><div id="board" role="img" aria-label="2048 board"></div><div class="stats"><div class="score" id="score"></div><p id="move"></p><p id="status"></p><div id="probabilities"></div><p id="value"></p><p class="muted" id="spawn"></p></div></div>
<div class="controls"><button id="first" aria-label="First move">First</button><button id="previous" aria-label="Previous move">Previous</button><button id="play">Play</button><button id="next" aria-label="Next move">Next</button><button id="last" aria-label="Last move">Last</button><label>Speed <select id="speed"><option value="650">Slow</option><option value="250" selected>Normal</option><option value="70">Fast</option></select></label></div>
<input id="seek" type="range" min="0" value="0" aria-label="Replay move"><p class="muted" id="position"></p>
<footer>Tile labels are powers of two; blue outlines mark the newly spawned tile. Probabilities and value describe the decision immediately before the displayed move. Value uses the model's discounted reward/1024 units. This replay is one episode, not an estimate of overall performance. No network connection is needed.</footer>
<script id="replay-data" type="application/json">__REPLAY_JSON__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('replay-data').textContent),steps=data.steps;
const get=id=>document.getElementById(id),seek=get('seek');let position=0,timer=null;
seek.max=steps.length;
get('meta').textContent=`${data.evaluation.policy} · ${data.evaluation.action_selection} · ${data.evaluation.split} seed ${data.evaluation.seed} · episode ${data.episode_id}`;
const colors=['#cfc2b5','#eee4da','#ede0c8','#f2b179','#f59563','#f67c5f','#f65e3b','#edcf72','#edcc61','#edc850','#edc53f','#edc22e'];
function stop(){if(timer!==null)clearInterval(timer);timer=null;get('play').textContent='Play';}
function render(){const step=position?steps[position-1]:null,board=step?step.board_after:data.initial_board;get('board').replaceChildren();board.flat().forEach((exp,index)=>{const tile=document.createElement('div');tile.className='tile'+(step&&step.spawn&&step.spawn.position===index?' spawn':'');tile.textContent=exp?String(2**exp):'';tile.style.background=colors[Math.min(exp,11)];if(exp>2)tile.style.color='#fff8ef';if(exp>9)tile.style.fontSize='23px';if(exp>13)tile.style.fontSize='18px';get('board').append(tile);});get('score').textContent=`Score ${(step?step.score:data.initial_score).toLocaleString()}`;get('move').textContent=step?`${step.action_name.toUpperCase()} · +${step.raw_reward}`:'Initial board';get('status').textContent=data.complete?'Complete episode':`Incomplete episode · ${data.interruption_reason||'interrupted'}`;get('status').className='badge'+(data.complete?'':' partial');get('probabilities').replaceChildren();data.action_names.forEach((name,index)=>{const row=document.createElement('div');row.className='prob'+(step&&step.action===index?' chosen':'');const label=document.createElement('span');label.textContent=name;const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=step?`${100*step.probabilities[index]}%`:'0%';track.append(fill);const number=document.createElement('span');number.textContent=step?`${(100*step.probabilities[index]).toFixed(1)}%`:'—';row.append(label,track,number);get('probabilities').append(row);});get('value').textContent=step&&step.value!==null?`V(s) ${step.value.toFixed(4)}`:'V(s) unavailable';get('spawn').textContent=step&&step.spawn?`Spawned ${step.spawn.tile} at row ${step.spawn.row+1}, column ${step.spawn.column+1}`:'';seek.value=position;get('position').textContent=`Move ${position} of ${steps.length} · use left/right arrow keys to step`;get('previous').disabled=position===0;get('next').disabled=position===steps.length;}
function jump(value){position=Math.max(0,Math.min(steps.length,value));render();}
get('first').onclick=()=>{stop();jump(0);};get('last').onclick=()=>{stop();jump(steps.length);};get('previous').onclick=()=>{stop();jump(position-1);};get('next').onclick=()=>{stop();jump(position+1);};seek.oninput=()=>{stop();jump(Number(seek.value));};get('play').onclick=()=>{if(timer!==null){stop();return;}if(position===steps.length)jump(0);if(!steps.length)return;get('play').textContent='Pause';timer=setInterval(()=>{jump(position+1);if(position===steps.length)stop();},Number(get('speed').value));};get('speed').onchange=()=>{if(timer!==null){stop();get('play').click();}};document.addEventListener('keydown',event=>{if(event.target.tagName==='INPUT'||event.target.tagName==='SELECT')return;if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();stop();jump(position+(event.key==='ArrowRight'?1:-1));}});render();
</script></html>'''
