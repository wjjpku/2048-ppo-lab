"""Portable fresh-run entry point around the archived PPO numerical core.

This command never loads the packaged demonstrations automatically. It creates
a new output directory and publishes only complete update barriers.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time
import torch
from .budget import Budget, BudgetExpired
from .config import Config
from .env import VectorEnv
from .model import ActorCritic
from .ppo import make_optimizer, ppo_update
from .rewards import validate_reward_base
from .rollout import collect_rollout
from .runtime import setup

def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)

def train(config: Config, output, *, reward_base=2, device="cpu", max_seconds=60):
    config.validate()
    reward_base = validate_reward_base(reward_base)
    if not 0 < max_seconds < float("inf"):
        raise ValueError("max_seconds must be finite and positive")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    sample_rng, shuffle_rng = setup(config.seed, config.threads, True, device)
    model = ActorCritic(config.hidden_size).to(device)
    optimizer = make_optimizer(model, config)
    env = VectorEnv(config.num_envs, config.seed, split_id=0, max_episode_steps=None)
    budget = Budget(max_seconds, reserve=0)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    iteration = 0
    optimizer_steps = 0
    def save_barrier():
        weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        export = dict(format_version=1, model_id="fresh-run", iteration=iteration,
                      hidden_size=config.hidden_size, reward_base=reward_base, state_dict=weights)
        atomic_save(export, root / "model.pt")
        atomic_save(dict(model=weights, optimizer=optimizer.state_dict(), env=env.state_dict(),
                         config=asdict(config), iteration=iteration,
                         optimizer_steps=optimizer_steps, total_steps=iteration * config.batch_size,
                         reward_base=reward_base, sample_rng=sample_rng.get_state(),
                         shuffle_rng=shuffle_rng.get_state()), root / "latest-training.pt")
    try:
        save_barrier()
        for j in range(config.total_iterations):
            if stopped:
                break
            budget.check()
            rollout = collect_rollout(model, env, config, device, sample_rng, j,
                                      budget=budget, reward_base=reward_base)
            metrics = ppo_update(model, optimizer, rollout, config, shuffle_rng, budget=budget)
            iteration = j + 1
            optimizer_steps += metrics["optimizer_steps"]
            save_barrier()
            row = dict(iteration=iteration, total_steps=iteration * config.batch_size,
                       reward_base=reward_base, **metrics)
            with (root / "updates.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
    except BudgetExpired:
        # A partial update is deliberately not exported; the prior barrier remains.
        stopped = True
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = dict(completed_updates=iteration, requested_updates=config.total_iterations,
                   total_steps=iteration * config.batch_size, optimizer_steps=optimizer_steps,
                   stopped=stopped, device=str(device), elapsed_seconds=budget.elapsed,
                   note="Time checks are cooperative; saving/a current operation may exceed the deadline slightly.")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
