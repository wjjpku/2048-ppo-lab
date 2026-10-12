import argparse
import json
from pathlib import Path
import torch
from .config import Config
from .evaluate import evaluate, verify_replay
from .inference import load_model, model_manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description="2048 PPO Lab — CPU demos and complete replays")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list bundled fixed inference models")
    for name in ("demo", "evaluate"):
        p = sub.add_parser(name, help="run frozen inference; never updates model weights")
        p.add_argument("--model", default="exp4to8-7640", choices=[r["id"] for r in model_manifest()["models"]])
        p.add_argument("--weights", type=Path, help="optional custom tensor-only inference export")
        p.add_argument("--episodes", type=int, default=1)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--bank", type=int, default=16)
        p.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
        p.add_argument("--device", default="cpu")
        p.add_argument("--max-seconds", type=float, default=60)
        p.add_argument("--output", type=Path, default=Path("demo-output"))
    p = sub.add_parser("verify-replay")
    p.add_argument("path", type=Path)
    p = sub.add_parser("train", help="explicit fresh training, bounded by updates and elapsed time")
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reward-base", type=float, choices=(2, 3, 3.3, 3.5, 4), default=2)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "models":
        result = model_manifest()
    elif args.command == "verify-replay":
        result = verify_replay(args.path)
    elif args.command == "train":
        from .train import train
        c = Config(seed=args.seed, hidden_size=args.hidden_size, num_envs=args.num_envs,
                   rollout_steps=args.rollout_steps, minibatch_size=args.minibatch_size,
                   total_iterations=args.updates)
        result = train(c, args.output, reward_base=args.reward_base, device=args.device,
                       max_seconds=args.max_seconds)
    else:
        model, info = load_model(args.model, weights=args.weights, device=args.device)
        args.output.mkdir(parents=True, exist_ok=False)
        result = evaluate(model, device=args.device, episodes=args.episodes, seed=args.seed,
                          split_id=args.bank, mode=args.mode, max_seconds=args.max_seconds,
                          replay_path=args.output / "replay.json")
        result["fixed_model"] = info
        result["purpose"] = "local demonstration; reusing a bank is not new independent evidence"
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        result = {k: result[k] for k in ("complete", "completed_episodes", "mean_score", "max_tile_counts", "fixed_model", "purpose")}
        result["replay"] = str(args.output / "replay.html")
    print(json.dumps(result, indent=2, ensure_ascii=False))
