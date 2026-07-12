"""
Entry point. Model is set via SLURM script.

    # train (single-player, one model)
    python -m score_rl_parallel.main train \
        --model_id meta-llama/Llama-3.1-8B-Instruct

    # evaluate a checkpoint (or omit --adapter for the baseline)
    python -m score_rl_parallel.main eval \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --adapter models/score_rl/final
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from .config import (
    MULTIPLAYER_GAMES,
    SINGLE_PLAYER_GAMES,
    SCoReConfig,
)


def _add_config_args(p: argparse.ArgumentParser):
    """Expose the common config knobs; anything omitted keeps its default."""
    p.add_argument("--model_id", type=str)
    p.add_argument("--model_registry_name", type=str)
    p.add_argument("--games", type=str,
                   help="'single', 'multi', 'all', or comma-separated game names")
    p.add_argument("--instances_name", type=str)
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta1", type=float)
    p.add_argument("--beta2", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--length_norm", choices=["sequence", "constant"])
    p.add_argument("--stage1_epochs", type=int)
    p.add_argument("--stage2_epochs", type=int)
    p.add_argument("--episodes_per_update", type=int)
    p.add_argument("--episodes_per_game_per_epoch", type=int)
    p.add_argument("--val_episodes_per_game", type=int)
    p.add_argument("--episode_timeout_s", type=float,
                   help="per-episode wall-clock guard (covers y1+y2); raise for "
                        "long games like adventuregame")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max_new_tokens", type=int)
    p.add_argument("--max_train_turn_tokens", type=int,
                   help="drop episodes before backward if any scored turn has "
                        "prompt+completion longer than this; 0 disables")
    # Tri-state: unset -> keep model-default template behavior (config None);
    # 'false' is REQUIRED for Qwen3.5, which thinks by default and would otherwise
    # leak <think>...</think> into every answer and abort the game parser.
    p.add_argument("--enable_thinking", choices=["true", "false"], default=None,
                   help="force chat-template thinking on/off; omit for model default")
    p.add_argument("--reward_mode", choices=["success_binary", "bench_normalized"])
    p.add_argument("--fast_inference", action="store_true")
    p.add_argument("--parallel_gpus", type=int,
                   help="number of replicas for parallel episode generation "
                        "(one per GPU; 1 = single-GPU bitsandbytes path). "
                        "Produces ONE model trained on all games.")
    p.add_argument("--rollout_scheduler", choices=["dynamic", "round_robin"],
                   help="parallel rollout assignment strategy; dynamic keeps "
                        "replicas busier when instances have uneven runtimes")
    p.add_argument("--rollout_round_size", type=int,
                   help="parallel path: generate at most this many episodes "
                        "before optimizing/broadcasting; omit for whole-game "
                        "rollout buffers")
    p.add_argument("--multiplayer_teacher",
                   choices=["self_play_all_roles", "self_play_frozen"],
                   help="multiplayer self-play policy; default trains the same "
                        "learner in every player slot")
    p.add_argument("--seed", type=int)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--log_dir", type=str)
    p.add_argument("--run_dir", type=str)


def _resolve_games(value: str):
    if value == "single":
        return SINGLE_PLAYER_GAMES
    if value == "multi":
        return MULTIPLAYER_GAMES
    if value == "all":
        return SINGLE_PLAYER_GAMES + MULTIPLAYER_GAMES
    return tuple(g.strip() for g in value.split(",") if g.strip())


def _build_config(args) -> SCoReConfig:
    valid = {f.name for f in fields(SCoReConfig)}
    overrides = {}
    for k, v in vars(args).items():
        if k in ("cmd", "adapter"):
            continue
        if v is None or v is False:  # unset flags keep defaults
            continue
        if k == "games":
            overrides["games"] = _resolve_games(v)
        elif k == "enable_thinking":
            overrides["enable_thinking"] = (v == "true")
        elif k in valid:
            overrides[k] = v
    return SCoReConfig(**overrides)


def main():
    parser = argparse.ArgumentParser(prog="score_rl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="run SCoRe two-stage training")
    _add_config_args(p_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint (or baseline)")
    _add_config_args(p_eval)
    p_eval.add_argument("--adapter", type=str, default=None,
                        help="path to a saved LoRA adapter; omit for baseline")

    p_merge = sub.add_parser(
        "merge",
        help="merge a trained adapter to full weights + register for `playpen eval`")
    _add_config_args(p_merge)
    p_merge.add_argument("--adapter", type=str, required=True,
                         help="path to the trained LoRA adapter to merge")
    p_merge.add_argument("--registered_name", type=str, required=True,
                         help="model_name to add to model_registry.json")
    p_merge.add_argument("--merged_dir", type=str, default=None,
                         help="output dir for merged weights (default: <adapter>_merged)")
    p_merge.add_argument("--registry_path", type=str, default="model_registry.json")

    args = parser.parse_args()
    cfg = _build_config(args)

    if args.cmd == "train":
        from .trainer import train
        train(cfg)
    elif args.cmd == "eval":
        from .evaluate import evaluate
        evaluate(cfg, adapter_path=args.adapter)
    elif args.cmd == "merge":
        from .merge import merge_and_register
        merge_and_register(
            cfg,
            adapter_path=args.adapter,
            registered_name=args.registered_name,
            merged_dir=args.merged_dir,
            registry_path=args.registry_path,
        )


if __name__ == "__main__":
    main()
