"""
Evaluation -- Frozen weights, no gradients.

Reports the metrics that check whether SCoRe is working, per game and aggregate:
    win_rate (y1, y2), abort_rate, mean_reward,
    self_correction_rate  = P(y2 wins | y1 did not win)   <- the SCoRe signal
    regression_rate       = P(y2 loses | y1 won)          <- the SCoRe failure mode

Run against a checkpoint adapter to track progress, or with adapter_path=None to
reproduce the pre-training baseline for comparison.
"""

from __future__ import annotations

from statistics import mean
from typing import Optional

from .config import SCoReConfig
from .env import GameHandle, LearnerBackend
from .model import load_for_eval
from .rollout import collect_episode_guarded


def _aggregate(eps: list) -> dict:
    if not eps:
        return {"n": 0}
    y1_fail = [e for e in eps if not e.y1.success]
    y1_win = [e for e in eps if e.y1.success]
    return {
        "n": len(eps),
        "win_rate_y1": mean(float(e.y1.success) for e in eps),
        "win_rate_y2": mean(float(e.y2.success) for e in eps),
        "abort_rate_y1": mean(float(e.y1.aborted) for e in eps),
        "abort_rate_y2": mean(float(e.y2.aborted) for e in eps),
        "mean_reward_y1": mean(e.y1.reward for e in eps),
        "mean_reward_y2": mean(e.y2.reward for e in eps),
        # conditional rates (None when denominator empty)
        "self_correction_rate": (mean(float(e.y2.success) for e in y1_fail)
                                 if y1_fail else None),
        "regression_rate": (mean(float(not e.y2.success) for e in y1_win)
                            if y1_win else None),
    }


def evaluate(cfg: SCoReConfig, adapter_path: Optional[str] = None) -> dict:
    model, tokenizer = load_for_eval(cfg, adapter_path)
    learner = LearnerBackend(model, tokenizer, cfg)
    model.eval()

    per_game, all_eps = {}, []
    for game in cfg.games:
        try:
            handle = GameHandle(game, cfg)
        except Exception as e:
            print(f"[eval][error] load {game}: {e!r}")
            continue
        with handle:
            ids = handle.instance_keys[: cfg.val_episodes_per_game]
            eps = []
            for gid in ids:
                ep = collect_episode_guarded(handle, learner, gid, cfg)
                if ep is not None:
                    eps.append(ep)
            per_game[game] = _aggregate(eps)
            all_eps.extend(eps)
        print(f"[eval] {game}: {per_game[game]}")

    overall = _aggregate(all_eps)
    result = {"overall": overall, "per_game": per_game,
              "adapter": adapter_path or "(baseline)"}
    print(f"[eval] OVERALL: {overall}")
    return result
