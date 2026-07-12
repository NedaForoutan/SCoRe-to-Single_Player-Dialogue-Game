"""
SCoRe two-attempt rollout on a dialogue game.

An episode plays the SAME instance twice with the SAME learner:
    y1: fresh attempt.
    y2: after a self-correction preamble that shows the model its y1 transcript
        and its outcome, and asks it to do better. The preamble is prepended to
        every y2 turn as prompt context (never scored -- only completion tokens
        enter the policy gradient).

Reward is read from the game outcome. Reward is already normalized
to [0, 1]; the SCoRe shaping bonus is applied later in losses.py.

Infra failures (a hang inside generate, a runner crash) are dropped as `None`,
NOT recorded as aborts -- an abort is a rule violation with learning signal.
"""

from __future__ import annotations

import concurrent.futures
from typing import Optional

from .config import SCoReConfig
from .env import GameHandle, LearnerBackend
from .episode import Attempt, Episode


def _reflection_preamble(y1: Attempt, cfg: SCoReConfig) -> list[dict]:
    """
    Build the self-correction prompt from the y1 transcript. We reconstruct the
    learner's turns from captured tokens (decoded) so the model sees what it did.
    """
    outcome = (
        "You SUCCEEDED in your previous attempt."
        if y1.success
        else ("Your previous attempt was ABORTED due to a rule violation."
              if y1.aborted else "Your previous attempt was UNSUCCESSFUL.")
    )
    instruction = (
        f"{outcome} Below is a summary of what you did last time. Carefully review "
        f"it, identify what went wrong or could be improved, and play better this "
        f"time. A fresh attempt at the same task begins now."
    )
    return [{"role": "user", "content": instruction}]


def collect_episode(
    handle: GameHandle,
    learner: LearnerBackend,
    instance_key: tuple[str, int],
    cfg: SCoReConfig,
    player_models: Optional[list] = None,
) -> Optional[Episode]:
    """
    Play y1 then y2 on `instance_key` (experiment_name, game_id). Returns an
    Episode, or None if either attempt produced no learner tokens (infra drop).
    """
    ep = Episode(game=handle.game_name, game_id=instance_key)

    # y1 -- no prompt
    learner.prefix_messages = []
    ep.y1 = handle.play_attempt(learner, instance_key, player_models=player_models)

    # y2 -- reflection derived from y1
    learner.prefix_messages = _reflection_preamble(ep.y1, cfg)
    ep.y2 = handle.play_attempt(learner, instance_key, player_models=player_models)
    learner.prefix_messages = []

    if not ep.y1.has_tokens() or not ep.y2.has_tokens():
        print(
            f"[drop-detail] {handle.game_name} {instance_key}: "
            f"y1tok={ep.y1.n_completion_tokens()} y1_abort={ep.y1.aborted} "
            f"y1_reward={ep.y1.reward}; "
            f"y2tok={ep.y2.n_completion_tokens()} y2_abort={ep.y2.aborted} "
            f"y2_reward={ep.y2.reward}",
            flush=True,
        )
        return None
    return ep


def collect_episode_guarded(
    handle: GameHandle,
    learner: LearnerBackend,
    instance_key: tuple[str, int],
    cfg: SCoReConfig,
    player_models: Optional[list] = None,
) -> Optional[Episode]:
    """
    Wall-clock guard around one episode. If generation hangs past
    cfg.episode_timeout_s, abandon the worker thread and drop the episode.

    Diagnosis for dropped episodes from timestamps.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(
            collect_episode, handle, learner, instance_key, cfg, player_models
        )
        try:
            ep = fut.result(timeout=cfg.episode_timeout_s)
            if ep is None:
                print(f"[drop] {handle.game_name} {instance_key}: no learner "
                      f"tokens in y1 or y2", flush=True)
            return ep
        except concurrent.futures.TimeoutError:
            print(f"[drop] {handle.game_name} {instance_key}: timeout "
                  f"(> {cfg.episode_timeout_s:.0f}s)", flush=True)
            return None
        except Exception as e:
            print(f"[drop] {handle.game_name} {instance_key}: {type(e).__name__}: "
                  f"{e}", flush=True)
            return None
