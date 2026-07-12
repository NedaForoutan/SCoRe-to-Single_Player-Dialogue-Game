"""
Central configuration for SCoRe-on-Playpen training.

Everything model-specific is a field here so a SLURM job only has to override
`--model_id` to swap models. The defaults match the SCoRe reference implementation except
where the dialogue-game setting demands a change.

Design decisions :-
  * Algorithm  : SCoRe two-stage REINFORCE. alpha=10, beta1=0.01, beta2=0.1.
  * Reference  : same weights via `model.disable_adapter()` -> no second model in VRAM.
  * Games      : in-process clemcore GameBenchmark (no clem serve, no OpenEnv server).
  * Stability  : batched updates + length normalization + per-game reward -> [0,1].
  * Speed      : plain transformers + bitsandbytes replicas for rollout
                 generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional



# Single-player only, one model, no teacher.
SINGLE_PLAYER_GAMES: tuple[str, ...] = (
    "textmapworld",
    "textmapworld_graphreasoning",
    "textmapworld_specificroom",
    "wordle",
    "adventuregame",
)

# Multiplayer games use multiple player slots. The default training setup is self-play with the same learner policy occupying every slot, so one adapter learns every role the game asks it to play.
MULTIPLAYER_GAMES: tuple[str, ...] = (
    "codenames",
    "dond",
    "guesswhat",
    "hot_air_balloon",
    "imagegame",
    "matchit_ascii",
    "privateshared",
    "referencegame",
    "taboo",
    "wordle_withcritic",
    "wordle_withclue",
)



# Reward is normalized to [0, 1] before the SCoRe shaping bonus is applied. Games have different scales, and alpha=10 * (r2 - r1) produced gradients that differ by 1000x across games.
# Normalizing first makes alpha=10 safe and comparable everywhere and does  not blow up the gradients.
# Two modes:
#   "success_binary"  -> reward = 1.0 if the game reports SUCCESS else 0.0,
#                        EXCEPT abort -> abort_reward (default -1.0). Lose == 0; abort is penalized so cold-start models get a signal.
#   "bench_normalized"-> reward = clip(BENCH_SCORE / bench_max[game], 0, 1). Relies on the per-game maxima below being correct.
REWARD_MODE_SUCCESS_BINARY = "success_binary"
REWARD_MODE_BENCH_NORMALIZED = "bench_normalized"

# Per-game BENCH_SCORE maxima. Aborts are NaN and always map to `abort_reward`; lose maps to 0. 
    "wordle": 100.0,            # SPEED_SCORES max
    "wordle_withclue": 100.0,
    "wordle_withcritic": 100.0,
    "taboo": 100.0,
    "referencegame": 100.0,
    "imagegame": 100.0,
    # single-player defaults
    "adventuregame": 100.0,
    "textmapworld": 100.0,
    "textmapworld_graphreasoning": 100.0,
    "textmapworld_specificroom": 100.0,
}


@dataclass
class SCoReConfig:
    # model can be modifies in SLURM.
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Name the game registry uses for this model as an opponent/teacher. Falls
    model_registry_name: Optional[str] = None
    # lora  configs
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ScoRe configs
    alpha: float = 10.0   # Stage II self-correction bonus: r2 + alpha*(r2 - r1)
    beta1: float = 0.01   # Stage II KL weight (both attempts)
    beta2: float = 0.1    # Stage I  KL weight (attempt 1 only)
    # Length normalization for PG and KL terms.
    #"sequence" -> divide by actual generated length 
    length_norm: str = "constant"

    # optimization 
    lr: float = 1e-5
    max_grad_norm: float = 1.0          
    stage1_epochs: int = 3
    stage2_epochs: int = 5
    
    # Batched updates  This (plus length norm + reward norm) keeps graident in control and not explode
    episodes_per_update: int = 8        # gradient accumulation across episodes
    use_8bit_optimizer: bool = True

    # generation parameter
    temperature: float = 0.8
    max_new_tokens: int = 256
    # Backward pass guard. Generation can legally produce very long multi-turn transcripts (especially hot_air_balloon/adventuregame), but scoring one huge prompt+completion turn with gradients can OOM a 24GB A30.     # Treat such trajectories as infra drops instead of crashing the run. 0 disables.
    max_train_turn_tokens: int = 8192

    fast_inference: bool = False
    enable_thinking: Optional[bool] = None  # auto per model

    # reward modes 
    reward_mode: str = REWARD_MODE_SUCCESS_BINARY
    abort_reward: float = -1.0

    # parallelism #
    # Number of model replicas used to GENERATE episodes concurrently, one per GPU. 
    parallel_gpus: int = 1
    # Parallel rollout scheduling:
    #   "dynamic" -> each replica pulls the next unfinished instance when it becomes free. This is better for dialogue games as durations vary wildly, so fixed shards leave GPUs idle.

    rollout_scheduler: str = "dynamic"
    # In the parallel trainer, generate at most this many episodes before an optimizer phase. A small multiple of episodes_per_update keeps rollouts fresher and broadcasts more often.
    rollout_round_size: Optional[int] = None

    
    # "self_play_all_roles" is the main training path: every clembench player slot is backed by the same learner backend, so completions from every role become TurnTokens and receive SCoRe credit.
    multiplayer_teacher: str = "self_play_all_roles"

   
    games: tuple[str, ...] = SINGLE_PLAYER_GAMES
    instances_name: Optional[str] = None   # clemcore instances file; None = default
    episodes_per_game_per_epoch: Optional[int] = None  # None/0 => full pool once
    val_episodes_per_game: int = 8
    seed: int = 0


    # In-process games can still hang inside model.generate on a pathological input, a per-episode wall-clock guard drops (not aborts) such episodes.
    episode_timeout_s: float = 5000.0

    # ---- io -------------------------------------------------------------- #
    output_dir: str = "models/score_rl"
    log_dir: str = "runs/score_rl"
    run_dir: str = "runs/score_rl/state"

    checkpoint_every_n_updates: int = 8

    def registry_name(self) -> str:
        return self.model_registry_name or self.model_id.rsplit("/", 1)[-1]

    def is_multiplayer(self, game: str) -> bool:
        return game in MULTIPLAYER_GAMES
