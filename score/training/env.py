"""
In-process clemcore environment: run one game instance, capture the learner's
tokens, read the outcome. 

Confirmed API (via probes):
  * Learner plugs in as a `clemcore.backends.Model`; Player routes non-Custom/
    models straight to `generate_response(messages) -> (prompt, resp, text)`.
  * Games are driven by `clemcore.clemgame.runners.sequential.run(benchmark,
    instances, [models], callbacks=...)`. batch_size=1 => sequential runner.
  * `on_game_end(game_master, game_instance, exception, rewards)` gives us the
    game master, whose `state.outcome` is one of Outcome.SUCCESS/FAILURE/ABORTED.
  * Benchmarks: GameRegistry -> GameSpec -> GameBenchmark.load_from_spec (context
    manager); instances via GameInstances.from_game_spec, filtered by game_id.

Token capture happens here (per learner turn) so the loss can recompute per-token
log-probs on the trainable model without any chat-template misalignment.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch

from clemcore.backends import Model, ModelSpec
from clemcore.clemgame import (
    GameBenchmark,
    GameBenchmarkCallback,
    GameBenchmarkCallbackList,
    GameInstances,
)
from clemcore.clemgame.master import Outcome
from clemcore.clemgame.registry import GameRegistry
from clemcore.clemgame.runners import sequential

from .config import SCoReConfig
from .episode import Attempt, TurnTokens



# Learner backend: wraps our HF model as a clemcore Model
class LearnerBackend(Model):
    """
    A clemcore backend that generates with trainable model and records the
    exact (prompt_ids, completion_ids) of every turn into `current_attempt`.

    `prefix_messages` lets the SCoRe second attempt prepend a reflection preamble
    (the y1 transcript + a self-correction instruction) to every turn -- those are
    prompt tokens, so they never enter the policy-gradient (only completion tokens do).
    """

    def __init__(self, hf_model, tokenizer, cfg: SCoReConfig):
        super().__init__(ModelSpec(model_name=cfg.registry_name()))
        self.set_gen_args(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens)
        self._hf = hf_model
        self._tok = tokenizer
        self._cfg = cfg
        self.current_attempt: Attempt = Attempt()
        self.prefix_messages: list[dict] = []
       
        gc = getattr(self._hf, "generation_config", None)
        if gc is not None:
            gc.max_length = None


    def supports_batching(self) -> bool:
        return False

    def new_attempt(self, prefix_messages: Optional[list[dict]] = None) -> Attempt:
        self.current_attempt = Attempt()
        self.prefix_messages = prefix_messages or []
        return self.current_attempt

    @property
    def _thinking_kwargs(self) -> dict:
        if self._cfg.enable_thinking is None:
            return {}
        return {"enable_thinking": self._cfg.enable_thinking}

    def generate_response(self, messages: list[dict]):
        model_messages = self.prefix_messages + list(messages)

        prompt_ids = self._tok.apply_chat_template(
            model_messages,
            add_generation_prompt=True,
            tokenize=True,
            **self._thinking_kwargs,
        )
    
        if hasattr(prompt_ids, "ids"):              # tokenizers.Encoding
            prompt_ids = prompt_ids.ids
        elif hasattr(prompt_ids, "input_ids"):      # BatchEncoding
            prompt_ids = prompt_ids.input_ids
        elif isinstance(prompt_ids, dict):          # plain dict
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = list(prompt_ids[0])        # unwrap a [1, T] batch
        prompt_ids = [int(t) for t in prompt_ids]
        input_ids = torch.tensor([prompt_ids], device=self._hf.device)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = self._hf.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._cfg.max_new_tokens,
                do_sample=self._cfg.temperature > 0.0,
                temperature=self._cfg.temperature if self._cfg.temperature > 0 else None,
                pad_token_id=self._tok.pad_token_id,
            )
        completion_ids = out[0, input_ids.shape[1]:].tolist()
        # drop trailing pad tokens (keep a generated eos: it is a real action token)
        while completion_ids and completion_ids[-1] == self._tok.pad_token_id:
            completion_ids.pop()

        text = self._tok.decode(completion_ids, skip_special_tokens=True).strip()

        if os.environ.get("SCORE_DEBUG_GEN"):
            raw = self._tok.decode(completion_ids, skip_special_tokens=False)
            print("\n----- [SCORE_DEBUG_GEN] learner turn -----")
            print(f"  prompt tokens : {len(prompt_ids)}")
            print(f"  completion tok: {len(completion_ids)}")
            print(f"  RAW (with specials):\n{raw!r}")
            print(f"  CLEAN text (what GM parses):\n{text!r}")
            print("------------------------------------------\n", flush=True)

        self.current_attempt.turns.append(
            TurnTokens(prompt_ids=list(prompt_ids), completion_ids=completion_ids)
        )
        # (prompt, response_object, response_text) -- first two are for GM logging only
        return {"messages": model_messages}, {"text": text}, text


class TeacherBackend(Model):
    """
    Frozen self-play teacher for multiplayer rollouts.

    It shares the replica's HF model/tokenizer with the learner, so it adds no
    extra model copy. Rollouts are collected under no_grad and the trainer does
    not update weights until a rollout round ends, so the teacher is effectively
    frozen for the full set of episodes generated from that snapshot. Crucially,
    it does NOT append TurnTokens: SCoRe optimizes learner completions only.
    """

    def __init__(self, hf_model, tokenizer, cfg: SCoReConfig):
        super().__init__(ModelSpec(model_name=f"{cfg.registry_name()}-teacher"))
        self.set_gen_args(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens)
        self._hf = hf_model
        self._tok = tokenizer
        self._cfg = cfg
        gc = getattr(self._hf, "generation_config", None)
        if gc is not None:
            gc.max_length = None

    def supports_batching(self) -> bool:
        return False

    @property
    def _thinking_kwargs(self) -> dict:
        if self._cfg.enable_thinking is None:
            return {}
        return {"enable_thinking": self._cfg.enable_thinking}

    def generate_response(self, messages: list[dict]):
        prompt_ids = self._tok.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            **self._thinking_kwargs,
        )
        if hasattr(prompt_ids, "ids"):
            prompt_ids = prompt_ids.ids
        elif hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        elif isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = list(prompt_ids[0])
        prompt_ids = [int(t) for t in prompt_ids]

        input_ids = torch.tensor([prompt_ids], device=self._hf.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self._hf.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._cfg.max_new_tokens,
                do_sample=self._cfg.temperature > 0.0,
                temperature=self._cfg.temperature if self._cfg.temperature > 0 else None,
                pad_token_id=self._tok.pad_token_id,
            )
        completion_ids = out[0, input_ids.shape[1]:].tolist()
        while completion_ids and completion_ids[-1] == self._tok.pad_token_id:
            completion_ids.pop()
        text = self._tok.decode(completion_ids, skip_special_tokens=True).strip()
        return {"messages": list(messages)}, {"text": text}, text



# Capture callback: grabs the game master (for outcome) and any exception
class _CaptureCallback(GameBenchmarkCallback):
    def __init__(self):
        self.game_master = None
        self.exception: Optional[Exception] = None
        self.rewards = None
        self.fired = False

    def on_benchmark_start(self, game_benchmark):
        pass

    def on_benchmark_end(self, game_benchmark):
        pass

    def on_game_start(self, game_master, game_instance):
        self.game_master = game_master

    def on_game_end(self, game_master, game_instance, exception=None, rewards=None):
        self.game_master = game_master
        self.exception = exception
        self.rewards = rewards
        self.fired = True


# Reward from outcome
def _reward_from_outcome(outcome, cfg: SCoReConfig) -> tuple[float, bool, bool]:
    """Returns (reward in [0,1], success, aborted). success_binary mode."""
    success = outcome == Outcome.SUCCESS
    aborted = outcome == Outcome.ABORTED
    if aborted:
        return cfg.abort_reward, False, True
    return (1.0 if success else 0.0), success, False



# Game handle: open a benchmark once, play many instances against it
class GameHandle:
    """Holds an opened GameBenchmark + its instances for one game."""

    def __init__(self, game_name: str, cfg: SCoReConfig):
        self.game_name = game_name
        self.cfg = cfg
        registry = GameRegistry.from_directories_and_cwd_files()
        specs = registry.get_game_specs_that_unify_with(game_name)
        if not specs:
            raise LookupError(f"No game spec unifies with {game_name!r}")
        self.game_spec = list(specs)[0]
        if cfg.instances_name:
            self.game_spec.instances = cfg.instances_name
        self._benchmark_cm = GameBenchmark.load_from_spec(self.game_spec)
        self.benchmark: GameBenchmark = self._benchmark_cm.__enter__()
        self.instances = GameInstances.from_game_spec(self.game_spec)
   
        self.instance_keys: list[tuple[str, int]] = [
            (str(row["experiment"]["name"]), int(row["game_instance"]["game_id"]))
            for row in self.instances
        ]

    def close(self):
        try:
            self._benchmark_cm.__exit__(None, None, None)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def play_attempt(
        self,
        learner: LearnerBackend,
        instance_key: tuple[str, int],
        player_models: Optional[list] = None,
    ) -> Attempt:
        """
        Play ONE instance to completion; return the populated Attempt (tokens +
        normalized reward + success/abort). `instance_key` is (experiment_name,
        game_id) -- both are required because game_id repeats across experiments.
        player_models defaults to [learner] (single-player). Multiplayer passes
        the ordered clembench player slots; in the default self-play-all-roles
        mode those slots are all this same learner backend, so every role's
        generated turns are captured into the same Attempt.
        """
        exp_name, game_id = instance_key
        attempt = learner.new_attempt(prefix_messages=learner.prefix_messages)
        single = self.instances.filter(
            lambda row: str(row["experiment"]["name"]) == exp_name
            and int(row["game_instance"]["game_id"]) == int(game_id)
        )
        capture = _CaptureCallback()
        models = player_models if player_models is not None else [learner]

        sequential.run(
            self.benchmark,
            single,
            models,
            callbacks=GameBenchmarkCallbackList([capture]),
        )

        if capture.game_master is not None and capture.exception is None:
            reward, success, aborted = _reward_from_outcome(
                capture.game_master.state.outcome, self.cfg
            )
        else:
            # runner raised before/at end -> treat as abort (infra, not learning signal)
            reward, success, aborted = self.cfg.abort_reward, False, True

        attempt.reward = reward
        attempt.success = success
        attempt.aborted = aborted
        return attempt
