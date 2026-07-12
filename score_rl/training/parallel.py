"""
Parallel episode generation for SCoRe.

`cfg.parallel_gpus = N` spins up N *replicas* of the learner, one per GPU. Each
replica generates its shard of the game's instances concurrently; the weight
update runs on replica 0 ONLY; the new LoRA is then broadcast to replicas 1..N-1
before the next generation round. The output is one adapter trained on ALL games
-- replicas are a throughput trick for on-policy rollouts.

Generate the whole game's episodes while all replicas are in 
sync -> every rollout is on-policy w.r.t. the pre-round weights -> run all
of that game's updates on replica 0 -> broadcast once before the next game. One
broadcast per game, not per update.
"""

from __future__ import annotations

import concurrent.futures
import queue
from typing import Optional

import torch

from .config import SCoReConfig
from .env import GameHandle, LearnerBackend, TeacherBackend
from .model import load_learner_plain, build_optimizer
from .rollout import collect_episode_guarded


# Player slots for multiplayer Playpen games. In the default self-play-all-roles
# mode, every slot is backed by the same learner backend and all generated turns
# are optimized by SCoRe. The role labels are only used by the optional frozen
# debug mode below.
MULTIPLAYER_PLAYER_ORDER: dict[str, tuple[str, ...]] = {
    "taboo": ("teacher", "learner"),
    "referencegame": ("teacher", "learner"),
    "imagegame": ("teacher", "learner"),
    "wordle_withcritic": ("learner", "teacher"),
    "clean_up": ("learner", "teacher"),
    "codenames": ("learner", "teacher"),
    "dond": ("learner", "teacher"),
    "guesswhat": ("learner", "teacher"),
    "hot_air_balloon": ("learner", "teacher"),
    "matchit_ascii": ("learner", "teacher"),
    "privateshared": ("learner",),
    "wordle_withclue": ("learner", "teacher"),
}


class _Replica:
    """One learner copy pinned to one GPU, with its own backend + game handles."""

    def __init__(self, idx: int, device: str, model, tokenizer, cfg: SCoReConfig):
        self.idx = idx
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.backend = LearnerBackend(model, tokenizer, cfg)
        self.teacher = TeacherBackend(model, tokenizer, cfg)
        self._handles: dict[str, GameHandle] = {}
        self._cfg = cfg

    def handle(self, game: str) -> GameHandle:
        """Open (once) and cache this replica's own GameHandle for `game`."""
        h = self._handles.get(game)
        if h is None:
            h = GameHandle(game, self._cfg)
            h.__enter__()
            self._handles[game] = h
        return h

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def player_models(self, game: str) -> list:
        if not self._cfg.is_multiplayer(game):
            return [self.backend]
        order = MULTIPLAYER_PLAYER_ORDER.get(game)
        if order is None:
            raise NotImplementedError(
                f"No multiplayer player order configured for {game!r}"
            )
        if self._cfg.multiplayer_teacher == "self_play_all_roles":
            return [self.backend for _ in order]
        if self._cfg.multiplayer_teacher != "self_play_frozen":
            raise ValueError(
                "multiplayer_teacher must be 'self_play_all_roles' or "
                f"'self_play_frozen', got {self._cfg.multiplayer_teacher!r}"
            )
        models = []
        for role in order:
            if role == "learner":
                models.append(self.backend)
            elif role == "teacher":
                models.append(self.teacher)
            else:
                raise ValueError(f"Unknown multiplayer role {role!r}")
        return models


class ParallelLearner:
    """
    N replicas for parallel generation; replica 0 is the trainable "main" copy.

    Public surface used by the trainer:
      * .main_model / .main_tokenizer  -- replica 0 (what the optimizer + losses see)
      * .optimizer                      -- built over replica 0's params
      * .generate_game(game, keys)      -- returns list[Episode], generated in parallel
      * .mark_updated()                 -- call after each optimizer.step()
      * .close()
    """

    def __init__(self, cfg: SCoReConfig):
        self.cfg = cfg
        n = max(1, int(cfg.parallel_gpus))
        avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if avail and n > avail:
            print(f"[parallel] requested {n} replicas but only {avail} GPUs "
                  f"visible -- using {avail}")
            n = avail
        self.n = n

        self.replicas: list[_Replica] = []
        for i in range(n):
            device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            device_map = {"": i} if torch.cuda.is_available() else {"": "cpu"}
            print(f"[parallel] loading replica {i} on {device} ...", flush=True)
            model, tok = load_learner_plain(cfg, device_map=device_map)
            self.replicas.append(_Replica(i, device, model, tok, cfg))

        # Replica 0 is the training copy.
        self.optimizer = build_optimizer(self.replicas[0].model, cfg)
        self._dirty = False  # True once replica 0 has updates not yet broadcast

        # Thread pool for fan-out generation (one worker per replica). Generation
        # is CUDA-bound on distinct devices, so the GIL is released during the
        # heavy work and the replicas truly overlap.
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=n)

    @property
    def main_model(self):
        return self.replicas[0].model

    @property
    def main_tokenizer(self):
        return self.replicas[0].tokenizer

    @property
    def main_device(self) -> str:
        return self.replicas[0].device

    def mark_updated(self):
        """Record that replica 0's weights changed (broadcast lazily before gen)."""
        self._dirty = True

    # weight broadcast  
    def _sync_replicas(self):
        """Copy replica 0's LoRA weights onto every other replica."""
        if not self._dirty or self.n == 1:
            self._dirty = False
            return
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        sd = get_peft_model_state_dict(self.replicas[0].model)
        for r in self.replicas[1:]:
            sd_dev = {k: v.detach().to(r.device) for k, v in sd.items()}
            set_peft_model_state_dict(r.model, sd_dev)
        self._dirty = False

    # parallel generation
    def _shard(self, keys: list) -> list[list]:
        """
        Round-robin the instance keys across replicas. Round-robin (not
        contiguous blocks) spreads long/short instances evenly so one replica
        doesn't get stuck with all the slow games
        """
        buckets: list[list] = [[] for _ in range(self.n)]
        for i, k in enumerate(keys):
            buckets[i % self.n].append(k)
        return buckets

    def _generate_shard(self, replica: _Replica, game: str, keys: list) -> list:
        """Play every key in this replica's shard; return its episodes."""
        handle = replica.handle(game)
        player_models = replica.player_models(game)
        out = []
        for key in keys:
            replica.model.eval()
            ep = collect_episode_guarded(
                handle, replica.backend, key, self.cfg, player_models=player_models
            )
            if ep is not None:
                out.append(ep)
        return out

    def _generate_dynamic(self, replica: _Replica, game: str,
                          work: "queue.Queue") -> list:
        """
        Each replica owns exactly one thread and one GameHandle, 
        so clemcore/model state is never used concurrently within
        a replica. Fast replicas pull more instances; slow instances no longer
        determine an entire fixed shard's completion time.
        """
        handle = replica.handle(game)
        player_models = replica.player_models(game)
        out = []
        while True:
            try:
                key = work.get_nowait()
            except queue.Empty:
                return out
            try:
                replica.model.eval()
                ep = collect_episode_guarded(
                    handle, replica.backend, key, self.cfg,
                    player_models=player_models,
                )
                if ep is not None:
                    out.append(ep)
            finally:
                work.task_done()

    def generate_game(self, game: str, keys: list) -> list:
        """
        Generate episodes for one game across all replicas, in parallel.

        Replicas are synced to replica 0's current weights first, so every
        rollout in this round is on-policy w.r.t. the same weights. Returns a
        flat list[Episode] (drops are already filtered out by the guard).
        """
        self._sync_replicas()

        if self.n == 1:
            return self._generate_shard(self.replicas[0], game, list(keys))

        if self.cfg.rollout_scheduler == "round_robin":
            shards = self._shard(list(keys))
            futures = [
                self._pool.submit(
                    self._generate_shard, self.replicas[i], game, shards[i]
                )
                for i in range(self.n)
            ]
        elif self.cfg.rollout_scheduler == "dynamic":
            work: queue.Queue = queue.Queue()
            for key in keys:
                work.put(key)
            futures = [
                self._pool.submit(self._generate_dynamic, r, game, work)
                for r in self.replicas
            ]
        else:
            raise ValueError(
                "rollout_scheduler must be 'dynamic' or 'round_robin', got "
                f"{self.cfg.rollout_scheduler!r}"
            )
        episodes: list = []
        for f in concurrent.futures.as_completed(futures):
            episodes.extend(f.result())
        return episodes

    def close(self):
        self._pool.shutdown(wait=False)
        for r in self.replicas:
            r.close()
