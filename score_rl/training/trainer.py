"""
SCoRe two-stage trainer.

Loop shape (per stage, per epoch, per game):
    set eval -> collect `episodes_per_update` episodes (rollout, no grad)
    set train -> one batched REINFORCE update (grad accumulation over the batch)

Stability comes from following things - 
  * batched updates  -> low-variance gradients
  * length normalization in losses.py
  * per-game reward normalized to [0,1] before the alpha shaping

Checkpointing is at (stage, epoch, game) granularity with  progress.json.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional

import torch

from .config import SCoReConfig
from .env import GameHandle, LearnerBackend
from .losses import stage_backward
from .model import build_optimizer, load_learner_plain
from .parallel import ParallelLearner
from .rollout import collect_episode_guarded


def _chunks(seq, size):
    """Yield successive `size`-length chunks of `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _rollout_rounds(keys, cfg: SCoReConfig):
    """
    Split rollout collection into bounded rounds when requested. This keeps
    SCoRe updates closer to the policy that generated them and gives replicas a
    chance to receive fresh LoRA weights before the whole game is exhausted.
    """
    round_size = cfg.rollout_round_size
    if not round_size or round_size <= 0:
        yield list(keys)
        return
    for chunk in _chunks(list(keys), int(round_size)):
        yield chunk


# resume state 
@dataclass
class ProgressState:
    completed: list = field(default_factory=list)  # keys "stage/epoch/game"
    update_step: int = 0

    @staticmethod
    def key(stage: int, epoch: int, game: str) -> str:
        return f"{stage}/{epoch}/{game}"

    def is_done(self, stage: int, epoch: int, game: str) -> bool:
        return self.key(stage, epoch, game) in self.completed

    def mark(self, stage: int, epoch: int, game: str):
        k = self.key(stage, epoch, game)
        if k not in self.completed:
            self.completed.append(k)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ProgressState":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()


# one batched update
def _episode_max_turn_tokens(ep) -> int:
    turns = list(ep.y1.turns) + list(ep.y2.turns)
    if not turns:
        return 0
    return max(len(t.prompt_ids) + len(t.completion_ids) for t in turns)


def _filter_trainable_batch(batch, cfg, stage: int, game: str,
                            n_drop: int) -> tuple[list, int]:
    limit = int(getattr(cfg, "max_train_turn_tokens", 0) or 0)
    if limit <= 0:
        return list(batch), n_drop

    keep = []
    for ep in batch:
        mx = _episode_max_turn_tokens(ep)
        if mx > limit:
            print(
                f"[drop] {game} {ep.game_id}: train-turn-too-long "
                f"({mx} > {limit} tokens) before stage{stage} backward",
                flush=True,
            )
            n_drop += 1
        else:
            keep.append(ep)
    return keep, n_drop


def _update_guarded(model, optimizer, batch, stage, cfg, device, game: str,
                    n_drop: int) -> tuple[dict | None, int]:
    batch, n_drop = _filter_trainable_batch(batch, cfg, stage, game, n_drop)
    if not batch:
        return None, n_drop
    try:
        return _update(model, optimizer, batch, stage, cfg, device), n_drop
    except torch.OutOfMemoryError as e:
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[drop] {game}: CUDA OOM during stage{stage} backward for "
            f"{len(batch)} episode(s): {e}",
            flush=True,
        )
        return None, n_drop + len(batch)


def _update(model, optimizer, batch, stage, cfg, device) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses, agg = [], {}
    n = len(batch)
    for ep in batch:
        # Accumulate this episode's grads turn-by-turn (scaled by 1/n for the
        # batch mean). Peak VRAM is bounded to a single turn's graph, which is
        # what keeps many-turn games from OOMing.
        total, m = stage_backward(stage, model, ep, cfg, device, scale=1.0 / n)
        losses.append(total)
        for k, v in m.items():
            agg.setdefault(k, []).append(v)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
    )
    optimizer.step()

    out = {k: mean(v) for k, v in agg.items()}
    out["loss/mean"] = mean(losses)
    out["grad_norm"] = float(grad_norm)
    out["corrected_rate"] = mean(float(ep.corrected()) for ep in batch)
    out["regressed_rate"] = mean(float(ep.regressed()) for ep in batch)
    out["reward/win_y1"] = mean(float(ep.y1.success) for ep in batch)
    out["reward/win_y2"] = mean(float(ep.y2.success) for ep in batch)
    out["reward/mean_y1"] = mean(ep.y1.reward for ep in batch)
    out["reward/mean_y2"] = mean(ep.y2.reward for ep in batch)
    # abort_rate_y1 is the critical health signal: if y1 almost always aborts,
    # r1=0 for nearly every episode and SCoRe has no first attempt to correct
    # FROM -> the self-correction objective gets no gradient. Watch this first.
    out["abort_rate_y1"] = mean(float(ep.y1.aborted) for ep in batch)
    out["abort_rate_y2"] = mean(float(ep.y2.aborted) for ep in batch)
    return out


# One stage
def _run_stage(stage, epochs, model, tokenizer, learner, optimizer, cfg,
               progress, progress_path, writer, device):
    rng = random.Random(cfg.seed + stage)
    for epoch in range(epochs):
        for game in cfg.games:
            if progress.is_done(stage, epoch, game):
                print(f"[skip] stage{stage} epoch{epoch} {game} (already done)")
                continue

            try:
                handle = GameHandle(game, cfg)
            except Exception as e:
                print(f"[error] could not load game {game}: {e!r} -- skipping")
                progress.mark(stage, epoch, game)
                progress.save(progress_path)
                continue

            with handle:
                if not handle.instance_keys:
                    print(f"[warn] {game} has no instances")
                    progress.mark(stage, epoch, game)
                    progress.save(progress_path)
                    continue

                # Single-player pools are small: play EVERY instance once per
                # epoch (shuffled) rather than sampling with replacement. If
                # episodes_per_game_per_epoch is set (>0) it caps/pads the count
                ids = list(handle.instance_keys)
                rng.shuffle(ids)
                cap = cfg.episodes_per_game_per_epoch
                if cap:
                    ids = [ids[i % len(ids)] for i in range(cap)]
                buffer, n_drop = [], 0

                for i, gid in enumerate(ids):
                    model.eval()
                    ep = collect_episode_guarded(handle, learner, gid, cfg)
                    if ep is None:
                        n_drop += 1
                        continue
                    buffer.append(ep)

                    if len(buffer) >= cfg.episodes_per_update:
                        m, n_drop = _update_guarded(
                            model, optimizer, buffer, stage, cfg, device, game, n_drop
                        )
                        buffer = []
                        if m is None:
                            continue
                        progress.update_step += 1
                        _log(writer, f"stage{stage}/{game}", m, progress.update_step)
                        print(f"[s{stage} e{epoch} {game} u{progress.update_step}] "
                              f"loss={m['loss/mean']:.3f} gn={m['grad_norm']:.2f} "
                              f"win1={m['reward/win_y1']:.2f} win2={m['reward/win_y2']:.2f} "
                              f"ab1={m['abort_rate_y1']:.2f} ab2={m['abort_rate_y2']:.2f} "
                              f"corr={m['corrected_rate']:.2f} reg={m['regressed_rate']:.2f} "
                              f"drop={n_drop}")
                        if progress.update_step % cfg.checkpoint_every_n_updates == 0:
                            _save_checkpoint(model, cfg, progress.update_step)

                if buffer:  # flush remainder
                    m, n_drop = _update_guarded(
                        model, optimizer, buffer, stage, cfg, device, game, n_drop
                    )
                    if m is not None:
                        progress.update_step += 1
                        _log(writer, f"stage{stage}/{game}", m, progress.update_step)

            # Persist weights BEFORE marking done, so the resume/ adapter always
            # corresponds to (or leads) the completed-games list.
            _save_resume(model, cfg)
            progress.mark(stage, epoch, game)
            progress.save(progress_path)
            print(f"[done] stage{stage} epoch{epoch} {game} (dropped {n_drop})")


# One stage with parallel generation varaint 
def _run_stage_parallel(stage, epochs, plearner, optimizer, cfg,
                        progress, progress_path, writer, device):
    """
    Same stage semantics as `_run_stage`, but each game's episodes for the epoch
    are GENERATED across all replicas at once (on-policy w.r.t. replica 0's
    current weights), then consumed in `episodes_per_update`-sized batches by the
    UNCHANGED `_update` on replica 0. After each optimizer step we flag the
    weights dirty; the next `generate_game` broadcasts them before generating.

    Produces exactly one model (replica 0's adapter). The math per update is
    identical to the single-GPU path -- only WHERE/WHEN episodes are generated
    changes.
    """
    model = plearner.main_model
    rng = random.Random(cfg.seed + stage)
    for epoch in range(epochs):
        for game in cfg.games:
            if progress.is_done(stage, epoch, game):
                print(f"[skip] stage{stage} epoch{epoch} {game} (already done)")
                continue

            # Instance keys come from replica 0's handle (all replicas share the
            # same instance file, so keys are identical across replicas).
            try:
                keys = list(plearner.replicas[0].handle(game).instance_keys)
            except Exception as e:
                print(f"[error] could not load game {game}: {e!r} -- skipping")
                progress.mark(stage, epoch, game)
                progress.save(progress_path)
                continue

            if not keys:
                print(f"[warn] {game} has no instances")
                progress.mark(stage, epoch, game)
                progress.save(progress_path)
                continue

            rng.shuffle(keys)
            cap = cfg.episodes_per_game_per_epoch
            if cap:
                keys = [keys[i % len(keys)] for i in range(cap)]

            n_drop = 0
            for round_idx, round_keys in enumerate(_rollout_rounds(keys, cfg), start=1):
                # Parallel generation for this bounded round. If cfg.rollout_round_size
                # is unset, this is exactly the old whole-game behavior.
                episodes = plearner.generate_game(game, round_keys)
                n_drop += len(round_keys) - len(episodes)
                rng.shuffle(episodes)  # decorrelate batch composition from scheduler order

                for batch in _chunks(episodes, cfg.episodes_per_update):
                    m, n_drop = _update_guarded(
                        model, optimizer, batch, stage, cfg, device, game, n_drop
                    )
                    if m is None:
                        continue
                    plearner.mark_updated()
                    progress.update_step += 1
                    _log(writer, f"stage{stage}/{game}", m, progress.update_step)
                    print(f"[s{stage} e{epoch} {game} r{round_idx} "
                          f"u{progress.update_step}] "
                          f"loss={m['loss/mean']:.3f} gn={m['grad_norm']:.2f} "
                          f"win1={m['reward/win_y1']:.2f} "
                          f"win2={m['reward/win_y2']:.2f} "
                          f"ab1={m['abort_rate_y1']:.2f} "
                          f"ab2={m['abort_rate_y2']:.2f} "
                          f"corr={m['corrected_rate']:.2f} "
                          f"reg={m['regressed_rate']:.2f} drop={n_drop}")
                    if progress.update_step % cfg.checkpoint_every_n_updates == 0:
                        _save_checkpoint(model, cfg, progress.update_step)

            # Rolling resume snapshot in lockstep with progress (see _save_resume).
            _save_resume(model, cfg)
            progress.mark(stage, epoch, game)
            progress.save(progress_path)
            print(f"[done] stage{stage} epoch{epoch} {game} (dropped {n_drop})")


# logging and checkpoints
def _log(writer, tag, metrics: dict, step: int):
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(f"{tag}/{k}", v, step)


def _save_adapter(model, out_dir: Path):
    """
    Atomically write a PEFT adapter to `out_dir`: save to `<out_dir>.tmp` first,
    then swap it into place. A crash mid-save can corrupt a half-written dir, so
    the swap guarantees `out_dir` is always either the previous good adapter or
    the new complete one -- never a partial one that would break resume.
    """
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp))
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    tmp.replace(out_dir)


def _save_checkpoint(model, cfg: SCoReConfig, step: int):
    out = Path(cfg.output_dir) / f"checkpoint-{step}"
    _save_adapter(model, out)
    print(f"[ckpt] saved {out}")


def _save_resume(model, cfg: SCoReConfig):
    """
    Save a rolling `resume/` adapter at each (stage, epoch, game) boundary, in
    lockstep with progress.json's `mark`.
    """
    _save_adapter(model, Path(cfg.output_dir) / "resume")


def _load_adapter_into(model, adapter_dir: Path) -> bool:
    """
    Reload saved LoRA weights into an existing PEFT model in place. Returns False
    if no adapter file is present (nothing to resume from). Keys written by
    save_pretrained match set_peft_model_state_dict's expected format.
    """
    from peft import set_peft_model_state_dict

    adapter_dir = Path(adapter_dir)
    safe = adapter_dir / "adapter_model.safetensors"
    binf = adapter_dir / "adapter_model.bin"
    if safe.exists():
        from safetensors.torch import load_file
        sd = load_file(str(safe))
    elif binf.exists():
        sd = torch.load(str(binf), map_location="cpu")
    else:
        return False

    dev = next(model.parameters()).device
    sd = {k: v.to(dev) for k, v in sd.items()}
    set_peft_model_state_dict(model, sd)
    return True


def _resume_weights(model, cfg: SCoReConfig, progress: "ProgressState") -> None:
    """
    If progress.json shows completed games, reload the matching `resume/` adapter
    so the skipped games' training isn't lost. 
    """
    if not progress.completed:
        return
    resume_dir = Path(cfg.output_dir) / "resume"
    if _load_adapter_into(model, resume_dir):
        print(f"[resume] loaded adapter from {resume_dir} "
              f"({len(progress.completed)} game-epoch(s) already done)")
    else:
        print(f"[resume][WARN] progress.json lists {len(progress.completed)} "
              f"completed game-epoch(s) but no adapter at {resume_dir} -- those "
              f"weights are LOST. Clear {cfg.run_dir}/progress.json to retrain "
              f"from base, or restore a checkpoint-* dir as resume/.")


# entry point
def train(cfg: SCoReConfig):
    if cfg.parallel_gpus and cfg.parallel_gpus > 1:
        return _train_parallel(cfg)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    model, tokenizer = load_learner_plain(cfg, device_map=device_map)
    optimizer = build_optimizer(model, cfg)
    learner = LearnerBackend(model, tokenizer, cfg)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        writer = None

    progress_path = Path(cfg.run_dir) / "progress.json"
    progress = ProgressState.load(progress_path)
    # Resuming a killed run: reload the weights matching the completed-games list
    _resume_weights(model, cfg, progress)

    print(f"=== SCoRe training: {cfg.model_id} on {list(cfg.games)} ===")
    _run_stage(1, cfg.stage1_epochs, model, tokenizer, learner, optimizer,
               cfg, progress, progress_path, writer, device)
    _run_stage(2, cfg.stage2_epochs, model, tokenizer, learner, optimizer,
               cfg, progress, progress_path, writer, device)

    final = Path(cfg.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    if writer is not None:
        writer.close()
    print(f"=== done. final adapter at {final} ===")


def _train_parallel(cfg: SCoReConfig):
    """
    Multi-GPU path: N replicas generate episodes in parallel, replica 0 trains
    and broadcasts. 
    """
    plearner = ParallelLearner(cfg)
    device = plearner.main_device

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        writer = None

    progress_path = Path(cfg.run_dir) / "progress.json"
    progress = ProgressState.load(progress_path)
    # Resume: reload the adapter into replica 0, then mark dirty so the next
    # generate_game broadcasts it to replicas 1..N-1 before any rollout (keeping
    # every replica on-policy w.r.t. the resumed weights).
    _resume_weights(plearner.main_model, cfg, progress)
    if progress.completed:
        plearner.mark_updated()

    print(f"=== SCoRe training [parallel x{plearner.n}]: {cfg.model_id} "
          f"on {list(cfg.games)} ===")
    try:
        _run_stage_parallel(1, cfg.stage1_epochs, plearner, plearner.optimizer,
                            cfg, progress, progress_path, writer, device)
        _run_stage_parallel(2, cfg.stage2_epochs, plearner, plearner.optimizer,
                            cfg, progress, progress_path, writer, device)

        final = Path(cfg.output_dir) / "final"
        final.mkdir(parents=True, exist_ok=True)
        plearner.main_model.save_pretrained(str(final))
        print(f"=== done. final adapter at {final} ===")
    finally:
        if writer is not None:
            writer.close()
        plearner.close()
