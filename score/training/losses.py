"""
SCoRe two-stage REINFORCE losses.

  * Reference policy = the SAME model with LoRA disabled (`model.disable_adapter()`).
    No second model is ever held in VRAM.
  * KL uses the K3 (Schulman) unbiased estimator: exp(d) - d - 1, d = logp_ref - logp.
  * Length normalization (per attempt) is the primary stability mechanism -- with
    it, alpha=10 does not explode; the reference does not even clip gradients.
  * Selective log-softmax over the completion region only; no [T, V] softmax over
    the full sequence.

Stage I  (init):  optimize logp(y2)*r2 ; KL-anchor y1.           (beta2)
Stage II (amplify): optimize logp(y1)*r1 + logp(y2)*r2_shaped ;  KL-anchor y1 & y2. (beta1)
"""

from __future__ import annotations

import torch

from .config import SCoReConfig
from .episode import Attempt, Episode, TurnTokens


# Per-turn token log-probs on the trainable model (differentiable)
def _selective_log_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    log p(target) = logit[target] - logsumexp(logits), computed WITHOUT
    materializing the full [T, V] log-softmax tensor. A full
    fp32 log_softmax over Llama's 128k vocab, held in the autograd graph for
    every turn, is a large share of the OOM.
    """
    chosen = logits.gather(1, targets.unsqueeze(1)).squeeze(1)   # [T_comp]
    return chosen - torch.logsumexp(logits, dim=-1)              # [T_comp]


def _turn_token_logprobs(model, turn: TurnTokens, device) -> torch.Tensor:
    """
    Log-prob of each completion token under `model`. Returns a 1-D tensor of
    length len(completion_ids), with gradient attached.

    `use_cache=False`: with the KV cache on, HF silently
    disable gradient checkpointing (the two are incompatible), so full-network
    activations are retained across every turn's forward -> OOM. Turning the
    cache off restores checkpointing and is correct for a training forward
    """
    full_ids = turn.prompt_ids + turn.completion_ids
    prompt_len = len(turn.prompt_ids)
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    logits = model(ids, use_cache=False).logits[0]  # [T, V]
    # predict token t from position t-1: completion tokens live at [prompt_len-1 : -1]
    comp_logits = logits[prompt_len - 1: -1]         # [T_comp, V]
    targets = torch.tensor(turn.completion_ids, dtype=torch.long, device=device)

    return _selective_log_softmax(comp_logits, targets)  # [T_comp]


def _attempt_ref_turn_logprobs(model, attempt: Attempt, device) -> list:
    """
    Reference (LoRA-disabled) per-token log-probs, ONE detached tensor per
    scored turn (turns with completion tokens). No grad, so nothing is retained
    across turns -- we materialize all of them cheaply and index by turn below.
    """
    out = []
    with torch.no_grad(), model.disable_adapter():
        for t in attempt.turns:
            if t.completion_ids:
                out.append(_turn_token_logprobs(model, t, device).detach())
    return out



# Normalizers and terms
def _norm_denominator(n_tokens: int, cfg: SCoReConfig) -> float:
    if cfg.length_norm == "constant":
        return float(cfg.max_new_tokens)
    return float(max(n_tokens, 1))            # "sequence"



# Per-attempt gradient accumulation (memory bounded to ONE turn's graph)
def _attempt_denominator(attempt: Attempt, cfg: SCoReConfig) -> float:
    """
    Length-normalizer for the whole attempt, computed from token counts alone
    (no forward). constant -> max_new_tokens; sequence -> total completion tokens.
    """
    n = sum(len(t.completion_ids) for t in attempt.turns if t.completion_ids)
    return _norm_denominator(n, cfg)


def _accumulate_attempt_grads(model, attempt: Attempt, pg_reward, kl_coeff: float,
                              cfg: SCoReConfig, device, scale: float) -> dict:
    """
    Backward the attempt's SCoRe contribution ONE TURN AT A TIME, accumulating
    into `.grad`. This is mathematically identical to building the full loss
    -(pg_reward/denom)*sum(logp) + (kl_coeff/denom)*sum(k3) and calling one
    backward, because both terms are linear sums over turns and `denom` is a
    per-attempt constant known from token counts. The point is memory: only one
    turn's autograd graph is ever alive, so peak VRAM no longer grows with the
    number of turns (the true cause of the many-turn OOM).

    pg_reward None or 0.0 -> no policy-gradient term for this attempt.
    kl_coeff 0.0          -> no KL anchor (skips the reference forwards).
    `scale` folds in the batch 1/n so callers need not rescale grads.

    Returns scalar metric sums (pg and kl loss contributions, pre-scale) as
    plain floats; holds no live tensors.
    """
    scored = [t for t in attempt.turns if t.completion_ids]
    denom = _attempt_denominator(attempt, cfg)
    do_pg = pg_reward is not None and pg_reward != 0.0
    do_kl = kl_coeff != 0.0

    ref_turns = _attempt_ref_turn_logprobs(model, attempt, device) if do_kl else []

    pg_sum, kl_sum = 0.0, 0.0
    for i, turn in enumerate(scored):
        token_lp = _turn_token_logprobs(model, turn, device)   # [T_comp], grad

        turn_loss = token_lp.new_zeros(())
        if do_pg:
            pg_contrib = (token_lp.sum() / denom) * pg_reward
            turn_loss = turn_loss - pg_contrib
            pg_sum += (-pg_contrib).item()
        if do_kl:
            d = ref_turns[i] - token_lp                        # log(ref/policy)
            k3 = torch.exp(d) - d - 1.0
            kl_contrib = kl_coeff * (k3.sum() / denom)
            turn_loss = turn_loss + kl_contrib
            kl_sum += kl_contrib.item()

        # Skip the backward if this turn contributes nothing differentiable
        # (e.g. pg_reward==0 and kl_coeff==0): keeps grad graph clean.
        if do_pg or do_kl:
            (turn_loss * scale).backward()
        del token_lp, turn_loss

    return {"pg": pg_sum, "kl": kl_sum}



# Stage updates (accumulate grads for one episode; caller steps the optimizer)
def stage1_backward(model, episode: Episode, cfg: SCoReConfig, device, scale: float):
    """
    max  E[ r(y2) - beta2 * KL(pi(y1) || ref) ]
    -> minimize  -pg(y2) + beta2 * kl(y1)
    Only y2 receives a policy-gradient signal; y1 is held near the reference.
    Grads for this episode are accumulated (scaled by `scale`); no optimizer step.
    """
    m2 = _accumulate_attempt_grads(model, episode.y2, episode.y2.reward, 0.0,
                                   cfg, device, scale)
    m1 = _accumulate_attempt_grads(model, episode.y1, None, cfg.beta2,
                                   cfg, device, scale)
    total = m2["pg"] + m1["kl"]
    metrics = {
        "loss/total": total,
        "loss/pg_y2": m2["pg"],
        "loss/kl_y1": m1["kl"],
        "reward/y1": episode.y1.reward,
        "reward/y2": episode.y2.reward,
    }
    return total, metrics


def stage2_backward(model, episode: Episode, cfg: SCoReConfig, device, scale: float):
    """
    max  E[ r(y1) + r_shaped(y2) - beta1*(KL(pi(y1)||ref) + KL(pi(y2)||ref)) ]
    with r_shaped(y2) = r(y2) + alpha*(r(y2) - r(y1)).
    Both attempts receive a policy-gradient signal; both are KL-anchored.
    """
    r1, r2_shaped = episode.shaped_rewards(cfg.alpha)

    m1 = _accumulate_attempt_grads(model, episode.y1, r1, cfg.beta1,
                                   cfg, device, scale)
    m2 = _accumulate_attempt_grads(model, episode.y2, r2_shaped, cfg.beta1,
                                   cfg, device, scale)
    total = m1["pg"] + m2["pg"] + m1["kl"] + m2["kl"]
    metrics = {
        "loss/total": total,
        "loss/pg_y1": m1["pg"],
        "loss/pg_y2": m2["pg"],
        "loss/kl_y1": m1["kl"],
        "loss/kl_y2": m2["kl"],
        "reward/y1": r1,
        "reward/y2_raw": episode.y2.reward,
        "reward/y2_shaped": r2_shaped,
        "reward/bonus": r2_shaped - episode.y2.reward,
    }
    return total, metrics


def stage_backward(stage: int, model, episode: Episode, cfg: SCoReConfig, device,
                   scale: float):
    fn = stage1_backward if stage == 1 else stage2_backward
    return fn(model, episode, cfg, device, scale)
