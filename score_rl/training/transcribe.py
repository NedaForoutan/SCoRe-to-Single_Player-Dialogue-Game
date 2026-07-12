"""
score_rl/transcribe.py

Post-hoc transcript generation: replay SCoRe y1 -> y2 rollouts with a saved
MERGED checkpoint.

Inference only: no gradients, no optimizer, no LoRA handling needed since
`--model_path` already points at merged weights. Reuses the same in-process
clemcore GameBenchmark rollout path as training (env.py / rollout.py), so the
model sees IDENTICAL prompt structure to what it was trained under -- these
are fresh rollouts with the final model, not a replay of specific training
episodes (decoded text was never saved during training, only token ids used
for the loss).

Usage:
    python -m score_rl.transcribe \
        --model_path /path/to/final_merged \
        --games textmapworld_specificroom wordle \
        --n_instances 10 \
        --out_dir transcripts/ \
        --only_corrected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import SCoReConfig
from .env import GameHandle, LearnerBackend
from .rollout import _reflection_preamble


class TranscribingBackend(LearnerBackend):
    """
    Same generation behavior as LearnerBackend (so the model sees IDENTICAL
    prompts to training), but additionally keeps a decoded, human-readable
    transcript of every turn for the CURRENT attempt. Reset on new_attempt(),
    same as the token-id Attempt it wraps.

    `messages` passed into generate_response() is the FULL running
    conversation history each call (clemcore convention) -- crucially, this
    means the model's OWN previous reply reappears inside `messages` on the
    *next* call (clemcore feeds it back as context). So we log new turns by
    diffing history length, and must NOT also manually append the reply right
    after generating it -- that reply will get picked up by the diff on the
    next call anyway. The one exception is the attempt's FINAL reply, which
    has no "next call" to catch it -- that one is flushed explicitly via
    flush_pending() after play_attempt() returns.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.transcript: list[dict] = []
        self._prefix_logged = False
        self._history_len = 0
        self._pending_text: str | None = None

    def new_attempt(self, prefix_messages=None):
        self.transcript = []
        self._prefix_logged = False
        self._history_len = 0
        self._pending_text = None
        return super().new_attempt(prefix_messages)

    def generate_response(self, messages: list[dict]):
        if self.prefix_messages and not self._prefix_logged:
            for m in self.prefix_messages:
                self.transcript.append(dict(m))
            self._prefix_logged = True

        new_msgs = list(messages)[self._history_len:]
        for m in new_msgs:
            self.transcript.append(dict(m))
        self._history_len = len(messages)

        prompt_obj, resp_obj, text = super().generate_response(messages)
        # Do NOT append here -- if there's a next turn, the diff above will
        # log this reply then (clemcore re-feeds it as part of `messages`).
        self._pending_text = text
        return prompt_obj, resp_obj, text

    def flush_pending(self):
        """Call once after an attempt finishes, to log its final reply
        (the one no subsequent generate_response() call ever re-surfaces)."""
        if self._pending_text is not None:
            self.transcript.append({"role": "assistant", "content": self._pending_text})
            self._pending_text = None


def transcribe_episode(handle: GameHandle, backend: TranscribingBackend,
                        instance_key, cfg: SCoReConfig):
    """One instance, y1 then y2 (self-correction), with full text captured."""
    backend.prefix_messages = []
    y1 = handle.play_attempt(backend, instance_key)
    backend.flush_pending()
    y1_transcript = backend.transcript
    if not y1.has_tokens():
        return None

    backend.prefix_messages = _reflection_preamble(y1, cfg)
    y2 = handle.play_attempt(backend, instance_key)
    backend.flush_pending()
    y2_transcript = backend.transcript
    backend.prefix_messages = []
    if not y2.has_tokens():
        return None

    return {
        "game": handle.game_name,
        "instance_key": list(instance_key),
        "y1": {"transcript": y1_transcript, "success": y1.success,
               "aborted": y1.aborted, "reward": y1.reward},
        "y2": {"transcript": y2_transcript, "success": y2.success,
               "aborted": y2.aborted, "reward": y2.reward},
        "corrected": (not y1.success) and y2.success,   # matches Episode.corrected()
        "regressed": y1.success and (not y2.success),   # matches Episode.regressed()
    }


def to_markdown(ep: dict) -> str:
    lines = [f"# {ep['game']} — instance {ep['instance_key']}", ""]
    tag = ("SELF-CORRECTED (y1 failed -> y2 succeeded)" if ep["corrected"] else
           "REGRESSED (y1 succeeded -> y2 failed)" if ep["regressed"] else
           "no change")
    lines.append(f"**Outcome:** {tag}\n")
    for label, attempt in (("Attempt 1 (y1)", ep["y1"]),
                            ("Attempt 2 (y2, after self-correction prompt)", ep["y2"])):
        lines.append(f"## {label}")
        lines.append(f"*success={attempt['success']}, aborted={attempt['aborted']}, "
                     f"reward={attempt['reward']}*\n")
        for turn in attempt["transcript"]:
            role = turn.get("role", "?")
            content = turn.get("content", "")
            lines.append(f"**{role}:** {content}\n")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="path to MERGED weights (e.g. final_merged/)")
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--n_instances", type=int, default=10)
    p.add_argument("--out_dir", default="transcripts")
    p.add_argument("--only_corrected", action="store_true",
                   help="save only episodes where y1 failed and y2 succeeded")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="lower (e.g. 0.3) for cleaner, more reproducible "
                        "showcase transcripts; 0.8 matches the training regime")
    args = p.parse_args()

    cfg = SCoReConfig(model_id=args.model_path, games=tuple(args.games),
                       enable_thinking=False, temperature=args.temperature)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb, device_map={"": 0},
        dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    backend = TranscribingBackend(model, tokenizer, cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for game in args.games:
        with GameHandle(game, cfg) as handle:
            keys = handle.instance_keys[: args.n_instances]
            for key in keys:
                ep = transcribe_episode(handle, backend, key, cfg)
                if ep is None:
                    print(f"[skip] {game} {key}: no tokens (infra drop)")
                    continue
                if args.only_corrected and not ep["corrected"]:
                    continue
                fname = f"{game}_{key[0]}_{key[1]}"
                (out_dir / f"{fname}.md").write_text(to_markdown(ep))
                (out_dir / f"{fname}.json").write_text(json.dumps(ep, indent=2))
                print(f"[saved] {fname}  corrected={ep['corrected']} "
                      f"regressed={ep['regressed']}")


if __name__ == "__main__":
    main()