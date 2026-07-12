"""
Model loading.

A SLURM job sets `--model_id` and nothing else changes. 
  * 4-bit QLoRA fits any of Llama-3.1-8B / Gemma / Qwen on an A30 GPU room
    to spare (the reference policy is the SAME weights via disable_adapter()).

Reference policy note: we NEVER load a second model. KL uses
`model.disable_adapter()` (see losses.py). On 2x  this leaves an entire GPU
free for the multiplayer teacher later.
"""

from __future__ import annotations

import torch

from .config import SCoReConfig


def _as_text_tokenizer(tokenizer):
    """
    Multimodal checkpoints (e.g. Qwen3.5-9B) load as a `Processor`, whose
    `apply_chat_template` expects message content to be a list of typed parts
    ({"type": "text", ...}) and blows up on our plain-string messages with
    `TypeError: string indices must be integers`. The Processor wraps the real
    text tokenizer at `.tokenizer` -- unwrap it so the text-only path works
    unchanged. Plain tokenizers have no `.tokenizer`, so they pass through.
    """
    inner = getattr(tokenizer, "tokenizer", None)
    return inner if inner is not None else tokenizer


def load_learner(cfg: SCoReConfig, max_seq_length: int = 4096):
    """
    Returns (model, tokenizer). The model has a trainable LoRA adapter attached
    and, when cfg.fast_inference, a colocated vLLM engine for generation.
    """
    from unsloth import FastLanguageModel  # imported here so config import stays light

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=cfg.fast_inference,   # vLLM path
        max_lora_rank=cfg.lora_r,
        dtype=None,                          # Unsloth auto: bf16 where supported
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    tokenizer = _as_text_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_learner_plain(cfg: SCoReConfig, device_map, max_seq_length: int = 4096):
    """
    Plain-transformers QLoRA loader for the PARALLEL path (cfg.parallel_gpus > 1).
    `AutoModelForCausalLM` + bitsandbytes 4-bit + PEFT 

    Numerics match the Unsloth 4-bit path as closely as the stack allows: nf4 +
    double-quant + bf16 compute, LoRA on the same 7 projections, dropout 0. 

    `device_map` places this replica: pass a specific GPU (e.g. {"": 3}) so each
    replica lives entirely on one device. Returns (model, tokenizer) with a fresh
    trainable LoRA adapter attached.
    """
    import torch as _torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb,
        device_map=device_map,
        dtype=_torch.bfloat16,
    )
    # QLoRA needs the frozen base prepared (fp32 layernorms, input-grad hook,
    # gradient checkpointing) before adapters are attached.
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer = _as_text_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.max_length = None

    return model, tokenizer


def load_for_eval(cfg: SCoReConfig, adapter_path: str | None = None,
                  max_seq_length: int = 4096):
    """
    Load a model for evaluation (no new trainable adapter).
      * adapter_path given -> loads base + that saved LoRA adapter (a checkpoint).
      * adapter_path None  -> loads the bare base model (the pre-training baseline).
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path or cfg.model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        dtype=None,
    )
    tokenizer = _as_text_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_optimizer(model, cfg: SCoReConfig):
    """8-bit AdamW over LoRA params only."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.use_8bit_optimizer:
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(params, lr=cfg.lr)
    return torch.optim.AdamW(params, lr=cfg.lr)
