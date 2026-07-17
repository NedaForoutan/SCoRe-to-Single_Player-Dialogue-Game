# **Applying Self-Correction via Reinforcement Learning (SCoRe) to Multi-Step Dialogue Games**

This repository adapts **Self-Correction via Reinforcement Learning (SCoRe)** to the **PLAYPEN** benchmark for training Large Language Models (LLMs) on **single-player multi-turn dialogue games**.

Unlike the original SCoRe work, which focuses on **single-step mathematical reasoning and code generation**, this project demonstrates how iterative self-correction can improve **long-horizon conversational agent behavior**.

---

## Overview

Large Language Models are increasingly used as autonomous agents that interact with environments over multiple conversational turns. Existing post-training methods are primarily designed for static reasoning or single-turn instruction following.

In this work, we adapt **SCoRe** to the **PLAYPEN** dialogue game benchmark by training **Qwen3.5-9B** with iterative self-correction and interaction-level reinforcement learning.

The resulting model significantly improves interactive gameplay performance while preserving general language capabilities.

---

## Features

- ✅ Two-stage SCoRe reinforcement learning
- ✅ Single-player PLAYPEN dialogue game training
- ✅ Self-correction prompt generation
- ✅ Episode-level reinforcement learning rewards
- ✅ 4-bit QLoRA training
- ✅ Publicly available trained model

---

## Training Games

The model is trained on five PLAYPEN single-player games:

- AdventureGame
- TextMapWorld
- TextMapWorld GraphReasoning
- TextMapWorld SpecificRoom
- Wordle

---

## Results

|          Model       | ClemScore | StatScore |
|----------------------|----------:|----------:|
| Qwen3.5-9B           |   41.18   | **54.35** |
| **SCoRe-Qwen3.5-9B** | **43.31** |   52.71   |

The SCoRe fine-tuned model

- improves **ClemScore** by **+5.98**
- preserves and slightly improves **StatScore**

showing that self-correction reinforcement learning improves interactive dialogue-game performance without sacrificing general language understanding.

---

## Training

Training follows the original two-stage SCoRe algorithm.

### Stage I

Constrained initialization using KL-divergence to improve the second attempt while keeping the first attempt close to the base model.

### Stage II

Reward-shaped reinforcement learning encourages successful self-correction by rewarding improvements between the first and second attempts while penalizing regressions.

For each episode

1. the model plays the dialogue game once,
2. receives a self-correction prompt summarizing the outcome,
3. plays the same game a second time,
4. receives an episode-level reward.

---

---

## Repository Structure

```
.
├── score_rl/                  # SCoRe reinforcement learning implementation
│   ├── trainer.py             # RL training loop
│   ├── rollout.py             # On-policy rollout generation
│   ├── losses.py              # SCoRe loss functions
│   ├── evaluate.py            # Evaluation utilities
│   ├── config.py              # Training configuration
│   ├── env.py                 # Dialogue game environment
│   ├── model.py               # Model loading and inference
│   ├── main.py                # Main training entry point
│   └── ...
│
├── playpen-eval/              # Evaluation results on the PLAYPEN benchmark
│   ├── Qwen3.5-9B/            # Results of the base Qwen3.5-9B model
│   └── score-Qwen3.5-9B-sp/   # Results of the SCoRe fine-tuned model
│
├── transcripts/               # Dialogue game transcripts generated during evaluation
│
├── playpen/                   # PLAYPEN game interface and utilities
│
├── clembench/                 # CLEMbench benchmark framework
│
├── examples/                  # Example scripts and usage examples
│
└── README.md
```

### Main Directories

| Directory | Description |
|-----------|-------------|
| `score_rl/` | Implementation of the SCoRe reinforcement learning algorithm, including training, rollouts, reward computation, and evaluation. |
| `playpen-eval/` | Evaluation outputs on the PLAYPEN benchmark for both the base Qwen3.5-9B model and the SCoRe fine-tuned model. |
| `transcripts/` | Example dialogue-game transcripts generated during evaluation, illustrating model interactions with the game environment. |
| `playpen/` | PLAYPEN environment and game wrappers used during training and evaluation. |
| `clembench/` | CLEMbench framework used for executing and evaluating dialogue games. |
| `examples/` | Example scripts demonstrating how to run training and evaluation. |

---

## Released Model

The fine-tuned model is available on Hugging Face:

**https://anonymous-hf.up.railway.app/a/447autv89swb/**

---
## Resources

- PLAYPEN Benchmark: https://huggingface.co/datasets/colab-potsdam/playpen-data
- LM Playschool Challenge: https://lm-playschool.github.io/challenge/
- Released Model: https://anonymous-hf.up.railway.app/a/447autv89swb/

---

## License

This repository is released under the MIT License.
