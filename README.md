# RL Fine-Tuning for LLMs

CS224R final project repository for studying reinforcement learning methods for post-training language models on mathematical reasoning tasks.

## Overview

This project explores how reinforcement learning can improve the reasoning ability of language models after supervised fine-tuning. The default task is **Countdown**, where the model is given a set of numbers and a target value, and must generate a sequence of arithmetic operations that reaches the target.

The project follows a staged fine-tuning pipeline:

1. **Supervised Fine-Tuning (SFT)**  
   Warm-start the model using supervised examples.

2. **Preference Optimization (IPO/DPO-style)**  
   Fine-tune the model using pairwise preference data.

3. **Online RL with RLOO**  
   Improve the model using verifier-based rewards and online policy-gradient updates.

4. **Research Extension**  
   Explore an additional idea to improve performance, stability, efficiency, or generalization.

## Project Structure

```text
rl-finetuning-llms/
├── default_proj/
│   ├── evaluation/              # Countdown verifier and evaluation utilities
│   ├── ipo_trainer/             # Preference optimization training code
│   ├── rloo_trainer/            # Online RL / RLOO training code
│   ├── sft_trainer/             # Supervised fine-tuning code
│   ├── modal_train.py           # Modal training entry point
│   ├── modal_requirements.txt   # Modal dependencies
│   ├── pyproject.toml
│   └── README.md
└── README.md
