# COCONUT Reproduction

Reproducing Meta AI's COCONUT (Chain of Continuous Thought) paper with Llama 3.1 8B on GSM8k.

## Paper
**Title**: "Training Large Language Models to Reason in a Continuous Latent Space"
**ArXiv**: https://arxiv.org/html/2412.06769v1
**Date**: December 2024

## Quick Start

1. **Review the plan**: Read `PLAN.md` for detailed milestone breakdown
2. **Start with Milestone 1**: Run data loading script
   ```bash
   cd experimental/learning/reproduce_coconut
   python step1_data.py
   ```
3. **Follow milestones sequentially**: Each builds on the previous

## Structure

```
reproduce_coconut/
├── README.md              # This file
├── PLAN.md               # Detailed implementation plan
├── step1_data.py         # Milestone 1: Data loading
├── step2_tokens.py       # Milestone 2: Add special tokens
├── step3_stage0.py       # Milestone 3: CoT baseline
├── step4_continuous.py   # Milestone 4: Continuous thought mechanism
├── step5_stage1.py       # Milestone 5: Stage 1 training
├── step6_eval.py         # Milestone 6: Evaluation
└── runs/                 # Checkpoints and logs
    ├── stage0/           # CoT baseline
    └── stage1/           # COCONUT with continuous thoughts
```

## Goals

- **Stage 0**: Fine-tune Llama 3.1 8B on GSM8k with standard chain-of-thought
- **Stage 1**: Train with continuous thoughts (hidden states as embeddings)
- **Evaluation**: Compare accuracy and reasoning efficiency

## Key Innovation

Instead of generating reasoning steps as text tokens, COCONUT feeds hidden states directly back as input embeddings, enabling:
- More efficient reasoning (fewer tokens)
- Breadth-first search patterns (multiple paths encoded in latent space)
- Better performance on problems requiring backtracking

## Timeline

- Coding: ~3 hours (6 milestones)
- Training: ~2 hours (Stage 0 + Stage 1)
- Total: ~5 hours for minimal reproduction
