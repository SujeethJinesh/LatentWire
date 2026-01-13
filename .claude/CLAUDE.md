# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: January 2025

## Project Overview

Telepathy is a cross-model communication system for MLSys 2025. It enables a "sender" LLM (Llama) to transmit information to a "receiver" LLM (Mistral) through trained soft tokens, achieving significant latency reduction over text-based communication.

## Repository Structure

```
LatentWire/
├── latentwire/                 # Core library
│   ├── train.py                # Training loop
│   ├── eval.py                 # Evaluation
│   ├── models.py               # Encoder, Adapter, LMWrapper
│   ├── losses.py               # Loss functions
│   └── data.py                 # Dataset loading
├── telepathy/                  # Paper experiments
│   ├── train_telepathy.py      # Unified training script
│   ├── eval_telepathy.py       # Unified evaluation script
│   ├── run_baselines.py        # All baselines (zeroshot, fewshot, lora, prompt_tuning)
│   ├── run_benchmarks.py       # Latency/throughput benchmarks
│   ├── linear_probe_baseline.py
│   ├── run_enhanced_paper_evaluation.py
│   ├── submit_enhanced_paper_eval.slurm  # HPC submission script
│   └── paper_writing/          # LaTeX source
├── scripts/                    # Analysis utilities
├── latent_bridge_v15.py        # Bridge architecture classes
├── requirements.txt
└── runs/                       # Output directory (created at runtime)
```

## Development Environment

**Split development and execution environment:**

- **Local development**: MacBook for code editing, analysis, reviewing logs
- **Training execution**: HPC cluster with H100 GPUs

**Workflow:**
1. Develop/modify code locally
2. Push to git
3. Run on HPC: `git pull && PYTHONPATH=. sbatch telepathy/submit_enhanced_paper_eval.slurm`
4. HPC pushes results back to git
5. Pull and analyze locally: `git pull`

## SLURM Job Submission

**CRITICAL SLURM settings for Marlowe HPC:**

| Setting | Value |
|---------|-------|
| `--account` | `marlowe-m000066` |
| `--partition` | `preempt` |
| Working dir | `/projects/m000066/sujinesh/LatentWire` |

**Example commands:**
```bash
# On HPC:
cd /projects/m000066/sujinesh/LatentWire
git pull
sbatch telepathy/submit_enhanced_paper_eval.slurm

# Monitor:
squeue -u $USER
tail -f runs/enhanced_eval_*.log
```

## Key Scripts

### Training
```bash
# Train bridge on SST-2
python telepathy/train_telepathy.py --dataset sst2 --soft_tokens 8 --steps 2000

# Supported datasets: sst2, agnews, trec, banking77
```

### Evaluation
```bash
python telepathy/eval_telepathy.py --checkpoint runs/sst2/bridge_sst2.pt --dataset sst2
```

### Baselines
```bash
# Zero-shot, few-shot, LoRA, prompt tuning
python telepathy/run_baselines.py --baseline zeroshot --dataset sst2
python telepathy/run_baselines.py --baseline fewshot --dataset sst2 --shots 5
python telepathy/run_baselines.py --baseline lora --dataset sst2 --rank 8
python telepathy/run_baselines.py --baseline prompt_tuning --dataset sst2 --soft_tokens 8
```

### Benchmarks
```bash
python telepathy/run_benchmarks.py --benchmark latency --checkpoint runs/sst2/bridge_sst2.pt
```

## Development Principles

1. **NEVER create new files unless explicitly requested** - edit existing files
2. **Always commit and push after completing tasks**
3. **Always `git pull` before analyzing results**
4. **Use Opus subagents for complex analysis tasks**
5. **Scripts must work end-to-end from scratch** - no pre-existing checkpoints

## Common Commands

```bash
# Set environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify Python files compile
python3 -m py_compile latentwire/*.py telepathy/*.py

# Run full paper evaluation on HPC
sbatch telepathy/submit_enhanced_paper_eval.slurm
```
