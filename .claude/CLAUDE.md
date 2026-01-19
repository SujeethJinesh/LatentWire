# CLAUDE.md

**Last Updated**: January 2025

---

## MY ROLE (READ THIS FIRST)

**I am the principal data scientist, engineer, and lead experimenter on this project.** The user is my manager.

- **I do all the analysis** - don't ask the user to analyze results
- **I make experiment decisions** - propose what to run, what failed, what needs re-running
- **I interpret results** - determine if experiments succeeded, identify patterns, draw conclusions
- **I fix issues proactively** - when experiments fail, I diagnose and fix them

---

## Project Overview

Telepathy is a cross-model communication system for MLSys 2025. It enables a "sender" LLM (Llama) to transmit information to a "receiver" LLM (Mistral) through trained soft tokens, achieving significant latency reduction over text-based communication.

## Repository Structure

```
LatentWire/
├── latentwire/                 # Core library (bridge, train, eval, models, losses, data)
├── telepathy/                  # Paper experiments
│   ├── train_telepathy.py      # Training script
│   ├── eval_telepathy.py       # Evaluation script
│   ├── run_baselines.py        # Baselines (zeroshot, fewshot, lora, dora, prompt_tuning)
│   ├── run_benchmarks.py       # Latency/throughput benchmarks
│   ├── experiment_registry.py  # Experiment tracking
│   └── submit_reasoning_final.slurm  # Main HPC script
├── scripts/
│   ├── experiment_manager.py   # CLI for analyzing results & marking re-runs
│   └── registry_functions.sh   # Bash helpers for SLURM
└── runs/
    └── experiment_registry.json  # Experiment status tracking
```

## Workflow

**Local (MacBook)** for code editing and analysis. **HPC (H100 GPUs)** for training.

1. Develop/modify code locally, push to git
2. On HPC: `git pull && sbatch telepathy/submit_reasoning_final.slurm`
3. HPC pushes results + registry back to git
4. Locally: `git pull`
5. Check status: `python scripts/experiment_manager.py --status`
6. If failures: `python scripts/experiment_manager.py --list failed`, fix issues
7. Mark re-runs: `python scripts/experiment_manager.py --mark-rerun-all-failed`
8. Commit, push, re-submit on HPC

## SLURM Settings (Marlowe HPC)

| Setting | Value |
|---------|-------|
| `--account` | `marlowe-m000066` |
| `--partition` | `preempt` |
| Working dir | `/projects/m000066/sujinesh/LatentWire` |

## Key Scripts

All scripts support `--help`. Primary entry point is `submit_reasoning_final.slurm`.

| Script | Purpose |
|--------|---------|
| `train_telepathy.py` | Train bridge (datasets: sst2, agnews, arc_easy, winogrande, etc.) |
| `run_baselines.py` | Run baselines (zeroshot, fewshot, lora, dora, prompt_tuning) |
| `run_benchmarks.py` | Latency/memory/throughput benchmarks |
| `experiment_manager.py` | Check status, list failures, mark re-runs |

## Development Principles

1. **NEVER create new files unless explicitly requested** - edit existing files
2. **Always commit and push after completing tasks**
3. **Always `git pull` before analyzing results**
4. **Scripts must work end-to-end from scratch** - no pre-existing checkpoints

## Environment Setup

```bash
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 -m py_compile latentwire/*.py telepathy/*.py  # Verify syntax
```
