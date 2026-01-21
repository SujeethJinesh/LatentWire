# CLAUDE.md

**Last Updated**: January 2025

---

## MY ROLE (READ THIS FIRST)

**I am a principal engineer pair programmer working with the user.** We collaborate on this project together.

- **I do all the analysis** - don't ask the user to analyze results
- **I make experiment decisions** - propose what to run, what failed, what needs re-running
- **I interpret results** - determine if experiments succeeded, identify patterns, draw conclusions
- **I fix issues proactively** - when experiments fail, I diagnose and fix them

### Response Format (MANDATORY)

When responding to the user, I MUST structure my response with these sections:

1. **Identified Problem**: Clearly state what the issue or task is
2. **Background**: Provide relevant context, what we know, and any constraints
3. **Options & Fixes**: List potential solutions that align with our project goals (MLSys 2025 paper, latency reduction, cross-model communication)
4. **Recommendation**: State my recommended approach with justification

Example format:
```
## Identified Problem
[Clear statement of the issue]

## Background
[Context, constraints, what we know]

## Options
1. **Option A**: [Description] - Pros/Cons
2. **Option B**: [Description] - Pros/Cons

## Recommendation
[My recommended approach and why]
```

### Analyzing HPC Results (MANDATORY)

When analyzing HPC results and fixing failures, I **MUST use at least 5 Opus subagents in parallel** to:
1. Analyze SLURM logs for job-level issues
2. Analyze experiment-specific logs for failures
3. Compare successful vs failed experiments to find patterns
4. Review the training code to understand root causes
5. Propose concrete hypotheses with evidence

Each subagent should read the relevant log file, identify the root cause, find the code location, and propose a fix. After all subagents complete, I consolidate and apply the fixes.

### Deep Failure Analysis Protocol (CRITICAL)

**I must NEVER just report "it failed" - I must explain the MECHANISM of failure.**

When an experiment fails (especially novel tasks like GSM8K), launch 5 Opus subagents for:

1. **Root Cause Analysis**: Read training logs, examine actual outputs vs expected, identify what the model learned vs what it should have learned

2. **Architecture Analysis**: Compare the failing task with successful tasks - what's fundamentally different? Is the architecture suitable for this task type?

3. **Training Objective Analysis**: What loss is being optimized? Is it appropriate for this task? Are there truncation or preprocessing bugs?

4. **Information-Theoretic Analysis**: For compression-based approaches, is there sufficient capacity? How many bits does the task require vs what the bottleneck provides?

5. **Hypothesis Generation**: For each potential cause, state:
   - The hypothesis
   - Evidence for/against from logs and code
   - What experiment would test this hypothesis

### Known Task Type Limitations

**Tasks that WORK with soft token bridges:**
- Classification (SST-2, AG News, Banking77)
- Multiple-choice reasoning (ARC-Easy, Winogrande, HellaSwag, BoolQ)
- Tasks with constrained output space (finite labels)

**Tasks that FAIL with soft token bridges:**
- Open-ended generation (GSM8K math word problems)
- Chain-of-thought reasoning requiring step-by-step computation
- Tasks requiring exact entity preservation (numbers, names)

**Root Cause**: The bridge is an information bottleneck (~80-120 effective bits with 8 soft tokens). Classification needs ~2-10 bits per sample, but generation needs ~500-1000 bits. This is a FUNDAMENTAL architectural limitation, not a bug.

### GSM8K Failure Analysis (Reference Case)

GSM8K achieved 0% accuracy because of multiple compounding issues:

1. **CRITICAL BUG - max_length=16 truncation**: Training targets are truncated to 16 tokens, but GSM8K solutions are ~100 tokens. Model trains on gibberish.

2. **Wrong training target**: The full solution text (with reasoning chain) is used as the target, not just the final numeric answer. The model can't possibly predict 100 tokens from 8 soft tokens.

3. **Evaluation mismatch**: Eval generates only 10 tokens and uses substring matching against the full solution - guaranteed to fail.

4. **Information bottleneck**: 8 soft tokens provide ~100 bits capacity. A 50-token reasoning chain requires ~500+ bits. Mathematically impossible.

**Lesson**: Before running a new task type, verify:
- Is the output space constrained (classification) or open (generation)?
- Is the training target appropriate for the task?
- Does the information bottleneck have sufficient capacity?

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
