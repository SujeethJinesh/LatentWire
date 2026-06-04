# plans/LOCAL_ASSETS.md — local assets (FILL IN the real paths before launch)

> Host for pre-launch: **MacBook (Apple Silicon, MPS, 64 GB, NO CUDA)** → do CPU/cached/small-model work only; PARK all CUDA / W4A16 / 30B / long-decode / confirmation / profiler jobs as `parked: needs_gpu` for the GPU node.

## Repo layout
- Channel-Set / outlier_migrate code: `experimental/outlier_migrate/`
- LatentWire code and historical score caches: repo root plus `results/source_private_*`
- Python env + activate command: `source venv_arm64/bin/activate`

## Models (small, local — for the Mac use ≤ ~1B on CPU/MPS)
- LatentWire scoring: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-0.6B`, `microsoft/Phi-3-mini-4k-instruct`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Channel-Set reasoning models (GPU-node only — do NOT load on the Mac): `ibm-granite/granite-4.0-h-small`, `nvidia/Nemotron-3-Nano-30B-A3B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `tiiuae/Falcon-H1-0.5B`

## Existing caches to REUSE (verified present — do NOT regenerate)
- Channel-Set: experimental/outlier_migrate/phase*/results/ (activation / KL / ParoQuant / drift packets) — ~6.3 G
- LatentWire: results/source_private_* , results/latentwire_colm_v3_* (source-private score caches, controls) — ~5.6 G
- IMPORTANT: run the Stage-0 cache-leakage protocol (plans/MAC_PRELAUNCH.md) over these before screening.

## Datasets (local paths or HF)
- Channel-Set GPU-node datasets: AIME-2024/2025, MATH-500, GPQA-Diamond; do not generate new long-reasoning traces on this Mac pass.
- LatentWire Mac datasets/caches: existing `results/source_private_*` frozen rows, plus public ARC-Challenge, OpenBookQA, MMLU-Pro, and GSM8K only for dev/gate slices if a local runner explicitly needs fresh small-model scoring.

## Hardware
- Now: MacBook (MPS, 64 GB, no CUDA).
- Later: 1x RTX Pro 6000 (~96 GB), GPU work serialized; CPU screens wide.
