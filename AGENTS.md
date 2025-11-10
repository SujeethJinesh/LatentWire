# Repository Guidelines

## Project Structure & Module Organization
All scoped work lives in `paper_writing/`. `cross_attention.py` is the DDP training driver that translates Mistral hidden states into soft tokens for Llama, while `run_ablations.sh` orchestrates the multi-config sweeps. `PLAN.md`, `EXPERIMENTS_SUMMARY.md`, and `README.md` track timelines plus expected metrics—update them whenever a run completes so downstream writing stays accurate. Outputs are stored under `paper_writing/runs/ablations_YYYYMMDD_HHMMSS/CFG_NAME/` with checkpoints, `train.log`, and any auto-generated analyzers.

## Build, Test, and Development Commands
Run the full Week‑1 suite with `PYTHONPATH=. bash paper_writing/run_ablations.sh` (expects 4× H100; recreates `runs/`). For single configs, use `torchrun --nproc_per_node=4 paper_writing/cross_attention.py --soft_tokens 64 --depth 8 --lr 1e-4 --train_steps 3000 --bf16` and override flags as needed. Post-training analysis is two-stage: `python paper_writing/analyze_compression.py --checkpoint paper_writing/runs/ablations_*/1a_stable_64tok/checkpoint.pt --num_samples 200`, then `cd paper_writing/runs/ablations_* && python analyze_ablations.py` to emit `ablation_results.json` and `summary.log`. Keep `PYTHONPATH=.` in every command so translator imports resolve.

## Coding Style & Naming Conventions
Python sources use 4-space indentation, type hints, and module-level dataclasses for configs; mirror the layout already in `cross_attention.py`. Deterministic training (manual seeds, disabled cuDNN benchmark, rank-aware seeding) is part of the contract—keep those guards intact. Name experiments with the existing scheme `(<index><letter>)_<descriptor>_<tokens>` and leave timestamps in `runs/ablations_YYYYMMDD_HHMMSS` for chronological diffs.

## Testing Guidelines
There is no standalone unit suite; “tests” are experiment validations. Treat `run_ablations.sh` as the integration harness, then verify each run via `summary.log`, per-run `train.log`, and the JSON emitted by `analyze_ablations.py`. Before claiming success, confirm bridged accuracy, degradation (peak minus final), and KV-cache savings match `EXPERIMENTS_SUMMARY.md`, and paste the supporting log snippet or metrics table into your PR. Re-run `analyze_compression.py` whenever you change quantization knobs or checkpoint formats.

## Commit & Pull Request Guidelines
Recent history follows `type: short imperative` (e.g., `fix: Improve model loading`, `refactor: Clean up generation`). Keep subject lines under ~70 chars, use lowercase types (`feat|fix|refactor|docs|chore`), and summarize metrics or log paths in the body whenever experiments are updated. Pull requests should reference the corresponding plan item, describe datasets/configs touched, link the relevant `runs/.../train.log` or JSON artifact, and flag any pending items (plots, reruns, missing baselines).

## Experiment Hygiene & Configuration Tips
Start from a clean `paper_writing/runs/` for reproducible ablations (`rm -rf paper_writing/runs` before multi-hour jobs). All sanctioned configs use `seed=1234`, InfoNCE weight `0.05`, early-stopping patience `5`, and repetition penalty `1.1`; deviations must be justified in `PLAN.md`. Monitor logs every few hours (`tail -f runs/.../train.log`) to catch divergence early, and upload checkpoints plus analysis artifacts before pruning local state.
