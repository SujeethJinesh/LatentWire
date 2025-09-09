# Repository Guidelines

## Project Structure & Module Organization
- `latentwire/` — core Python package: encoder, adapters, wrappers (`models.py`), data loaders, train/eval.
- `scripts/` — convenience entrypoints: setup, smoke, train, eval.
- `tests/` — CPU‑friendly smoke tests mirroring `latentwire/` modules.
- Top‑level: `README.md`, `requirements.txt`, `Makefile`, optional `RESEARCH_PROPOSAL.md` (read first).

## Build, Test, and Development Commands
- Environment: Python 3.10+ (prefer 3.11; setup scripts auto-detect `python3.11`).
- Install: `make setup-mac` (macOS) or `make setup-linux` (Linux). Creates `.venv` and installs deps.
- Smoke test: `make smoke` or `bash scripts/run_smoke_cpu.sh` (validates inputs_embeds → decode path).
- Train (small demo): `make train` → saves encoder/adapters to `./ckpt`.
- Evaluate: `make eval` → prints EM/F1, NLL/token, compression, latency, joint‑pick.
- Direct pytest: `pytest -q`.
- Prefetch assets (optional): `make prefetch` or `bash scripts/prefetch_assets.sh <llama_id> <qwen_id>`.

## Coding Style & Naming Conventions
- Language: Python only. Indentation 4 spaces; max line length ~100.
- Modules: snake_case (`data.py`, `train.py`); classes in PascalCase; functions/vars in snake_case.
- Formatting/Linting: prefer Black (`black .`) and Ruff (`ruff check .`) if installed; keep imports tidy.
- Type hints where helpful; concise docstrings for public functions.

## Testing Guidelines
- Tests live in `tests/` and mirror package layout; name files `test_*.py`.
- Keep tests deterministic; avoid network I/O beyond HF model/dataset pulls used by smoke tests.
- Run locally via `pytest -q` or `make smoke` for quick checks.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
  - Example: `feat(encoder): add latent_len sweep flag`.
- PRs: include a clear description, linked issues, and relevant logs/metrics screenshots (eval output block).
- Keep diffs focused; update docs/tests alongside code; pass smoke/pytest before requesting review.

## Security & Configuration Tips
- Do not commit secrets. For gated HF models, authenticate locally with `huggingface-cli login`.
- Large checkpoints/datasets are user‑installed; never vendor binaries into the repo.
- macOS MPS is supported by PyTorch; CUDA/4‑bit (bitsandbytes) is optional and Linux‑specific.

## Agent‑Specific Instructions
- Read `RESEARCH_PROPOSAL.md` (if present) before major changes.
- Prefer `Makefile`/`scripts` entrypoints; avoid destructive commands.
- Validate changes with `make smoke` (fast) and `make eval` (reports figures of merit) before merging.
