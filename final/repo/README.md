# LatentWire

This repo tracks the `latent_bridge` research program for cross-model
communication / latent transfer.

The current paper status, live branches, and benchmark order live in:

- `paper/README.md`
- `paper/experiment_ledger_20260421.md`
- `paper/benchmark_expansion_order_20260422.md`

Use `latent_bridge/README.md` for implementation-facing method notes. Treat
`latent_bridge/HANDOFF.md` as historical context, not the current project
state.

The active code lives in:

- `latent_bridge/` — main implementation
- `rotalign/` — compatibility alias for older imports
- `scripts/` — thin CLI wrappers for calibration, evaluation, ablations, and control runs
- `tests/` — regression coverage for the current layout
- `data/` — calibration and evaluation inputs
- `references/` — downloaded papers and citation manifest
- `paper/` — active paper workspace, ledgers, reviewer notes, and the official `iclr2026` template

For the current experiment surface and writing artifacts, start at
[paper/README.md](paper/README.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On this machine, prefer `venv_arm64` if `.venv` drifts across shells:

```bash
source venv_arm64/bin/activate
```

On Apple Silicon, the current small-model control runs are intended to use
`--device mps`.

## Common Commands

```bash
source .venv/bin/activate

# Run the test suite
pytest -q

# Calibrate a translator
python scripts/calibrate.py --help

# Evaluate a checkpoint
python scripts/evaluate.py --help

# Run the focused held-out control suite
python scripts/run_control_suite.py --help
```

## Repo Scope

Legacy `telepathy`, `latentwire`, and unrelated baseline codepaths have been
removed so the root repo matches the current RotAlign-KV work.
