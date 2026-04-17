# LatentWire

This repo now tracks the `latent_bridge` / RotAlign-KV project only.

The active code lives in:

- `latent_bridge/` — main implementation
- `rotalign/` — compatibility alias for older imports
- `scripts/` — thin CLI wrappers for calibration, evaluation, ablations, and control runs
- `tests/` — regression coverage for the current layout
- `data/` — calibration and evaluation inputs
- `references/` — downloaded papers and citation manifest

For the detailed method, experiment plan, and current controls, read
[latent_bridge/README.md](latent_bridge/README.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
