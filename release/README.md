# OutlierMigrate Reproduction Package

This folder reproduces the empirical claims in the workshop draft
`Channel-Set Drift in Long-Reasoning W4A16 LLMs`. The paper studies
whether high-magnitude activation channels remain stable during long reasoning
decodes, and evaluates W4A16 channel-protection methods, a ParoQuant rotation
baseline, and mechanism analyses.

## Install

```bash
cd release
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.3
python -m pip install -e ".[dev]"
```

For full GPU reproduction, install the tested inference stack:

```bash
python -m pip install -e ".[full,dev]"
```

## Quick Start

```bash
source .venv/bin/activate
python src/scripts/verify_environment.py
python src/scripts/reproduce_set_leaving.py --config configs/granite_small.yaml --fast-verify
```

## Verified Reproduction

Fast verification replays the frozen paper claims and validates tolerances:

```bash
bash src/scripts/reproduce_all.sh --fast-verify
```

The minimal release does not yet include the full model-inference runners.
Running an individual reproduction script without `--fast-verify` or
`--dry-run` fails explicitly rather than pretending to run GPU reproduction.
The archived experimental packets in the main repository contain the full
runner outputs used for the paper.

## Hardware

Fast verification is CPU-only and runs in under 5 minutes. The original
experiment packets were produced on a single high-memory NVIDIA GPU with CUDA
12.8, PyTorch 2.8.0, Transformers 4.57.6, and vLLM 0.10.2.

## Citation

```bibtex
@inproceedings{outliermigrate2026,
  title = {Channel-Set Drift in Long-Reasoning W4A16 LLMs: Mechanisms and a Budget-Tuned Remedy},
  author = {Anonymous},
  booktitle = {Workshop submission},
  year = {2026}
}
```

## License

Code in this release package is Apache-2.0. Model checkpoints retain their
original licenses.
