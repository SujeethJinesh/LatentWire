# OutlierMigrate Reproduction Package

This folder reproduces the empirical claims in the workshop draft
`Decode-Position Channel Drift at Long-Decode Reasoning`. The paper studies
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

## Full Reproduction

Fast verification replays the frozen paper claims and validates tolerances:

```bash
bash src/scripts/reproduce_all.sh --fast-verify
```

Full-fidelity GPU reproduction uses the same scripts without `--fast-verify`.
It requires one A100/H100-class GPU with at least 80 GB VRAM, approximately
600 GB disk for checkpoints and outputs, and 30-50 GPU hours for all paper
claims.

## Hardware

Fast verification is CPU-only and runs in under 5 minutes. Full reproduction
was tested on a single high-memory NVIDIA GPU with CUDA 12.8, PyTorch 2.8.0,
Transformers 4.57.6, and vLLM 0.10.2.

## Citation

```bibtex
@inproceedings{outliermigrate2026,
  title = {Decode-Position Channel Drift at Long-Decode Reasoning},
  author = {Anonymous},
  booktitle = {Workshop submission},
  year = {2026}
}
```

## License

Code in this release package is Apache-2.0. Model checkpoints retain their
original licenses.
