# SinkAware

SinkAware is a bounded experiment for testing whether attention sink tokens can
be handled as a static prior in long-context attention kernels.

This directory is self-contained inside the main repository. Use the repo-local
virtual environment at `./venv_arm64` from the repository root for current Mac
work. The older `experimental/sinkaware/.venv` is kept only as local scratch
state.

## Local Setup

From the repository root:

```bash
./venv_arm64/bin/python -m pip install -r experimental/sinkaware/requirements.txt
```

Keep external reference repositories and downloaded artifacts outside git, for
example under `experimental/sinkaware/external/` and
`experimental/sinkaware/artifacts/`.

## Current Gate

The exact static-prior branch is killed: fixed sink-token logits remain
query-dependent. The only live branch is approximate per-head low-rank
sink-logit prediction while keeping all non-sink scores exact.

The current 48-trace distilgpt2 probe is weakly positive at the aggregate layer
level (`rank2 output rel-L2=0.141` versus `position=0.170`) but mixed at
layer-head granularity (`+0.0297 +/- 0.0378` output rel-L2 improvement, 20/72
head wins). Treat this as a gate to more rigorous correctness and repeatability
checks, not as a quality or speed result.

Phase 4 Macbook kernel work must run through `TRITON_INTERPRET=1` against a CPU
reference. Interpreter-mode correctness is not GPU performance evidence, and
native speed claims require NVIDIA hardware.
