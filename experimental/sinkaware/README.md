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
head wins). Simple validation head selection failed. A randomized split/seed
repeat kept all-head rank-2 positive across three token splits
(`+0.0368 +/- 0.0006` output rel-L2 improvement), but the layer-head win rate
remained low (`0.282 +/- 0.024`). A bounded length/sink sweep over
`max_length={64,96}` and `sink_tokens={2,4}` kept all-head rank-2 positive
(`+0.0366 +/- 0.0024`, minimum config `+0.0342`), while preserving the same
per-head caveat. A trace-level frozen split repeat on 24 traces also kept
all-head rank-2 positive (`+0.0398 +/- 0.0014`, minimum split `+0.0387`), but
the head win rate stayed low (`0.287 +/- 0.018`). Treat this as a
correctness/repeatability gate, not as a quality or speed result.

Phase 4 Macbook kernel work must run through `TRITON_INTERPRET=1` against a CPU
reference. Interpreter-mode correctness is not GPU performance evidence, and
native speed claims require NVIDIA hardware.
