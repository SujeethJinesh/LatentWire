# SinkAware

SinkAware is a bounded experiment for testing whether attention sink tokens can
be handled as a static prior in long-context attention kernels.

This directory is self-contained inside the main repository. Use the
project-local virtual environment at `experimental/sinkaware/.venv`.

## Local Setup

From the repository root:

```bash
cd experimental/sinkaware
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Keep external reference repositories and downloaded artifacts outside git, for
example under `experimental/sinkaware/external/` and
`experimental/sinkaware/artifacts/`.

## Current Gate

Phase 0 and Phase 1 are not started. The first gate is a code-level audit of
existing attention kernels to determine whether any implementation already
performs sink-as-precomputed-prior handling.

Kill criterion: an existing kernel implements the sink contribution as a
precomputed prior/bias and skips score computation for fixed sink positions.
