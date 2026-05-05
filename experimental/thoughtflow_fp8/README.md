# ThoughtFlow-FP8

Minimal project scaffold for the ThoughtFlow-FP8 experiment.

## Scope

ThoughtFlow-FP8 tests whether reasoning-aware KV cache compression can retrofit
onto existing open-weight reasoning models without retraining. The proposed
method combines FP8 KV quantization, sink-anchor protection, and
reasoning-phase-aware eviction.

## Local Workflow

This project lives inside the main LatentWire repository. Use this single
checkout for the experiment.

Use a per-project virtual environment:

```bash
cd experimental/thoughtflow_fp8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Keep downloaded papers, cloned competitor repos, caches, generated traces, and
large outputs out of git. Put scratch artifacts under ignored local directories
such as `external/`, `data/`, `traces/`, `results/`, or `scratch/`.

## Phase Gates

Phase completion requires the deliverables listed in
`../03_thoughtflow_fp8.md`, plus an update to `progress.md`. Do not mark a
phase complete until the deliverables exist and any relevant checks have been
run.

Current required early gates:

- Phase 0: local setup, references/repos/datasets identified or fetched.
- Phase 1: LongFlow failure hypothesis, competitive matrix, and DeepSeek V4
  retrofit differentiation.

The project should be killed or pivoted if Phase 1 cannot identify a concrete
LongFlow failure mode that ThoughtFlow-FP8 directly addresses.
