# HybridKernel

Bounded experiment for a fused attention-to-SSM boundary kernel in hybrid LLMs.

This folder is self-contained inside the main LatentWire checkout. Keep local
dependencies, scratch clones, and generated artifacts under
`experimental/hybridkernel/` unless a shared repo asset is explicitly needed.

## Current Handoff Status

Local Mac work is saturated. The project has the config audit, literature/source
audit, architecture map, pre-GPU threshold model, fixed-request driver,
profiler-analysis parser, artifact verifier, synthetic packet fixture, tests,
native packet checklist, and a reproducible local environment preflight. Do not
add more local kernels, scaffolds, or paper claims until native NVIDIA profiler
data exists.

The current Mac ARM64 preflight is recorded in
`phase0/local_preflight.json` and `phase0/local_preflight.md`: PyTorch is
available with MPS, CUDA is unavailable, and `triton==3.7.0+git270e696d` is
importable from a repo-local `triton-cpu` source build. `pip index versions`
still finds no matching wheel distributions for `triton`, `triton-cpu`, or
`triton-nightly` in `./venv_arm64`; the working path is the source-install
readout in `../triton_cpu_source_install_20260506.md`. Local Phase 4 Triton
interpreter correctness is unblocked, but this is not a kernel-performance
result.

The next exact gate is a user-operated NVIDIA/vLLM packet that passes
`phase2/check_profiler_run_artifacts.py` and is then reduced by
`phase2/analyze_profiler_metrics.py`. The checker requires the saved
`profiler_analysis_gate.json`/`.md` outputs to match the metric rows being
checked. Promotion requires repeated server-side Nsight evidence of separable
attention/SSM boundary overhead clearing the 3% recoverable-gain gate. If that
evidence is absent, kill or shelve this branch.

The native artifact checker rejects generated skeleton TODOs, client-only
profile scopes, stale analyzer outputs, duplicated run IDs, and tiny or
placeholder Nsight exports. Matching profiler filenames alone are not
admissible evidence.

## Local Setup

Use the repo-local ARM64 virtual environment when available:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
source ./venv_arm64/bin/activate
python -m pytest experimental/hybridkernel/phase2/tests
```

If that environment is unavailable, create a per-project virtual environment:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
python3 -m venv experimental/hybridkernel/.venv
source experimental/hybridkernel/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experimental/hybridkernel/requirements.txt
```

Use Mac CPU/MPS for setup, literature audit, architecture mapping, and reference
tests. Do not run remote GPU work from this repo; write a local runbook for any
future 5090/H100 gate.

Record the local dependency surface before handoff:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
PIP_CACHE_DIR=.debug/pip_cache ./venv_arm64/bin/python \
  experimental/hybridkernel/phase0/preflight_environment.py \
  --check-pip-index \
  --output-json experimental/hybridkernel/phase0/local_preflight.json \
  --output-md experimental/hybridkernel/phase0/local_preflight.md
```

This command does not install packages. It only records torch/CUDA/MPS/Triton
availability and whether Triton candidates are visible from the active pip
index.

## Scope

Primary question: do hybrid attention/SSM layer transitions create enough
decode overhead to justify a fused boundary kernel?

Initial targets:

- Granite-4.0-H-Tiny and Granite-4.0-H-Small
- Nemotron-H-8B-Base
- Nemotron-3-Nano-30B-A3B
- Apriel-H1-15B-Thinker
- Qwen3-Next-80B-A3B config only

## Gates

Phase 0 and Phase 1 are not complete until their deliverables exist and are
recorded in `progress.md`.

- Phase 0: local environment, external references, and model configs available.
- Phase 1: literature/code audit confirms no existing fused boundary kernel.
- Phase 2: architecture map shows theoretical fusion benefit is plausibly at or
  above the project threshold.
- Phase 3: CPU reference implementation matches canonical behavior.
- Phase 4: `TRITON_INTERPRET=1` kernel skeleton matches a CPU reference. A
  missing local Triton dependency blocks Phase 4 completion rather than
  proving or disproving the systems idea.

Phase 0-4 Mac-side deliverables are now review-ready as a pre-GPU handoff, not
as performance evidence. Do not proceed with implementation until the native
packet gate clears.
