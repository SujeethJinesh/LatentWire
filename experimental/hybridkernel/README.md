# HybridKernel

Bounded experiment for a fused attention-to-SSM boundary kernel in hybrid LLMs.

This folder is self-contained inside the main LatentWire checkout. Keep local
dependencies, scratch clones, and generated artifacts under
`experimental/hybridkernel/` unless a shared repo asset is explicitly needed.

## Local Setup

Create a per-project virtual environment:

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
- Phase 4: Triton skeleton compiles/lowers enough to justify GPU profiling.

Do not proceed to GPU spend until Phase 0-4 deliverables are present and reviewed.
