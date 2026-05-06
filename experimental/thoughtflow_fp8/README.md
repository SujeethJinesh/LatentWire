# ThoughtFlow-FP8

Diagnostic falsification package for the stopped ThoughtFlow-FP8 experiment.

Current status: the original anchor/recent/phase/math policy family and the
pre-registered `rdu_topk`, `psi_topk`, and `vwac_topk` successors are stopped
or killed on the available Mac-local sparse-cache surfaces. The folder is now a
falsification/diagnostic harness unless a genuinely new pre-registered signal
is evaluated once on a fresh/larger frozen sparse-cache surface with
matched-budget quality wins, same/cross-family separation, paired uncertainty,
and oracle/headroom diagnostics.

Use `phase2/current_decision_manifest_20260506.md` and
`paper/reviewer_pack.md` as the current review entry points. Older artifacts
with `ALIVE` or `PROMOTED` statuses are historical and superseded by the stop
decision.

## Scope

ThoughtFlow-FP8 now reports a stopped diagnostic branch rather than a proposed
method. The original target combined FP8 KV quantization, sink-anchor
protection, and reasoning-phase-aware eviction, but current tracked evidence is
CPU sparse-cache scoring plus an int8/Triton-interpreter reference primitive.
No real FP8, CUDA, latency, throughput, or Blackwell result is claimed.

## Local Workflow

This project lives inside the main LatentWire repository. Use this single
checkout for the experiment.

Use the repo-local virtual environment that the current paper and reviewer pack
cite:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pip install -r experimental/thoughtflow_fp8/requirements.txt
```

Historical per-project `.venv` notes in older progress entries are superseded
by the repo-local `./venv_arm64` workflow above.

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

Phase 4 Macbook kernel work must run through `TRITON_INTERPRET=1` against a CPU
reference. Interpreter-mode correctness is not GPU performance evidence.

Stable owned-test command:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

Avoid broad recursive test collection over external/vendor folders. The
non-interpreter Triton CPU backend can require Homebrew GCC/Darwin linker
details and is not a stable evidence gate for this project.
