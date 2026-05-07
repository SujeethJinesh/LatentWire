# ThoughtFlow-FP8

Diagnostic falsification package for the stopped ThoughtFlow-FP8 experiment.

Current status: the original anchor/recent/phase/math policy family and the
pre-registered `rdu_topk`, `psi_topk`, and `vwac_topk` successors are stopped
or killed on the available Mac-local sparse-cache surfaces. The folder is now a
falsification/diagnostic harness unless a genuinely new pre-registered signal
is evaluated once on a fresh/larger frozen sparse-cache surface with
nominal-budget quality wins with achieved keep rates reported,
stopped-family/proxy-baseline separation, paired uncertainty, and
oracle/headroom diagnostics.

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

## Completion Estimate And Roadmap

Estimated completion as a diagnostic workshop note: **93%**.

Estimated completion as a positive method paper: **0% on the current branch**.

What is already complete:

- original anchor/recent/phase/math branch evaluation;
- `rdu_topk`, `psi_topk`, and `vwac_topk` preregistrations and fresh-surface
  failures;
- sparse-cache falsification ladder, decision manifest, reviewer pack,
  COLM-style draft, and Triton interpreter primitive.

What remains for the diagnostic note:

- **10%**: optional copyediting only after venue/page constraints are known.

What remains for a positive method:

- a genuinely new utility family, preregistered before measurement;
- one no-retune evaluation on a fresh/larger frozen sparse-cache surface;
- nominal-budget quality wins over R-KV-like and ThinKV-like controls with
  achieved keep rates reported;
- stopped-family/proxy-baseline separation, paired uncertainty, and
  oracle/headroom diagnostics.

Do not spend GPU time on the current branch set. Only Mac-local documentation,
packet hygiene, and camera-ready paper polish remain for this branch family.

## Current Stop And Reopen Rules

Status: **STOP / diagnostic only**. There is no live positive method branch.
The original anchor/recent/phase/math family and the consumed `rdu_topk`,
`psi_topk`, and `vwac_topk` successors must not be retuned or rerun as revival
attempts. Historical `ALIVE` or `PROMOTED` artifacts are superseded by
`phase2/current_decision_manifest_20260506.md`.

Do not run GPU or native FP8 work for:

- `rdu_topk`;
- `psi_topk`;
- `vwac_topk`;
- anchor/recent/phase/math sweeps;
- latency, throughput, CUDA, or Blackwell claims.

GPU time is only justified after a new preregistered CPU sparse-cache utility
family clears the reopen gate below.

## Reopen Quickstart

### 1. Write A New Preregistration Before Measurement

Create a new file under `experimental/thoughtflow_fp8/phase2/`:

```text
preregister_<signal_slug>_utility_<YYYYMMDD>.md
```

It must define:

- the new utility family and exact policy transform;
- forbidden inputs, including continuation loss, trace labels, prior frozen
  outcomes, and any retune after seeing fresh-surface results;
- the fresh/larger frozen input surface;
- the one allowed evaluation command;
- output paths under
  `experimental/thoughtflow_fp8/results/thoughtflow_fp8_reopen_<YYYYMMDD>_<signal_slug>_<surface_slug>/`;
- promotion and kill rules.

### 2. Local Preflight Before Any Reopened Run

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests \
  experimental/tests/test_mac_complete_readiness.py -rs
```

### 3. One-Shot Evaluation Rule

A reopened signal gets exactly one no-retune evaluation on the preregistered
fresh/larger frozen sparse-cache surface. Store outputs under the result
directory declared in the preregistration. The result packet must include:

- the preregistration file;
- frozen input trace manifest and hashes;
- raw scored outputs;
- aggregate table against R-KV-like and ThinKV-like controls;
- stopped-family/proxy-baseline split;
- paired uncertainty;
- oracle/headroom diagnostics;
- explicit promote/kill decision.

### 4. Promote / Kill

Promote only if the single preregistered run shows nominal-budget quality wins
over both R-KV-like and ThinKV-like with achieved keep rates reported, strict
stopped-family/proxy-baseline separation, paired uncertainty supporting the win,
and oracle/headroom diagnostics showing usable remaining signal.

If any condition fails, kill that exact signal. Do not tune its formula, mix it
with consumed RDU/PSI/VWAC components, change the surface, rerun on another
fresh surface, or move to GPU.

## Diagnostic Packet Hygiene

The current diagnostic packet is tracked at
`phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/`. Regenerate
it only from a clean `experimental/thoughtflow_fp8` path:

```bash
./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py
```

The builder refuses dirty-path regeneration, and the saved-artifact tests assert
that the manifest records a clean ThoughtFlow path. This protects the
falsification packet from mixing stale JSON evidence with uncommitted paper or
script edits. The fresh C2C input JSONL files referenced by the PSI/VWAC
artifacts are force-tracked under the root `results/` directory so a clean
checkout can resolve all hashed packet inputs.

If a future preregistered sparse-cache signal promotes, the next native gate is
limited GPU validation of that exact frozen policy only, not GPU-side method
search.

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

## Historical Phase Gates (Superseded)

The phase notes below are retained for auditability. They are no longer live
project guidance: the current branch is stopped as a positive method and
positioned only as a falsification-methodology artifact.

Phase completion requires the deliverables listed in
`../03_thoughtflow_fp8.md`, plus an update to `progress.md`. Do not mark a
phase complete until the deliverables exist and any relevant checks have been
run.

Historical early gates:

- Phase 0: local setup, references/repos/datasets identified or fetched.
- Phase 1: LongFlow failure hypothesis, competitive matrix, and DeepSeek V4
  retrofit differentiation.

This historical kill/pivot rule has fired for the positive-method branch. It is
superseded by the current diagnostic-paper stop rule and reopen gate.

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
