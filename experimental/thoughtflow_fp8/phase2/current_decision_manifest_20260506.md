# ThoughtFlow-FP8 Current Decision Manifest

- date: 2026-05-06
- status: **STOP / diagnostic only**
- current live method branch: none

## Current Decision

The original anchor/recent/phase/math policy family and the pre-registered
`rdu_topk` successor are stopped on the available Mac-local sparse-cache
surfaces. `rdu_topk` cleared the first frozen 74-trace gate and reproduced on
the same deterministic slice, but then failed stricter checks:

- alternate surface: a stopped same-family sparse row beat `rdu_topk` by 0.006
  NLL;
- independent saved traces: R-KV-like was best compressed at NLL 3.981, while
  `rdu_topk` reached 4.014 on 89 scored traces.

## Current Claim

ThoughtFlow-FP8 is a falsification ladder for sparse-cache retention ideas. It
is not a positive KV-compression method, not a real FP8 method, and not a GPU
systems result.

## Stale Positive Artifacts

Older artifacts such as `frozen_sparse_cache_probe.md` and
`rdu_robustness_diagnostic.md` contain historical `ALIVE` or `PROMOTED`
statuses. Those are preserved for auditability only. The current decision is
the demotion recorded here, in `progress.md`, in
`phase2/stop_pivot_decision_20260506.md`, and in the COLM-style draft.

## Allowed Next Work

Allowed:

- diagnostic packaging and consistency checks;
- a fresh preregistration for a genuinely different utility family, before any
  measurement;
- a one-shot evaluation only on a fresh/larger frozen sparse-cache surface.

Not allowed:

- retuning `rdu_topk` on the current traces;
- spending GPU time on the current branch;
- claiming FP8, CUDA, latency, throughput, or Blackwell evidence from the
  current artifacts.
