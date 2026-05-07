# ThoughtFlow-FP8 Current Decision Manifest

- date: 2026-05-06
- status: **STOP / diagnostic only**
- current live method branch: none

## Current Decision

The original anchor/recent/phase/math policy family and the repo-local registered
`rdu_topk`, `psi_topk`, and `vwac_topk` successors are stopped on the available
Mac-local sparse-cache surfaces. `rdu_topk` cleared the first frozen 74-trace
gate and reproduced on the same deterministic slice, but then failed stricter
checks:

- alternate surface: a stopped same-family sparse row beat `rdu_topk` by 0.006
  NLL;
- independent saved traces: R-KV-like was best compressed at NLL 3.981, while
  `rdu_topk` reached 4.014 on 89 scored traces.

The later `psi_topk` prefix-surprisal successor was registered in-repo before
measurement and evaluated once on the fresh C2C GSM70 saved-trace surface. It
failed decisively: `psi_topk` reached NLL 7.899 versus ThinKV-like 3.906 and
R-KV-like 3.960 on 70 scored traces.

The later `vwac_topk` value-weighted attention-contribution successor was also
registered in-repo before measurement and evaluated once on the fresh C2C SVAMP70
surface. It failed: `vwac_topk` reached NLL 4.336 versus R-KV-like 4.096 and
ThinKV-like 4.162 on 64 scored traces.

## Current Claim

ThoughtFlow-FP8 is a falsification ladder for sparse-cache retention ideas. It
is not a positive KV-compression method, not a real FP8 method, and not a GPU
systems result.

## Stale Positive Artifacts

Older artifacts such as `frozen_sparse_cache_probe.md`,
`rdu_robustness_diagnostic.md`, and `rdu_no_retune_reproduction_check.md`
contain historical `ALIVE`, `REPRODUCED`, or `PROMOTED` statuses. They now
carry supersession banners and are preserved for
auditability only. The current decision is the demotion recorded here, in `progress.md`, in
`phase2/stop_pivot_decision_20260506.md`, and in the COLM-style draft.

## Allowed Next Work

Current reopen state: **all consumed successor registrations are stopped and no
fresh utility signal is registered, so there is no runnable successor gate.**
A future gate starts with a new repo-local registration artifact, not with another
measurement on the current traces or the consumed RDU/PSI/VWAC surfaces.

Allowed:

- diagnostic packaging and consistency checks;
- a fresh registration for a genuinely different utility family, before any
  measurement;
- a one-shot evaluation only on a fresh/larger frozen sparse-cache surface.

Not allowed:

- retuning `rdu_topk` on the current traces;
- retuning `psi_topk`, adding recency to it, or rerunning it on a new surface
  after seeing the failure;
- retuning `vwac_topk`, changing value/attention normalization, or rerunning it
  on a new surface after seeing the failure;
- spending GPU time on the current branch;
- claiming FP8, CUDA, latency, throughput, or Blackwell evidence from the
  current artifacts.
