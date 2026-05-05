# SinkAware Reviewer Pack

- status: presentable only as a narrow, weakly positive systems spinout gate
- current decision: exact static reuse killed; rank-2 approximate prediction is
  alive but fragile

## Paper Link

- Draft outline: `experimental/sinkaware/paper/outline.md`

## Current Claim

Fixed early-token sinks cannot be reused exactly without query-dependent
`QK_sink`. A per-head rank-2 predictor improves aggregate softmax/output drift
over a position-only predictor on Mac-local distilgpt2 traces, but the new
layer-head paired readout shows the gains are concentrated rather than
uniform. This justifies interpreter/readiness work, not a speed or quality
claim.

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| exact static prior | counterexample fails exactness | exact branch killed |
| real QK sink-logit probe | rank-8 hidden+pos R2 0.712 vs position-only 0.153 | approximate branch alive |
| cost model | rank-2 cost ratio 0.531x exact four-sink QK with R2 0.420 | rank-2 is viable compromise |
| softmax/output probe | 48 traces; aggregate rank-2 output rel-L2 0.141 vs position-only 0.170 | weakly positive |
| layer-head paired probe | rank-2 output rel-L2 improvement +0.0297 +/- 0.0378; 20/72 head wins | fragile; needs repeats or head gating |
| Triton readiness | `TRITON_INTERPRET=1` set, `triton` not importable, CUDA unavailable on Mac | no interpreter pass yet |

## Reviewer Risks

- The method is approximate and may hurt downstream quality.
- Aggregate output improvements are not yet robust per head.
- Existing attention-sink systems occupy the broad novelty frame.
- distilgpt2 is only a Mac-local probe, not a modern benchmark.
- No GPU latency or memory claim exists yet.

## Next Experiment

First make the `TRITON_INTERPRET=1` approximate-attention scaffold runnable in
`./venv_arm64` or in a Linux GPU environment and verify exact-prediction
correctness. Then repeat the softmax/output probe with split seeds or a
head-selective rank-2 gate before running the native NVIDIA comparison in
`experimental/sinkaware/phase2/gpu_gate_runbook.md`.
