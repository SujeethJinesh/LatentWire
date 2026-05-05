# SinkAware Reviewer Pack

- status: presentable as a narrow systems spinout idea
- current decision: alive only as approximate low-rank sink-logit prediction

## Paper Link

- Draft outline: `experimental/sinkaware/paper/outline.md`

## Current Claim

Fixed early-token sinks cannot be reused exactly without query-dependent
`QK_sink`, but a per-head low-rank predictor can approximate sink logits well
enough on Mac-local distilgpt2 traces to justify one native GPU gate.

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| exact static prior | counterexample fails exactness | exact branch killed |
| real QK sink-logit probe | rank-8 hidden+pos R2 0.712 vs position-only 0.153 | approximate branch alive |
| cost model | rank-2 cost ratio 0.531x exact four-sink QK with R2 0.420 | rank-2 is viable compromise |
| softmax/output probe | 48 traces; rank-2 output rel-L2 0.141 vs position-only 0.170 | enough for native GPU gate |

## Reviewer Risks

- The method is approximate and may hurt downstream quality.
- Existing attention-sink systems occupy the broad novelty frame.
- distilgpt2 is only a Mac-local probe, not a modern benchmark.
- No GPU latency or memory claim exists yet.

## Next Experiment

Run `experimental/sinkaware/phase2/gpu_gate_runbook.md` on native NVIDIA
hardware. The comparison must include exact attention, exact fixed-sink
decomposition that still computes `QK_sink`, rank-2 prediction, and
position-only prediction.
