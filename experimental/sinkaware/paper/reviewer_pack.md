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
| layer-head paired probe | rank-2 output rel-L2 improvement +0.0297 +/- 0.0378; 20/72 head wins | fragile; needs repeats or better stability |
| validation head-selective gate | selected 19/72 heads; held-out output rel-L2 0.2035 vs 0.1724 position and 0.1419 all-rank2 | simple head selection weakened |
| split/seed all-rank2 gate | 3 randomized token splits; rank-2 output rel-L2 improvement +0.0368 +/- 0.0006; all seeds positive | all-rank2 repeatable but still per-head fragile |
| length/sink all-rank2 sweep | lengths 64/96, sinks 2/4, 3 seeds each; mean improvement +0.0366 +/- 0.0024; min config +0.0342 | alive but bounded |
| trace-level frozen split gate | 48 traces, 3 whole-trace splits; mean improvement +0.0379 +/- 0.0014; min split +0.0367 | stronger repeatability, still bounded |
| held-out/cross-family repeat | measured 48 traces, 3 whole-trace split seeds, distilgpt2 plus facebook/opt-125m; output rel-L2 improvements +0.0306 +/- 0.0023 and +0.0788 +/- 0.0069 | alive but bounded; not predictor transfer, GPU speed, or end-to-end quality evidence |
| cross-family length stability | measured 48 traces, lengths 64/96, 3 whole-trace split seeds, distilgpt2 plus facebook/opt-125m; all 4 model/length rows positive; mean output rel-L2 improvement +0.0535 +/- 0.0262; min row +0.0301 | alive but bounded; stronger Mac-local stability, still no downstream or speed claim |
| Triton readiness | `TRITON_INTERPRET=1`, repo-local `triton-cpu` source install, CUDA unavailable on Mac | interpreter correctness passes; no GPU claim |

## Reviewer Risks

- The method is approximate and may hurt downstream quality.
- Aggregate output improvements survive token split repeats, a small
  length/sink sweep, and a 48-trace frozen split repeat, but are not robust per
  head.
- Existing attention-sink systems occupy the broad novelty frame.
- distilgpt2 is no longer the only model-family probe, and the OPT row now
  survives measured 48-trace, three-seed held-out repeats at lengths 64 and 96,
  but it is still a Mac-local diagnostic and not a benchmark-backed result.
- No GPU latency or memory claim exists yet.

## Next Experiment

The `TRITON_INTERPRET=1` approximate-attention scaffold now runs in
`./venv_arm64` through the local `triton-cpu` source build and verifies
exact-prediction correctness. The next Mac-feasible gate is a downstream
quality/control diagnostic that preserves strict same-family versus OPT-family
separation and compares exact attention, position-only replacement, and rank-2
replacement. Simple validation head selection is no longer a good rescue.
Native NVIDIA comparison remains gated by
`experimental/sinkaware/phase2/gpu_gate_runbook.md`.
