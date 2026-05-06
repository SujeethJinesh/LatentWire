# SinkAware Reviewer Pack

- status: Mac-complete for the current branch; presentable only as a narrow,
  weakly positive systems spinout gate
- current decision: exact static reuse killed; rank-2 approximate prediction is
  alive but bounded pending native timing

## Paper Link

- Draft PDF: `experimental/sinkaware/paper/sinkaware_colm2026.pdf`
- Draft TeX: `experimental/sinkaware/paper/sinkaware_colm2026.tex`
- Draft outline: `experimental/sinkaware/paper/outline.md`

## Current Claim

Fixed early-token sinks cannot be reused exactly without query-dependent
`QK_sink`. A per-head rank-2 predictor improves aggregate softmax/output drift
and downstream causal-LM patch drift over a position-only predictor on
Mac-local distilgpt2/OPT-125M traces. The gains are not uniform per head and
top-1 disagreement remains non-negligible, so this justifies native timing
readiness only, not a benchmark, speed, or preservation guarantee.

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
| Triton readiness | `TRITON_INTERPRET=1`, `TRITON_CPU_BACKEND=1`, repo-local `TRITON_HOME`, repo-local `triton-cpu` source install, CUDA unavailable on Mac | interpreter correctness passes; no GPU claim |
| downstream causal-LM control smoke | distilgpt2 and facebook/opt-125m, 24 traces, split seeds 0/1/2; exact replacement is a no-op; rank-2 improves absolute loss drift over position-only by +0.0393 +/- 0.0134 and +0.1225 +/- 0.0284 | alive but bounded; superseded by larger downstream repeats |
| downstream length/sink sweep | lengths 64/96 and sink counts 2/4; all four config rows positive with minimum model loss improvement >= +0.0272 | stronger Mac-local quality-control surface; still no benchmark or speed claim |
| larger downstream repeats | 48 traces, sink counts 2/4, lengths 64/96, split seeds 0/1/2; exact replacement remains no-op; rank-2 beats position-only by loss and KL on distilgpt2 and OPT-125M; min model loss improvement is +0.0263 at sink2/length64 and remains positive in all larger rows | Mac-local downstream control surface saturated; native timing is next |
| downstream rank frontier | 48 traces, length 96, sink 4, ranks 1/2/4/8; abs loss deltas 0.137/0.096/0.062/0.044 and top-1 disagreement 0.143/0.125/0.095/0.080 improve monotonically, but rank4/rank8 exceed exact four-sink QK multiply-add cost | rank2 remains the only live systems compromise |
| native packet validator | `check_native_gpu_packet.py` validates returned native packet metadata, measured model/sequence shapes, quality drift, per-head drift, same-shape latency repeats, NCU summary, cross-artifact shape consistency, and decision file | Mac-side admissibility guard only; not performance evidence |

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
- Rank-2 still changes top-1 predictions in the downstream patch gate
  (roughly 0.07--0.15 in the larger rows), so the paper must describe the
  result as a quality-control diagnostic rather than a preservation guarantee.

## Next Experiment

The downstream causal-LM control now has larger 48-trace repeats at lengths
64/96 with sink counts 2/4. It favors rank-2 over position-only in every model
row with exact replacement as a no-op control. The 48-trace downstream rank
frontier also explains why not to promote rank4/rank8 before GPU timing: they
improve quality but lose the simple multiply-add wedge. This saturates the
useful Mac-local downstream-control work for the current branch. Native NVIDIA
comparison remains gated by `experimental/sinkaware/phase2/gpu_gate_runbook.md`;
returned packets must pass `experimental/sinkaware/phase2/check_native_gpu_packet.py`
before any paper claim can cite native timing or HBM numbers.
