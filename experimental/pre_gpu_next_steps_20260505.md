# Pre-GPU Next Steps For Side Systems Projects

These projects remain outside COLM_v3 claims. The goal here is to improve the
odds that any later NVIDIA run is worth doing.

## HybridKernel

Current status: **WEAKLY ALIVE; local work saturated until native profiling**.

What can still be done before NVIDIA:

- finalize an Nsight/vLLM profiler runbook;
- identify exact kernel names and counters to inspect around attention/SSM
  boundaries;
- extend config-only threshold modeling to more hybrid models if public configs
  are available;
- do a source-line audit only for evidence of explicit boundary conversion or
  materialization.

Current local gate result:

- Granite requires about 25% genuinely avoidable boundary traffic at 60%
  recovery to clear a 3% proxy gain.
- Qwen3-Next requires about 10.4%, but its boundary type is less directly
  matched to the Granite Mamba2 fusion story.
- The native packet checker now rejects stale analysis: future packets must
  include `profiler_analysis_gate.{json,md}` generated from the same
  `profiler_metrics.json`.

Decision: no more Mac kernel implementation. Move only to profiler preparation.

## SinkAware

Current status: **ALIVE as approximate, not exact**.

What can still be done before NVIDIA:

- per-head QK-sink approximation error;
- softmax/output-quality error from replacing exact sink logits with rank-2
  approximations;
- CPU reference for exact fused sink+tail composition with an approximate
  branch;
- sequence-length and sink-token-count sweeps;
- a small model-family repeat beyond distilgpt2 if cheap.

Current local gate result:

- exact static sink reuse remains killed;
- real distilgpt2 QK-sink logits are predictable from low-rank hidden features;
- rank-2 is the current cost/accuracy compromise: `0.531x` exact four-sink QK
  estimated multiply-adds with `R2=0.420`;
- rank-8 is much more accurate (`R2=0.712`) but likely not cheaper than exact
  four-sink QK;
- simple validation-selected head gating failed held-out: selected rank-2 heads
  had output rel-L2 `0.2035`, worse than position-only `0.1724` and all-rank2
  `0.1419`;
- the larger 48-trace frozen split gate keeps all-head rank-2 positive:
  output rel-L2 improvement `+0.0379 +/- 0.0014`, minimum split `+0.0367`,
  but head win rate remains low at `0.278 +/- 0.016`.

Latest local gate:

- Triton is still unavailable in `./venv_arm64`, so the fallback was a
  12-trace, one-seed held-out/model-family smoke gate.
- Rank-2 remained positive on `distilgpt2` (`+0.0329` output rel-L2
  improvement) and `facebook/opt-125m` (`+0.0709`), for aggregate model-row
  improvement `+0.0519 +/- 0.0372`.
- This fits predictors separately per model; it is not cross-model predictor
  transfer and not promotion evidence.

Decision: continue only as approximate low-rank SinkAware. The next pre-GPU
gate is Triton-interpreter correctness for the approximate operator or a larger
repeated cross-family falsification gate, not kernel coding.

## ThoughtFlow-FP8

Current status: **REVIVED on the pre-registered recurrence-distance gate**.

What can still be done before NVIDIA:

- replace text-marker rules with a hidden/KV saliency policy;
- test actual quality/perplexity impact under cache dropping on CPU/MPS;
- run a stricter LongFlow/ThinKV/R-KV-like comparison using model-derived
  token importance;
- add a real long-CoT trace set if available locally.

Current local gate result:

- no keep-rate band from 0.10 to 0.35 beats the strongest proxy;
- protected markers beat attention-received saliency but still tie the
  LongFlow-like importance proxy;
- CPU sparse-cache dropping now gives ThoughtFlow-saliency-recent the best mean
  compressed-row NLL (`3.372`), but it clears ThinKV-like by only `0.017` NLL
  and its paired interval versus R-KV-like crosses zero;
- the larger frozen 74-trace sparse-cache probe weakened the stopped policy
  family: ThinKV-like is best at `3.900` NLL versus frozen sparse ThoughtFlow
  `3.908`.
- the one allowed successor, `rdu_topk`, was then evaluated once and clears the
  pre-registered sparse-cache gate: NLL `3.779`, paired delta `-0.121`
  `[-0.211,-0.037]` versus ThinKV-like and `-0.160` `[-0.264,-0.050]`
  versus R-KV-like.

Decision: do not tune anchor/recent/phase/math weights further on the current
saved traces. The live branch is now recurrence-distance utility. The next
pre-GPU gate is larger frozen/seeded reproduction with strict family separation
and oracle/headroom diagnostics, not local retuning.
