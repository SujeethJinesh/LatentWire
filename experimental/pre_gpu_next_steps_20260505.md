# Pre-GPU Next Steps For Side Systems Projects

These projects remain outside COLM_v3 claims. The goal here is to improve the
odds that any later NVIDIA run is worth doing.

## HybridKernel

Current status: **WEAKLY ALIVE**.

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
  `0.1419`.

Decision: continue only as approximate low-rank SinkAware. The next pre-GPU
gate is a repeatable all-rank2 quality/stability result or a better stability
mechanism than validation head selection, not kernel coding.

## ThoughtFlow-FP8

Current status: **MIXED/WEAKENED**.

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
  and its paired interval versus R-KV-like crosses zero.

Decision: do not move to GPU until a hidden/KV policy beats the strongest local
proxy by at least the `0.03` NLL promotion margin and shows quality impact under
actual cache dropping.
