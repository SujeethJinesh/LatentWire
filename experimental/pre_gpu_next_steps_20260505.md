# Pre-GPU Next Steps For Side Systems Projects

> **Superseded on 2026-05-07.** This file is historical planning context. The
> active scope is exactly HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8.
> Use `experimental/project_status_20260506.md` and
> `experimental/native_gpu_handoff_20260506.md` for current gates. In
> particular, HybridKernel must run the native profiler packet before any
> fused-kernel implementation, and ThoughtFlow is diagnostic-only unless a new
> preregistered utility signal is opened.

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
- A packet skeleton generator now creates the native run directory shape and
  the checker rejects any unfilled `TODO_NATIVE_PROFILE_FILL` sentinel.
- The checker also rejects tiny or placeholder Nsight artifacts by default, so
  synthetic fixtures are schema-only unless native artifact validation is
  explicitly disabled.
- A local preflight artifact now records the Triton state:
  Torch `2.6.0` imports, CUDA is unavailable, MPS is available, PyPI/index
  checks for `triton`, `triton-cpu`, and `triton-nightly` still report no
  compatible distribution for this macOS arm64 `venv_arm64`, but the
  experimental `triton-cpu` source build is importable as
  `triton==3.7.0+git270e696d`.
- The HybridKernel Phase 4 interpreter tests pass locally under
  `TRITON_INTERPRET=1`; this is correctness evidence only, not a reason to add
  more Mac kernels.

Decision: no more Mac kernel implementation. Move only to profiler preparation
and native evidence collection.

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

Latest local gates:

- Before the source build, Triton was unavailable in `./venv_arm64`, so the
  fallback was a held-out/model-family gate.
- The earlier 12-trace smoke has now been scaled through 24 traces to 48 traces
  and split seeds `0,1,2`.
- Rank-2 remained positive on `distilgpt2` (`+0.0306 +/- 0.0023` output rel-L2
  improvement) and `facebook/opt-125m` (`+0.0788 +/- 0.0069`), for aggregate
  model-row improvement `+0.0547 +/- 0.0472` and minimum model-row improvement
  `+0.0306`.
- This fits predictors separately per model; it is not cross-model predictor
  transfer and not promotion evidence.
- The Phase 4 approximate attention and sink-decomposition interpreter tests
  now pass locally under the `triton-cpu` source install. This removes the
  Mac correctness blocker but does not add downstream quality or speed evidence.
- A cross-family length-stability gate at max lengths `64` and `96` kept all
  four model/length rows positive, with aggregate output rel-L2 improvement
  `+0.0535 +/- 0.0262`, minimum model/length row `+0.0301`, and head win rate
  `0.982 +/- 0.008`.

Decision: continue only as approximate low-rank SinkAware. The next pre-GPU
gate is a downstream quality/control diagnostic under the same GPT2/OPT
separation, followed by native timing only if quality remains bounded.

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
- cached deterministic splits support the mean margin but expose remaining
  uncertainty: 4/4 half-size partitions keep positive mean margins, but only
  2/4 clear both paired CI highs below zero.
- a measured no-retuning rerun of the same frozen 74-trace gate exactly
  reproduces the cached promoted row: `rdu_topk` NLL `3.779`, paired deltas
  `-0.121 [-0.211,-0.037]` versus ThinKV-like and `-0.160 [-0.264,-0.050]`
  versus R-KV-like, with measured-minus-cached NLL drift `0.000` for every
  policy.
- the measured compressed oracle is still better (`3.634` NLL), leaving a
  `0.145` NLL gap from `rdu_topk` to per-trace compressed oracle and a `0.419`
  oracle-hit rate.
- the harder alternate no-retuning surface (`max_length=112`,
  `continuation_tokens=32`) weakens the strict claim: `rdu_topk` still beats
  R-KV-like (`+0.087` NLL margin) and ThinKV-like (`+0.256`), but a stopped
  same-family sparse row is better by `0.006` NLL (`3.588` versus `3.594`).
- The Phase 4 anchor/phase quantization interpreter tests now pass locally
  under the `triton-cpu` source install. This validates kernel logic for the
  quantization scaffold only; it does not rescue the weakened sparse-cache
  method claim.

Decision: do not tune anchor/recent/phase/math weights further on the current
saved traces. The live branch is recurrence-distance utility, but it is
weakened by the alternate-surface result. The next pre-GPU gate is a larger or
independently seeded frozen reproduction that must beat both cross-family
baselines and stopped same-family rows with oracle/headroom diagnostics.
