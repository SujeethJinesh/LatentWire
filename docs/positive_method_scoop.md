# Positive-Method Scoop Addendum: Unified Sensitivity/Budget/Hysteresis Family

Date: 2026-05-28

## Scope

This note checks the unified WJAC/LAMBDA/HYST family against nearby
quantization and long-decode work. It is a scoop-risk addendum only: no GPU
jobs were run, no ledger state was edited, and no claims below are experimental
evidence for LatentWire.

Checked family:

```text
P*_t = TopK_{l,i} h_{l,i}(t) * x_{l,i}(t)^2
s_{l,i}(t) = q_{l,i} * EMA(x_{l,i}(t)^2) - lambda * 1[i notin P_{l,t-1}]
q_{l,i} = ||W_{l,:,i}||_2^2  for M-WJAC
```

## Bottom Line

No direct primary-source scoop was found for the exact three proposed elements:

1. dynamic decode-time WJAC/weight-column-norm EMA scoring for protected
   activation channels;
2. layerwise waterfilling of a fixed protected activation-channel budget for
   W4A16 long-decode;
3. hysteretic entry/exit updates for protected activation-channel sets in
   W4A16 long-decode.

The novelty claim must still be narrow. DecDEC already owns dynamic per-decode
salient-channel residual fetching. AWQ already owns activation-aware static
weight-channel saliency. ChanMix/KVTuner/KVmix already own sensitivity-aware
mixed-precision allocation in KV-cache settings. Xu 2026 already owns the
activation-sensitivity framing as a unifying PTQ principle.

## Search Surface

Targeted queries included:

- `AWQ activation aware weight quantization weight saliency activation statistics`
- `ChanMix OpenReview yjr2jX41qO KV cache channel mixing`
- `Activation Sensitivity Xu 2026 quantization criteria taxonomy LLM`
- `dynamic decode-time weight norm channel protection LLM quantization`
- `weight Jacobian channel quantization LLM activation sensitivity`
- `W4A16 dynamic activation channel protection decode-time EMA`
- `layerwise waterfilling LLM quantization activation channels`
- `hysteresis protected activation quantization LLM`
- `long-decoding reasoning LLM quantization W4A16 activation outliers`

Primary sources checked were arXiv, OpenReview, USENIX/OSDI, AAAI proceedings,
and project pages when available.

## Source Ledger

| Source | What the source actually claims | Relevance | Scoop status |
|---|---|---|---|
| AWQ, Lin et al. 2023/2024, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | AWQ is a low-bit weight-only LLM quantization method. It identifies salient weight channels using offline activation statistics and protects them through equivalent scaling rather than hardware-inefficient mixed precision. | Directly relevant to activation-aware saliency and W4A16-adjacent baselines. | Important prior art, not a direct scoop. M-WJAC protects activation channels during decode using EMA state and a weight-column-norm sensitivity proxy; AWQ protects static weight channels during weight-only PTQ. |
| DecDEC, Park et al. 2025, [USENIX OSDI PDF](https://www.usenix.org/system/files/osdi25-park-yeonhong.pdf), [arXiv:2412.20185](https://arxiv.org/abs/2412.20185) | DecDEC stores the full-precision minus quantized weight residual on CPU and dynamically fetches residuals for salient channels identified from activation outliers at each decoding step. | Closest dynamic decode-time channel method found. | Adjacent and a strong reviewer comparator, not a direct scoop. It is reactive per-step residual fetching for quantized weights, not EMA/hysteretic protected activation-channel selection or WJAC sensitivity scoring. |
| ChanMix, Liao and Wen 2026, [OpenReview PDF](https://openreview.net/pdf?id=yjr2jX41qO) | ChanMix is an ICLR 2026 KV-cache mixed-precision quantization framework. It finds channel sensitivity asymmetry in KV cache, gives more bits to retrieval/outlier channels and fewer bits to robust/subnormal channels, and implements channel reordering plus custom Triton kernels. | Directly relevant to sensitivity-aware channel allocation under long-context inference. | Important prior art, not a direct scoop. M-LAMBDA allocates a protected activation-channel budget for W4A16 GEMM inputs, not KV-cache bitwidths or packed KV storage. |
| Activation Sensitivity, Xu 2026, [arXiv:2601.11663](https://arxiv.org/abs/2601.11663) | Formalizes activation sensitivity as expected loss impact from channel-wise perturbations. It frames sensitivity as squared norm of gradient-weighted activations, connects AWQ/GPTQ/Fisher/Hessian criteria, and explicitly says it is a conceptual framework rather than a new quantization algorithm. | Directly relevant to the WJAC rationale and taxonomy. | Must cite as theory prior. Not a direct scoop because WJAC is a specific deployable proxy and ablation protocol, but do not claim the sensitivity theory as new. |
| ParoQuant, Liang et al. 2026, [OpenReview](https://openreview.net/forum?id=1USeVjsKau), [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Pairwise Rotation Quantization uses optimizable Givens rotations and channel-wise scaling to reduce outliers for reasoning LLM inference. It emphasizes long chain-of-thought error accumulation and reports gains over AWQ on reasoning tasks. | Important current W4A16/reasoning baseline context. | Adjacent, not a direct scoop. It transforms/scales the quantized weight-only inference path; it does not maintain dynamic protected activation-channel sets. |
| KVTuner, Li et al. 2025, [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) | Searches layer-wise mixed-precision key/value cache configurations using sensitivity analysis, then uses offline-searched hardware-friendly precision pairs during inference. | Relevant to M-LAMBDA and sensitivity-aware layer allocation. | Adjacent. It is layer-wise KV-cache precision selection, not protected activation-channel waterfilling for W4A16 GEMM inputs. |
| KVmix, Li et al. 2026, [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/40422), [arXiv:2506.08018](https://arxiv.org/abs/2506.08018) | Uses gradient-based layer importance for mixed-precision KV-cache quantization and dynamically keeps recent pivotal KV pairs in full precision while compressing older ones. | Relevant because it combines gradient sensitivity, layer-specific allocation, and dynamic long-context KV handling. | Adjacent and important. It acts on KV cache and recent token selection, not W4A16 activation-channel protection with WJAC EMA or hysteresis. |
| High-Rate Quantized Matrix Multiplication II, Ordentlich and Polyanskiy 2026, [arXiv:2605.13768](https://arxiv.org/abs/2605.13768) | Analyzes quantized matrix multiplication with calibration covariance. Connects weight-only LLM PTQ to weighted MSE source coding and classical reverse waterfilling for rate allocation. | Relevant to the word "waterfilling" and rate-allocation theory. | Theory-adjacent, not a direct scoop. It is weight-only quantization/rate allocation, not empirical layerwise protected activation-channel budget allocation. |
| Activation Function Informed Quantization, Lew and Aamodt 2025/2026, [OpenReview](https://openreview.net/forum?id=zxAPHZNFt8) | Selectively computes channels in W4A4 or W8A8 based on activation-function gradient behavior; channel selection is calibrated from a single example and used for future inference. | Relevant to selective channel-wise quantized computation. | Adjacent. Selection is static after calibration and tied to activation-function sign/gradient, not decode-time EMA, WJAC, or hysteretic protected sets. |
| Rotated Runtime Smooth, Yi et al. 2024/2025, [arXiv:2409.20361](https://arxiv.org/abs/2409.20361) | Provides runtime activation smoothing with rotation for INT4 inference, using channel-wise maximums to handle outliers. | Relevant to runtime activation handling. | Adjacent. Runtime smoothing is not protected-set selection, layerwise waterfilling, or WJAC scoring. |
| ASER, Zhao et al. 2024/2025, [arXiv:2411.07762](https://arxiv.org/abs/2411.07762) | Combines activation smoothing/outlier extraction with low-rank error reconstruction for low-bit LLM quantization, including W4A8 per-channel settings. | Relevant to activation smoothing and error reconstruction. | Adjacent. It is offline smoothing/reconstruction, not long-decode dynamic protected activation-channel selection. |
| MCAP, Das 2026, [arXiv:2604.21026](https://arxiv.org/abs/2604.21026) | Uses load-time Monte Carlo activation profiling to drive per-layer precision dispatch, e.g. W4A8 vs W4A16, and memory residency decisions on target hardware. | Relevant to deployment-time layer profiling and W4A16 decisions. | Adjacent. It profiles layers and precision/residency tiers, not channel-level protected activation sets during decode. |
| LAQuant, Choi et al. 2026, [arXiv:2605.08755](https://arxiv.org/abs/2605.08755) | Long-decoding reasoning models can lose accuracy under representative quantization recipes despite preserving perplexity. LAQuant uses layer-wise weight-only QAT with a one-layer lookahead loss and reasoning-domain calibration. | Relevant to long-decode quantization evaluation pressure. | Adjacent. It supports the need for long-decode benchmarks but does not scoop W4A16 protected activation-channel EMA/waterfilling/hysteresis. |

## Requested Differentiation Checks

### M-WJAC vs AWQ

Status: differentiated, with wording constraints.

AWQ already claims that activation statistics are better than weight magnitudes
for finding salient weight channels, and it protects those channels with offline
scaling in a weight-only PTQ method. M-WJAC must not be framed as "the first
activation-aware channel saliency method." The defensible distinction is:
M-WJAC is a decode-time protected activation-channel policy whose score
combines EMA activation energy with a cheap local downstream-sensitivity proxy,
`||W_{:,i}||_2^2`, and is evaluated under long-decode drift with matched
shuffle/random controls.

Safe wording:

> AWQ establishes static activation-aware weight saliency for weight-only PTQ.
> We instead test whether a dynamic protected activation-channel policy can
> preserve W4A16 long-decode behavior when channel energy drifts across the
> generated sequence. Our WJAC score is a cheap proxy for downstream
> sensitivity, not a claim of Fisher-optimal channel selection.

### M-LAMBDA vs ChanMix/KVTuner/KVmix

Status: differentiated, but crowded.

ChanMix, KVTuner, and KVmix make sensitivity-aware allocation in KV-cache
compression a crowded comparison space. M-LAMBDA must not be described as the
first layerwise or channel-sensitive allocation method. The safe distinction is
the target object and decision variable: M-LAMBDA waterfills a fixed budget of
protected activation channels across layers for W4A16 GEMM inputs. ChanMix
reallocates KV-cache bit precision by channel type; KVTuner searches layer-wise
KV precision pairs; KVmix uses gradient-based layer importance for KV-cache
mixed precision and dynamic recent-pivotal KV retention.

Safe wording:

> Prior KV-cache methods allocate precision across layers or channels to reduce
> long-context cache error. Our LAMBDA ablation asks a narrower question:
> holding the total protected W4A16 activation-channel budget fixed, can
> calibration marginal curves reallocate protected channels across layers better
> than a flat per-layer budget?

### M-WJAC/HYST vs Activation Sensitivity

Status: differentiated only as implementation/evaluation, not theory.

Xu 2026 owns the general activation-sensitivity lens and explicitly connects
activation magnitude, downstream error propagation, Fisher, Hessian, GPTQ, and
AWQ. WJAC should cite this as motivation. The paper should not claim a new
unifying criterion. HYST is not covered by the Xu taxonomy except indirectly as
a policy-stability mechanism.

Safe wording:

> We use activation-sensitivity theory as motivation, then test a deployable
> proxy that avoids backward graphs. Hysteresis is evaluated separately as a
> churn-control policy for the protected set, not as a new sensitivity metric.

## Direct Scoop Check By Proposed Element

### Dynamic Decode-Time WJAC EMA

Direct scoop found: no.

Closest sources:

- DecDEC dynamically detects salient channels at each decoding step from
  activation outliers and fetches residual weights.
- Rotated Runtime Smooth uses runtime activation smoothing.
- AFIQ chooses selective quantized computation channels after calibration.
- MCAP uses load-time activation profiling for layer-level precision/residency.

None of the checked primary sources combines decode-time EMA of squared
activation energy with a weight-column-norm/Jacobian proxy to choose protected
activation channels for W4A16 long-decode.

### Layerwise Waterfilling Protected Activation Channels

Direct scoop found: no.

Closest sources:

- ChanMix, KVTuner, and KVmix allocate precision in KV-cache settings.
- Layer-wise quantization work assigns different quantization levels to layers.
- High-Rate Quantized Matrix Multiplication II uses reverse waterfilling for
  weight-only quantized matrix multiplication under a covariance model.

None of the checked primary sources waterfills a fixed protected activation
channel count across layers for W4A16 GEMM input protection.

### Hysteretic Protected-Set Updates

Direct scoop found: no.

Closest sources:

- DecDEC is dynamic but does not use entry/exit hysteresis.
- Runtime smoothing methods update activation handling online but do not keep a
  protected-set memory with hysteretic enter/exit thresholds.

None of the checked primary sources uses hysteresis to reduce protected-channel
churn for W4A16 long-decode activation protection. This is an absence finding
over the targeted search surface, not a proof of global absence.

## Reviewer-Risk Notes

- Do not claim novelty for "activation-aware saliency"; AWQ and the broader PTQ
  literature already cover it.
- Do not claim novelty for "dynamic salient channels"; DecDEC is too close.
- Do not claim novelty for "sensitivity-aware mixed precision"; ChanMix,
  KVTuner, KVmix, and Activation Sensitivity make that space crowded.
- Do not claim theory novelty for activation sensitivity; cite Xu 2026 and
  frame WJAC as a cheap tested proxy.
- Do claim, if experiments support it, the narrower combination: dynamic
  long-decode W4A16 protected activation-channel selection with explicit
  marginal ablations for sensitivity, layer budget, and churn control.

## Paper-Safe Claim Template

Use only if the preregistered ablations pass:

> Existing quantization methods already show that activation statistics,
> sensitivity-aware allocation, and long-decode outlier control matter. Our
> contribution is narrower: we evaluate a dynamic protected activation-channel
> family for W4A16 long-decode inference. The family separates three mechanisms
> under matched controls: a WJAC sensitivity proxy, layerwise fixed-budget
> waterfilling, and hysteretic protected-set updates. This isolates whether
> protected activation channels can be selected as a stable decode-time policy,
> rather than as static weight saliency or KV-cache mixed precision.

Use if the ablations fail or are mixed:

> The scoop check shows that the proposed family is not directly preempted, but
> the surrounding literature is crowded enough that a negative result should be
> reported as a protocol/falsification finding rather than as a new quantization
> method. The result would still clarify whether dynamic activation-channel
> protection adds value beyond static saliency and KV-cache allocation methods.

## Decision

Proceed with the preregistered M-KLLOOK, M-LAMBDA, M-HYST, and M-WJAC sequence
only if the active GPU gate is free and the existing preregistration order is
respected. The scoop check does not justify widening the method branch or
skipping marginal ablations.
