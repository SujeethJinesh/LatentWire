# Positive-Method Scoop Check

Date: 2026-05-27

## M-PRED: Kalman/AR(1) Predictive Channel Tracker

### Search Queries

- `site:arxiv.org Kalman LLM quantization channel protection outlier decode-time`
- `site:arxiv.org AR(1) predictor outlier channel LLM quantization decode time`
- `"Kalman filter" "large language model" "quantization"`
- `"AttentionPredictor" "KV cache" quantization arxiv`
- `"dynamic" "salient channels" "decoding step" "DecDEC"`
- `"runtime" "EMA" "outlier" "LLM quantization"`

### Sources Checked

| Work | Source | Relevance | Scoop status |
|---|---|---|---|
| DecDEC | `https://arxiv.org/abs/2412.20185`, `https://www.usenix.org/system/files/osdi25-park-yeonhong.pdf` | Dynamically identifies salient channels at each decoding step from activation outliers. | Adjacent, not a scoop. DecDEC is reactive per-step residual fetching; it does not use one-step-ahead Kalman/AR prediction or budgeted EMA top-K protection. |
| AttentionPredictor | `https://arxiv.org/abs/2502.04077` | Predicts next-token attention scores for KV-cache compression using temporal patterns. | Adjacent, not a scoop. It predicts KV-token attention scores, not activation-channel protection sets for W4A16 GEMM inputs. |
| Rotated Runtime Smooth | `https://arxiv.org/abs/2409.20361` | Runtime activation smoothing plus rotation for INT4 inference. | Adjacent, not a scoop. It smooths activation maxima/group scales rather than predicting future top-K protected channels. |
| ASER | `https://arxiv.org/abs/2411.07762` | Activation smoothing and error reconstruction for low-bit quantization. | Adjacent, not a scoop. It uses smoothing/outlier extraction plus reconstruction, not decode-time predictive channel tracking. |
| GuidedQuant | `https://arxiv.org/abs/2505.07004` | End-loss/gradient-guided PTQ objective. | Adjacent to M-FISH, not M-PRED. It does not implement temporal prediction of channel sets. |

### Finding

No verified 2024--2026 paper found in this search directly matches M-PRED's
proposed mechanism: estimating per-channel decode-time state and innovation,
then applying a one-step-ahead Kalman/AR-style prediction before the next
activation quantization decision.

The most relevant prior-art distinction is:

- DecDEC shows dynamic per-step channel salience and motivates non-static
  selection.
- AttentionPredictor validates temporal prediction as useful for LLM inference,
  but on KV-token attention rather than activation-channel protection.
- M-PRED would test whether the temporal prediction idea transfers to W4A16
  long-decode activation-channel protection.

### Decision

Proceed to M-PRED preregistration unless a later, more targeted source search
finds a direct protected-channel Kalman/AR predictor.

## M-FISH: Fisher-Curvature-Weighted Protection

Date: 2026-05-28

### Search Queries

- `Fisher channel W4A16 quantization LLM`
- `Hessian channel decode-time quantization LLM`
- `gradient guided channel protection LLM quantization`
- `loss guidance activation channel quantization W4A16`

### Sources Checked

| Work | Source | Relevance | Scoop status |
|---|---|---|---|
| GuidedQuant | `https://arxiv.org/abs/2505.07004` | Uses end-loss gradient information in PTQ objectives while preserving cross-weight dependencies. | Adjacent and important prior art, not a direct scoop. It is a quantization-objective method, not decode-time top-K channel protection under long-decode drift. |
| QQQ | `https://arxiv.org/abs/2406.09904` | Uses adaptive smoothing and Hessian-based compensation for W4A8 LLM quantization. | Adjacent, not a direct scoop. It targets W4A8 smoothing/compensation and kernels, not W4A16 endpoint protected-set selection. |
| CW-HAWQ | `https://arxiv.org/abs/2008.08284` | Channel-wise Hessian-aware trace weighting for mixed-precision quantization. | Foundational prior art for channel-wise curvature sensitivity, but not LLM long-decode W4A16 protected-channel selection. |
| HAS-VQ | `https://arxiv.org/abs/2601.06959` | Hessian-adaptive vector quantization for LLM compression. | Adjacent 2026 Hessian-aware compression prior, not a direct decode-time channel-set method. |

### Feasibility Finding

No direct scoop was found for M-FISH as proposed, but the faithful experiment is
not a small patch on the current runner. Existing M11b/M-PRED packets cache
activation magnitudes at decode positions; they do not preserve autograd graphs
or per-channel loss gradients. Computing diagonal Fisher at the same activation
surface would require a new gradient runner that either retains the full
10K-token decode computation graph or reconstructs it layer-by-layer under
teacher forcing. That is a materially different infrastructure path from the
current endpoint scorer.

### Decision

Defer M-FISH as `DEFERRED_INFRA_MFISH_BACKWARD_GRAPH` unless the human
explicitly authorizes a new gradient-capture runner. Do not substitute a
Hessian/Fisher proxy from cached magnitudes; that would not test the stated
method.

## Unified Sensitivity+Budget+Hysteresis Family

Date: 2026-05-28

### Status

The M-FISH-first plan is superseded by the oracle-derived unified family:

```text
P*_t = TopK_{l,i} h_{l,i}(t) * x_{l,i}(t)^2
s_{l,i}(t) = q_{l,i} * EMA(x_{l,i}(t)^2) - lambda * 1[i notin P_{l,t-1}]
```

The next sensitivity branch is M-WJAC, where
`q_{l,i} = ||W_{l,:,i}||_2^2`. This tests a downstream-sensitivity proxy without
gradient capture.

### Search Queries

- `AWQ activation aware weight quantization weight saliency activation statistics`
- `dynamic decode-time weight norm channel protection LLM quantization`
- `weight Jacobian channel quantization LLM activation sensitivity`
- `ChanMix KV cache mixed precision channel mixing`
- `Activation Sensitivity Xu 2026 LLM quantization`

### Sources Checked

| Work | Source | Relevance | Scoop status |
|---|---|---|---|
| AWQ | `https://arxiv.org/abs/2306.00978` | Activation-aware weight quantization protects salient weights based on activation statistics. | Important prior art for static saliency. Not a scoop for M-WJAC because M-WJAC dynamically ranks protected activation channels during decode with EMA state and matched drift controls. |
| DecDEC | `https://arxiv.org/abs/2412.20185` | Shows dynamic salient channels during decoding and motivates non-static selection. | Adjacent. It does not test layerwise budget waterfilling, hysteresis, or weight-Jacobian dynamic EMA scoring. |
| ChanMix | OpenReview ID `yjr2jX41qO` | Mixed precision/channel allocation in KV-cache setting. | Adjacent. M-LAMBDA allocates layerwise protected activation-channel budgets for W4A16 long-decode GEMM inputs, not KV-cache storage. |
| Activation Sensitivity taxonomy | user-flagged as Xu 2026 | Motivates sensitivity-aware criteria. | Adjacent. The planned experiment tests a concrete decode-time WJAC proxy with controls rather than a static taxonomy. |
| GuidedQuant | `https://arxiv.org/abs/2505.07004` | Gradient/end-loss-guided PTQ objective. | Adjacent to full Fisher. M-WJAC avoids backward graphs and tests a cheaper weight-norm sensitivity proxy. |

### Differentiation to Preserve in Paper

- **M-WJAC vs AWQ:** static activation-aware weight saliency is already known;
  the new test is dynamic decode-time protected activation-channel selection
  under long-reasoning drift.
- **M-LAMBDA vs ChanMix:** mixed-precision KV-cache allocation is adjacent, but
  this experiment allocates layerwise protection budget for W4A16 activation
  channel sets.
- **M-WJAC vs full Fisher/GuidedQuant:** this branch is not claiming gradient
  optimality; it tests whether a free local Jacobian proxy captures enough
  sensitivity to rescue weak regimes.

### Decision

Proceed with M-KLLOOK, M-LAMBDA, M-HYST, and M-WJAC after V1 completes. Do not
build M-FISH gradient infrastructure unless WJAC demonstrates that the
sensitivity axis is useful but the weight-norm proxy is too coarse.
