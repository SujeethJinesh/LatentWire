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
