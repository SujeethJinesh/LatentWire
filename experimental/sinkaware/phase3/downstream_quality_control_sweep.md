# SinkAware Downstream Quality/Control Sweep

- date: 2026-05-06
- status: alive but bounded
- models: `distilgpt2`, `facebook/opt-125m`
- traces: 24
- train fraction: 0.67
- seeds: 0, 1, 2

This sweep extends the downstream causal-LM control smoke across two sequence
lengths and two fixed-sink counts. Each row patches full model attention during
causal-LM evaluation and compares exact baseline attention, exact sink-logit
replacement, position-only replacement, and rank-2 replacement. Predictors are
fit separately per model/config/split. This is not benchmark accuracy,
predictor transfer, or GPU speed evidence.

Positive values mean rank-2 is closer to exact baseline behavior than
position-only.

| Max length | Sink tokens | Loss-delta improvement | KL improvement | Minimum model loss improvement | Exact no-op? |
|---:|---:|---:|---:|---:|---|
| 64 | 2 | +0.0544 +/- 0.0505 | +0.0485 +/- 0.0693 | +0.0287 | yes |
| 64 | 4 | +0.0809 +/- 0.0815 | +0.0825 +/- 0.1078 | +0.0393 | yes |
| 96 | 2 | +0.0433 +/- 0.0317 | +0.0408 +/- 0.0541 | +0.0272 | yes |
| 96 | 4 | +0.0728 +/- 0.0619 | +0.0769 +/- 0.0904 | +0.0412 | yes |

## Decision

Rank-2 remains positive on every model/config row under this small downstream
control sweep. This strengthens the Mac-local quality-control surface enough to
make native timing the next systems gate. It still does not support benchmark,
latency, HBM, throughput, or cross-model predictor-transfer claims.
