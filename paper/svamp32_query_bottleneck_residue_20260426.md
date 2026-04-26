# SVAMP32 Query-Bottleneck Residue Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: a source-derived positive method plus medium,
  seed-repeat, uncertainty, source-control, and cross-family gates
- current story: the C2C-derived residue sidecar remains a bound, but the
  tested source-side predictors do not recover clean source-necessary IDs
- blocker: source-derived residue prediction under strict controls

## Gate

This gate implemented the next branch selected after the recovered-branch
audit: a learned query-bottleneck residue predictor. It uses the existing
strict SVAMP32 candidate-pool decoder and controls, but replaces ridge residue
classification with learned output-query slots over all-layer source summary
tokens.

Configuration:

- probe model: `query_bottleneck`
- feature layers: `all`
- feature dimension: `44800`
- query slots: `8`
- query epochs: `80`
- query learning rate: `0.01`
- query weight decay: `0.001`
- query seed: `0`
- cross-fitting: leave-one-ID-out

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 9/32 | 0/6 | 2/3 |
| zero_source | 14/32 | 0/6 | 3/3 |
| shuffled_source | 10/32 | 0/6 | 1/3 |
| label_shuffled | 14/32 | 0/6 | 3/3 |
| target_only | 14/32 | 0/6 | 3/3 |
| slots_only | 8/32 | 0/6 | 0/3 |

Failing criteria:

- `min_correct`
- `preserve_fallback_floor`
- `min_clean_source_necessary`

## Decision

Fail the query-bottleneck smoke gate. This exact learned slot variant does not
improve over pooled/all-layer ridge readout and still recovers `0/6` clean
source-necessary IDs. Do not scale this summary-token query bottleneck upward.

The broader query-bottleneck idea is weakened, not globally killed: this probe
queries layer-summary tokens, not full token/layer traces or C2C cache-residual
targets. The next highest-value branch should either:

- train a token/layer-level C2C-residual distillation target, or
- use full source token traces with stronger cross-fit regularization and a
  rate/slot curve before any generation.

## Artifacts

See `results/svamp32_query_bottleneck_residue_20260426/manifest.md`.
