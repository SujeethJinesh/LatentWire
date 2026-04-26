# SVAMP32 Source-Token All-Layer Bottleneck Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: a deployable source-derived positive method plus
  medium, seed-repeat, uncertainty, source-control, and cross-family gates
- current story: the C2C-derived residue sidecar remains a useful bound, but
  source-state residue predictors do not recover clean source-necessary IDs
- blocker: source-derived residue prediction under strict controls

## Gate

After the recovered-branch audit, the remaining query-bottleneck question was
whether the older learned source-token probe failed because it used only
`mid,last` layers. This run tested full source-token all-layer features with
the existing cross-fitted learned syndrome analyzer.

Configuration:

- analyzer: `scripts/analyze_svamp32_learned_syndrome_probe.py`
- feature layers: `all`
- query count: `4`
- hidden dim: `16`
- epochs: `80`
- outer folds: `8`
- seed: `2`
- controls: zero-source, shuffled-source, label-shuffled, same-norm-noise,
  target-only, and slots-only

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 7/32 | 0/6 | 2/3 |
| zero_source | 14/32 | 0/6 | 3/3 |
| shuffled_source | 10/32 | 0/6 | 1/3 |
| label_shuffled | 13/32 | 0/6 | 3/3 |
| same_norm_noise | 14/32 | 0/6 | 3/3 |
| target_only | 14/32 | 0/6 | 3/3 |
| slots_only | 8/32 | 0/6 | 0/3 |

Failing criteria:

- `min_correct`
- `preserve_fallback_floor`
- `min_clean_source_necessary`

## Decision

Kill source-token query-bottleneck residue prediction for the current SVAMP32
syndrome surface. The full all-layer source-token variant is worse than the
previous `mid,last` source-token probe and still recovers `0/6` clean
source-necessary IDs.

Do not spend more cycles on source-token residue predictors unless a new
teacher signal or source surface changes the hypothesis.

## Audit Synthesis

Subagent and local artifact audits found no overlooked real benchmark-positive
branch:

- query-pool/idweighted SVAMP variants remain below target self-repair and
  recover at most `1/6` clean residual IDs
- raw GSM70 dynalign is seed-fragile and not source-specific under later
  runtime controls
- process repair and stochastic route generation have real oracle/selector
  headroom, but not a source-control-clean communication claim
- the strongest under-incorporated positives are toy interface results:
  quotient/GPA/sparse dictionary plus sequence-aligned byte sidecars, which
  should be treated as a cross-family interface-stress direction rather than a
  same-family Qwen rescue

## Next Gate

Switch from tuning dead SVAMP32 source-state residue predictors to one of:

1. process-repair/selector source-control diagnostic on a strict clean surface,
   only promoting if it recovers source-derived clean IDs beyond target
   self-repair without target-self losses; or
2. real cross-family tokenizer/interface stress gate for the quotient/GPA
   sparse dictionary plus sequence-aligned byte sidecar lane.

Both are source-surface discovery gates, not ICLR-positive claims yet.

## Artifacts

See
`results/svamp32_source_token_all_layers_bottleneck_20260426/manifest.md`.
