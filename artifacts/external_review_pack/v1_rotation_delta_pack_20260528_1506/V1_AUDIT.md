# V1 Audit

## Same Baseline As Nemotron M11b

Yes. V1 reuses BF16 traces, BF16 score cache, static top-1% score cache, and
protected sets from:

`experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`

The source artifact hashes are copied in
`experiments/v1_paroquant_nemotron/source_artifacts.json`.

## Why 10/12 Traces Are Included

The recovery metric is computed only where static top-1% is worse than BF16,
so there is a positive recoverable static gap. Two traces had no positive
recoverable gap and are listed in `tables/excluded_traces.csv`.

Excluded traces:

- nemotron_trace_3 / opencompass_AIME2025_I_3: no positive static-1pct recoverable gap, gap=-0.0596918
- nemotron_trace_4 / opencompass_AIME2025_I_4: no positive static-1pct recoverable gap, gap=-2.79888e-05

## How Recovery Can Exceed 1

Recovery is `1 - (L_method - L_bf16) / (L_static - L_bf16)`, using perplexity
losses in this packet. If the method is better than BF16 on the scored window,
`L_method < L_bf16`, the numerator is negative and recovery exceeds 1. This is
not clipped; the paper should describe the metric as unclipped recovery.

## Outlier Check

Included ParoQuant recoveries range from 0.960 to 1.952.
The largest trace is `opencompass_AIME2025_I_9` with recovery
1.952. The result is not solely
driven by that trace because every included trace has recovery above
0.960, and the bootstrap CI lower bound remains above 1.

## No-Gap Traces

There are 2 no-gap traces
(0.167 fraction). They are
excluded from recovery aggregation and listed explicitly.

## Baseline / Cache / Version Mismatches

No baseline mismatch is visible in the artifact metadata. V1 uses the same
prompt SHA, model ID, model snapshot, BF16 trace artifact, BF16 score cache,
and static top-1% score cache as the Nemotron M11b reference. The V1 checker
passed with `artifact_complete=true`.

Model snapshot: `cbd3fa9f933d55ef16a84236559f4ee2a0526848`

Implementation caveat: V1 is marked
`algorithmic_reproduction_not_full_upstream`, so it is a ParoQuant-style
algorithmic reproduction rather than the upstream runtime kernel.

## Check Script

The check script passed. See
`experiments/v1_paroquant_nemotron/check_script_output.txt`.

## Reproducibility From Score Caches

The pack includes the compact score caches under
`experiments/v1_paroquant_nemotron/source_score_cache/`. The checker result is
also included. Re-running the checker in the source repository on the original
run directory reproduces the decision.

Source artifact count: 4
