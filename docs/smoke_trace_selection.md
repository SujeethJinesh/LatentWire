# Smoke Trace Selection for GPU Phase 2

Date: 2026-05-28

## Scope

This note fixes a deterministic smoke trace set for the next GPU Phase 2 run
from cached V2 M11b DeepSeek/Falcon artifacts only. No random draw was used.

The selection uses:

- per-trace recovery and static-gap rows from `per_trace_metrics.json`;
- BF16 trace lengths and EOS positions from `bf16_trace_manifest.json`;
- activation drift from `activation_magnitudes.jsonl.gz`;
- model-level protected-set drift metadata from `protected_trajectories.json`
  and `protected_sets.json`.

For activation drift, I used a CPU-only readout: for each trace and layer,
compute the Jaccard distance between the top 10% activation-magnitude channels
at decode positions 100 and 10000, then average over layers. Protected-set
trajectories are model-level rather than trace-level, so they are recorded as
run context, while trace stress ranking uses per-trace activation drift.

## Source Artifacts

| Model | Run directory | Primary source files |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z` | `per_trace_metrics.json`, `bf16_trace_manifest.json`, `activation_magnitudes.jsonl.gz`, `protected_sets.json`, `protected_trajectories.json` |
| Falcon-H1-0.5B-Instruct | `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z` | `per_trace_metrics.json`, `bf16_trace_manifest.json`, `activation_magnitudes.jsonl.gz`, `protected_sets.json`, `protected_trajectories.json` |

Model-level protected-set trajectory drift from decode position 100 to 10000:

| Model | Regime | Mean Jaccard drift | Median Jaccard drift | Layers |
|---|---|---:|---:|---:|
| DeepSeek | `m11b_top10` | 0.4308 | 0.4638 | 28 |
| DeepSeek | `m11b_top5` | 0.5027 | 0.4902 | 28 |
| DeepSeek | `m11b_top1` | 0.5253 | 0.6087 | 28 |
| Falcon | `m11b_top10` | 0.4455 | 0.4214 | 36 |
| Falcon | `m11b_top5` | 0.4540 | 0.4478 | 36 |
| Falcon | `m11b_top1` | 0.4676 | 0.4810 | 36 |

## Exact GPU Phase 2 Smoke Set

Use these trace IDs, in this order:

```json
{
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": [
    "opencompass_AIME2025_I_5",
    "opencompass_AIME2025_I_11",
    "opencompass_AIME2025_I_8"
  ],
  "tiiuae/Falcon-H1-0.5B-Instruct": [
    "opencompass_AIME2025_I_7",
    "opencompass_AIME2025_I_1",
    "opencompass_AIME2025_I_11"
  ]
}
```

Equivalent prompt indices:

```json
{
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": [5, 11, 8],
  "tiiuae/Falcon-H1-0.5B-Instruct": [7, 1, 11]
}
```

## DeepSeek Selections

| Role | Prompt index | Trace ID | Why selected |
|---|---:|---|---|
| Known positive-gap trace | 5 | `opencompass_AIME2025_I_5` | Best stable positive M11b trace: `m11b_top10` recovery 0.7206, `static_top10` recovery 0.4873, margin +0.2334. Positive recoverable static gap: 0.002091. BF16 first EOS at 3708, so this is not the long stress case. |
| Long/high-drift stress trace | 11 | `opencompass_AIME2025_I_11` | No EOS before the fixed 20000-token BF16 cap, and highest DeepSeek per-trace activation top-10 drift: 0.8246. Recovery is near neutral/negative (`m11b_top10` -0.0924), making it a stress/control trace rather than a cherry-picked win. |
| Representative average trace | 8 | `opencompass_AIME2025_I_8` | `m11b_top10` recovery 0.3354, exactly the DeepSeek included-trace median; no EOS before 20000 tokens; large recoverable static gap 0.09656; activation drift 0.8134, close enough to stress behavior to remain useful. |

Rejected near-miss: `opencompass_AIME2025_I_10` has a large apparent
M11b-vs-static margin (+1.8939), but the recoverable static gap is only
0.0000443, so it is too denominator-sensitive for a smoke anchor.

## Falcon Selections

| Role | Prompt index | Trace ID | Why selected |
|---|---:|---|---|
| Known positive-gap/high-drift trace | 7 | `opencompass_AIME2025_I_7` | Positive `m11b_top10` recovery 0.3826 with `static_top10` recovery 0.3257, margin +0.0569. It also has the highest Falcon per-trace activation top-10 drift: 0.8426, so it combines the positive and drift strata. |
| Long stress trace | 1 | `opencompass_AIME2025_I_1` | Latest Falcon BF16 EOS by a wide margin: first EOS at decode position 9900, versus at most 2865 for the other selected Falcon candidates. It also has a large recoverable static gap: 0.2090. `m11b_top10` recovery is negative (-0.1430), so it tests long-trace robustness rather than a known win. |
| Representative average trace | 11 | `opencompass_AIME2025_I_11` | Closest Falcon trace to the median `m11b_top10` recovery and median activation drift: recovery 0.0324 versus median 0.0439, activation drift 0.7921 versus median 0.7973. BF16 first EOS at 2865, the second-latest selected Falcon decode. |

Rejected near-miss: `opencompass_AIME2025_I_0` has the highest Falcon
`m11b_top10` recovery (0.4532), but `opencompass_AIME2025_I_7` is almost as
positive while also being the highest-drift trace, so it is the better smoke
anchor under the stratification goal.

## Notes for Phase 2

- Keep DeepSeek and Falcon separate in reporting; the cached M11b evidence is
  ambiguous on DeepSeek and Falcon, not a cross-family positive-method pass.
- Do not replace these IDs with random traces. If Phase 2 needs fewer traces,
  keep the first two per model and drop only the representative average row.
- If Phase 2 can afford extra Falcon coverage, the next deterministic add-on is
  `opencompass_AIME2025_I_0` as a second positive Falcon trace.
