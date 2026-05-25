# Statistical Rigor Audit

Date: 2026-05-25

Scope: paper statistics, release metrics, and Phase 9 result packets.

## Findings

| Severity | Finding | Status |
|---|---|---|
| SUBSTANTIAL | Many method comparisons use only 8-10 positive-gap traces after no-gap filtering. | Paper reports included trace counts and no-gap fractions; keep this prominent. |
| SUBSTANTIAL | M11b Granite CI overlaps zero and is very wide. | Paper calls M11b partial and reports the CI; no clean generalization claim is made. |
| MINOR | Multiple methods were tried across Phase 9. | Paper frames the result as a preregistered method queue and reports KILLs; no post-hoc winner-only reporting observed. |

## Bootstrap And Seeds

- Phase 9 packets report `bootstrap_samples: 1000`.
- M11b Granite uses bootstrap seed `20260527`.
- M11b Nemotron uses bootstrap seed `20260604`.
- Release configs expose bootstrap seeds but FAST_VERIFY does not recompute CIs.

## Positive-Gap Filtering

Recovery is undefined when BF16 and static are effectively tied. The paper and
packet metrics report no-gap counts:

- M11b Granite: 4/12 no recoverable static gap; 8 included traces.
- M11b Nemotron: 2/12 no recoverable static gap; 10 included traces.
- Phase 4 no-gap fraction: 0.375.

This filtering is defensible only if presented as positive-gap recovery, which
the current paper does.

## Effect Sizes

Headline medians are reported with CIs or explicit control comparisons:

- M11b Granite top-5: `0.449284091125`, CI `[-1.300900018791, 1.000790626547]`.
- M11b Nemotron top-5: `0.456736183270`, CI `[0.345887792388, 0.794986745913]`.
- M11b Nemotron top-10: `0.814739798903`, CI `[0.254439288178, 0.922552264880]`.
- ParoQuant Granite: `0.753776848891`, CI `[0.477044452405, 1.003769554323]`.

No statistical audit issue requires changing the paper, provided the CI honesty
and positive-gap scope stay in the abstract/results/limitations.
