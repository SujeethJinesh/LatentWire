# V2 M11b DeepSeek Result

## Decision

`AMBIGUOUS_V2_M11B_DEEPSEEK`

## Run Directory

`experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z`

## Headline Results

All results use the 12-trace AIME-2025 deterministic slice. One trace had no
recoverable static gap, leaving 11 included traces for recovery statistics.

| Regime | Median recovery | CI95 | Interpretation |
|---|---:|---:|---|
| M11b top-1 | 0.001 | [-0.378, 0.333] | no useful recovery |
| M11b top-5 | -0.177 | [-1.079, 0.369] | negative median |
| M11b top-10 | 0.335 | [-0.409, 0.494] | positive median, wide CI |
| Static top-10 | 0.377 | [-1.142, 0.538] | matched static budget slightly higher median |

## Rationale

M11b top-10 clears the raw median floor of 0.30 but does not beat static top-10
by the preregistered 0.15 margin and has a confidence interval crossing zero.
The result is not a PASS, but it is also not a clean KILL because top-10 still
shows positive median recovery while top-1/top-5 do not.

This weakens the architecture-agnostic M11b-positive story. It supports the
current architecture-dependent framing: M11b is strong on Nemotron, ambiguous on
DeepSeek, and still needs Falcon before the cross-model positive-method gate can
be decided.

## Queue Impact

Continue V2 with Falcon-H1 under the remaining V2 cap. If Falcon also fails to
PASS, M11b should be framed as a Nemotron-strong, architecture-dependent method
rather than a broadly transferable remedy.
