# Vacation Decision: M11b Outcome and Cross-Domain Branching

Date: 2026-05-18 UTC

## Situation

M11b budget scaling landed with checker decision
`PASS_M11B_BUDGET_MATTERS` on Granite-4-H-Small:

- Run dir:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`
- Pass regime: `m11b_top5`
- Top-5 median recovery: `0.4492840911245966`
- Top-5 CI95: `[-1.3009000187907436, 1.00079062654749]`
- Static-top10 median recovery: `-0.05176717655558488`
- Top-5 minus static-top10 median: `0.5010512676801815`
- Included positive-static-gap traces: `8/12`

## Human-Defined Outcome Cases

Case 1, clean `PASS_M11B_BUDGET_MATTERS`: top5 median `>= 0.30`, CI95 lower
`> 0.10`, top5 beats static-top10 by `>= 0.15`, and random-walk control does
not beat top5. Paper trajectory becomes a positive-method story, and priority
shifts to M11b replication on Nemotron-3-Nano and DeepSeek-R1-Distill-Qwen-1.5B.

Case 2, `PASS_M11B_BUT_STATIC_MATCHES`: top5 passes but static-top10 recovers
similarly. Paper trajectory becomes a simpler budget-suffices story with lower
ceiling.

Case 3, `AMBIGUOUS_M11B`: top5 median is intermediate or the CI is very wide.
Continue the queued mechanism work: M26, M27, ParoQuant, and KL accumulation.
M31 remains conditional on KL decay; M33 remains conditional on M26 partial
signal.

Case 4, `KILL_M11B`: top5 median is below `0.10` or fails controls. Continue
the negative-mechanism queue and conditional M31/M33 logic.

## Classification of the Landed Result

The checker mechanically returns `PASS_M11B_BUDGET_MATTERS` because top5
median recovery is above `0.30` and top5 beats static-top10 by more than
`0.15`. However, under the human's stricter provisional interpretation, this
is not a clean Case 1 because the top5 CI lower bound is
`-1.3009000187907436`, far below `0.10`.

I therefore treat the result as:

- **Mechanical decision:** `PASS_M11B_BUDGET_MATTERS`
- **Scientific branching posture:** replication-required, Case 3-style
  ambiguous positive signal

This preserves the checker result without overclaiming the method.

## Branching Decision

Do not immediately authorize M31 or M33 from the cross-domain list. Continue:

1. M26 stable-core size check and M26 if suitable.
2. M27 layer-stratified protection.
3. ParoQuant baseline.
4. KL accumulation.

If the human wants a positive-method-first path, the most direct alternative is
to replicate M11b top5 on Nemotron-3-Nano and DeepSeek-R1-Distill-Qwen-1.5B
before M26/M27. I am not switching to that path without an explicit human
instruction because the latest queue placed M26/M27/ParoQuant/KL next.

## Synthetic-Validation Caveats

Synthetic checks for M30/M31/M33 are informative but not definitive. They use
independent-channel and linear-error assumptions that do not match real LLM
activation dynamics, and their spectra are easier than the empirical Granite
spectral readout. The real arbiter remains the landed packet data.

## What Would Invalidate This Decision

This decision should be revisited if:

- M11b replicates on a second model.
- M26 returns a partial stable-core signal that authorizes M33.
- KL accumulation returns a high decay parameter supporting M31.
- The human decides that mechanical PASS is enough to prioritize replication
  over the currently queued mechanism experiments.
