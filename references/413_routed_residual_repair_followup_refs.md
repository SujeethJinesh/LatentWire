# Routed Residual Repair Follow-Up Refs (2026-04-21)

Purpose: capture the strongest routed or conditional residual-repair references
for LatentWire now that simple one-path residual variants have mostly failed on
the frozen same-pair contract.

## Strongest Sources

1. S'MoRE
   Link: https://arxiv.org/abs/2504.06426
   Why it matters: hierarchical low-rank residual trees are the closest direct
   precedent for routed residual experts.

2. StructMoE
   Link: https://proceedings.mlr.press/v262/sarwar24a.html
   Why it matters: low-rank experts under a tight compute budget.

3. ReMix
   Link: https://openreview.net/forum?id=zNqc0li5Dl
   Why it matters: router collapse is a real risk; route quality must be logged
   instead of assumed.

4. Gate to the Vessel
   Link: https://openreview.net/pdf?id=Zfk5IoAtP0
   Why it matters: confidence-gated residual experts only activate when repair
   is likely to help.

5. ResMoE
   Link: https://openreview.net/forum?id=YF6CVtbh5U
   Why it matters: reconstruct experts around a shared barycenter rather than
   storing each expert densely.

6. ProDA
   Link: https://openreview.net/forum?id=e3uQFSdJP0
   Why it matters: multiplicative modulation can be better than flat additive
   correction when the missing signal is structured.

7. LoRA-X
   Link: https://openreview.net/forum?id=6cQ6cBqzV3
   Why it matters: cross-model adapter transport is feasible if routing respects
   subspace similarity.

## Exact Next Ablations

1. `dynalign + routed residual repair`
   Why now: activate repair only on uncertain or mismatched residuals instead
   of paying the same correction everywhere.

2. `dynalign + no-op / flat residual / routed residual`
   Why now: the minimal control ladder for whether routing is actually useful.

3. `dynalign + routed residual + adaptive canonical wrapper`
   Why now: only if the routed lane survives by itself on the frozen contract.

## Interpretable Telemetry

- route entropy
- expert utilization and dead-route rate
- gated fraction of tokens / positions
- help vs harm conditioned on router confidence
- latency and bytes per token

## Current Read

- The live same-pair lane is still a single-path residual row:
  `dynalign_module_replace_residrank16 = 0.1250`.
- The current failure pattern suggests the next lift may need selective repair,
  not another global residual tweak.
