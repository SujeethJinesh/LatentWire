# Routed Residual Real-Lane Refs (2026-04-21)

Purpose: tighten the next routed-repair branch around the live same-pair lane
`dynalign_module_replace_residrank16` instead of treating routing as a generic
MoE detour.

## Strongest Sources

1. ResMoE
   Link: https://arxiv.org/abs/2503.06881
   Why it matters: shared-barycenter restoration is the closest direct template
   for adding a compact corrective branch on top of a working dense base.

2. M2R2
   Link: https://arxiv.org/abs/2502.02040
   Why it matters: conditional residual spending is a better fit than paying a
   uniform correction everywhere.

3. S'MoRE
   Link: https://arxiv.org/abs/2504.06426
   Why it matters: hierarchical residual experts are a strong precedent for
   routing low-rank repair while keeping the backbone fixed.

4. Attractor Patch Networks
   Link: https://arxiv.org/abs/2602.06993
   Why it matters: routed low-rank patch experts are the cleanest direct
   inspiration for a tiny query-conditioned repair bank.

5. SwitchCodec
   Link: https://arxiv.org/abs/2601.20362
   Why it matters: adaptive residual experts can be kept compact if routing and
   capacity are decoupled.

6. Value Residual Learning
   Link: https://aclanthology.org/2025.acl-long.1375/
   Why it matters: if the remaining failure is value starvation, a value-side
   bypass may work better than another hidden-state-only dense residual.

## Exact Next Ablations

1. `dynalign + routed residual repair`
   Why now: smallest selective-repair branch that keeps the current dense lane
   intact and only spends extra capacity on mismatched samples.

2. `dynalign + routed residual repair + no-op control`
   Why now: distinguishes true routing benefit from plain extra parameters.

3. `dynalign + value-side routed residual bypass`
   Why now: tests whether the remaining same-pair gap is more about value-path
   information loss than hidden-state mismatch.

## Interpretable Telemetry

- route entropy
- expert utilization and dead-route rate
- routed fraction of samples / tokens
- residual-to-base norm ratio
- help / harm conditioned on router confidence
- numeric extraction coverage
- bytes and latency deltas versus the plain rank16 row

## Current Read

- The live real same-pair ceiling is still
  `dynalign_module_replace_residrank16 = 0.1250`.
- Fixed wrappers, preserve-tail splits, eigenspace projection, and simple
  saliency weighting all fail to retain that lift on the frozen GSM8K32
  contract.
- Routed residual repair is now the smallest branch that changes the actual
  repair policy instead of just reweighting the same dense correction.
