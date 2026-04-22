# Value-Routed Repair Refs (2026-04-22)

Purpose: focus the next exact same-pair branch on value-side selective repair
instead of another global residual or geometry tweak.

## Strongest Sources

1. Value Residual Learning
   Link: https://aclanthology.org/2025.acl-long.1375/
   Why it matters: the best direct clue that residual help may belong on the
   value path rather than a uniform hidden-state correction.

2. ResMoE
   Link: https://arxiv.org/abs/2503.06881
   Why it matters: strong template for restoring a good dense base with a
   small shared corrective path instead of replacing it wholesale.

3. M2R2
   Link: https://arxiv.org/abs/2502.02040
   Why it matters: conditional residual spending is a better fit than paying a
   uniform correction everywhere.

4. S'MoRE
   Link: https://arxiv.org/abs/2504.06426
   Why it matters: residual experts can stay tiny and still be structurally
   useful if routing is selective.

5. Attractor Patch Networks
   Link: https://arxiv.org/abs/2602.06993
   Why it matters: routed low-rank patch experts are the cleanest small-graft
   precedent for selective repair on top of a fixed backbone.

6. SwitchCodec
   Link: https://arxiv.org/abs/2601.20362
   Why it matters: compact residual-expert banks are viable if routing and
   repair capacity are decoupled.

## Exact Next Ablations

1. `dynalign + value-routed repair`
2. `dynalign + K/V routed bank` as the bigger follow-up only if the value-side
   row survives the same frozen contract

## Interpretable Telemetry

- per-example win/loss/tie vs target
- routed fraction on `V`
- route entropy and expert utilization if the branch becomes multi-expert
- residual-to-base norm ratio for `K` and `V` separately
- bytes and latency versus plain `dynalign_module_replace_residrank16`

## Current Read

- On the frozen GSM8K32 contract,
  `dynalign_value_routed_module_replace_residrank16 = 0.1250` with full
  numeric extraction coverage (`32/32`).
- That ties the live `dynalign_module_replace_residrank16 = 0.1250` row rather
  than improving it, so value-side routing is now a live branch and control,
  not a new ceiling.
- The next routed step should be multi-expert or value-bank repair, not
  another single-path dense route blend.
