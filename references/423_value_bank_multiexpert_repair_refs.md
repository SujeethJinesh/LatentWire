# Value-Bank Multi-Expert Repair Refs (2026-04-22)

Purpose: focus the next exact same-pair branch on richer value-side repair
after the single-path value route tied, but did not beat,
`dynalign_module_replace_residrank16`.

## Strongest Sources

1. ResMoE
   Link: https://arxiv.org/abs/2503.06881
   Why it matters: strongest precedent for restoring a good dense base with a
   small residual expert family instead of replacing the base mapping.

2. M2R2
   Link: https://arxiv.org/abs/2502.02040
   Why it matters: supports spending repair capacity selectively rather than
   uniformly across all samples.

3. S'MoRE
   Link: https://arxiv.org/abs/2504.06426
   Why it matters: compact residual experts can stay cheap if routing is
   sparse and stable.

4. Attractor Patch Networks
   Link: https://arxiv.org/abs/2602.06993
   Why it matters: low-rank routed patches are the cleanest small-graft
   template for expertized repair on top of a frozen backbone.

5. Value Residual Learning
   Link: https://aclanthology.org/2025.acl-long.1375/
   Why it matters: best direct clue that useful repair can live mostly on the
   value path instead of a symmetric hidden-state correction.

6. SwitchCodec
   Link: https://arxiv.org/abs/2601.20362
   Why it matters: strong precedent for decoupling compact residual experts
   from their routing logic.

## Exact Next Ablations

1. `dynalign + value-bank routed repair`
2. `dynalign + top-2 value-bank repair` with entropy/load regularization
3. `dynalign + value-bank repair` under a fixed bytes budget matched to the
   live dense residual row

## Interpretable Telemetry

- per-example win/loss/tie versus target
- expert load and route entropy
- top-1 gate margin
- residual-to-base norm ratio on `V`
- bytes and latency versus `dynalign_module_replace_residrank16`
- numeric extraction coverage and empty-prediction rate

## Current Read

- `dynalign_value_routed_module_replace_residrank16 = 0.1250` kept the live
  same-pair lift with full coverage, which means selective value-side repair is
  a real lane rather than another negative control.
- The next routed step should be multi-expert/value-bank repair, not another
  single-gate blend or geometry wrapper.
