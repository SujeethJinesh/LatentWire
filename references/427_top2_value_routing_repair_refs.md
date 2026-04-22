# Top-2 Value Routing Repair Refs (2026-04-22)

Purpose: focus the next exact same-pair branch on stronger expertized V-side
repair after `dynalign_value_bank_module_replace_residrank16 = 0.0938` and
`dynalign_value_routed_module_replace_residrank16 = 0.1250`.

## Strongest Sources

1. ResMoE
   Link: https://arxiv.org/abs/2503.06881
   Why it matters: strongest residual-restoration template for preserving a
   good dense base and restoring only what is still missing.

2. S'MoRE
   Link: https://arxiv.org/abs/2504.06426
   Why it matters: good structural template for stacking residual experts
   instead of relying on a single corrective tail.

3. Attractor Patch Networks
   Link: https://arxiv.org/abs/2602.06993
   Why it matters: low-rank patch experts are the cleanest small-graft
   analogue for routed repair on top of a fixed backbone.

4. ERMoE
   Link: https://arxiv.org/abs/2511.10971
   Why it matters: strongest recent source for stable expert initialization and
   interpretable routing.

5. DirMoE
   Link: https://openreview.net/forum?id=a15cDnzr6r
   Why it matters: useful for separating expert choice from mixture weight,
   which maps well to top-2 V-bank routing.

6. Dense Backpropagation Improves Routing for Sparsely-Gated Mixture-of-Experts
   Link: https://openreview.net/forum?id=huy8g3iKy0
   Why it matters: best reference for approximating nonselected expert output
   from selected experts in sparse routing.

7. Mixture of Thoughts
   Link: https://openreview.net/forum?id=x9tSyvnD8o
   Why it matters: motivates expert aggregation at latent level rather than
   only hard expert selection.

## Exact Next Ablations

1. `dynalign_value_bank_module_replace_residrank16` with top-1 vs top-2 value
   routing, holding `K` fixed.
2. Same branch with random vs barycenter/SVD vs cluster-prototype expert init.
3. `dynalign_value_patch_module_replace_residrank16` as a patch-expert V-only
   follow-up if top-2 still ties or regresses.

## Minimal Telemetry

- exact GSM8K32 score
- win/tie/loss versus `target_alone`
- numeric extraction coverage
- top-2 mass and route entropy
- expert load balance histogram
- `||ΔV|| / ||V||` and `||ΔK|| / ||K||`
- prompt-cluster purity for selected experts

## Current Read

- The bank-only route currently hits `0.0938`, while the simpler value-routed
  branch preserves `0.1250`.
- The next smallest plausible lift is sparse top-2 V-bank routing rather than
  another dense bank blend.
