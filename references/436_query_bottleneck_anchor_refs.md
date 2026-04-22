# Query Bottleneck And Anchor Refs (2026-04-22)

Purpose: capture the strongest cross-domain ideas that still look additive to
the live dynalign residual lane rather than another nearby negative control.

## Strongest Interface / Connector Sources

1. Flamingo
   Link: https://arxiv.org/abs/2204.14198
   Why it matters: query-token bottlenecks for selective transfer.

2. BLIP-2
   Link: https://arxiv.org/abs/2301.12597
   Why it matters: learned query bridge between incompatible spaces.

3. InstructBLIP
   Link: https://arxiv.org/abs/2305.06500
   Why it matters: stronger query-driven connector supervision.

4. MM1
   Link: https://arxiv.org/abs/2403.09611
   Why it matters: practical connector choices under multimodal mismatch.

5. Dense Connector
   Link: https://arxiv.org/abs/2405.13800
   Why it matters: explicit connector design instead of assuming a fixed bridge.

6. The Vision Wormhole
   Link: https://arxiv.org/abs/2602.15382
   Why it matters: strong recent connector/resampler inspiration for mismatched
   latent interfaces.

## Strongest Preserve-Core / Tail-Repair Sources

1. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: preserve anchors, compress tail.

2. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: explicit additive residual correction.

3. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: eigenspace-aware repair inspiration, even though the naive
   direct transplant failed on our contract.

4. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: importance-aware preservation.

5. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: compact codebooks for the non-anchor tail.

6. SERQ
   Link: https://arxiv.org/abs/2603.08185
   Why it matters: selective residual reconstruction over a compressed path.

## Strongest Routed / Expert Repair Sources

1. ResMoE
   Link: https://arxiv.org/abs/2503.06881
   Why it matters: residual experts rather than one dense repair path.

2. S'MoRE
   Link: https://arxiv.org/abs/2504.06426
   Why it matters: sparse mixture repair and stronger expert specialization.

3. Attractor Patch Networks
   Link: https://arxiv.org/abs/2602.06993
   Why it matters: localized selective correction instead of global residuals.

4. M2R2
   Link: https://arxiv.org/abs/2502.02040
   Why it matters: multi-route repair as a more serious alternative to the
   simple routed-bank variants.

## Exact Next Branches Suggested By The Combined Read

1. Anchor-preserving codebook tail on top of
   `dynalign_module_replace_residrank16`
2. Query-guided bottleneck / resampler sidecar with preserved anchors
3. Stronger multi-expert value-side repair with top-2 or learned sparse routing
4. Better verifier-triggered repair only if the trigger signal is stronger than
   the current one-gate sidecar

## Current Read

- The live same-pair clue still looks like “right basis first, then selective
  repair,” not “more geometry.”
- The most plausible additive next step is preserve-anchor + tail-model, with
  query-guided bottlenecks as the best lateral interface redesign.
