# Verifier-Gated V-Sidecar Refs (2026-04-22)

Purpose: keep verifier-gated and disagreement-triggered refinement as the
smallest lateral additive branch on top of the live dynalign residual lane.

## Strongest Sources

1. Goedel-Prover-V2
   Link: https://arxiv.org/abs/2508.03613
   Why it matters: verifier-guided self-correction with a compact scaffold.

2. PAG
   Link: https://arxiv.org/abs/2506.10406
   Why it matters: selective revision only when a verifier says the answer is
   wrong.

3. Adaptive Coopetition
   Link: https://arxiv.org/abs/2510.18179
   Why it matters: coarse verifier signals for deciding when to collaborate or
   abstain.

4. ProgCo
   Link: https://arxiv.org/abs/2501.01264
   Why it matters: explicit verification and self-correction logic.

5. FLASH
   Link: https://arxiv.org/abs/2505.12728
   Why it matters: latent-aware speculative refinement template for a sidecar.

6. Speculative Verification
   Link: https://arxiv.org/abs/2509.24328
   Why it matters: accept/reject via information gain rather than unconditional
   refinement.

7. Latent Refinement Decoding
   Link: https://arxiv.org/abs/2510.11052
   Why it matters: belief-state refinement in latent space.

8. Value Residual Learning
   Link: https://aclanthology.org/2025.acl-long.1375/
   Why it matters: direct evidence that additive correction may belong on the
   value path.

## Exact Next Ablations

1. `dynalign_module_replace_residrank16 + verifier-gated V-side sidecar`
2. `dynalign_module_replace_residrank16 + disagreement-triggered routing`
3. `dynalign_module_replace_residrank16 + speculative latent refinement sidecar`

## Minimal Telemetry

- gate rate and acceptance rate
- disagreement score histogram
- verifier score before/after repair
- residual norm delta by layer
- per-expert load balance if routing is used
- numeric extraction coverage, exact match, latency, bytes

## Current Read

- If expertized V repair keeps tying or regressing, verifier-gated V-side
  repair is the cleanest lateral branch that still matches the paper story.
