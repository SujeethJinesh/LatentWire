# Value Verifier Sidecar Real-Lane Refs (2026-04-22)

Purpose: keep the next same-pair branch tightly focused on value-only repair
 that is accepted selectively rather than applied unconditionally.

## Strongest Sources

1. Goedel-Prover-V2
   Link: https://arxiv.org/abs/2508.03613
   Why it matters: strongest recent template for verifier-guided correction.

2. PAG
   Link: https://arxiv.org/abs/2506.10406
   Why it matters: explicit accept-or-abstain framing for selective revision.

3. FLASH
   Link: https://arxiv.org/abs/2505.12728
   Why it matters: closest latent-side refinement pattern to a compact
   sidecar that only patches when justified.

4. Speculative Verification
   Link: https://arxiv.org/abs/2509.24328
   Why it matters: accept/reject logic grounded in whether refinement adds
   information.

5. Value Residual Learning
   Link: https://aclanthology.org/2025.acl-long.1375/
   Why it matters: direct support for putting additive repair on the value path.

## Exact Next Ablations

1. single-sidecar V repair with learned accept gate
2. disagreement-triggered V repair with a fixed threshold control
3. top-2 verifier-gated V repair if the single-sidecar lane preserves the live row

## Minimal Telemetry

- sidecar acceptance rate
- wins/ties/losses vs `target_alone`
- coverage and empty-prediction rate
- `||delta V|| / ||V||`
- latency and extra bytes

## Current Read

- This is the cleanest lateral branch after routed-bank saturation because it
  tests selective repair directly instead of adding more continuous mixing.
