# Anchor Tail Follow-up Refs (2026-04-22)

Purpose: keep the preserve-anchor / codebook-tail backup branch explicit if
expertized repair does not move the frozen same-pair contract.

## Strongest Sources

1. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: closest preserve-core plus reconstruction-tail framing.

2. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: protected core plus mixed-precision residual repair.

3. SERQ
   Link: https://arxiv.org/abs/2603.08185
   Why it matters: saliency-aware tail repair over a compact correction path.

4. QERA
   Link: https://arxiv.org/abs/2410.06040
   Why it matters: clean decomposition of protected anchor versus repair tail.

5. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: simplest importance-budgeting reference for what to protect.

6. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: strongest additive codebook-tail analogue.

7. TurboQuant
   Link: https://arxiv.org/abs/2504.19874
   Why it matters: lightweight residual-correction inspiration for a repaired
   tail representation.

8. AnTKV
   Link: https://arxiv.org/abs/2506.19505
   Why it matters: direct anchor-aware compression precedent, even though it is
   on KV cache rather than bridge residuals.

## Exact Next Ablations

1. frozen GSM8K32 anchor-fraction sweep with fixed tail rank
2. tail-format sweep: dense residual tail vs codebook tail vs hybrid codebook +
   low-rank repair
3. routed-tail sweep: global tail vs expertized tail vs V-only routed tail

## Minimal Telemetry

- `anchor_fraction`, `anchor_ids`, `anchor_overlap_jaccard`
- `tail_format`, `tail_rank`, `tail_codebook_bits`
- `core_residual_norm`, `tail_residual_norm`, `core_tail_error_ratio`
- `codebook_utilization`, `dead_codes`, `assignment_entropy`
- `win_loss_tie`, `numeric_extraction_coverage`, `bytes_per_example`, `latency_ms`

## Current Read

- The preserve-anchor family is still the best backup if stronger expertized
  repair does not beat the frozen same-pair contract.
