# Tokenizer Interface Denoise Refs (2026-04-22)

Purpose: capture the strongest interface redesign ideas that attack tokenizer
and transport mismatch directly instead of adding another nearby residual loss.

## Strongest Sources

1. Cross-Tokenizer LLM Distillation through a Byte-Level Interface
   Link: https://arxiv.org/abs/2604.07466
   Why it matters: direct byte-level teacher/student interface rather than
   token remap heuristics.

2. CTPD
   Link: https://arxiv.org/abs/2601.11865
   Why it matters: practical cross-tokenizer distillation signal.

3. ByT5
   Link: https://arxiv.org/abs/2105.13626
   Why it matters: byte-level modeling as a universal interface baseline.

4. TokAlign
   Link: https://openreview.net/forum?id=NEMEyZ3oc2
   Why it matters: learned tokenizer alignment instead of fixed remap logic.

5. FOCUS
   Link: https://arxiv.org/pdf/2305.14481
   Why it matters: vocabulary adaptation under distribution shift.

6. BLIP-2
   Link: https://arxiv.org/abs/2301.12597
   Why it matters: fixed learned query bottleneck between mismatched spaces.

7. Perceiver-VL
   Link: https://arxiv.org/abs/2211.11701
   Why it matters: learned resampling / bottleneck connectors.

8. Diffusion-Link
   Link: https://arxiv.org/abs/2510.11330
   Why it matters: denoising bridge rather than one-shot regression.

9. Denoising Diffusion Bridge Models
   Link: https://openreview.net/forum?id=1qu9RHtrC1
   Why it matters: principled bridge-model framing for iterative repair.

## Exact Next Interface Ablations

1. byte-level distillation head versus current byte sidecar
2. learned tokenizer alignment versus fixed byte/span remap
3. fixed query-bottleneck connector (`K={8,16,32}` queries) versus sparse
   dictionary
4. one-step versus three-step denoising bridge on top of the live residual lane

## Minimal Telemetry

- byte-level KL / exact byte recovery
- token fertility and boundary F1
- OOV mass and prompt-length drift
- query utilization entropy and effective rank
- stepwise help/harm for denoising bridges
- same-pair and cross-family bytes/latency frontier

## Current Read

- The strongest interface-side pivots are byte-level distillation, learned
  tokenizer alignment, fixed query bottlenecks, and denoising bridges.
- These are better bets than another simple routed bank or one-gate sidecar if
  the current real-lane query-bank branch only ties or regresses.
