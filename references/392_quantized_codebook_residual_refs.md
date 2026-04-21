# Quantized Residual / Codebook Communication Refs (2026-04-21)

Purpose: capture the most relevant fresh references for the next
compression/communication branch after the frozen GSM8K32 checkpoint sweep. The
current read is that byte-only alignment and the first SAE proxy are negative
on the exact same-pair contract, so the next codec-like move should preserve
answer-bearing transport directions rather than just compress activations.

## Highest-Signal References

1. Multi-Way Representation Alignment
   Link: https://arxiv.org/abs/2602.06205
   Why it matters: gives a better shared coordinate system across models, which
   is a stronger precondition for any learned discrete or quantized channel.

2. Cross-Tokenizer LLM Distillation through a Byte-Level Interface
   Link: https://arxiv.org/abs/2604.07466
   Why it matters: makes byte-level alignment a learned interface objective
   rather than the fixed byte-overlap heuristic that failed on the same-pair
   contract.

3. DWA-KD
   Link: https://arxiv.org/abs/2602.21669
   Why it matters: turns dynamic alignment quality into a teacher-weighting
   signal, which is directly relevant to our best current real proxy lane.

4. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: suggests preserving dominant directions first and pushing
   quantization into the residual, which fits our “transport the answer bits”
   problem better than weight-only serving quantization.

5. GlowQ
   Link: https://arxiv.org/abs/2603.25385
   Why it matters: another strong residual-aware quantization reference that
   reinforces the idea that correction should target the hard residual rather
   than the whole latent.

6. CommVQ
   Link: https://arxiv.org/abs/2506.18879
   Why it matters: gives a commutative vector-quantized interface that is much
   closer to a true communication codec than AWQ/EXL2-style model compression.

7. Transporting Task Vectors across Different Architectures without Training
   Link: https://arxiv.org/abs/2602.12952
   Why it matters: points to functional residual transport as a different
   object to compress and move, not just activations or weights.

## Non-Redundant Ablations

1. Output-aware alignment + low-rank residual correction
   Why now: the checkpoint sweep says output-aware alignment is the only live
   same-pair real proxy, so the next compression-side bet should repair that
   transport residual directly.

2. Shared-basis codebook communication initialized from the sparse dictionary
   Why now: the first SAE proxy was a clean negative, so the next discrete
   channel should be a true rate-controlled codebook experiment, not just a
   sparse adapter swap.

## Recommended Next Experiment

Run `dynalign_module_replace + low-rank residual correction` first on the
frozen GSM8K32 contract with a small rank sweep, then compare that against a
learned codebook / adaptive-K channel only if the residual-correction branch
still leaves a significant gap to `C2C`.
