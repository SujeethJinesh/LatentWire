# Anchor Tail Codec References

Date: `2026-04-22`

## Why This Memo Exists

The strongest codec-side clue in the repo is still:

- preserving a small anchor set can help a lot on the toy side
- naive codebook-tail variants are not enough yet
- the real same-pair lane needs a bytes-aware branch, not just another dense
  residual tweak

This memo narrows the next codec-side branch to anchor-preserving selective
precision and stronger tail coding.

## Strongest References

- [AWQ](https://arxiv.org/abs/2306.00978)
  Activation-aware protected channels; direct inspiration for target-aware
  anchor protection.
- [OWQ](https://arxiv.org/abs/2306.02272)
  Outlier-aware mixed precision; useful for preserving a small anchor set.
- [QuIP#](https://arxiv.org/abs/2402.04396)
  Rotation plus lattice/codebook quantization; strong “rotate first, then
  codebook the tail” reference.
- [AQLM](https://arxiv.org/abs/2401.06118)
  Additive / multi-codebook quantization; closest direct reference for a
  residual tail codebook.
- [VPTQ](https://arxiv.org/abs/2409.17066)
  Structured vector PTQ with residual/outlier handling; useful codebook-tail
  comparator.
- [Preserve-Then-Quantize](https://openreview.net/forum?id=cHUXlmsSXK)
  Dominant-subspace preservation before compression; closest conceptual match
  to the anchor-preserving branch.
- [ResQ](https://openreview.net/forum?id=4qIP1sXcR1)
  Low-rank residuals plus mixed precision; strong template for “protect core,
  quantize rest.”
- [SERQ](https://openreview.net/forum?id=nFjj8NEBqv)
  Saliency-aware low-rank error reconstruction; good reference if importance
  must be learned instead of magnitude-based.
- [QTIP](https://arxiv.org/abs/2406.11235)
  Trellis-coded quantization with incoherence preprocessing; useful if the tail
  needs stronger structured coding.
- [Residual vector quantization for KV cache compression in large language model](https://arxiv.org/abs/2410.15704)
  Direct RVQ-for-KV reference; strongest discrete tail-coding analogue.
- [KIVI](https://arxiv.org/abs/2402.02750)
  Canonical KV-cache quantization baseline.
- [MatryoshkaKV](https://arxiv.org/abs/2410.14731)
  Progressive / nested KV compression; useful for depth-aware selective
  precision.
- [TurboQuant](https://arxiv.org/abs/2504.19874)
  Modern low-distortion compression reference for fast transport.
- [KV Cache Transform Coding for Compact Storage in LLM Inference](https://arxiv.org/abs/2511.01815)
  Strongest transform-coding analogue for anchor plus coded tail transport.

## Ranked Next Experiments

1. Anchor-preserving selective precision on top of
   `dynalign_module_replace_residrank16`
   - keep top-k critical channels or positions at high precision
   - quantize only the tail
   - main question: can we preserve `0.1250` while reducing bytes?
2. Sensitivity-selected anchors instead of magnitude-selected anchors
   - use calibration loss impact or source/target sensitivity
   - main question: is the “important anchor” story structural or just an
     outlier heuristic?
3. Two-stage residual codebooks
   - coarse anchor / template stage plus RVQ-style tail stage
   - main question: does a discrete tail model outperform the current dense
     residual tail?

## Reviewer-Driven Telemetry To Log

- exact example IDs and slice hash
- source-alone, target-alone, text relay, communicated row, and oracle bound on
  the same IDs
- paired delta vs `target_alone`, bootstrap interval, and seed spread
- protected-anchor accounting:
  - which channels / heads / positions were protected
  - overlap between magnitude-selected and sensitivity-selected anchors
  - protected mass coverage
- compression accounting:
  - bytes
  - bits per protected anchor
  - bits per tail element
  - latency
  - peak memory
- tail-code telemetry:
  - codebook size
  - usage entropy / perplexity
  - dead-code rate
  - residual norm captured by each stage

## Smallest Real Benchmark Run

On the live `dynalign_module_replace_residrank16` lane:

1. preserve a small anchor set, for example top `k=8`
2. quantize the tail to `4` bits
3. compare against the exact same frozen IDs and report bytes together with
   accuracy

If the row stays at `0.1250`, the selective-precision story is alive. If it
falls back to `0.0938`, the current tail codec is still not the missing piece.
