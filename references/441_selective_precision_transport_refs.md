# Selective Precision / Transport Compression References

Date: `2026-04-22`

## Why This Memo Exists

The live same-pair gain is still narrow, and most simple routed or geometry
variants fall back to the old ceiling. The most plausible next codec-side idea
is still:

- preserve the important anchors exactly or at higher precision
- compress only the tail
- make the transport target-aware rather than magnitude-only

## Strongest References

- [AWQ](https://arxiv.org/abs/2306.00978)
  Activation-aware selective precision; direct inspiration for protected
  channels or positions.
- [QuIP#](https://arxiv.org/abs/2402.04396)
  Incoherence plus codebooks; useful if we want rotation-before-codebook.
- [SpinQuant](https://arxiv.org/abs/2405.16406)
  Learned rotations for outlier suppression; good for basis-selection before
  selective precision.
- [AQLM](https://arxiv.org/abs/2401.06118)
  Additive / residual codebook quantization; strong match for codebook tails.
- [Preserve-Then-Quantize](https://openreview.net/forum?id=cHUXlmsSXK)
  Dominant-subspace preservation followed by low-rank reconstruction; closest
  direct match to anchor-preserving tails.
- [ResQ](https://openreview.net/forum?id=4qIP1sXcR1)
  Residual low-rank plus mixed precision; strong template for “protect the
  core, quantize the rest.”
- [VPTQ](https://arxiv.org/abs/2409.17066)
  Vector PTQ with residual/outlier handling; useful for structured tails.
- [PatternKV](https://arxiv.org/abs/2510.05176)
  Pattern-aligned residual quantization for KV caches; relevant if the bridge
  state should be coded relative to a pattern bank.
- [KVzip](https://arxiv.org/abs/2505.23416)
  KV compression with context reconstruction under unknown future queries.
- [KV Cache Transform Coding](https://openreview.net/forum?id=aNVKROYpLB)
  Strong transform-coding baseline for KV transport.
- [GuidedQuant](https://arxiv.org/abs/2505.07004)
  End-loss-guided quantization; useful if importance must be target-aware.
- [TurboBoA](https://arxiv.org/abs/2602.04929)
  Attention-aware selective precision and broken-down channel handling.

Practical comparator:

- [EXL2 / exllamav2](https://github.com/turboderp-org/exllamav2)
  Worth treating as an implementation-style mixed-bit comparator even though it
  is not the core paper direction.

## Best Near-Term Ablations

1. Anchor-preserving tail quantization on top of `dynalign + resid16`.
2. Sensitivity-chosen protected channels instead of magnitude-only protected
   channels.
3. Two-stage residual codebooks for the transport tail.
4. Pattern-aligned residual transport relative to a small pattern bank.
5. Learned orthogonal rotation plus selective precision.

## Smallest Real Benchmark Experiment

On top of the live `dynalign_module_replace_residrank16` lane:

- keep the top `k` critical channels or positions in full precision
- quantize the remaining tail to `4` bits
- compare against the current `0.1250` row on the exact same frozen IDs

If the row stays at `0.1250` while reducing bytes, the selective-precision story
is real. If it falls back to `0.0938`, the tail codec is not yet the missing
piece.
