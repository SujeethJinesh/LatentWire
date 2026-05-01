# HellaSwag Anchor-Variant Scout

## Status

This scout weakens the anchor/common-basis rescue branch. Dense hidden innovation
remains the live positive method; anchor-derived common-basis variants do not yet
recover enough of the dense signal to justify an all-slice promotion run.

## Why This Gate Was Run

The five-slice dense hidden-innovation gate passed with a fixed 2-byte raw
source-private packet, but the anchor-relative common-basis gate failed. This
scout asked whether the failure was specifically due to using the wrong anchor
coordinates, rather than evidence that the hidden signal is unusable.

In lay terms: the previous method worked when the sender could privately look at
its own hidden vectors. This scout tried several ways to translate those vectors
into a shared map. The map did not preserve enough of the useful information.

## Result

Artifact:
`results/source_private_hellaswag_anchor_variant_scout_20260501_qwen05_validation4096_5120/hellaswag_anchor_variant_scout.json`

Decision slice: HellaSwag validation rows `4096:5120`.

| Variant | Accuracy | Delta vs best label-copy | CI95 low vs label-copy | Delta vs score-only | Scout pass |
| --- | ---: | ---: | ---: | ---: | --- |
| cosine_full | 0.512695 | +0.012695 | -0.002930 | +0.015625 | false |
| signed_topk_hash16x64 | 0.497070 | -0.002930 | -0.015161 | 0.000000 | false |
| cosine_topk16 | 0.497070 | -0.002930 | -0.016138 | 0.000000 | false |
| rbf_topk16 | 0.497070 | -0.002930 | -0.016602 | 0.000000 | false |
| spectral32 | 0.499023 | -0.000977 | -0.013672 | +0.001953 | false |
| qjl_sign32 | 0.497070 | -0.002930 | -0.015625 | 0.000000 | false |

Best label-copy baseline on this slice: `0.500000`.
Score-only bagged control: `0.497070`.
Dense hidden-innovation reference: `0.503125` weighted over the five-slice run.

## Interpretation

The full cosine anchor variant has a nonzero signal on this particular slice,
but it is not strong enough to clear the promotion rule: it misses the
`+0.02` delta requirement against best label-copy, misses the `+0.02` delta
against score-only, and its paired CI lower bound crosses zero. The local
top-k, RBF, spectral, QJL-sign, and signed-hash variants collapsed to score-only
or near score-only.

This makes the next high-value branch a dense sketch or learned sparse basis,
not more hand-designed anchor features. The anchor family should be treated as
demoted unless a new mechanism explicitly changes the representation class,
such as a learned crosscoder/SAE-style shared dictionary or a QJL-style dense
random projection of residuals.

## Contribution Accounting

Current ICLR contributions still alive:

1. Fixed-byte source-private hidden-innovation packet: positive on 5 frozen
   HellaSwag slices, with label-copy, score-only, zero-hidden, wrong-hidden, and
   candidate-roll controls.
2. Evaluation protocol: contiguous frozen slices, train-sample bagging, paired
   bootstrap intervals, and source-destroying controls that separate real source
   communication from target-cache effects.
3. Systems contract: 2-byte raw / 5-byte framed packet accounting and native
   systems ledger, with GPU-native latency/throughput still pending.

Needs more work:

- Common-basis/shared-coordinate method: anchor-relative and anchor-variant
  scouts are negative or underpowered.
- Cross-family falsification: still required before ICLR claims.
- Native systems row: Mac-local evidence exists, but NVIDIA/vLLM or equivalent
  serving evidence remains a blocker for a systems-heavy ICLR story.

## Decision

Promote: dense hidden innovation remains live.

Weaken: anchor-relative common-basis, local top-k anchors, RBF anchors, spectral
anchor basis, anchor QJL-sign sketches, and signed top-k anchor hashing.

Next exact gate: run a dense residual sign-sketch / QJL scout from existing
cached hidden residuals. If that fails, move to learned sparse/crosscoder-style
shared dictionaries rather than further hand-designed anchor charts.
