# CommonsenseQA Score-Packet and Fusion Probe

Date: 2026-05-01

## Status

This branch is not promotable as an ICLR headline result.

The positive fact remains useful: CommonsenseQA has real non-science source
signal. The negative fact is more important for the paper: simple score,
rank, confidence, and receiver-fusion packets do not beat a strict source-label
text control on the held-out validation parity split.

## Why We Ran This

The previous CommonsenseQA fixed-packet probe showed matched source-private
packet accuracy of `0.440` against target/index-prior accuracy of `0.206`, but
same-byte structured text also reached `0.440`. In plain terms, on this
benchmark the current packet mostly acts like “the source picked option B,”
and short answer strings make that easy for a same-byte text control to copy.

This follow-up asked whether the source score distribution carries extra
usable information beyond the top answer label.

## Runs

### Rank-bin score packet

Artifact:
`results/source_private_commonsenseqa_score_packet_headroom_20260501_qwen05_validation_rankbin/`

- source-label text heldout accuracy: `0.4377`
- top-2 threshold score-packet heldout accuracy: `0.4410`
- threshold packet delta over source-label text: `+0.0033`
- best rank-bin packet: `top2_margin_4bin`
- best rank-bin heldout accuracy: `0.4377`
- best rank-bin delta over source-label text: `+0.0000`
- source top-2 oracle heldout accuracy: `0.6721`
- pass gate: `false`

Interpretation: the top-2 oracle is large, so the source score list contains
latent headroom. However, margin bins do not identify when the runner-up should
replace the source top answer. Every learned rank bin selected rank 0.

### Quantized source-score fusion packet

Artifact:
`results/source_private_commonsenseqa_score_fusion_packet_probe_20260501_qwen05_qwen3_validation/`

- packet: one clipped 4-bit source z-score per candidate, `3B` raw / `6B`
  record
- receiver side information: Qwen3-0.6B local choice log-likelihood scores
- source-label text heldout accuracy: `0.4377`
- receiver-label heldout accuracy: `0.3246`
- best top-label pair rule: `always_source`
- best top-label pair heldout accuracy: `0.4377`
- calibrated fusion source weight: `1.0`
- quantized fusion heldout accuracy: `0.4295`
- fusion delta over source-label text: `-0.0082`
- source/receiver union top-2 oracle heldout accuracy: `0.7279`
- pass gate: `false`
- receiver scoring latency on CPU: `574.9s`

The first receiver run on MPS failed with an Apple MPS matmul shape error, so
the recorded artifact uses CPU. This is a systems warning: native accelerator
claims still require a proper NVIDIA/vLLM path and cannot rely on this Mac MPS
execution path.

### Cached margin-conditioned DFS

Using the saved source and receiver score caches, I tested margin-conditioned
fusion weights by source-margin and receiver-margin bins.

- full-precision, 2x2 margin bins: heldout `0.4393`, delta `+0.0016`
- full-precision, 4x4 margin bins: heldout `0.4197`, delta `-0.0180`
- full-precision, 8x8 margin bins: heldout `0.3902`, delta `-0.0475`
- quantized, 2x2 margin bins: heldout `0.4361`, delta `-0.0016`
- quantized, 4x4 margin bins: heldout `0.4180`, delta `-0.0197`
- quantized, 8x8 margin bins: heldout `0.3836`, delta `-0.0541`

Interpretation: the branch overfits quickly and does not provide a
reproducible selector for the top-2 oracle headroom.

## Decision

Ruled out for now:

- top-2 margin thresholding as the promoted CommonsenseQA method
- rank-bin score packets
- global Qwen2.5-0.5B -> Qwen3-0.6B score fusion
- margin-conditioned score fusion with the current two score caches

Still alive:

- a distribution-QJL packet ISA that encodes centered source distribution
  evidence as a sign sketch over candidate residuals
- a harder non-science benchmark where same-byte text cannot cheaply copy the
  selected answer string

Promoted next gate:

- HellaSwag validation bridge with the current fixed-packet controls. It is a
  non-science, four-choice commonsense completion benchmark with longer answer
  endings, which should make same-byte text a much stricter control than
  CommonsenseQA short answer phrases.

## Paper Impact

For ICLR, CommonsenseQA should be framed as a saturation diagnostic and reviewer
threat model, not as a headline positive benchmark. The paper can use it to show
we are explicitly separating real source communication from answer-copying.

For COLM, this is useful as a limitations/result-quality section, but only if
the positive story remains centered on ARC/OpenBookQA and not on non-science
generality.
