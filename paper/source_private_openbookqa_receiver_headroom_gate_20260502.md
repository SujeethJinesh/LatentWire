# Source-Private OpenBookQA Receiver/Headroom Gate, 2026-05-02

## Status

- current paper readiness: COLM is now strong; ICLR full paper is closer but
  still not comfortable.
- current story: LatentWire has fixed-byte source-private packet transfer on
  ARC/OpenBookQA and now a held-out OpenBookQA receiver-fusion method that
  improves over packet-only.
- exact blocking gap: the receiver needs stronger per-seed/control stability,
  ARC replication, native NVIDIA systems rows, and a less label-copy-like
  common-basis or learned connector.

Artifact:
`results/source_private_openbookqa_receiver_headroom_gate_20260502/`

## What This Gate Does

Plain-language version: the source model sends a tiny `3B` hint. The receiver
has its own train-split public question/candidate scorer. A validation-only
selector learns when the hint looks worth trusting and when to fall back to the
receiver scorer. Test labels are held out until the final readout.

This is a receiver/headroom gate, not a new model-forward run. It reconstructs
the promoted OpenBookQA `3B` source-private packet from the answer-free
source-choice cache, trains the public target scorer on OpenBookQA train,
selects the packet/target fusion rule on OpenBookQA validation, and evaluates
once on OpenBookQA test.

## Results

Default seed `47` on OpenBookQA test:

| Row | Accuracy |
|---|---:|
| packet-only | 0.378 |
| train-split public target scorer | 0.372 |
| packet+target receiver | 0.424 |
| same-byte text receiver control | 0.378 |
| best no-source receiver control | 0.372 |

The default receiver improves over packet-only by `+0.046` with paired CI95
`[+0.008, +0.084]`, and improves over the target-public scorer by `+0.052`.
Across five packet projection seeds, every receiver delta over packet-only is
positive; the row-bootstrap aggregate delta is `+0.0336` with CI95 low
`+0.0004`. Strict per-seed paired-CI promotion is weaker: `2/5` seeds have
positive CI lower bounds.

## Control Readout

The default seed is the cleanest row:

- same-byte structured text receiver: `0.378`, below matched receiver `0.424`
- zero-source receiver: `0.372`
- shuffled-source receiver: `0.364`
- random same-rate packet receiver: `0.370`
- target-derived sidecar receiver: `0.352`
- candidate-derangement receiver: `0.292`
- source-label-copy receiver: `0.372`

Two cautions matter for reviewer framing:

- `label_permutation` is not destructive for this index-based MCQ receiver
  because the current runner preserves answer indices; candidate derangement is
  the meaningful candidate-control row.
- the current `3B` packet is still a compact source-selected-candidate sketch.
  The safe claim is source-private evidence fusion, not universal latent
  language or a hidden-state common basis.

## Decision

Promote this as the current positive method branch for COLM and as a necessary
but not sufficient ICLR result.

Do not claim ICLR readiness yet. The next exact gate is to replicate this
receiver on ARC-Challenge and rerun OpenBookQA with a stricter fixed selector
or nested validation protocol that reduces same-byte/control sensitivity across
all seeds.
