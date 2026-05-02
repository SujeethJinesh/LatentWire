# Source-Private ARC-Challenge Receiver/Headroom Gate, 2026-05-02

## Status

- current paper readiness: COLM remains strong; ICLR full paper is still not
  comfortable.
- current story: LatentWire has fixed-byte source-private packet transfer on
  ARC/OpenBookQA and a positive OpenBookQA packet/target receiver, but that
  receiver does not yet generalize back to ARC.
- exact blocking gap: the current validation-selected receiver gate is not a
  cross-benchmark method. We need a train-source/nested selector, a cleaner
  common-basis packet, or a systems-facing hidden-innovation codec before this
  can be an ICLR headline contribution.

Artifact:
`results/source_private_arc_challenge_receiver_headroom_gate_20260502/`

## What This Gate Does

Plain-language version: the source model sends a tiny `12B` hint for each ARC
science question. The receiver has its own public train-split scorer. A
validation-only selector learns when to trust the hint and when to use the
receiver scorer, then we evaluate once on ARC test.

This is the direct ARC replication of the OpenBookQA receiver-fusion protocol:
the packet is reconstructed from the answer-key-forbidden Qwen2.5-0.5B
source-choice cache, the target scorer trains only on ARC train, the selector
is selected only on ARC validation, and ARC test labels are held out until the
final readout.

## Results

Default seed `47` on ARC-Challenge test:

| Row | Accuracy |
|---|---:|
| packet-only | 0.344 |
| train-split public target scorer | 0.277 |
| packet+target receiver | 0.340 |
| same-byte text receiver control | 0.311 |
| best no-source receiver control | 0.277 |

The default receiver stays well above target-public (`+0.062`) but hurts the
stronger packet-only row by `-0.004` with paired CI95
`[-0.009, +0.001]`. Across five packet projection seeds, the receiver hurts
packet-only every time: aggregate row-mean delta `-0.0142`, CI95
`[-0.0218, -0.0068]`, and strict per-seed CI pass count `0/5`.

## Diagnostic Readout

The failure is not lack of complementarity. A post-hoc oracle choosing the
correct answer from either packet-only or target-public reaches about `0.524`
accuracy on test for each seed, about `+0.180` above packet-only. The problem
is selector generalization: the validation-selected receiver learns to override
packet predictions, but on test those overrides help fewer rows than they hurt.

Cheap validation-selected rule probes gave the same answer. The best simple
target-confidence rule selected on ARC validation improved validation only
slightly, then reached `0.339` on test versus packet-only `0.344`
(`-0.005`). That weakens the idea that a trivial confidence threshold fixes the
ARC failure.

## Decision

Do not promote receiver-fusion as a cross-benchmark ICLR method yet.
OpenBookQA remains a real positive row, but ARC narrows the claim to
benchmark-specific source-private evidence fusion.

Next exact gate: either generate an ARC train source-choice cache and train the
packet/target selector on train with validation-only threshold/model selection,
or move to the higher-value common-basis branch: a Fourier/anchor-syndrome
packet on the same frozen ARC surface. The latter is cleaner if the paper needs
a distinct technical contribution beyond selective packet routing.
