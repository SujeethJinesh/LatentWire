# Reference Memo 748: OpenBookQA Hardening And Sparse-Basis Boundaries

Date: 2026-05-04

## Why This Memo Exists

The live LatentWire decision surface now has a caveated OpenBookQA
packet+target receiver positive row, while the ARC sparse/behavior atom
branches still fail strict source-causality controls. The next ICLR gate should
therefore harden the OpenBookQA receiver before spending more time on generic
SAE/crosscoder/prefix variants.

## Primary Sources And Boundaries

- [OpenBookQA](https://aclanthology.org/D18-1260/) is a science QA benchmark
  built to require open-book scientific facts plus common reasoning. LatentWire
  uses it as a second benchmark surface because it is smaller and more
  Mac-local than full long-context evaluations, but a positive row still needs
  strict source-choice and wrong-row controls before becoming ICLR-grade.
- [Cache-to-Cache](https://openreview.net/forum?id=LeatkxrBCi) is the closest
  dense cache-fusion baseline. LatentWire should not claim raw accuracy,
  latency, HBM, energy, or throughput superiority without native measurement;
  the defensible contrast is low-rate source-private packets and utility per
  byte.
- [Communicating Activations Between Language Model Agents](https://arxiv.org/abs/2501.14082)
  directly pressures broad "latent communication" novelty. LatentWire should
  claim packetized, source-private communication under destructive controls,
  not generic activation exchange.
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600),
  [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410), and
  [Universal Sparse Autoencoders](https://arxiv.org/abs/2502.03714) make
  sparse/common feature discovery non-novel by itself. A LatentWire sparse
  basis must be judged by downstream source-causal utility per byte.
- [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/) and
  [Transcoders](https://arxiv.org/abs/2406.11944) are strong priors for
  shared/private atom and causal-feature bases. They are useful method
  inspiration, but the novelty must be packet transfer with atom-shuffle,
  wrong-row, target-derived, and source-choice controls.
- [Relative Representations](https://openreview.net/forum?id=SrC-nwieGJ) and
  [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)
  motivate common geometry. LatentWire should cite them as motivation and
  baselines, not as proof that a common packet basis works.

## Hardening Outcome

The OpenBookQA gate was hardened with same-source-choice wrong-row packets and
candidate-roll packets. The default matched receiver still reaches `0.424`
accuracy, but the same-source-choice wrong-row control reaches `0.422`, leaving
only a `0.002` default gap. The row therefore should not be promoted as a clean
second-benchmark positive method. It is useful evidence that current
packet+target fusion mostly tracks source-choice structure.

## Consequence For The Next Gate

The OpenBookQA hardening gate now weakens score/choice receiver fusion as an
ICLR path:

1. Keep the train-only source-private `3B` packet and target-public receiver
   contract.
2. Add same-source-choice wrong-row, source-index/rank/score, candidate-roll,
   source-row shuffle, target-derived sidecar, same-byte text, and random
   same-byte controls.
3. Require positive paired uncertainty beyond the strongest control, not only
   beyond packet-only.
4. Report helps/harms, coverage, utility per byte, and packet/framed/DMA-rounded
   bytes.

The next exact ICLR gate is a fresh ARC n32 tokenwise source-evidence cache with
target-loss connector preflight. If local model loading is not feasible on the
Mac, fall back to a target-side behavior-transcoder feasibility proof before
source packetization. The project should not run another shallow selector or
mean-hidden packet until one of those gates produces source-causal signal.
