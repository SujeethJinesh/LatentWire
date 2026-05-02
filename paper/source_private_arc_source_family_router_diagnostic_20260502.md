# ARC Source-Family Router Diagnostic

Date: 2026-05-02

## Status

- Current paper readiness: COLM remains plausible with honest scope; ICLR is
  still blocked by source-family robustness.
- Current story: fixed-byte source-private packets work on ARC/OpenBookQA, and
  the ARC Fourier/anchor-syndrome row gives a public common-basis packet with
  mismatch controls. The source-family falsification shows the ARC method is
  not yet robust to TinyLlama-vs-Qwen source disagreement.
- Exact gap: convert the observed source/packet oracle headroom into a
  train-selected positive method, or move to a learned connector/stronger
  source-family gate.

## Experiment

Artifact:
`results/source_private_arc_challenge_source_family_router_diagnostic_20260502/`

Command:

```bash
./venv_arm64/bin/python scripts/build_source_private_arc_challenge_source_family_router_diagnostic.py \
  --output-dir results/source_private_arc_challenge_source_family_router_diagnostic_20260502 \
  --bootstrap-samples 500
```

Lay explanation: TinyLlama and Qwen often chose different source answers on
ARC. I tested whether the receiver could look at simple packet-confidence
signals and decide which tiny packet to trust. This is like asking whether a
student can tell which of two short hints is reliable just by looking at how
confident the final answer looks.

## Result

The validation-selected router does not repair the strict disagreement slice:

- selected metric: `best_score`
- validation accuracy: `0.394`
- test TinyLlama packet accuracy: `0.269`
- test Qwen-substituted packet accuracy: `0.317`
- test router accuracy: `0.315`
- router minus Qwen-substituted mean/min: `-0.002` / `-0.008`
- minimum paired CI95 low versus Qwen-substituted: `-0.023`
- packet oracle accuracy on the same rows: `0.586`
- source-choice oracle accuracy on the full ARC test set: `0.455`

## Interpretation

This is useful because it separates two failure modes. The oracle numbers show
that the disagreement rows contain complementary information: sometimes the
TinyLlama packet is right when the Qwen-substituted packet is wrong. However,
receiver-only packet confidence does not identify those cases reliably. Cheap
selective classification is therefore not enough for the current ARC
source-family repair.

The next high-value gate should use one of:

1. source-side score/confidence caches from TinyLlama and Qwen, still under the
   answer-key-forbidden source-cache audit;
2. a learned common-basis connector trained on public train rows and evaluated
   on frozen ARC disagreement rows;
3. a stronger non-Qwen source on NVIDIA so the alternate source has enough
   competence to support cross-family transfer.

## Positioning

This result narrows, rather than weakens, the paper story. It says LatentWire is
not just prefix/prompt tuning and not a high-rate KV-cache fusion method. C2C
projects and fuses source KV caches inside the target model, while KVComm and
KVCOMM transmit or reuse KV states. TurboQuant compresses vectors/KV cache
states. Our current method is a fixed-byte, source-private packet across a
discrete boundary; the open problem is learning or selecting a robust common
language under that severe byte and source-privacy constraint.

Relevant primary sources:

- C2C: https://arxiv.org/abs/2510.03215
- KVComm: https://arxiv.org/abs/2510.03346
- KVCOMM: https://arxiv.org/abs/2510.12872
- TurboQuant: https://arxiv.org/abs/2504.19874
- Selective classification: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

## Decision

Rule out receiver-only scalar packet-confidence routing as the next ICLR repair
for ARC source-family disagreement. Promote source-side confidence routing or a
learned connector as the next exact gate.
