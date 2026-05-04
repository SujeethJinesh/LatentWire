# 2026-05-04 HellaSwag Receiver-Calibrated Top1/Top2 Ambiguity-Code Gate

## Readiness

- Current paper readiness: COLM remains plausible with narrow packet/system
  claims; ICLR is still blocked by the lack of a learned positive receiver.
- Current story: fixed-byte source-private packets and destructive controls are
  defensible, but shallow learned receivers have not yet extracted source
  information beyond packet/source-choice baselines.
- Exact remaining blocker: a learned source-private receiver must beat
  packet-only/fixed-hybrid and destructive controls with positive paired
  uncertainty on frozen held-out slices.

## Lay Explanation

This gate asks whether TinyLlama can send Qwen a tiny clue that says more than
"my favorite answer is A." The new one-byte packet stores:

- source top-1 candidate id: `2` bits;
- source top-2 candidate id: `2` bits;
- sparse common-basis atom slot: `4` bits.

Qwen then uses its own scores and target-side common-basis atom values to choose
among source top-1, source top-2, Qwen target top-1, Qwen mean-zscore, and Qwen
hybrid. If the sparse atom is real communication, it should beat a top1/top2
packet without the atom, and it should break under wrong-row, candidate-roll,
target-derived, atom-permutation, random-byte, zero-source, and label-permuted
controls.

## Artifact

`results/source_private_hellaswag_receiver_calibrated_top2_ambiguity_code_gate_20260504_validation1024_2048/`

Key files:

- `hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.json`;
- `ambiguity_rows.csv`;
- `ambiguity_config_rows.csv`;
- `ambiguity_blocks.csv`;
- `frontier_rows.csv`;
- `control_rows.csv`;
- `manifest.json`.

## Method

The run extends
`scripts/build_source_private_hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.py`.
It reuses the previous train-only CCA/common-basis and decision-supervised
SAE-style sparse encoder, but adds a receiver-calibrated top1/top2 ambiguity
packet. The source-side packet still exposes no source text, KV cache, raw
hidden vectors, raw score vectors, or logits.

Run command:

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.py \
  --output-dir results/source_private_hellaswag_receiver_calibrated_top2_ambiguity_code_gate_20260504_validation1024_2048 \
  --pca-dims 64 \
  --shared-dims 8 \
  --sae-atoms 64 \
  --sae-topks 2 \
  --decoder-ridges 0.01,0.1,1,10,100,1000,10000 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

## Result

The gate fails.

| Row | Accuracy | Delta vs packet | CI95 low | Helps | Harms |
|---|---:|---:|---:|---:|---:|
| Packet-only | `0.501953` | `0.000000` | `0.000000` | `0` | `0` |
| Receiver-calibrated top1/top2 ambiguity code | `0.500977` | `-0.000977` | `-0.002930` | `0` | `1` |
| Source-pair no-atom ambiguity decoder | `0.501953` | `0.000000` | `0.000000` | `0` | `0` |
| Target-derived source-pair ambiguity control | `0.501953` | `0.000000` | `0.000000` | `0` | `0` |
| Wrong-row ambiguity-code control | `0.500977` | `-0.000977` | `-0.007812` | `5` | `6` |
| Candidate-roll ambiguity-code control | `0.501953` | `0.000000` | `-0.003906` | `2` | `2` |
| Atom-slot permutation control | `0.500977` | `-0.000977` | `-0.003442` | `0` | `1` |
| Random same-byte control | `0.498047` | `-0.003906` | `-0.008790` | `2` | `6` |
| Label-permutation decoder control | `0.497070` | `-0.004883` | `-0.011719` | `4` | `9` |
| Source top1/top2 oracle diagnostic | `0.693359` | `+0.191406` | `+0.160620` | `235` | `39` |

Block deltas versus packet-only are `0.000000`, `0.000000`, `0.000000`,
`-0.004878`, and `0.000000`, so block stability also fails.

## Interpretation

The top1/top2 oracle remains large, which means source-side information exists:
the correct answer is often in TinyLlama's top two choices. But this specific
receiver-calibrated sparse/common-basis ambiguity packet does not extract it.
The learned sparse atom causes one harmful override and no helpful overrides;
removing the atom, deriving the source pair from the target, or rolling the
candidates performs as well or better.

This weakens the shallow sparse/common-basis ambiguity-code branch. It does not
kill all latent transfer, but it rules out the current version as an ICLR
positive method. The next method branch should move to the Qwen-to-Phi
official-train scaffold with a stronger source-specific packet, or to an
explicit syndrome/error-correction audit if a near-miss appears.

## Decision

Demote the receiver-calibrated top1/top2 ambiguity code as implemented. Do not
claim SAE/common-basis latent communication, sparse atom causality, or
source-specific ambiguity resolution from this result.

The live next gate should use the Qwen-to-Phi official-train harm-controlled or
receiver-calibrated scaffold, because that path has the reviewer-relevant
fixed-hybrid baseline and native source/target separation. It must include
score-only, rank-only, label-copy, top1/top2-only, target-derived packet,
wrong-row, candidate-roll, atom/feature-id shuffle, atom knockout, same-byte
random, and source-score/logit-fusion controls.
