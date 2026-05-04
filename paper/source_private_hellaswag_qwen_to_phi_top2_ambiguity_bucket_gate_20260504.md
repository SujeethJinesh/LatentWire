# HellaSwag Qwen-To-Phi Top1/Top2 Ambiguity Bucket Gate

Current paper readiness: COLM remains plausible as a scoped source-private
packet/evaluation paper; ICLR remains blocked. Estimated ICLR readiness is
roughly 35-45%.

Current paper story: LatentWire has strong machinery for source-private
fixed-byte packets, destructive controls, byte accounting, and Mac-local
systems evidence. The missing ICLR result is still a learned receiver that uses
source evidence to improve a frozen target beyond source-index/rank and
target-derived controls.

Exact gap blocking submission: the learned receiver must beat fixed hybrid,
candidate-only/source-index style rows, target-only, no-syndrome top-pair
packets, and source-destroying controls with paired uncertainty on frozen
validation slices.

## Purpose

The previous receiver-calibrated top1/top2 ambiguity-code gate failed because
the extra 4-bit sparse atom did not add causal source-specific evidence. This
gate moved the top1/top2 ambiguity idea into the stricter Qwen-to-Phi
official-train scaffold:

- Qwen source may inspect its own scores.
- The packet exposes only source top1/top2 IDs and quantized source-side
  decision-syndrome bins.
- Phi may use Phi-local score bins as decoder side information.
- The bucket receiver is selected on official-train fit/dev rows and frozen on
  validation `1024:2048`.
- Pass requires beating fixed hybrid, candidate-only, no-syndrome/source-index
  controls, and destructive controls.

Lay explanation: Qwen sends Phi its two best guesses plus a few tiny confidence
clues. Phi is allowed to switch away from the safe fixed answer only for clue
patterns that helped on training questions. If the same behavior appears when
the clues are shuffled, zeroed, or made from Phi itself, the packet is not real
cross-model communication.

## Run

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.py \
  --output-dir results/source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate_20260504_validation1024_2048 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

## Result

| Row | Accuracy | Delta vs fixed hybrid | CI95 low | Helps | Harms |
|---|---:|---:|---:|---:|---:|
| fixed hybrid | 0.467448 | 0.000000 | 0.000000 | 0 | 0 |
| top2 ambiguity bucket | 0.467448 | 0.000000 | 0.000000 | 0 | 0 |
| no-syndrome top-pair control | 0.467448 | 0.000000 | 0.000000 | 0 | 0 |
| source top1 | 0.411458 | -0.055990 | -0.078809 | 22 | 65 |
| raw source-score logit fusion | 0.391927 | -0.075521 | -0.100260 | 33 | 91 |
| source top1/top2 oracle | 0.675781 | 0.208333 | 0.177083 | 174 | 14 |

The official-train selector chose `no_op` with zero eligible buckets. The
selected packet therefore ties fixed hybrid and the no-syndrome top-pair
control. The best destructive control also ties fixed hybrid because the
selected model does not act.

Slice stability is neutral, not positive:

| Slice | Rows | Method acc. | Fixed hybrid acc. | Delta |
|---:|---:|---:|---:|---:|
| 1024 | 384 | 0.486979 | 0.486979 | 0.000000 |
| 1536 | 384 | 0.447917 | 0.447917 | 0.000000 |

## Interpretation

This kills the current top1/top2 score-level ambiguity-bucket branch on the
strongest available Qwen-to-Phi decision surface. The source top1/top2 oracle
still has large headroom, but the official-train low-harm receiver cannot find
stable buckets that generalize even to held-out official-train dev rows.

The important distinction is that the headroom is conditional and fragile. It
exists if an oracle knows when to choose source top1 or top2, but shallow
receiver-visible score bins do not identify those rows safely. This means more
top-pair bucket variants are low expected value unless they introduce new
source-specific evidence beyond rank/score bins.

Decision: demote score-level top1/top2 ambiguity packets. The next live gate
should be an equal-byte quantized source-score / QJL-TurboQuant-style
comparator on cached scores. That closes a reviewer-requested missing baseline
and tests whether a structured quantized score packet can recover any of the
oracle headroom before returning to heavier target-native latent receivers.

## Artifacts

- Script:
  `scripts/build_source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.py`
- Test:
  `tests/test_build_source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.py`
- Results:
  `results/source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate_20260504_validation1024_2048/`
- References:
  `references/710_qwen_to_phi_top2_ambiguity_bucket_refs_20260504.md`
