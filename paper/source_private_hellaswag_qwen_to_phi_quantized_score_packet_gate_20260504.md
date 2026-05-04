# HellaSwag Qwen-To-Phi Quantized Score Packet Gate

Current paper readiness: COLM remains plausible as a scoped
source-private packet/evaluation paper; ICLR remains blocked. Estimated ICLR
readiness is roughly 35-45%.

Current paper story: LatentWire has strong machinery for fixed-byte
source-private packets, destructive controls, byte accounting, and Mac-local
systems evidence. The missing ICLR result is still a positive receiver method
that uses source evidence to improve a frozen target beyond target-cache,
source-index/rank, and equal-byte destructive controls.

Exact gap blocking submission: a live receiver must beat fixed hybrid by a
nontrivial margin with positive paired uncertainty, remain nonnegative on
frozen slices, and beat source-destroying controls under strict same-family and
cross-family separation.

## Purpose

Reviewer feedback left one important shallow baseline unresolved: can a tiny
quantized source score vector explain the remaining source top1/top2 oracle
headroom? This gate closes that comparator on the frozen Qwen-to-Phi
HellaSwag validation `1024:2048` surface.

The sender exposes no source text, KV cache, hidden vector, or raw logits. It
compresses Qwen's four candidate scores into `1B/2B/4B/8B` packets. The
receiver reconstructs the score vector, combines it with Phi-local scores, and
uses official-train calibration to choose a frozen blend/threshold policy.

Lay explanation: Qwen is allowed to send Phi a tiny compressed version of how
much it likes each answer. Phi then checks whether that tiny score message
helps it safely switch answers. If the same behavior works when the message is
zeroed, shuffled, or made from Phi itself, the message is not useful
cross-model communication.

## Run

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate.py \
  --output-dir results/source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate_20260504_validation1024_2048 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

## Result

The gate fails.

| Row | Accuracy | Delta vs fixed hybrid | CI95 low | Helps | Harms |
|---|---:|---:|---:|---:|---:|
| fixed hybrid | 0.467448 | 0.000000 | 0.000000 | 0 | 0 |
| best quantized score packet | 0.468750 | 0.001302 | -0.003906 | 3 | 2 |
| candidate only | 0.455729 | -0.011719 | -0.022135 | 5 | 14 |
| raw source-score logit fusion | 0.391927 | -0.075521 | -0.100260 | 33 | 91 |
| source top1/top2 oracle | 0.675781 | 0.208333 | 0.175781 | 174 | 14 |

Best selected packet:

- method: `quantized_score_packet_rotated_uniform_zscore_2B`;
- raw payload: `2B`;
- framed record: `5B`;
- bits per candidate coordinate: `4`;
- alpha: `2.0`;
- threshold: `5.896644`.

Budget curve:

| Raw bytes | Framed bytes | Best method | Accuracy | Delta | CI95 low |
|---:|---:|---|---:|---:|---:|
| 1 | 4 | `quantized_score_packet_uniform_zscore_1B` | 0.464844 | -0.002604 | -0.010417 |
| 2 | 5 | `quantized_score_packet_rotated_uniform_zscore_2B` | 0.468750 | 0.001302 | -0.003906 |
| 4 | 7 | `quantized_score_packet_rotated_uniform_zscore_4B` | 0.468750 | 0.001302 | -0.003906 |
| 8 | 11 | `quantized_score_packet_rotated_uniform_zscore_8B` | 0.468750 | 0.001302 | -0.003906 |

Slice stability is not acceptable:

| Slice | Rows | Method acc. | Fixed hybrid acc. | Delta | CI95 low | Helps | Harms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 384 | 0.494792 | 0.486979 | 0.007812 | 0.000000 | 3 | 0 |
| 1536 | 384 | 0.442708 | 0.447917 | -0.005208 | -0.013021 | 0 | 2 |

The best destructive control is
`zero_source_packet_uniform_zscore_1B_control` at `0.467448`, tied with fixed
hybrid. Official-train/eval content overlap is `0`.

## Interpretation

This closes the equal-byte source-score quantization baseline and weakens the
score-packet branch on Qwen-to-Phi. The source top1/top2 oracle still shows
large conditional headroom, but even structured 2-8 byte score packets do not
identify the safe switch rows reliably. The first slice has a small positive
effect, but the second slice goes negative, so this is not an ICLR-promotable
positive method.

The important scientific read is that score-level information is not absent;
it is not sufficient. The receiver needs richer source-specific evidence about
why source top1 or source top2 should override Phi, not just a compressed
version of source confidence.

Systems implication: the best candidate would have been attractive at `2B`
payload / `5B` framed per request, but no native systems claim is allowed
because the accuracy gate failed and NVIDIA/vLLM/SGLang serving is still
pending.

## Decision

Demote shallow score-level packet methods on Qwen-to-Phi:

- top1/top2 ambiguity buckets failed;
- raw source-score logit fusion is harmful;
- equal-byte quantized score packets fail paired uncertainty and slice
  stability.

The next highest-priority live branch should pivot back to target-native
receiver state, but with a stricter objective than prior soft-slot attempts:
selective hidden/logit resonance on answer-relevant positions, plus explicit
wrong-row, source-row-shuffle, candidate-roll, target-derived, zero-source,
and same-byte random controls.

## Artifacts

- Script:
  `scripts/build_source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate.py`
- Test:
  `tests/test_build_source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate.py`
- Results:
  `results/source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate_20260504_validation1024_2048/`
- References:
  `references/711_qwen_to_phi_quantized_score_packet_refs_20260504.md`
