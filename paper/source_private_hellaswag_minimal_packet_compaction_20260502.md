# HellaSwag Minimal Packet Compaction

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM is stronger after this systems compaction
  result; ICLR still needs a positive receiver/common-language method, broader
  benchmark coverage, and native GPU serving rows.
- Current story: HellaSwag hidden-innovation packets already transfer useful
  source-private decision information. This gate shows the runtime packet can
  be smaller without changing any promoted predictions.
- Exact remaining blocker: the receiver branch still does not beat packet-only
  cross-family, and native NVIDIA/vLLM/SGLang systems rows remain pending.

## Lay Explanation

The old packet had two bytes: one byte worth of answer choice plus one
confidence/debug byte. This experiment checks whether the answer choice alone
is enough. It packs the selected candidate into one byte, decodes it, and
verifies that every decoded answer is exactly the same as before.

## Artifact

`results/source_private_hellaswag_minimal_packet_compaction_20260502/hellaswag_minimal_packet_compaction.json`

Supporting files:

- `results/source_private_hellaswag_minimal_packet_compaction_20260502/hellaswag_minimal_packet_compaction.md`
- `results/source_private_hellaswag_minimal_packet_compaction_20260502/manifest.json`

## Method

- Inputs:
  - Qwen full-validation global-stability HellaSwag packet artifact.
  - Qwen full-validation prediction JSONL.
  - TinyLlama full-validation HellaSwag packet artifact.
  - TinyLlama full-validation prediction JSONL.
- Candidate count: `4`.
- Theoretical payload: `2` bits.
- Implemented compact packet: `1B` raw candidate-id payload plus the existing
  `3B` framing assumption, for `4B` framed per request.
- Original packet: `2B` raw / `5B` framed.
- Pass rule: compact decoding must exactly reproduce every original selected
  prediction, inherited source gates must have passed, raw and framed bytes
  must both decrease, and no source text, KV cache, raw hidden vector, or raw
  score vector can be transmitted.

## Result

Gate outcomes:

- pass gate: `true`;
- prediction-equivalent rows: `30126/30126`;
- source exposure clear: `true`;
- native GPU claims allowed: `false`.

Packet accounting:

| Quantity | Original | Compact | Reduction |
|---|---:|---:|---:|
| raw payload bytes / request | `2B` | `1B` | `50%` |
| framed record bytes / request | `5B` | `4B` | `20%` |
| logical raw bytes / 10042-row validation | `20084B` | `10042B` | `50%` |
| logical framed bytes / 10042-row validation | `50210B` | `40168B` | `20%` |
| packed batch-64 framed bytes | `320B` | `256B` | `20%` |

Accuracy is unchanged because the decoded predictions are identical:

| Row | Compact Accuracy | Baseline | Delta | CI95 low |
|---|---:|---:|---:|---:|
| Qwen mean-zscore | `0.526688` | best label-copy `0.480880` | `+0.045808` | `+0.040131` |
| Qwen hybrid | `0.532464` | best label-copy `0.480880` | `+0.051583` | `+0.044812` |
| TinyLlama mean-zscore | `0.619199` | best label-copy `0.558753` | `+0.060446` | `+0.053921` |

## Interpretation

This is a promoted systems-side improvement. It preserves the strongest current
HellaSwag packet evidence while making the runtime packet smaller. The safe
claim is:

> On the promoted HellaSwag source-private packet rows, the confidence/debug
> byte is not needed for decoding. A one-byte candidate-id packet reproduces
> all selected predictions exactly, reducing logical raw payload by `50%` and
> framed payload by `20%`.

This does not solve the cross-family receiver/common-language gap. It also does
not claim GPU throughput, HBM traffic reduction, vLLM/SGLang serving speed, or
superiority to C2C/KVComm/KV-quantization methods.

## Contribution Status

Promoted:

1. A stricter rate point for the source-private HellaSwag packet method:
   `1B` raw / `4B` framed instead of `2B` raw / `5B` framed.
2. Exact prediction-equivalence evidence across Qwen mean-zscore, Qwen hybrid,
   and TinyLlama full-validation packet rows.
3. A clearer systems table that separates logical packet bytes, packed batch
   bytes, and cacheline/DMA-rounded single-request bytes.

Still blocked:

1. Positive receiver improvement over packet-only.
2. A second benchmark with the same minimal packet discipline.
3. Native NVIDIA/vLLM/SGLang rows and matched comparisons against C2C/KVComm,
   QJL/TurboQuant, KIVI, and KVQuant.

## Decision

Use the `1B` raw / `4B` framed packet as the default HellaSwag systems row
going forward. Keep receiver/common-language claims disabled until a method
beats packet-only under official-train calibration and destructive controls.
