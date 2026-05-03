# HellaSwag Non-Qwen Receiver-Family Packet Gate

Date: 2026-05-03

## Readiness

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: the fixed-byte packet has real source-private utility beyond
  Qwen-family receivers, but learned target-aware receiver improvement is still
  missing.
- Exact remaining blocker: a receiver or connector must beat packet-only and
  source-index/source-score baselines, not merely preserve source packet utility.

## Lay Explanation

This experiment checks whether a packet produced from TinyLlama still helps a
different target model family, Phi-3-mini. Think of it as asking whether a tiny
hint from one model helps a different model answer HellaSwag questions. It
does on two adjacent frozen slices, but the learned receiver that tries to
combine the hint with Phi's own scores still does not beat simply trusting the
packet.

## Artifacts

`results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536/`

`results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048/`

`results/source_private_hellaswag_nonqwen_receiver_family_multislice_summary_20260503_validation1024_2048/`

Key files:

- `hellaswag_nonqwen_receiver_family_packet_gate.json`
- `hellaswag_nonqwen_receiver_family_packet_gate.md`
- `target_score_cache.json`
- `tinyllama_source_packet_slice_augmented.jsonl`
- `receiver_gate/`

Command:

```bash
PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_hellaswag_nonqwen_receiver_family_packet_gate.py \
  --output-dir results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536 \
  --slice-start 1024 \
  --slice-rows 512 \
  --train-prefix-rows 128 \
  --bootstrap-samples 500 \
  --target-lm-device mps \
  --target-lm-dtype float16 \
  --target-lm-prompt-mode continuation \
  --local-files-only \
  --run-date 2026-05-03
```

Second-slice command:

```bash
PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_hellaswag_nonqwen_receiver_family_packet_gate.py \
  --output-dir results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048 \
  --slice-start 1536 \
  --slice-rows 512 \
  --train-prefix-rows 128 \
  --bootstrap-samples 500 \
  --target-lm-device mps \
  --target-lm-dtype float16 \
  --target-lm-prompt-mode continuation \
  --local-files-only \
  --run-date 2026-05-03
```

Aggregate command:

```bash
./venv_arm64/bin/python \
  scripts/build_source_private_hellaswag_nonqwen_receiver_family_multislice_summary.py \
  --output-dir results/source_private_hellaswag_nonqwen_receiver_family_multislice_summary_20260503_validation1024_2048 \
  --run-date 2026-05-03
```

## Result

HellaSwag validation slice `1024:1536`, with `128` receiver train rows and
`384` receiver eval rows:

| Row | Eval Accuracy | Delta |
|---|---:|---:|
| Phi-3 target-only | `0.270833` | n/a |
| TinyLlama packet-only | `0.489583` | `+0.218750` vs target |
| Candidate ridge receiver | `0.481771` | `-0.007812` vs packet |
| Target-or-packet oracle | `0.611979` | `+0.122396` vs packet |

Gate readout:

- source utility gate: `True`
- target-family transfer gate: `True`
- receiver improvement gate: `False`
- receiver CI95 low vs packet-only: `-0.020833`
- receiver CI95 low vs target-only: `+0.141862`
- target score latency on Mac MPS: `175.2s`
- native systems complete: `False`

HellaSwag validation slice `1536:2048`, with `128` receiver train rows and
`384` receiver eval rows:

| Row | Eval Accuracy | Delta |
|---|---:|---:|
| Phi-3 target-only | `0.255208` | n/a |
| TinyLlama packet-only | `0.523438` | `+0.268229` vs target |
| Candidate ridge receiver | `0.473958` | `-0.049479` vs packet |
| Target-or-packet oracle | `0.627604` | `+0.104167` vs packet |

Gate readout:

- source utility gate: `True`
- target-family transfer gate: `True`
- receiver improvement gate: `False`
- receiver CI95 low vs packet-only: `-0.089909`
- receiver CI95 low vs target-only: `+0.147070`
- target score latency on Mac MPS: `198.7s`
- native systems complete: `False`

Two-slice aggregate over contiguous HellaSwag validation `1024:2048`:

| Row | Weighted Eval Accuracy | Delta |
|---|---:|---:|
| Phi-3 target-only | `0.263021` | n/a |
| TinyLlama packet-only | `0.506510` | `+0.243490` vs target |
| Candidate ridge receiver | `0.477865` | `-0.028646` vs packet |
| Target-or-packet oracle | `0.619792` | `+0.113281` vs packet |

Aggregate readout:

- source utility slices: `2/2`
- target-family transfer slices: `2/2`
- receiver-improvement slices: `0/2`
- total receiver eval rows: `768`
- packet contract: `2B` raw / `5B` framed
- source-private packet: `True`

## Decision

This is a useful receiver-family result but not an ICLR-positive learned
connector. The fixed TinyLlama packet transfers to a non-Qwen receiver family
on both adjacent HellaSwag slices, which strengthens the packet-utility story.
However, the learned receiver does not improve over packet-only, so it does not
solve the reviewer concern about source-index/source-choice dominance.

Do not promote this as cross-family latent reasoning. Use it as a strict
receiver-family packet-utility diagnostic. The next exact gate should target
the oracle gap with a conditional innovation packet or sparse/common-feature
packet that is explicitly compared against packet-only, source-index, quantized
source-score, and candidate-roll controls.
