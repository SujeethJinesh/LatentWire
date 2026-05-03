# ARC Conditional-Innovation Packet Gate

Date: 2026-05-03

## Status

This is not an ICLR promotion. It is a useful preflight result: the
receiver-conditioned innovation packet beats learned source-index and
quantized source-score decoder controls on the frozen ARC validation heldout
parity split, but it does not beat direct source-label text by enough and all
paired intervals still cross zero.

## Experiment

Artifact:

`results/source_private_arc_conditional_innovation_packet_gate_20260503_qwen05_qwen3_validation/`

Inputs:

- eval split:
  `results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl`
- source cache:
  `results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation/source_scores.json`
- receiver cache:
  `results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation/receiver_scores.json`

Protocol:

- calibration rows: even ARC validation row indices, `n=150`;
- heldout rows: odd ARC validation row indices, `n=149`;
- source: Qwen2.5-0.5B-Instruct local choice log-likelihood cache;
- receiver side information: Qwen3-0.6B local choice log-likelihood cache;
- packet: one 4-bit clipped innovation value per public candidate, where
  innovation is `source row-zscore - receiver row-zscore`;
- max candidate count: `5`;
- raw payload: `23` bits, `3` bytes;
- framed record with header/CRC: `6` bytes;
- decoder: candidate-wise ridge model fit on calibration rows and evaluated on
  heldout rows.

Lay explanation: the receiver already has its own opinion about each answer.
Instead of sending "the source picks B," this packet sends a tiny list of where
the source's answer scores disagree with the receiver's answer scores. The
question is whether that disagreement helps the receiver make a better answer
choice.

## Results

Heldout ARC validation parity rows:

| condition | accuracy |
|---|---:|
| source-label text | `0.389262` |
| receiver-label text | `0.335570` |
| source-index-only decoder | `0.362416` |
| quantized source-score packet | `0.362416` |
| matched conditional-innovation packet | `0.395973` |
| innovation plus source-index packet | `0.355705` |
| zero-innovation control | `0.355705` |
| row-shuffle innovation control | `0.308725` |
| candidate-roll innovation control | `0.315436` |

Paired bootstrap:

| comparison | mean | CI95 low | CI95 high |
|---|---:|---:|---:|
| matched minus source-label text | `+0.006711` | `-0.053691` | `+0.067114` |
| matched minus source-index decoder | `+0.033557` | `-0.026846` | `+0.093960` |
| matched minus quantized source-score packet | `+0.033557` | `-0.013423` | `+0.087248` |
| matched minus best control | `+0.033557` | `-0.026846` | `+0.093960` |

Flip audit against direct source-label text:

- same prediction: `120/149`;
- fixed source errors: `11/149`;
- broke source-correct rows: `10/149`;
- changed wrong-to-wrong: `8/149`;
- net correct versus source: `+1`.

Subgroup readout:

- On source/receiver disagreement rows (`n=59`), source-label text is
  `0.389831` and matched innovation is `0.372881`.
- On source-wrong rows (`n=91`), matched innovation recovers `0.120879`,
  equal to the quantized source-score packet and below receiver-only / shuffled
  controls.
- On source-right rows (`n=58`), matched innovation preserves only
  `0.827586`, while source-index-only preserves `0.896552` and direct
  source-label text is `1.000000` by definition.

## Decision

The branch is alive as a design direction but not promotable:

- It improves over learned source-index and quantized source-score decoder
  controls in mean accuracy.
- It does not meaningfully beat direct source-label text.
- Its net gain is only one heldout example.
- The subgroup audit suggests the decoder mostly tracks the source and is not
  reliably correcting source errors.
- The paired intervals cross zero, so this cannot be claimed as stable.

The next method should not be another scalar/score-surface decoder. The next
highest-value branch is a query-conditioned sparse innovation packet:

1. project source evidence into a public receiver-query or SAE/common-feature
   basis;
2. send top-k sparse innovation atoms or QJL-style sign sketches, not full
   candidate score rows;
3. require it to beat source-label/source-index, quantized source-score,
   same-byte text, target-derived packet, zero-source, row-shuffle,
   candidate-roll, and label-shuffle controls;
4. run seed repeats and paired uncertainty before widening benchmarks.

## Systems Boundary

The systems-side claim is still byte/exposure accounting only. The packet is a
different communicated object from KV transfer, C2C, KVComm, LMCache, and
CacheGen, but native TTFT/TPOT/ITL/goodput/memory claims require NVIDIA runs.

The minimal NVIDIA gate should compare target-only, LatentWire packet,
same-byte visible text, and LMCache/CacheGen-style KV/cache movement on vLLM
and at least one SGLang check, with paired quality deltas and request-level
serving metrics.

## Reviewer Framing

Cut or demote:

- global score-vector fusion;
- simple learned receiver tuning;
- claims that the current conditional-innovation packet is already a positive
  method.

Keep:

- the strict source-private packet contract;
- the byte/exposure accounting;
- the destructive controls;
- the conditional-innovation framing as the next method search path.
