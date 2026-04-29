# ICLR Strengthening Cycle

- date: `2026-04-29`
- scale rung: strict small receiver confirmation / systems and novelty planning
- live branch: source-private evidence packets with frozen target-decoder
  receiver

## Contribution Stack

Current technical contributions:

1. Source-private evidence-packet benchmark and source-destroying control
   protocol.
2. Learned Wyner-Ziv/source-side-information compact packets.
3. Byte-rate systems frontier versus structured text, query-aware text, and
   full hidden-log relay.
4. Canonical RASP candidate-relative packets for remap robustness.
5. Protected rotated residual codec as a quantization-inspired comparator and
   near-miss.
6. Frozen-model target-decoder receiver evidence.

What still needs work:

- endpoint TTFT/E2E throughput telemetry;
- `n=160` or `n=256` all-control receiver rows;
- bidirectional cross-family transfer, likely requiring a non-scalar
  anchor-relative sparse packet;
- clearer competitor positioning against C2C/KV/activation/latent-agent
  communication.

## Gate Run This Cycle

The highest-priority local gate was the frozen target-decoder receiver because
it addresses the hand-coded-decoder reviewer objection without needing remote
NVIDIA GPUs.

Results:

| Surface | n | Conditions | Matched | Target | Best control | Valid matched | Pass |
|---|---:|---|---:|---:|---:|---:|---|
| core seed29 | 32 | all six | 0.688 | 0.250 | 0.250 | 1.000 | true |
| holdout seed30 | 32 | all six | 0.750 | 0.250 | 0.281 | 1.000 | true |

Artifacts:

- `results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n32_all_controls_cpu_max24/`
- `results/source_private_tool_trace_target_decoder_progress_gate_20260429/holdout_seed30_qwen3_n32_all_controls_cpu_max24/`

This is a strict-small positive receiver row: the target is a frozen
`Qwen/Qwen3-0.6B` CPU model, not a hand-coded decoder, and matched 2-byte
packets beat target-only while shuffled-source, random same-byte, and
matched-byte structured text controls stay near the target prior.

## Diagnostics

- MPS remains blocked for this Qwen3 receiver path by the Apple MPS matmul shape
  error, even on an `n=4` probe.
- A short-decode `max_new_tokens=8` n32 diagnostic fails with valid prediction
  rate `0.000` because generated labels are truncated to a shared prefix. The
  valid rows use `max_new_tokens=24`.
- Holdout n16 all-control was strongly positive but narrowly failed the small-n
  random control cutoff (`0.312` versus target `0.250` and cutoff `0.300`). The
  n32 holdout row passes with best control `0.281`.

## Literature / Subagent Synthesis

Novelty scout conclusion: the broad claim "models communicate through latent or
non-text channels" is risky. C2C, KVComm/KVCOMM, DroidSpeak, activation
communication, LatentMAS/Interlat-style latent agents, CIPHER, and prompt
compression already cover large parts of that space. The defensible novelty is
the combined threat model: source-private evidence, decoder side information,
extreme byte budgets, source-destroying controls, candidate-relative packet
design, and frozen target decoding.

Systems scout conclusion: the strongest systems story is not "we beat KV
compression" generally. It is a rate/latency frontier where a 2-byte
source-private packet sits far left of structured text and full-log relay. The
next systems gate should report endpoint TTFT/E2E latency for packet,
target-only, query-aware compressed text, JSON/free-text relay, and full-log
relay.

New-mechanism scout conclusion: the next genuinely distinct technical branch
should be anchor-relative sparse innovation packets. Scalar WZ and canonical
RASP fail bidirectional cross-family; a sparse relative dictionary/crosscoder
packet directly targets that failure by transmitting top-k source-private atoms
in an anchor-defined coordinate system.

Reference memo:

- `references/488_iclr_strengthening_and_anchor_relative_refs.md`

## Decision

Promote the frozen target-decoder receiver from smoke to strict-small
supporting evidence. Do not claim full receiver generality yet.

Keep same-family/remap packets as the positive method core. Do not claim
bidirectional cross-family transfer. The next best full-paper strengthening
move is either:

1. an endpoint TTFT/E2E systems frontier on this Mac; or
2. an anchor-relative sparse innovation packet smoke on the existing
   core-to-holdout and holdout-to-core surfaces.

Given the user's request for deeper technical contributions, the next method
branch should be the anchor-relative sparse packet. Endpoint telemetry should
remain the next systems gate.
