# Qwen-to-Phi Error-Conditioned Syndrome Gate

## Status

This is a negative learned-receiver gate. It weakens the target-error
source-top2 syndrome branch as a near-term ICLR-positive method, even though
the oracle audit still shows real source-unique headroom.

## What Ran

- Script: `scripts/build_source_private_hellaswag_qwen_to_phi_error_conditioned_syndrome_gate.py`
- Test: `tests/test_build_source_private_hellaswag_qwen_to_phi_error_conditioned_syndrome_gate.py`
- Artifact: `results/source_private_hellaswag_qwen_to_phi_error_conditioned_syndrome_gate_20260504_validation1024_2048/`
- Input eval: cached HellaSwag validation rows `1024:2048`, with fit/select prefixes removed per slice
- Packet: Qwen source top1/top2 candidate IDs plus quantized source-side syndrome bins
- Receiver side information: Phi-local score bins only
- Raw source text, source KV, source hidden states, and raw source score/logit vectors: not exposed

## Main Result

The official-train selector found a dev-positive model:

- selected scheme: `compact_error_syndrome`
- selected focus: `fixed_and_phi_wrong`
- official-dev delta versus fixed hybrid: `+0.016129`
- official-dev accuracy: `0.526882`

That did not generalize:

| row | accuracy | delta vs fixed hybrid | CI95 low | helps | harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed Qwen-hybrid packet | `0.467448` | `0.000000` | `0.000000` | `0` | `0` |
| error-conditioned syndrome packet | `0.463542` | `-0.003906` | `-0.020833` | `20` | `23` |
| source-pair no-syndrome control | `0.467448` | `0.000000` | `0.000000` | `0` | `0` |
| Qwen candidate only | `0.455729` | `-0.011719` | `-0.023438` | `5` | `14` |
| Phi target only | `0.263021` | `-0.204427` | `-0.250033` | `105` | `262` |

Oracle headroom remains large but not learned:

| oracle diagnostic | accuracy | delta vs fixed hybrid | helps | harms |
| --- | ---: | ---: | ---: | ---: |
| fixed hybrid or Qwen top2 | `0.694010` | `+0.226562` | `174` | `0` |
| fixed hybrid or Phi top2 | `0.727865` | `+0.260417` | `200` | `0` |
| fixed hybrid or union top2 | `0.845052` | `+0.377604` | `290` | `0` |

The learned packet captured only `20 / 174` possible fixed-hybrid repair helps
and introduced `23` new harms.

## Controls

The selected packet beat the destructive controls numerically, but failed the
actual method gate because it underperformed fixed hybrid and tied/lost to the
no-syndrome source-pair control.

Best destructive control:

- `target_derived_source_packet_receiver_control`: `0.450521`

Other destructive controls were lower:

- source-score row shuffle before encoding: `0.449219`
- zero-source packet: `0.256510`
- source-row shuffle: `0.240885`
- random same-byte source: `0.225260`
- candidate-roll source: `0.196615`

## Decision

Demote target-error-conditioned Qwen-top2 syndrome repair from the primary
ICLR path. The source-unique top2 rows are real, but this score/bin receiver
cannot identify them safely on held-out Qwen-to-Phi.

Promote the target-resonance soft-prefix encoder branch as the highest-value
next method branch. The strongest current story is now:

1. Target-native soft prefixes have demonstrated capacity to recreate
   full-prompt behavior.
2. Existing source-score/top2 packet receivers saturate or fail under strict
   controls.
3. The next positive method should train a source-conditioned encoder to emit
   target-native soft slots under wrong-source, zero-source, target-derived,
   candidate-roll, source-index/rank/score, same-byte, and target-only-slot
   controls.

## Lay Explanation

We found cases where Qwen's two favorite answers include the right answer when
the current Qwen-to-Phi packet is wrong. This experiment tried to learn a rule
for when Phi should trust that tiny Qwen hint. The rule looked useful on
training/dev questions, but on new questions it made slightly more mistakes
than it fixed.
