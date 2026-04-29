# Source-Private Wyner-Ziv Packet Gate

- date: `2026-04-29`
- artifact: `results/source_private_wyner_ziv_packet_gate_20260429/`
- script: `scripts/build_source_private_wyner_ziv_packet_gate.py`
- test: `tests/test_build_source_private_wyner_ziv_packet_gate.py`
- scale rung: strict small / remapped learned-packet confirmation

## Purpose

This gate addresses the method-originality criticism. The deterministic
diagnostic packet is strong but protocol-shaped. Here the source encoder learns
a compact vector packet from private evidence, while the target decoder uses
public candidate side information. This is the Wyner-Ziv/Slepian-Wolf framing:
communicate only the source innovation needed by a decoder that already has
side information.

## Setup

- train/eval: `768/512`
- family set: `all -> all`
- candidate view: public slot side information
- remapped slot codebooks: `101`, `103`, `107`
- budgets: `2`, `4`, `6` bytes
- comparators: raw source sign sketch, QJL/TurboQuant-style residual packet,
  canonical RASP score packet, query-aware diagnostic-span text at the same
  budget
- controls: label-shuffled ridge, constrained shuffled source, answer-masked
  source, random same-byte

## Result

The aggregate passes: `9/9` remap/budget rows pass the scalar WZ packet rule.

| Remap | Budget | Scalar WZ | Target | Best scalar control | Raw sign | QJL | Canonical RASP | Query-aware text@budget |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 0.418 | 0.250 | 0.264 | 0.301 | 0.396 | 0.350 | 0.250 |
| 101 | 4 | 0.432 | 0.250 | 0.264 | 0.326 | 0.461 | 0.494 | 0.250 |
| 101 | 6 | 0.463 | 0.250 | 0.264 | 0.332 | 0.447 | 0.494 | 0.250 |
| 103 | 2 | 0.436 | 0.250 | 0.250 | 0.303 | 0.439 | 0.363 | 0.250 |
| 103 | 4 | 0.475 | 0.250 | 0.266 | 0.328 | 0.461 | 0.520 | 0.250 |
| 103 | 6 | 0.508 | 0.250 | 0.266 | 0.316 | 0.484 | 0.520 | 0.250 |
| 107 | 2 | 0.418 | 0.250 | 0.246 | 0.309 | 0.393 | 0.350 | 0.250 |
| 107 | 4 | 0.445 | 0.250 | 0.246 | 0.326 | 0.453 | 0.506 | 0.250 |
| 107 | 6 | 0.492 | 0.250 | 0.232 | 0.330 | 0.457 | 0.506 | 0.250 |

Headline:

- pass rows: `9/9`
- minimum passing scalar WZ accuracy: `0.418`
- minimum passing scalar-control margin: `+0.154`
- packet-vs-query-aware text oracle compression: `2.3x-7.0x`

## Interpretation

This is a stronger method contribution than the deterministic packet alone. It
shows a learned source-private syndrome can transmit useful evidence under
target candidate side information across remapped slot codebooks, with
source-destroying controls and compression-native comparators.

It is not yet a full cross-family solution. Canonical RASP is often stronger at
4-6 bytes, and cross-family rows elsewhere remain asymmetric. The paper should
frame this as a learned packet-family contribution that reduces the fixed-code
objection, not as broad latent transfer.

## Next Gate

Run the same learned WZ packet table on one bidirectional core/holdout
cross-family surface. Promotion to a headline learned-method claim requires both
directions to pass controls.
