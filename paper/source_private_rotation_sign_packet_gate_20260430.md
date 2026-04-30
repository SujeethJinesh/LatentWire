# Source-Private Rotation-Sign Packet Gate

Date: 2026-04-30

## Cycle Start

1. Current ICLR readiness and distance: stronger scoped positive-method paper,
   but not yet a broad ICLR latent-communication claim. Distance remains one
   robust non-hand-coded learned receiver or a final narrow systems/protocol
   framing with larger frozen evidence.
2. Current paper story: source-private packets can transmit useful private
   evidence to a target-side decoder with strict controls, and the systems value
   comes from tiny auditable packet traffic rather than text/KV/cache transfer.
3. Exact blocker: the promoted rows still rely on semantic anchors or scalar
   learned projections; a compression-native sign/JL packet must either pass
   controls or be pruned as only a weak comparator.
4. Current live branch: semantic-anchor receiver plus scalar Wyner-Ziv packet;
   candidate branch under test: controlled rotation-sign source packet.
5. Highest-priority gate: same-family/remapped strict packet gate with
   constrained-shuffle, answer-masked, permuted-bit, and random same-byte
   controls.
6. Scale-up rung: strict small/medium Mac CPU confirmation (`n=256`, three
   remapped codebooks, three byte budgets).

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_rotation_sign_packet_gate.py \
  --output-dir results/source_private_rotation_sign_packet_gate_20260430 \
  --remap-seeds 101 103 107 \
  --budgets 2 4 6 \
  --train-examples 512 \
  --eval-examples 256 \
  --feature-dim 512 \
  --train-seed 29 \
  --eval-seed 30
```

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_compression_baselines.py \
  tests/test_build_source_private_rotation_sign_packet_gate.py
```

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

No PID was present. The gate is CPU-only.

## Result

Artifact:
`results/source_private_rotation_sign_packet_gate_20260430/`.

Headline:

- pass gate: `False`
- rows: `9`
- pass rows: `0`
- remap seeds: `[101, 103, 107]`
- budgets: `[2, 4, 6]`
- max rotation-sign accuracy: `0.348`
- max rotation-control margin: `+0.066`
- scalar Wyner-Ziv range on the same rows: `0.449-0.570`
- target-only: `0.250`

Representative rows:

| Remap | Budget | Rotation-sign | Scalar WZ | Target | Best rotation control | Rotation-control | Rotation-scalar | p50 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 0.297 | 0.449 | 0.250 | 0.297 | 0.000 | -0.152 | 0.580 |
| 101 | 4 | 0.336 | 0.531 | 0.250 | 0.277 | 0.059 | -0.195 | 6.681 |
| 103 | 4 | 0.348 | 0.500 | 0.250 | 0.281 | 0.066 | -0.152 | 6.636 |
| 107 | 6 | 0.332 | 0.520 | 0.250 | 0.340 | -0.008 | -0.188 | 9.399 |

## Interpretation

The pure random-rotation sign sketch carries a weak source signal above
target-only, but not enough to be a method claim. It fails for two reasons:

1. Control separation is too small. The best source-destroying controls often
   match or nearly match the source row; the strongest margin is only `+0.066`,
   far below the `+0.150` requirement.
2. It is consistently worse than scalar Wyner-Ziv packets by `0.141-0.230`
   accuracy on the same byte budgets.

The 2-byte rows have acceptable CPU decode latency (`~0.58 ms`) but inadequate
accuracy/control margin. The 4/6-byte rows are still weak and have higher p50
latency (`6.6-9.5 ms`) in the current Python decoder, so they do not rescue the
systems story.

## Decision

Prune pure rotation-sign packets as a headline method. Keep them as a fair
compression-native ablation against scalar Wyner-Ziv and protected rotated
residual packets. This strengthens the paper by showing that the positive
packet evidence is not a generic random-sign sketch artifact.

Next exact gate: product-codebook packet or stronger frozen-activation receiver.
For the systems track, protected rotated residual and scalar WZ remain better
comparators than pure sign sketches.

## Literature / Uniqueness Update

New reference memo:
`references/513_rotation_sign_packet_refs_20260430.md`.

The useful framing is now sharper:

- QJL and TurboQuant motivate sign/JL and rotation/quantization baselines, but
  they optimize vector/KV compression, not source-private task communication
  under destructive controls.
- Product quantization is a stronger next compression-native packet branch than
  uncalibrated sign sketches.
- KIVI/SnapKV remain KV/cache systems baselines, not source-private packet
  methods.
- C2C is the closest direct communication competitor, but transfers or fuses
  cache state rather than tiny auditable packets.
- CUDA transfer-granularity guidance supports our raw-byte plus cache-line/DMA
  accounting caveat.

## Tests

`11 passed in 167.60s`.
