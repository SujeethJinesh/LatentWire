# Source-Private ARC-Challenge Fixed-Packet Gate, 2026-05-01

## Status

- code: `scripts/run_source_private_arc_challenge_fixed_packet_gate.py`
- validation artifact:
  `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/`
- test artifact:
  `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/`
- test: `tests/test_run_source_private_arc_challenge_fixed_packet_gate.py`
- references: `references/568_arc_challenge_fixed_packet_gate_refs_20260501.md`

## Method

The source scorer is local Qwen2.5-0.5B choice-text log-likelihood on ARC
question/candidate text. At evaluation time it sees only `question` and
`choices`. The sender converts the selected candidate residual into a fixed
`12B` random-projection packet. The receiver sees public candidate text and the
packet, then decodes by sparse candidate-residual similarity.

This is not a native C2C/KV-cache method. It is a public benchmark bridge for a
fixed byte-scale source-private packet protocol.

## Results

| Split | N | Target | Matched 12B | Shuffled | Same-byte text | Derangement | CI95 low vs target |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 299 | 0.244 | 0.385 | 0.274 | 0.348 | 0.197 | 0.067 |
| test | 1172 | 0.265 | 0.346 | 0.247 | 0.311 | 0.213 | 0.046 |

Both validation and test pass the gate. On test, the matched packet beats
target/zero-source by `+0.080`, beats the best destructive control by `+0.080`,
and beats same-byte structured text by `+0.035`.

## Interpretation

This partially clears the public benchmark blocker. The packet is not just
copying a label: label permutation preserves accuracy, while candidate
derangement collapses below target. Same-byte text remains a strong comparator
and must stay in the paper.

The main caveat is source-model framing. The source is a local Qwen
log-likelihood scorer over answer choices, not yet a native model-to-model
latent endpoint. For ICLR, this should be presented as public benchmark
evidence that the fixed-byte packet interface can carry useful source decisions
under controls, then followed by either a second benchmark or native endpoint
measurements.

## Next Gate

The immediate robustness gate has now been run:

- seed-stability memo:
  `paper/source_private_arc_challenge_seed_stability_20260501.md`
- validation/test artifacts:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_validation/`
  and
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_test/`
- outcome: `5/5` validation seeds and `5/5` test seeds pass.

Next, run one of:

- a second public benchmark with the same source scorer and fixed packet
  contract, or
- a true source/target endpoint variant where the packet is emitted from one
  model instance and decoded by a different target model, or
- native NVIDIA/vLLM systems rows for TTFT, TPOT, goodput, HBM, and bytes.
