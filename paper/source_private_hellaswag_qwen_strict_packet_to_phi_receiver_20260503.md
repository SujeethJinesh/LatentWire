# HellaSwag Qwen Strict Packet To Phi Receiver-Family Scout

Date: 2026-05-03

## Readiness

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: the strict Qwen-family HellaSwag hidden-innovation packet is a
  real fixed-byte source-private signal, but Phi-side receiver fusion does not
  yet turn that signal into cross-family latent reasoning.
- Exact remaining blocker: a receiver/common-basis method must beat packet-only
  under strict controls, not merely show that a source packet beats Phi target
  scoring.

## Lay Explanation

This experiment asks whether the strongest current Qwen hidden-innovation
packet can help Phi-3 on the same HellaSwag rows. It can help if Phi simply
trusts the packet, but the learned receiver still does not know how to combine
Phi's own answer with the packet better than packet-only.

## Artifacts

`results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/`

`results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/`

`results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_multislice_20260503_validation1024_2048/`

## Commands

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py \
  --source results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_rank_score_channel_qwen05_train512_validation1024_2048/bagged_gate/predictions.jsonl \
  --output results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/qwen_strict_packet_predictions_1024_1536.jsonl \
  --start-index 1 \
  --count 512 \
  --run-date 2026-05-03

PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_hellaswag_receiver_family_packet_gate.py \
  --output-dir results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/receiver_gate \
  --source-packet-jsonl results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/qwen_strict_packet_predictions_1024_1536.jsonl \
  --target-global-artifact results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536/target_global_artifact.json \
  --source-packet-artifact results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_rank_score_channel_qwen05_train512_validation1024_2048/hellaswag_hidden_innovation_eval_slice_stress.json \
  --source-family Qwen2.5 \
  --target-family Phi-3-mini \
  --control-fields wrong_example_hidden_prediction,candidate_roll_hidden_prediction,zero_hidden_prediction,source_label_prediction,source_rank_only_bagged_prediction,score_only_bagged_prediction,score_channel_roll_hidden_prediction \
  --train-prefix-rows 128 \
  --bootstrap-samples 500 \
  --run-date 2026-05-03
```

The second slice uses `--start-index 513`, output
`qwen_strict_packet_predictions_1536_2048.jsonl`, and the cached Phi
`validation1536_2048/target_global_artifact.json`.

Aggregate:

```bash
./venv_arm64/bin/python \
  scripts/build_source_private_hellaswag_receiver_family_multislice_summary.py \
  --output-dir results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_multislice_20260503_validation1024_2048 \
  --run-date 2026-05-03
```

## Result

Two-slice aggregate over HellaSwag validation `1024:2048`:

| Row | Weighted Eval Accuracy | Delta |
|---|---:|---:|
| Phi-3 target-only | `0.263021` | n/a |
| Qwen strict packet-only | `0.455729` | `+0.192708` vs target |
| Candidate ridge receiver | `0.401042` | `-0.054688` vs packet |
| Target-or-packet oracle | `0.593750` | `+0.138021` vs packet |

Slice readout:

| Slice | Packet | Receiver | Receiver - Packet | Target Transfer | Receiver Improvement |
|---|---:|---:|---:|---:|---:|
| `1024:1536` | `0.473958` | `0.471354` | `-0.002604` | `True` | `False` |
| `1536:2048` | `0.437500` | `0.330729` | `-0.106771` | `False` | `False` |

Controls include wrong-example hidden, candidate-roll hidden, zero-hidden,
source-label copy, source-rank/index-only, score-only, and score-channel-roll
hidden packets.

## Decision

This falsifies the current generic receiver as an ICLR receiver-family method.
The Qwen strict packet has useful source signal for Phi, but the receiver loses
substantial packet utility and fails to beat packet-only on both adjacent
slices. Do not promote this as cross-family latent reasoning.

The next exact gate should change the receiver feature space rather than rerun
the same ridge/confidence receiver. The highest-value Mac-local branch is a
score-simplex Fourier/SVD innovation receiver: use the shared four-candidate
multiple-choice basis, row-centered Tiny/Phi score contrasts, and strict
basis-permutation/sign-flip/source-score controls to see whether a common
candidate-score basis can capture part of the target-or-packet oracle headroom.
