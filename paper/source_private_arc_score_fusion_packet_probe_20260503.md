# ARC Score-Fusion Packet Probe

Date: 2026-05-03

## Readiness

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets are useful on ARC/OpenBookQA
  under destructive controls, but the current positive mostly preserves source
  choice.
- Exact remaining blocker: a source-score or learned connector must beat
  explicit source-index/source-label transfer with paired uncertainty.

## Lay Explanation

This experiment asks whether the source model can send more than just "I pick
option B." It sends a tiny quantized score profile over all candidate answers,
then lets the receiver combine that profile with its own candidate scores.

If this worked, it would mean the source's full preference shape carries useful
extra information beyond the top answer. It did not work on this ARC validation
gate.

## Artifact

`results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation/`

Key files:

- `arc_challenge_score_fusion_packet_probe.json`
- `arc_challenge_score_fusion_packet_probe.md`
- `source_scores.json`
- `receiver_scores.json`

Command:

```bash
PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ./venv_arm64/bin/python scripts/build_source_private_commonsenseqa_score_fusion_packet_probe.py \
  --output-dir results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation \
  --benchmark-name ARC-Challenge \
  --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl \
  --source-score-cache results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation/source_scores.json \
  --receiver-score-cache results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation/receiver_scores.json \
  --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --receiver-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --source-lm-device auto_cpu \
  --receiver-lm-device auto_cpu \
  --lm-dtype float32 \
  --lm-max-length 256 \
  --lm-normalization mean \
  --local-files-only \
  --run-date 2026-05-03
```

## Result

ARC validation parity split:

| Row | Heldout Accuracy |
|---|---:|
| Qwen2.5 source-label/source-index | `0.389262` |
| Qwen3 receiver-label | `0.335570` |
| Best top-label pair rule | `0.389262` |
| Quantized source-score fusion | `0.389262` |
| Source top-2 oracle | `0.577181` |
| Source/receiver union top-2 oracle | `0.711409` |

The calibrated fusion weight was `0.65`, but heldout fusion only tied the
source-label baseline:

- fusion minus source-label: `0.000000`
- fusion minus best top-label pair: `0.000000`
- pass gate: `False`

The source and receiver score passes took `85.5s` and `108.6s` on CPU,
respectively. The first run exposed a Qwen/Transformers RoPE compatibility bug
in the shared LM scoring helper; the helper now keeps Qwen `rope_parameters`
intact while preserving the existing Phi-3 compatibility workaround.

## Decision

This weakens the raw score-vector packet branch on ARC. The source/receiver
top-2 oracle remains large, so the candidate set has headroom, but global
score fusion does not extract it.

Do not spend more Mac cycles on global source-score fusion or margin-only
source-score packets. The next method branch must change the communicated
object: conditional innovation features, sparse/common-feature packets, or a
true resonance sketch with candidate-roll and source-index controls.
