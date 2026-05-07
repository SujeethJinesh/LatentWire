# SSQ-LR S3 Transfer Reproducibility Manifest

- date: 2026-05-07
- status: current recipe stopped for GPU handoff
- decision: the frozen `0,30` mixed INT3/MXFP4 recipe fails no-retuning
  transfer to Granite 350M; layer-0 rescue diagnostics also fail two-model S3
  because Granite Tiny and Granite 350M prefer different low-bit recipes.

## Frozen Inputs

| Item | Value |
|---|---|
| Prompt file | `experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl` |
| Prompt SHA-256 | `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0` |
| Preregistration | `experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md` |
| Preregistration SHA-256 | `e82292ec47200b5925bfbd264deb3c691891809c55302f1a612e8bed9b748d86` |
| Seed for S2 replays | `3718` |
| Seed for S3 local transfer packets | `20260507` |
| Max input / prefix tokens | `32 / 12` |
| Block size | `256` |

## Model Revisions

| Model | Revision | Role |
|---|---|---|
| `ibm-granite/granite-4.0-h-tiny` | `791e0d3d28c86e106c9b6e0b4cecdee0375b6124` | source/local reference |
| `ibm-granite/granite-4.0-h-350m` | `3b17b717b8f2f5d305b0a92c1491e239aeda19c8` | second complete local transfer model |

## Reproduction Commands

Run from the repository root with the repo-local virtual environment and local
HF cache:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.ssq_lr_s2_state_replay_scout \
  --model-id ibm-granite/granite-4.0-h-350m \
  --prompt-path experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl \
  --prompt-limit 12 \
  --max-input-tokens 32 \
  --prefix-tokens 12 \
  --primary-layers 0,30 \
  --block-size 256 \
  --seed 3718 \
  --output-dir experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507

HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.ssq_lr_s2_state_replay_scout \
  --model-id ibm-granite/granite-4.0-h-350m \
  --prompt-path experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl \
  --prompt-limit 12 \
  --max-input-tokens 32 \
  --prefix-tokens 12 \
  --primary-layers 0 \
  --block-size 256 \
  --seed 3718 \
  --output-dir experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507

HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.ssq_lr_s2_state_replay_scout \
  --model-id ibm-granite/granite-4.0-h-350m \
  --prompt-path experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl \
  --prompt-limit 12 \
  --max-input-tokens 32 \
  --prefix-tokens 12 \
  --primary-layers 30 \
  --block-size 256 \
  --seed 3718 \
  --output-dir experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer30_20260507

HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.ssq_lr_s2_state_replay_scout \
  --model-id ibm-granite/granite-4.0-h-tiny \
  --prompt-path experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl \
  --prompt-limit 12 \
  --max-input-tokens 32 \
  --prefix-tokens 12 \
  --primary-layers 0 \
  --block-size 256 \
  --seed 3718 \
  --output-dir experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507
```

Build the two local S3 transfer packets from those S2 rows:

```bash
./venv_arm64/bin/python -m experimental.shared.ssq_lr_s3_local_transfer_prefilter \
  --source-s2-dir experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507 \
  --transfer-s2-dirs experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507 \
  --recipe-id mixed_int3_mxfp4_low_error_25pct \
  --primary-layers 0 \
  --block-size 256 \
  --seed 20260507 \
  --output-dir experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507

./venv_arm64/bin/python -m experimental.shared.ssq_lr_s3_local_transfer_prefilter \
  --source-s2-dir experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507 \
  --transfer-s2-dirs experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507 \
  --recipe-id int3_primary_state_block_scaled \
  --primary-layers 0 \
  --block-size 256 \
  --seed 20260507 \
  --output-dir experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507
```

Validate:

```bash
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507 \
  --gate ssq_lr_s2

./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507 \
  --gate ssq_lr_s3

./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507 \
  --gate ssq_lr_s3
```

## Result Packets

| Packet | Decision | Key readout |
|---|---|---|
| `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507/` | `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | fallback `int8_primary_state_block64`, `1.984x`, accuracy CI high `0.0`, NLL CI high `0.008906` |
| `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507/` | `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | `int3_primary_state_block_scaled`, `5.224x`, accuracy CI high `0.0`, NLL CI high `0.015687` |
| `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer30_20260507/` | `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | fallback `int8_primary_state_block64`, `1.984x`, accuracy CI high `0.0`, NLL CI high `0.008893` |
| `experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507/` | `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | `mixed_int3_mxfp4_low_error_25pct`, `4.192x`, accuracy CI high `0.0`, NLL CI high `0.041124` |
| `experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507/` | `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER` | frozen recipe hash `sha256:e917c6211dd0a14806f76c373689207c54f02adc0e96c579c307dafdae1fd69d`, max accuracy delta `0.066667` |
| `experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507/` | `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER` | frozen recipe hash `sha256:a460bb6aecb7fd8ddde17c52d2c368d239e77a3e711a7fdc08329846c5a59154`, max accuracy delta `0.052632` |

The local `frozen_recipe.json` file hashes are:

- mixed25 packet: `2b2a53d94198b763df622570425b52463e1ecd568f5b1f85e1d3caa77a7fcd87`
- INT3 packet: `2e87175e79f8ddf7530c7042624d3f861c886539e259cd02968ca534456f98bc`

## Interpretation

This is a stop artifact, not a GPU handoff. The failed `0,30` transfer shows
that the source-selected recipe is not stable across two Granite-family hybrid
models. The layer-0 diagnostics are explicitly post-hoc and cannot promote:
they show that a lower-bit recipe can pass one model while failing the other.
Any revival needs a new preregistration and fresh prompt surface before more
rows are inspected.
