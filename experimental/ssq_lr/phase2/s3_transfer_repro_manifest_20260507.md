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

Packet input hashes:

| Packet | `config.json` | `raw_rows.jsonl` | `summary.json` | `decision.md` |
|---|---|---|---|---|
| `ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507` | `5995f853cc4ddd405d0bde7bf60961d20a4d426dd075f94df32e249f819edaf3` | `eed422ba3c5bb95da8c089a73d888b07c43594e08d6e2304dd01f10289a8a37d` | `d14700439bd08881217b9416e045a19b8f46f8815f1fc26857ffb3edbc01fc46` | `95fe42f3782f995aed6acf95b80364cf0331e1a5b0e2c99976bb6f5513113534` |
| `ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507` | `3abebba80a837dd0b8b60305cae8abb242b385730f8a7d729a9554a13a2968e8` | `1f5cab17529893af777c5ca0e53ee2c3690213e9377e388dd1b181deed553d25` | `d86251a00d9e351e449b256293e9e3027e71b99bb0c76cdc972787bf26ed7071` | `b791eef85c9354c1e01d66fce18c015fc38a8a2c8c41c335d2c027338a226696` |
| `ssq_lr_s3_transfer_granite_350m_12p_layer30_20260507` | `a53f614fb180c5efe19d9fd8a9ba64a84e9148a4ae0d1852763baee059c1cd7f` | `b7b3b7e0c6898c71c900af75ff26fb34367a5e7b4281afc0aa54f6e1e7de26f9` | `fed13fc833ea2d476e1f43201f5d32dbb891cf53997a797f62b24fbee54341dd` | `95fe42f3782f995aed6acf95b80364cf0331e1a5b0e2c99976bb6f5513113534` |
| `ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507` | `c794c6c6ebfc546bfe83df407f019edd3b18328b532032b3695ee004f522418b` | `c44b50b1d1534cd7f7e5e36f8289b8c58c627d49d3e22cace0eca4897f4a0b9a` | `5d58980572c48693045de5a05f7a191133a99ed34fa273ad6ee25aedbcd2d3a8` | `b791eef85c9354c1e01d66fce18c015fc38a8a2c8c41c335d2c027338a226696` |
| `ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507` | `27decd3fd7b52dc4bd46d31d8b9d26cda7af979a980cae78a064b32698492858` | `135f6704b90323ad5a6fd969cb2a6bba844a7f6c5cba96d6b01aeae43efbf5aa` | `2ec0e397d49f6494f1871ed8fdbf23ecbce0812d3866ff0f41b4961433c05b15` | `f7387e3ae87267ddd79aa82b82423d4594ea69b38080ec744545dea816cb4968` |
| `ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507` | `692491be29372654c9435c56742344457abdb760019ee886ade1630bd7846fd4` | `cae683ff826dbcec0681ba6f46fecf2457ff5cdd25a645af8386ee5347d63d17` | `2e235bc731498c0197cf29ea73eeb2efc4c36723631ef37eeca21f912690a7e8` | `f7387e3ae87267ddd79aa82b82423d4594ea69b38080ec744545dea816cb4968` |

Revalidate the S2 packets with:

```bash
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507 \
  --gate ssq_lr_s2
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507 \
  --gate ssq_lr_s2
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer30_20260507 \
  --gate ssq_lr_s2
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507 \
  --gate ssq_lr_s2
```

Revalidate the no-retuning transfer packets with:

```bash
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507 \
  --gate ssq_lr_s3
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507 \
  --gate ssq_lr_s3
```

## Interpretation

This is a stop artifact, not a GPU handoff. The failed `0,30` transfer shows
that the source-selected recipe is not stable across two Granite-family hybrid
models. The layer-0 diagnostics are explicitly post-hoc and cannot promote:
they show that a lower-bit recipe can pass one model while failing the other.
Any revival needs a new preregistration and fresh prompt surface before more
rows are inspected.
