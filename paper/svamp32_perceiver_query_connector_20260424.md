# SVAMP32 Perceiver-Query Connector Smoke - 2026-04-24

## Status

- status: `killed_matched_candidate_before_controls`
- readiness impact: negative for the current live row; useful as an architecture screen
- gate: `>=14/32` target-self-repair preservation, `>=2/6` clean residual IDs, exact ID parity, no coverage failure
- outcome: best matched gate reached `10/32`, recovered `0/6` clean residual IDs, and lost `4` versus target self-repair

## Motivation

The previous delta-memory plus source-control connector recovered no clean
SVAMP32 residual IDs. Literature and side-agent synthesis converged on a
smaller Q-Former/Perceiver-style bottleneck rather than more scalar tuning:
learned receiver-conditioned queries should first attend to source/target
memory, then expose a compact context to the target receiver query. This is
near the connector pattern used by BLIP-2, Flamingo, and Perceiver IO, but
tested here as cache transfer rather than multimodal transfer.

Local reference anchors:

- `references/450_lateral_connector_repair_refs.md`
- `references/356_multimodal_diffusion_latent_interface_refs.md`
- `references/384_competitor_contract_latest_refs.md`

## Implementation

Added default-off query-innovation connector topology:

- `TranslatorConfig.innovation_connector_mode`
- CLI flag `--innovation-connector-mode {single_query,perceiver_queries}`
- `single_query`: legacy behavior, receiver query attends directly over source/target rows plus learned slots
- `perceiver_queries`: bridge-bank rows are learned connector queries, cross-attend over live source/target memory, then are read by the receiver query
- validation rejects non-query-innovation use and requires `bridge_bank_size > 0`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_translator_core.py tests/test_calibrate_and_ablation.py -q
```

Result: `234 passed in 3.90s`.

## Commands

Calibration:

```bash
./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt \
  --bits 4 \
  --alignment grouped_subspace_transport \
  --quantization-correction bridge_ridge_qk_dynalign_query_innovation_resampler_replace \
  --quantization-correction-rank 16 \
  --bridge-bank-size 8 \
  --innovation-connector-mode perceiver_queries \
  --innovation-target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --innovation-positive-weight 16 \
  --innovation-default-weight 1.0 \
  --innovation-target-self-preserve-weight 16 \
  --innovation-value-loss-weight 0.0 \
  --innovation-conditional-delta-memory \
  --innovation-control-weight 0.30 \
  --innovation-control-mode zero_and_shuffle \
  --innovation-contrastive-margin 0.010 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Matched evaluation:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode sweep \
  --gate-values 0.10 0.125 0.15 0.175 0.20 \
  --methods rotalign \
  --innovation-memory-control combined \
  --prediction-output .debug/svamp32_perceiver_query_connector_20260424/preds/perceiver_queries_combined_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Clean-target analyzer:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_gate_sweep_clean_targets.py \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --candidate-jsonl .debug/svamp32_perceiver_query_connector_20260424/preds/perceiver_queries_combined_attention_gate_sweep.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_perceiver_query_connector_20260424/perceiver_queries_w030_m010_attention_clean_targets.json \
  --output-md results/svamp32_perceiver_query_connector_20260424/perceiver_queries_w030_m010_attention_clean_targets.md \
  --expected-n 32 \
  --min-clean-residual-recovered 2 \
  --target-self-correct 14
```

## Evidence

Calibration:

- prompts: `32`
- dynamic token-mixture samples: `1411`
- clean residual prompts matched: `6`
- target-self-preserve prompts matched: `3`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- connector queries: `8`
- source-control fit: `zero_and_shuffle`, weight `0.30`, margin `0.010`

Matched sweep:

| Method | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses | Status |
|---|---:|---:|---:|---:|---:|---|
| `rotalign_kv_gate_0.15` | 10/32 | 0/6 | 2 | -4 | 1 | below gate |
| `rotalign_kv_gate_0.20` | 9/32 | 0/6 | 2 | -5 | 3 | below gate |
| `rotalign_kv_gate_0.12` | 8/32 | 0/6 | 0 | -6 | 1 | below gate |
| `rotalign_kv_gate_0.10` | 8/32 | 0/6 | 1 | -6 | 2 | below gate |
| `rotalign_kv_gate_0.17` | 7/32 | 0/6 | 1 | -7 | 3 | below gate |

Accounting:

- average bits: `3,183,390`
- average bytes: `397,923.75`
- best latency among tested gates: `7.262815` seconds/example at gate `0.20`
- exact ordered ID parity: true
- numeric extraction coverage: `32/32`

## Artifacts

- `results/svamp32_perceiver_query_connector_20260424/perceiver_queries_w030_m010_attention_clean_targets.json`
- sha256: `fbccf197d063dfd133f584a0397322f2e35f6e6de710b8ec92cf5dc594335e3c`
- `results/svamp32_perceiver_query_connector_20260424/perceiver_queries_w030_m010_attention_clean_targets.md`
- sha256: `fd80e0f402bfa2e49166262d29b1950b3d2abb550bf28fc2c7b63d23e8b062e9`
- `.debug/svamp32_perceiver_query_connector_20260424/preds/perceiver_queries_combined_attention_gate_sweep.jsonl`
- sha256: `a45d014b712f1e315210335a899cd12f18ada8d24e11c40addd53609350927e0`
- `.debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt`
- sha256: `ad64ffd29b5e31f029e9a4d14d75ed6bcb64906d44dc7746532f1606146712f0`

## Decision

No source-zero, source-shuffle, or target-only memory controls were run because
the matched row failed the gate. This setting does not justify compute on
source-necessity controls.

## Hypothesis Update

- killed: this specific 8-query, K-only, source-control-trained Perceiver-query checkpoint as a same-pair positive row
- weakened: simply replacing slot-memory attention with learned connector queries is sufficient
- weakened: the current K-only innovation loss can discover clean residual SVAMP IDs under strong source controls
- still alive: receiver-conditioned connector architecture if trained with stronger teacher/residual signal or a less lossy integration point
- promoted: before another expensive run, add an oracle/teacher-forced feasibility diagnostic that asks whether clean residual IDs are linearly separable in connector contexts at all

## Next Exact Gate

Do not widen to larger slices, seed repeats, or cross-family yet. The next gate
is a cheaper feasibility diagnostic:

- train/evaluate the same connector on the 6 clean residual IDs with explicit C2C residual distillation or teacher-forced answer-token objective
- compare matched, zero-source, shuffled-source, and target-only memory in one small diagnostic
- promote only if at least `2/6` clean IDs are recovered in matched and collapse under source controls while target-self-repair IDs are preserved
