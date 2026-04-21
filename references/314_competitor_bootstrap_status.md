# Competitor Bootstrap Status

Date: 2026-04-21

This memo inventories the direct competitors and matched controls we should keep separate from LatentWire, records which repos are already vendored, and captures the current runnable status for the main Qwen pair.

## Inventory

### Direct competitors

| Method | Local repo status | Local path | Notes |
|---|---|---|---|
| C2C | Already cloned before this pass | `references/repos/C2C` | Direct cross-model communication peer; strongest semantic-transfer baseline. |
| KVComm / Q-KVComm | Already cloned before this pass | `references/repos/KVComm` | Direct communication peer; good selective-sharing comparator. |
| LatentMAS | Referenced in planning docs, not required for this bootstrap pass | `references/` memos only | Keep in the paper comparison bucket, but no local repo was required here. |

### Matched controls

| Method | Local repo status | Local path | Notes |
|---|---|---|---|
| `kvpress` | Already cloned before this pass | `references/repos/kvpress` | Same-model cache-compression control; wrapper already exists in `scripts/run_kvpress_eval.py`. |
| KVzip | Already cloned before this pass | `references/repos/KVzip` | Same-model query-agnostic compression control. |
| Quest | Already cloned before this pass | `references/repos/Quest` | Same-model query-aware sparsity control. |
| H2O | Cloned shallowly in this pass | `references/repos/H2O` | Cache pruning control; use as long-context/same-model control only. |
| StreamingLLM | Cloned shallowly in this pass | `references/repos/StreamingLLM` | Attention-sink control; same-model long-context control only. |
| SnapKV | Cloned shallowly in this pass | `references/repos/SnapKV` | Query-aware pruning control; best run through `kvpress` first for a cheap smoke. |
| PyramidKV | Cloned shallowly in this pass | `references/repos/PyramidKV` | Dynamic KV cache compression control; same-model only. |
| AdaKV | Cloned shallowly in this pass | `references/repos/AdaKV` | Asymmetric head-budget control; same-model only. |

## Clone commands used this pass

These were safe shallow clones into `references/repos`:

```bash
git clone --depth 1 https://github.com/FMInference/H2O references/repos/H2O
git clone --depth 1 https://github.com/mit-han-lab/streaming-llm references/repos/StreamingLLM
git clone --depth 1 https://github.com/FasterDecoding/SnapKV references/repos/SnapKV
git clone --depth 1 https://github.com/Zefan-Cai/PyramidKV references/repos/PyramidKV
git clone --depth 1 https://github.com/FFY0/AdaKV references/repos/AdaKV
```

## Runnable status for the main Qwen pair

### Pair

- Source: `Qwen/Qwen2.5-0.5B-Instruct`
- Target: `Qwen/Qwen3-0.6B`

### Status

Runnable locally and already executed for the asymmetric K/V route/value split.

Current best GSM30 artifact:

- `results/asym_kv_qwen_20260421/qwen_gsm30_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat_telemetry.jsonl`
- sidecar: `results/asym_kv_qwen_20260421/qwen_gsm30_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat_telemetry.jsonl.meta.json`

Observed GSM30 summary:

- target-alone accuracy: `0.0667`
- RotAlign/asym-KV accuracy: `0.0667`
- delta: `0.0000`
- paired flips: `1` method-only, `1` baseline-only, `1` both-correct, `27` both-wrong

Observed GSM5/GSM10 control evidence from the same run family:

- GSM5 attention/energy: `0.20 -> 0.40` (`+0.20`)
- GSM5 random/random: `0.20 -> 0.20`
- GSM10 attention/energy: `0.10 -> 0.10`

Additional runs executed after this memo was drafted:

| Method | Eval slice | Accuracy | Artifact |
|---|---|---:|---|
| C2C native | first 5 from `gsm8k_gate_search_30.jsonl` | 0.0000 | `results/competitor_bootstrap_20260421/c2c_qwen_gsm5_native_20260421.jsonl` |
| KVPress none | `gsm8k_5.jsonl` | 0.2000 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm5_none_20260421.jsonl` |
| KVPress expected_attention `0.5` | `gsm8k_5.jsonl` | 0.2000 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm5_expected_attention_c050_20260421.jsonl` |

Interpretation:

- C2C is now confirmed runnable on the exact Qwen pair through the published
  `nics-efc/C2C_Fuser` artifact, but the first native GSM5 smoke is negative.
- KVPress is confirmed runnable as a same-model compression control; on GSM5,
  expected-attention compression preserves accuracy relative to no press but is
  slower in the current MPS wrapper.
- These are smoke results only. Full paper comparison still needs GSM30/70 or
  a repo-native benchmark where the competitor was designed to run.

## Fairness caveats

- Do not compare direct communication peers (`C2C`, `KVComm`) to same-model compression controls as if they were the same task class.
- Do not mix GSM exact-match with LongBench, Needle, or passkey metrics in one table.
- Do not compare raw byte counts unless payload, metadata, and checkpoint overhead are normalized consistently.
- `Quest`, `H2O`, `StreamingLLM`, `SnapKV`, `PyramidKV`, and `AdaKV` should be treated as same-model long-context/compression controls, not cross-model communication methods.
- The local `C2C` and `KVComm` paths are useful for smoke and parity checks, but any published comparison should still verify their repo-native assumptions.
- `Quest` and the cache-compression repos are CUDA/FlashAttention oriented; MPS or eager-mode behavior may differ.

## Best next competitor runs to execute

1. **C2C on the exact Qwen pair and GSM30 slice**.
   - Why: direct semantic-communication peer, same source/target pair as LatentWire.
   - Expected artifact: `results/competitor_bootstrap_20260421/c2c_gsm30_native/...` plus the companion summary files.
   - Preferred command source: `references/306_competitor_benchmark_bootstrap.md` and `references/311_competitor_smoke_matrix.md`.

2. **KVPress GSM30 sweep with `none` and `expected_attention`**.
   - Why: cheapest same-model control on the same eval slice; establishes the floor for query-aware cache compression.
   - Expected artifacts: `results/competitor_bootstrap_20260421/kvpress_none_gsm30.jsonl` and `results/competitor_bootstrap_20260421/kvpress_expected_attention_gsm30.jsonl` with `.meta.json` sidecars.
   - Preferred command source: `scripts/run_kvpress_eval.py` and `references/311_competitor_smoke_matrix.md`.

3. **KVzip GSM30 repo-native smoke**.
   - Why: query-agnostic compression control that is complementary to KVPress and closer to a robustness story.
   - Expected artifact: repo-native GSM result directory under `references/repos/KVzip/results/` or a LatentWire mirror under `results/competitor_bootstrap_20260421/` if wrapped locally.
   - Preferred command source: `references/311_competitor_smoke_matrix.md`.

## Runnable command snippets for the Qwen pair and controls

### LatentWire Qwen pair (already runnable and already observed)

```bash
./venv_arm64/bin/python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --task-type generation \
  --device mps --dtype float32 --max-new-tokens 64 \
  --methods rotalign target \
  --gate-mode fixed --fixed-gate 0.10 \
  --fusion-rule static \
  --kv-transport both \
  --position-selection-metric attention \
  --position-selection-ratio 0.50 \
  --kv-route-selection-ratio 0.25 \
  --kv-value-selection-ratio 0.75 \
  --kv-route-selection-metric attention \
  --kv-value-selection-metric energy \
  --runtime-head-selection-metric attention_peak \
  --runtime-head-selection-ratio 1.0 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template --target-use-chat-template \
  --source-enable-thinking false --target-enable-thinking false \
  --prediction-output results/asym_kv_qwen_20260421/qwen_gsm30_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat_telemetry.jsonl
```

### C2C native smoke

```bash
python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --limit 5 \
  --prediction-output results/c2c_gsm70_20260420/qwen_gsm70_c2c.jsonl
```

### KVPress same-model smoke

```bash
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_expected_attention_gsm30.jsonl
```

## Bottom line

The repo inventory is now complete for the current competitor bucket: direct peers (`C2C`, `KVComm`) are locally available, the main matched controls (`kvpress`, `KVzip`, `Quest`) are present, and the long-context controls (`H2O`, `StreamingLLM`, `SnapKV`, `PyramidKV`, `AdaKV`) are now vendored shallowly for later smoke tests. The current Qwen pair run is runnable and already observed, but it is still neutral on GSM30, so the next competitor work should focus on fair external controls rather than claiming any method win yet.
