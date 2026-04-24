# SVAMP32 Target-Memory Query Codec Screen

Date: 2026-04-23

## Paper Status

- readiness: not ICLR-ready
- estimated distance: medium-high
- current story: target self-repair is a real decoder-side floor, but a
  publishable positive method still needs source-conditioned latent transfer
  that recovers clean C2C-only residual IDs under exact-ID parity
- blocking gap: no live branch has reached `>=2/6` clean residual recoveries on
  the frozen SVAMP32 exact-ID slice while preserving near `14/32` target
  self-repair
- turn gate: test whether appending target-prior K/V side information to the
  query-innovation memory clears the matched same-pair gate

## Decision Inputs

Latest local evidence before this run:

- target-alone exact-ID baseline: `8/32`
- C2C teacher: `16/32`
- target self-repair: `14/32`
- prior protected K-only query codec: `8/32`, clean residual `0/6`
- prior sparse source-attention V screen: best `10/32`, clean residual `1/6`
- promotion criterion: `>=2/6` clean residual IDs before source-destroying
  controls

Subagent recommendations converged on a receiver-conditioned bottleneck:

- literature/internet: target-side memory is the minimal Q-Former/Perceiver-like
  side-information test before adding broader benchmark scope
- ablation design: after a matched pass, only run source/target/delta memory
  masks if the combined method clears `>=2/6`
- repo audit: implement as a default-off flag to avoid altering existing
  checkpoints or previous rows

References checked by subagents:

- C2C: https://arxiv.org/abs/2510.03215
- KVComm: https://arxiv.org/abs/2510.03346
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
- Perceiver IO: https://arxiv.org/abs/2107.14795
- AWQ: https://arxiv.org/abs/2306.00978

## Code Changes

- `TranslatorConfig.innovation_conditional_target_memory`
  - default-off, valid only for
    `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- `latent_bridge/calibrate.py`
  - added `--innovation-conditional-target-memory`
  - validates the flag against the query-innovation correction
- `latent_bridge/translator.py`
  - passes target-prior K/V tensors into the query module fit
  - appends target-prior K/V rows to module memory at fit time and runtime
  - passes conditional target memory through calibration-time fit diagnostics
- `latent_bridge/evaluate.py`
  - forwards target prompt K/V cache into `translate_layer` when the flag is
    enabled
- tests added for CLI parsing, config validation, fit-time forwarding, and
  evaluation-time target-cache forwarding

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_translator_core.py \
  tests/test_calibrate_and_ablation.py \
  tests/test_evaluate_helpers.py -q
```

Result: `319 passed`

## Experiment

Calibration:

```bash
./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_targetmem_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_targetmem_konly_query_codec_r16_bank16_seed1.pt \
  --bits 4 \
  --alignment grouped_subspace_transport \
  --quantization-correction bridge_ridge_qk_dynalign_query_innovation_resampler_replace \
  --quantization-correction-rank 16 \
  --bridge-bank-size 16 \
  --innovation-target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --innovation-positive-weight 16 \
  --innovation-default-weight 1.0 \
  --innovation-target-self-preserve-weight 16 \
  --innovation-value-loss-weight 0.0 \
  --innovation-conditional-target-memory \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Matched gate:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp32_targetmem_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_targetmem_konly_query_codec_r16_bank16_seed1.pt \
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
  --gate-values 0.10 0.15 0.175 0.20 \
  --methods rotalign \
  --prediction-output .debug/svamp32_targetmem_query_codec_20260423/preds/targetmem_konly_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Analyzer:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_gate_sweep_clean_targets.py \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --candidate-jsonl .debug/svamp32_targetmem_query_codec_20260423/preds/targetmem_konly_attention_gate_sweep.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_targetmem_query_codec_20260423/targetmem_konly_attention_clean_targets.json \
  --output-md results/svamp32_targetmem_query_codec_20260423/targetmem_konly_attention_clean_targets.md \
  --expected-n 32 \
  --min-clean-residual-recovered 2 \
  --target-self-correct 14
```

## Evidence

- calibration prompts: `32`
- dynamic token-mixture samples: `1411`
- calibration matched clean residual prompts: `6`
- calibration matched target-self-preserve prompts: `3`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- matched readout status: `no_matched_gate_candidate_for_controls`
- exact ordered ID parity: true for every row
- numeric extraction coverage: `32/32` for every row
- best row: `rotalign_kv_gate_0.20`, `9/32`
- best clean residual recovered: `0/6`
- best teacher-only recovered: `1`, ID `575d7e83d84c1e67`
- best target losses: `1`, ID `c042f0a2949ff8e6`
- bytes: `397,923.75`
- latency at best row: `8.662850` seconds/example

Rows:

| Method | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses |
|---|---:|---:|---:|---:|---:|
| `rotalign_kv_gate_0.20` | 9/32 | 0 | 1 | -5 | 1 |
| `rotalign_kv_gate_0.17` | 8/32 | 0 | 1 | -6 | 1 |
| `rotalign_kv_gate_0.15` | 8/32 | 0 | 1 | -6 | 1 |
| `rotalign_kv_gate_0.10` | 8/32 | 0 | 1 | -6 | 1 |

Controls were not run because no matched row reached the `>=2/6` clean
residual promotion threshold.

## Artifacts

- checkpoint:
  - `.debug/svamp32_targetmem_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_targetmem_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `071fc28113d8a8b4829feb8fb5391cd4d158e2fcb623bd9ea773cf3142bf2d67`
- matched sweep:
  - `.debug/svamp32_targetmem_query_codec_20260423/preds/targetmem_konly_attention_gate_sweep.jsonl`
  - sha256: `e16b0526ada85956a4842ba7abd9f783a50fae6c2985a57d4fddf01a3153547b`
  - `.debug/svamp32_targetmem_query_codec_20260423/preds/targetmem_konly_attention_gate_sweep.jsonl.meta.json`
  - sha256: `e570934f84fcb9c6d773153a5c4dd4a11217d43ca79c82f7ac6235211e0d59cc`
- readout:
  - `results/svamp32_targetmem_query_codec_20260423/targetmem_konly_attention_clean_targets.json`
  - sha256: `3baed26ad7dbc7a60c73cd49342759901539def40db1bd37a4681df617bc2f4a`
  - `results/svamp32_targetmem_query_codec_20260423/targetmem_konly_attention_clean_targets.md`
  - sha256: `2fb014905d46d342e650a21e6b3e5f1fc9a75218dbad0cf5e9aa06b1828d3da0`

## Hypothesis Update

- killed for now: bounded target-prior K-only memory as a same-pair positive
  method candidate
- weakened: adding target side-information alone is sufficient without an
  explicit source-minus-target residual/delta channel
- still alive: full target-prior delta-memory branch
  `[source K/V, target-prior K/V, source-minus-target delta K/V, learned slots]`
- saturated: scalar gate sweeps, K-only value-loss suppression, and target
  no-op preservation on the current query-innovation architecture
- blocked: seed repeats, larger frozen slice, and cross-family falsification
  remain blocked until a same-pair row reaches `>=2/6` clean residual IDs

## Next Gate

Implement the full target-prior delta-memory screen with explicit runtime
memory masks:

- combined: `source + target + delta + slots`
- source-only
- target-only
- delta-only

Promotion remains `>=2/6` clean residual recoveries under exact-ID parity, with
target-only failing to recover the same IDs.
