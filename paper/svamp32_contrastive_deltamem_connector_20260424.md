# SVAMP32 Contrastive Delta-Memory Connector Smoke

- Date: 2026-04-24
- Status: `no_matched_gate_candidate_for_controls`
- Gate: same-family Qwen2.5-0.5B-Instruct -> Qwen3-0.6B, SVAMP32 exact-ID matched slice
- Branch: source-control-contrastive query-innovation connector with source K/V, target-prior K/V, source-minus-target delta K/V, and learned slots

## Paper Status

Not ICLR-ready. The project still lacks a source-necessary positive method. The
live story remains: target self-repair is a strong decoder-side-information
floor, C2C exposes cache-level source headroom, and direct source/text relays
do not recover the clean residual IDs.

Blocking gap: no method reaches `>=2/6` clean source-necessary SVAMP32 residual
IDs while preserving the `14/32` target-self-repair row.

## Top Moves Considered

1. Combined delta-memory plus zero/shuffle source-control contrast.
   - Why it matters: directly tests whether the existing learned-query
     infrastructure can encode source-specific innovation rather than target
     cache repair.
   - Why it might fail: stronger controls can suppress all residual innovation,
     and the current query-innovation module may be too shallow to mimic C2C.
   - Evidence gained: matched clean residual count before spending controls.
   - Cost: one calibration plus one matched sweep.
   - Helps: same-pair, robustness, interpretability.
2. Implement a deeper Q-Former/Perceiver-style connector.
   - Why it matters: closest to the publishable connector story from C2C,
     BLIP-2, Flamingo, and Perceiver IO.
   - Why it might fail: higher implementation cost before the cheap connector
     family is exhausted.
   - Evidence gained: whether a receiver-conditioned bottleneck can recover
     multiple clean residual IDs.
   - Cost: high.
   - Helps: same-pair, efficiency, interpretability, cross-family if it works.
3. Widen evaluation immediately.
   - Why it matters: the 32-example slice is only a smoke gate.
   - Why it might fail: widening a non-positive row wastes compute and weakens
     the method story.
   - Evidence gained: larger-slice variance, but not a method advance.
   - Cost: medium/high.
   - Helps: reproducibility only after a live row exists.

Chosen move: run the combined delta-memory/source-control smoke first, because
it is the smallest decisive test of the current blocker.

## Result

The branch failed the matched gate and no runtime controls were run.

| Row | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses | Status |
|---|---:|---:|---:|---:|---:|---|
| `rotalign_kv_gate_0.20` | 8/32 | 0/6 | 1 | -6 | 1 | `matched_candidate_below_clean_gate` |
| `rotalign_kv_gate_0.17` | 8/32 | 0/6 | 1 | -6 | 1 | `matched_candidate_below_clean_gate` |
| `rotalign_kv_gate_0.15` | 8/32 | 0/6 | 1 | -6 | 1 | `matched_candidate_below_clean_gate` |
| `rotalign_kv_gate_0.12` | 8/32 | 0/6 | 1 | -6 | 1 | `matched_candidate_below_clean_gate` |

The source/oracle diagnostic with this candidate gives:

- `contrastive_deltamem_w050_gate020`: `8/32`
- clean residual correct: `0/6`
- wins vs target: `1`
- losses vs target: `1`
- oracle `target_self_repair + contrastive_deltamem_w050_gate020`: `15/32`
- clean residual added to `target_self_repair`: `0`

Interpretation: adding stronger source-control contrast to explicit target-prior
delta memory suppresses the matched row back to target-alone level and exposes
no clean residual signal. Since matched fails `>=2/6`, zero-source,
shuffled-source, and memory-mask controls are not justified for this checkpoint.

## Commands

Calibration:

```bash
./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_contrastive_deltamem_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_contrastive_deltamem_query_connector_w050_m001_r16_b16_seed1.pt \
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
  --innovation-conditional-delta-memory \
  --innovation-control-weight 0.50 \
  --innovation-control-mode zero_and_shuffle \
  --innovation-contrastive-margin 0.001 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Matched sweep:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp32_contrastive_deltamem_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_contrastive_deltamem_query_connector_w050_m001_r16_b16_seed1.pt \
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
  --gate-values 0.125 0.15 0.175 0.20 \
  --methods rotalign \
  --innovation-memory-control combined \
  --prediction-output .debug/svamp32_contrastive_deltamem_connector_20260424/preds/combined_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Clean-target readout:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_gate_sweep_clean_targets.py \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --candidate-jsonl .debug/svamp32_contrastive_deltamem_connector_20260424/preds/combined_attention_gate_sweep.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_contrastive_deltamem_connector_20260424/combined_w050_m001_attention_clean_targets.json \
  --output-md results/svamp32_contrastive_deltamem_connector_20260424/combined_w050_m001_attention_clean_targets.md \
  --expected-n 32 \
  --min-clean-residual-recovered 2 \
  --target-self-correct 14
```

Source/oracle cross-check:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_oracle_bound.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --source source_alone=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --source text_to_text=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --baseline target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --baseline selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate contrastive_deltamem_w050_gate020=path=.debug/svamp32_contrastive_deltamem_connector_20260424/preds/combined_attention_gate_sweep.jsonl,method=rotalign_kv_gate_0.20 \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --expected-n 32 \
  --date 2026-04-24 \
  --output-json results/svamp32_contrastive_deltamem_connector_20260424/source_oracle_bound_with_contrastive_deltamem.json \
  --output-md results/svamp32_contrastive_deltamem_connector_20260424/source_oracle_bound_with_contrastive_deltamem.md
```

## Artifacts

- checkpoint:
  - `.debug/svamp32_contrastive_deltamem_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_contrastive_deltamem_query_connector_w050_m001_r16_b16_seed1.pt`
  - sha256: `6b22d1b62da5455134c4a7935252426617d6d556ad0317a8f98217409c607bb8`
- calibration log:
  - `.debug/svamp32_contrastive_deltamem_connector_20260424/logs/calibrate_w050_m001_seed1.log`
  - sha256: `4f984ce9d9774c85c1e6a61fa08e2535268eb1507f7b411fd2208c3fefe09347`
- matched sweep:
  - `.debug/svamp32_contrastive_deltamem_connector_20260424/preds/combined_attention_gate_sweep.jsonl`
  - sha256: `d9b4735d64503f485a05fa300c78352f8c17b9ea018971dc9394cbbb71162fc4`
- matched eval log:
  - `.debug/svamp32_contrastive_deltamem_connector_20260424/logs/evaluate_combined_attention_gate_sweep.log`
  - sha256: `bc14b55156cb46b71bd3238cc26e2263dd1784c21d49ac3b79c7c2c5efaed66c`
- clean-target readout:
  - `results/svamp32_contrastive_deltamem_connector_20260424/combined_w050_m001_attention_clean_targets.json`
  - sha256: `34d7086b75d0e034d5793b571fba459bd3c218849b9b3df0e5cd8586e7658d56`
  - `results/svamp32_contrastive_deltamem_connector_20260424/combined_w050_m001_attention_clean_targets.md`
  - sha256: `7127f373e67e9187745e8cdfc1bccc586394d66433d1923ec41fc4dda24293aa`
- source/oracle cross-check:
  - `results/svamp32_contrastive_deltamem_connector_20260424/source_oracle_bound_with_contrastive_deltamem.json`
  - sha256: `cbf5b293238ded7e178afb893acc730c7a303f4c6ffea13babc8dedc135178f1`
  - `results/svamp32_contrastive_deltamem_connector_20260424/source_oracle_bound_with_contrastive_deltamem.md`
  - sha256: `214155c75285cf0c4131bae5812f9f0554a5fa411ddf473c2c8b5b3ed911ed6b`

## Subagent Synthesis

The literature, ablation, repo-audit, and internet-creative subagents agreed
that the right branch is source-control-contrastive learned-query transport, but
the cheap existing implementation is likely too weak. Primary-source anchors:

- C2C cache fusion: https://arxiv.org/abs/2510.03215
- KVComm selective KV sharing: https://arxiv.org/abs/2510.03346
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
- Perceiver IO learned query bottleneck: https://arxiv.org/abs/2107.14795
- InfoNCE / CPC contrastive objective: https://arxiv.org/abs/1807.03748
- AWQ salient-channel protection: https://arxiv.org/abs/2306.00978
- KIVI asymmetric KV precision: https://arxiv.org/abs/2402.02750

## Hypothesis Update

- killed for now: cheap scalar tuning of combined delta-memory plus
  zero/shuffle source-control contrast under the current query-innovation module
- weakened: explicit source-minus-target delta rows can expose clean residual
  information when regularized for source specificity
- weakened: source-control contrast alone is enough without a stronger
  receiver-conditioned connector architecture
- still alive: source-control contrast and target-prior/delta controls as
  diagnostics for a deeper connector
- promoted: implement a true receiver-conditioned Q-Former/Perceiver-style
  connector with C2C residual distillation and source-destroying controls

## Next Exact Gate

Stop tuning this cheap connector family unless the architecture changes.

Implement a small receiver-conditioned learned-query connector:

- frozen source and target
- `8-16` learned connector queries
- query cross-attention over source K/V plus target-prior state
- C2C clean-residual distillation
- target-self-repair preservation
- matched-vs-zero/shuffled/target-only source controls
- explicit byte/latency accounting

Promotion threshold remains:

- `>=14/32` total correct
- `>=2/6` clean residual IDs
- at most `1` target-correct loss
- clean wins disappear under source controls
- exact ordered ID parity and numeric coverage `>=31/32`
