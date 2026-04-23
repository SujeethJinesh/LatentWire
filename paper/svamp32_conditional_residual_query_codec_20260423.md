# SVAMP32 Conditional Residual Query Codec Screen

## Status

Not ICLR-ready. The live positive-method story remains blocked: target
self-repair is a strong decoder-side floor (`14/32`), while source-conditioned
latent transfer has not recovered at least `2/6` clean C2C-only residual IDs on
the frozen SVAMP32 exact-ID surface.

Current paper story: source communication should transmit conditional residual
information above target self-repair, not replace the target cache. This turn
tested the smallest target-self-preserving version of the existing query
innovation codec.

Blocking gap: no matched source-conditioned row clears `>=2/6` clean residual
recoveries with exact ID parity.

## Decision Surface

Alive before this screen:

- query-innovation residual codec as the only branch with one clean
  source-necessary ID
- target-self-preserving conditional no-op losses
- K-only residual training, because prior value transport was noisy

Saturated before this screen:

- scalar fixed-gate retuning
- clean-ID weighting alone
- naive zero/shuffle contrastive controls
- full and sparse value transport over the prior checkpoint

Blocked:

- seed repeats, larger frozen slices, and cross-family falsification until the
  same-pair matched gate recovers `>=2/6` clean residual IDs

## Code Change

Implemented default-off support for a protected residual query-codec screen:

- `--innovation-target-self-preserve-weight` expands
  `ids.target_self_repair` from the target-set JSON into prompt weights and a
  zero-residual mask, forcing the residual module to learn no-op behavior on
  target-self-repair IDs.
- `--innovation-value-loss-weight` scales the value residual/logit/distillation
  path for query innovation; this screen used `0.0` for a K-only codec.
- Existing query-innovation behavior is unchanged when both flags use defaults.

## Verification

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_calibrate_and_ablation.py \
  tests/test_translator_core.py -q
```

Result: `216 passed`.

## Experiment

Calibration:

```bash
./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --output .debug/svamp32_conditional_residual_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_preserve_konly_query_codec_r16_bank16_seed1.pt \
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
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --seed 1
```

Matched gate sweep:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/svamp32_conditional_residual_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_preserve_konly_query_codec_r16_bank16_seed1.pt \
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
  --gate-values 0.05 0.10 0.125 0.15 0.175 0.20 0.25 \
  --methods rotalign \
  --prediction-output .debug/svamp32_conditional_residual_query_codec_20260423/preds/preserve_konly_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Readout:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_gate_sweep_clean_targets.py \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --candidate-jsonl .debug/svamp32_conditional_residual_query_codec_20260423/preds/preserve_konly_attention_gate_sweep.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_conditional_residual_query_codec_20260423/preserve_konly_attention_clean_targets.json \
  --output-md results/svamp32_conditional_residual_query_codec_20260423/preserve_konly_attention_clean_targets.md \
  --expected-n 32 \
  --min-clean-residual-recovered 2 \
  --target-self-correct 14
```

## Evidence

Calibration telemetry:

- calibration prompts: `32`
- dynamic token-mixture samples: `1411`
- clean residual prompts matched: `6`
- target-self-preserve prompts matched: `3`
- average fit quality: K cosine `0.951`, V cosine `0.734`

Matched readout:

| Gate | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 8/32 | 0/6 | 1 | -6 | 1 |
| 0.10 | 7/32 | 0/6 | 1 | -7 | 2 |
| 0.12 | 8/32 | 0/6 | 1 | -6 | 1 |
| 0.15 | 8/32 | 0/6 | 1 | -6 | 1 |
| 0.175 | 8/32 | 0/6 | 1 | -6 | 1 |
| 0.20 | 8/32 | 0/6 | 1 | -6 | 1 |
| 0.25 | 8/32 | 0/6 | 1 | -6 | 2 |

Verdict: `no_matched_gate_candidate_for_controls`.

Controls were not run because the matched row did not reach the promotion gate.

## Artifact Hashes

- checkpoint:
  `.debug/svamp32_conditional_residual_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_preserve_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `a6236dd37c2dd8caa0d3928d644286ce5843ee26ff8f6fb336dcbfd8e6e24eca`
- matched sweep:
  `.debug/svamp32_conditional_residual_query_codec_20260423/preds/preserve_konly_attention_gate_sweep.jsonl`
  - sha256: `dc8cf4cb13e210d19ae56462bd741c4043fc237d8e2e7311bd42181aef1fa167`
- matched sweep meta:
  `.debug/svamp32_conditional_residual_query_codec_20260423/preds/preserve_konly_attention_gate_sweep.jsonl.meta.json`
  - sha256: `091a0ea04cd543a6c321bf7d525fc3ae6e204d18834f95dad3d5114a212499b0`
- clean-target JSON:
  `results/svamp32_conditional_residual_query_codec_20260423/preserve_konly_attention_clean_targets.json`
  - sha256: `73256d6e51fbe9e09ef357c1939087c67db821cf8cf6b34f86fb5a90d0edeb11`
- clean-target Markdown:
  `results/svamp32_conditional_residual_query_codec_20260423/preserve_konly_attention_clean_targets.md`
  - sha256: `0207a238675f8cceca3a7e25f21906d9579a061b9e771fd19ba52e8942201f89`

## Hypothesis Update

Weakened:

- target-self no-op residual weighting plus K-only value-loss suppression is not
  enough to recover the known clean source-necessary ID, let alone a second one

Still alive:

- an actual target-conditioned memory variant with target prior K/V as side
  information, because this screen did not add target-side cache state to the
  query memory

Promoted next:

- implement the isolated conditional query-memory variant described by the repo
  audit: keep old query innovation untouched, add a new opt-in variant whose
  module memory is `[source K/V, target-prior K/V, learned slots]`, and rerun
  the same matched SVAMP32 gate.

## References

- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597
- Flamingo Perceiver Resampler / gated cross-attention: https://arxiv.org/abs/2204.14198
- Perceiver IO latent bottleneck: https://arxiv.org/abs/2107.14795
- AWQ selective activation-aware quantization: https://arxiv.org/abs/2306.00978
- Wyner-Ziv decoder side-information coding: https://doi.org/10.1109/TIT.1976.1055508
