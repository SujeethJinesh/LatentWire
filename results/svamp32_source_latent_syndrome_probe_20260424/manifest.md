# SVAMP32 Source-Latent Syndrome Probe Results - 2026-04-24

## Summary

- status: `source_latent_syndrome_probe_fails_gate`
- gate: a leave-one-ID-out ridge predictor from frozen source hidden summaries
  must replace the C2C oracle syndrome while preserving the strict target-side
  candidate-pool controls
- strict pass rule: matched `>=14/32`, target-self `3/3`, clean
  source-necessary `>=2/6`, zero/shuffle/label-shuffle/target-only/slots-only
  control clean union `0/6`, exact ordered ID parity, numeric coverage
  `>=31/32`
- last-layer features: matched `9/32`, target-only `14/32`, clean
  source-necessary `0/6`, target-self `2/3`
- mid+last features: matched `9/32`, target-only `14/32`, clean
  source-necessary `0/6`, target-self `3/3`

This weakens the direct linear source-hidden syndrome branch. The target
candidate pool remains viable, but the tested Qwen2.5-0.5B source hidden
summaries do not read out the C2C residue syndrome on this strict SVAMP32
surface.

## Tracked Artifacts

- `qwen25_05b_last_targetpool_probe.json`
- sha256: `afc769b2d3f56e450ba3e0d2a4f5df73975d4fceb564b80518fb6d653229410e`
- `qwen25_05b_last_targetpool_probe.md`
- sha256: `474da20224a744dfbccbd9e2c8e04e00b94fe5f99e8e290f1f850c6c372c7cf0`
- `qwen25_05b_mid_last_targetpool_probe.json`
- sha256: `2fe61a32c3cc872cc72887ae34a41716e27d887754dd627aff6807fc7e20e40f`
- `qwen25_05b_mid_last_targetpool_probe.md`
- sha256: `cbaa677e680291bd3f2b00ab4a313a36088b71292e455a9c704af555ea8af866`

## Scratch Logs

- `.debug/svamp32_source_latent_syndrome_probe_20260424/logs/qwen25_05b_last_targetpool_probe_rerun.log`
- sha256: `b5c426ca5aced18f76260fc4848b5ee0619c4b1316f47e5050915e7e2fe75327`
- `.debug/svamp32_source_latent_syndrome_probe_20260424/logs/qwen25_05b_mid_last_targetpool_probe_rerun.log`
- sha256: `daefd26eb32828582a2899a199bcfae21f5b098c4163e26d500e3f11f029c930`

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_latent_syndrome_probe.py
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_latent_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --shuffle-offset 1 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers last \
  --device mps \
  --dtype float32 \
  --date 2026-04-24 \
  --output-json results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.json \
  --output-md results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.md
```

Use the same command with `--feature-layers mid,last` and the
`qwen25_05b_mid_last_targetpool_probe.*` outputs for the richer feature
variant.

## Interpretation

The prior oracle sidecar remains a useful bound, but this run does not convert
it into a deployable source-latent method. A next branch should either train a
small query bottleneck against the residue target with held-out IDs, or move to
a stricter C2C-residual distillation setup that predicts the residue from
source cache transformations rather than pooled final hidden summaries.
