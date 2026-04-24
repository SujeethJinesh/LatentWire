# SVAMP32 Learned Syndrome Probe Results - 2026-04-24

## Summary

- status: `learned_syndrome_probe_fails_gate`
- gate: a cross-fitted query bottleneck over frozen source token states must
  replace the C2C oracle syndrome while preserving strict target-side
  candidate-pool controls
- strict pass rule: matched `>=14/32`, target-self `3/3`, clean
  source-necessary `>=2/6`, zero/shuffle/label-shuffle/same-norm-noise/
  target-only/slots-only clean union `0/6`, exact ordered ID parity, numeric
  coverage `>=31/32`
- `q=4`, `h=16`, `8` outer folds, `80` epochs: matched `10/32`,
  target-only `14/32`, same-norm-noise `14/32`, clean source-necessary `0/6`,
  target-self `2/3`
- `q=8`, `h=64`, `8` outer folds, `120` epochs: matched `9/32`,
  target-only `14/32`, same-norm-noise `14/32`, clean source-necessary `0/6`,
  target-self `2/3`

The learned source-token bottleneck does not convert the syndrome bound into a
method on this surface. The result weakens source-token residue prediction and
promotes C2C-residual distillation or a different source signal as the next
gate.

## Tracked Artifacts

- `qbottleneck_q4_h16_f8_seed1_targetpool_probe.json`
- sha256: `8115eeabe5c98d6699c3aad7dd477bcb1740c84fbd8d7f4927922106f9193908`
- `qbottleneck_q4_h16_f8_seed1_targetpool_probe.md`
- sha256: `97aad502f382b47dc30ee2a89f0fe7cbfd89886c2f306b7546e4930fdbbdcef0`
- `qbottleneck_q8_h64_f8_seed1_targetpool_probe.json`
- sha256: `ae61f6f4c4947a5b6596537c75a2ffe2b7733735ed3ea2fbfe72b76424f43052`
- `qbottleneck_q8_h64_f8_seed1_targetpool_probe.md`
- sha256: `c26c0c4a0785495c07b9552ca02e1adb39d9d1788a345d29962ca3e89909c491`

## Scratch Logs

- `.debug/svamp32_learned_syndrome_probe_20260424/logs/qbottleneck_q4_h16_f8_seed1_targetpool_probe.log`
- sha256: `c6254395b9e858fd07060aa607c3d56049ef5377a631884bc4ad31a0d0664ca0`
- `.debug/svamp32_learned_syndrome_probe_20260424/logs/qbottleneck_q8_h64_f8_seed1_targetpool_probe.log`
- sha256: `3a74a9f2260b5a1efd2200f94d250bb867db9b9e96389df0805b32268bc7c45f`

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_learned_syndrome_probe.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_learned_syndrome_probe.py
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_learned_syndrome_probe.py \
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
  --query-count 4 \
  --hidden-dim 16 \
  --epochs 80 \
  --outer-folds 8 \
  --seed 1 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers mid,last \
  --device mps \
  --train-device mps \
  --dtype float32 \
  --date 2026-04-24 \
  --output-json results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.json \
  --output-md results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.md
```

Use the same command with `--query-count 8`, `--hidden-dim 64`,
`--epochs 120`, and the `qbottleneck_q8_h64_f8_seed1_targetpool_probe.*`
outputs for the richer variant.

## Interpretation

The oracle sidecar remains a useful bound, but both learned source-token
bottleneck variants fail below the target-only floor and recover no clean
source-necessary IDs. Same-norm noise reaching `14/32` indicates the learned
head is not extracting the missing clean residue from matched source token
states on this slice.
