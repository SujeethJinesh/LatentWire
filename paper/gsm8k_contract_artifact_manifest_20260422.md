# GSM8K Contract Artifact Manifest

- date: `2026-04-22`
- live label: `dynalign_module_replace_residrank16`
- main blocker: `seed_stability_and_cross_family_falsification`

## Artifact Paths

- `smoke_contract_json`: `results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.json`
- `live_contract_json`: `results/gsm8k_contract_residual_rank16_dynalign_20260421/gsm8k_contract_residual_sweep_20260421.json`
- `matched_control_json`: `results/gsm8k_contract_residual_rank16_tokenbasis_20260421/gsm8k_contract_residual_sweep_20260421.json`
- `larger_slice_campaign_json`: `results/gsm8k_contract_campaign_slice128_seed0_20260422/gsm8k_contract_campaign.json`
- `seed1_health_json`: `checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_seed1.pt.health.json`

## Evidence Summary

- smoke contract: target=`0.0625`, rotalign=`0.0625`, c2c=`0.1250`, rotalign numeric coverage=`28`
- live 32-example row: `dynalign_module_replace_residrank16` accuracy=`0.1250`, wins=`2`, losses=`0`, coverage=`32`
- matched control: `tokenbasis_replace_residrank16` accuracy=`0.0625`, wins=`0`, losses=`0`, coverage=`32`
- larger frozen slice: candidate mean=`0.1143`, target=`0.0571`, c2c=`0.1286`, delta=`0.0571` [-0.0143, 0.1429]
- seed-1 health: nonfinite=`2381056`, first_bad_key=`W_V.8`, top_tensor=`W_V.8`

## Next Exact Gates

- finish the larger frozen same-pair seed-repeat campaign without non-finite checkpoints
- preserve or beat the old 0.0938 ceiling while keeping full numeric coverage
- run one strict matched cross-family falsification pair before widening benchmarks
