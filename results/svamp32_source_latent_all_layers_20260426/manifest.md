# SVAMP32 Source-Latent All-Layer Manifest

- date: `2026-04-26`
- scale-up rung: strict small exact-ID gate
- status: `fails_gate`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- teacher: `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl`
- target: `results/svamp_exactid_baselines32_20260423/target_alone.jsonl`
- target set:
  `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`
- feature layers: `all`
- feature dimension: `44800`
- ridge lambda: `1.0`

## Gate

Leave-one-ID-out ridge prediction from source hidden summaries to C2C residue
classes, decoded through the strict frozen target candidate pool. Promotion
requires matched `>=14/32`, target-self `3/3`, clean source-necessary `>=2/6`,
exact ordered ID parity, numeric coverage `>=31/32`, and zero clean recovery
by source-destroying controls.

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 9/32 | 0/6 | 2/3 |
| zero_source | 14/32 | 0/6 | 3/3 |
| shuffled_source | 10/32 | 0/6 | 1/3 |
| label_shuffled | 14/32 | 0/6 | 3/3 |
| target_only | 14/32 | 0/6 | 3/3 |
| slots_only | 8/32 | 0/6 | 0/3 |

Failing criteria:

- `min_correct`
- `preserve_fallback_floor`
- `min_clean_source_necessary`

## Artifacts

- JSON:
  `results/svamp32_source_latent_all_layers_20260426/qwen25_05b_all_layers_targetpool_probe.json`
  - sha256:
    `5d37613d648392c58f7b28a7274ba8dadffd3b76cd09cd75dd517df64990bce9`
- Markdown:
  `results/svamp32_source_latent_all_layers_20260426/qwen25_05b_all_layers_targetpool_probe.md`
  - sha256:
    `6dcce69bde942ff9e7616b8bfc446ff3df942399b241e37f27069ac6ce4900c2`
- Log:
  `.debug/svamp32_source_latent_all_layers_20260426/logs/qwen25_05b_all_layers_targetpool_probe.log`
  - sha256:
    `de495dbf68d4c943bea4eed4964909cb83308de3315074302f473ce8b6936014`

## Decision

Kill direct linear source-hidden syndrome readout, including all-layer pooled
source features. The next branch should be a query-bottleneck or token/layer
C2C-residual distillation method with the same source-destroying controls.
