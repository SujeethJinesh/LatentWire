# SVAMP32 Query-Bottleneck Residue Manifest

- date: `2026-04-26`
- scale-up rung: strict small exact-ID gate
- status: `fails_gate`
- code commit before run: `938dc9c7`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- teacher: `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl`
- target: `results/svamp_exactid_baselines32_20260423/target_alone.jsonl`
- target set:
  `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`
- probe model: `query_bottleneck`
- feature layers: `all`
- feature dimension: `44800`
- query slots: `8`
- query epochs: `80`
- query lr: `0.01`
- query weight decay: `0.001`
- query seed: `0`

## Gate

Leave-one-ID-out learned query-bottleneck prediction from all-layer source
summary tokens to C2C residue classes, decoded through the strict frozen target
candidate pool. Promotion requires matched `>=14/32`, target-self `3/3`, clean
source-necessary `>=2/6`, exact ordered ID parity, numeric coverage `>=31/32`,
and zero clean recovery by source-destroying controls.

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 9/32 | 0/6 | 2/3 |
| zero_source | 14/32 | 0/6 | 3/3 |
| shuffled_source | 10/32 | 0/6 | 1/3 |
| label_shuffled | 14/32 | 0/6 | 3/3 |
| target_only | 14/32 | 0/6 | 3/3 |
| slots_only | 8/32 | 0/6 | 0/3 |

## Artifacts

- JSON:
  `results/svamp32_query_bottleneck_residue_20260426/qwen25_05b_all_layers_query_slots8_probe.json`
  - sha256:
    `59964c426e13f61dc00805beb30574aaa376df70ba77377864d0eeb41bb9d7b3`
- Markdown:
  `results/svamp32_query_bottleneck_residue_20260426/qwen25_05b_all_layers_query_slots8_probe.md`
  - sha256:
    `9736a7e2558bfcab3e91ee316a858c25c54320c7abdca0e14b3d947d8a9170e8`
- Log:
  `.debug/svamp32_query_bottleneck_residue_20260426/logs/qwen25_05b_all_layers_query_slots8_probe.log`
  - sha256:
    `4c8a62cdde5629759edb83d874d0441616eac46c7bdfd1db2b1666d479e0183c`

## Decision

Fail the gate and do not scale summary-token query bottlenecks. The next
highest-value branch is token/layer-level C2C-residual distillation or a full
source-token query bottleneck with a rate/slot curve and the same controls.
