# SVAMP32 Source-Token All-Layer Bottleneck Manifest

- date: `2026-04-26`
- scale-up rung: strict small exact-ID gate
- status: `fails_gate`
- code commit before run: `9c390da6`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- teacher: `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl`
- target: `results/svamp_exactid_baselines32_20260423/target_alone.jsonl`
- target set:
  `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`
- feature layers: `all`
- query count: `4`
- hidden dim: `16`
- epochs: `80`
- outer folds: `8`
- seed: `2`

## Gate

Cross-fitted learned source-token bottleneck over all source hidden layers,
decoded through the strict frozen target candidate pool. Promotion requires
matched `>=14/32`, target-self `3/3`, clean source-necessary `>=2/6`, exact
ordered ID parity, numeric coverage `>=31/32`, and zero clean recovery by
source-destroying controls.

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 7/32 | 0/6 | 2/3 |
| zero_source | 14/32 | 0/6 | 3/3 |
| shuffled_source | 10/32 | 0/6 | 1/3 |
| label_shuffled | 13/32 | 0/6 | 3/3 |
| same_norm_noise | 14/32 | 0/6 | 3/3 |
| target_only | 14/32 | 0/6 | 3/3 |
| slots_only | 8/32 | 0/6 | 0/3 |

## Artifacts

- JSON:
  `results/svamp32_source_token_all_layers_bottleneck_20260426/qbottleneck_q4_h16_f8_seed2_all_layers_targetpool_probe.json`
  - sha256:
    `c09874826af09a957a7c467ee5afd54fa36ec2122e62b2455cd553eaf7064e6a`
- Markdown:
  `results/svamp32_source_token_all_layers_bottleneck_20260426/qbottleneck_q4_h16_f8_seed2_all_layers_targetpool_probe.md`
  - sha256:
    `dee99f9ac14137e1f8c1da8fdebceee4182f3ff32cdf463f8c5cbccb6dd6ffa8`
- Log:
  `.debug/svamp32_source_token_all_layers_bottleneck_20260426/logs/qbottleneck_q4_h16_f8_seed2_all_layers_targetpool_probe.log`
  - sha256:
    `dcd00fe42f1c137f64110882bde470aa259489ecb747b8a2a73045fbd3043de6`

## Decision

Fail the gate and kill source-token query-bottleneck residue prediction on the
current SVAMP32 syndrome surface. Next work should move to source-surface
discovery: process-repair/selector controls or real cross-family
tokenizer/interface stress.
