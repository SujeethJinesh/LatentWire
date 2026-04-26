# SVAMP32 C2C Mechanism Projection Probe Manifest

- date: `2026-04-26`
- scale-up rung: strict small diagnostic gate
- status: `fails_gate`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- residual projection dim: `16`

## Artifacts

- JSON:
  `results/svamp32_c2c_mechanism_projection_probe_20260426/prefill_residual_projection16_targetpool_probe.json`
  - sha256:
    `3fc5f51320dbe96de8940c28194becda9f81f6906ab84a06a87e414f22a4400f`
- Markdown:
  `results/svamp32_c2c_mechanism_projection_probe_20260426/prefill_residual_projection16_targetpool_probe.md`
  - sha256:
    `5ad4dfdbafc2f52127817b593e5d22be84a921ad42d73ffe770e99e16a9d03a9`
- Log:
  `.debug/svamp32_c2c_mechanism_projection_probe_20260426/logs/prefill_residual_projection16_targetpool_probe.log`
  - sha256:
    `f569d565b4f5f44db44ba725704d8f4f4f07c501ad996ac33f030b0efc473fea`

## Result

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 13/32 | 0/6 | 3/3 |
| zero_source | 14/32 | 0/6 | 3/3 |
| shuffled_source | 11/32 | 0/6 | 2/3 |
| label_shuffled | 14/32 | 0/6 | 3/3 |
| target_only | 14/32 | 0/6 | 3/3 |
| slots_only | 8/32 | 0/6 | 0/3 |

## Decision

Fail the gate. Signed residual projections improve the matched row by one over
the earlier residual-summary probe, but do not beat target/control floors and
recover `0/6` clean source-necessary IDs.

