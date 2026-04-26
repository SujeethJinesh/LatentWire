# Qwen2.5-Math -> Qwen3 Token-Layer C2C Residual Manifest

- date: `2026-04-26`
- git commit at run time: `5bc3d04fc2950888d22be1ec8f98a68096fbe80d`
- status: `c2c_mechanism_syndrome_probe_fails_gate`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- feature family: `c2c_prefill_token_layer_tail_residual`
- probe model: `query_bottleneck`
- decision surface:
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`

## Result Summary

- matched: `8/32`
- target-only: `8/32`
- clean source-necessary IDs: `0/6`
- control clean union: `1/6`
- feature shape: `[32, 229376]`
- token shape: `[224, 1024]`

## Artifacts

| Path | SHA256 |
|---|---|
| `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.json` | `b2bfb8605b07c7a9f9d98d31fb35091e06457b42580e884f440be1684fba0b6e` |
| `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.md` | `83ba897e191dd62b51706c2859a443cf2760a6fd6230a8ea7998d374c6c5b440` |

## Decision

Do not scale C2C token-local mechanism readouts on this surface. They do not
recover clean C2C-only IDs beyond target/control behavior.

