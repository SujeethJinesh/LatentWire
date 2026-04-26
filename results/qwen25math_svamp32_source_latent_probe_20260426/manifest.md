# Qwen2.5-Math -> Qwen3 SVAMP32 Source-Latent Probe Manifest

- date: `2026-04-26`
- git commit at run time: `ecd8b0946956636073875186add64d16c68af7a6`
- status: `source_latent_syndrome_probe_fails_gate`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- decision surface:
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`

## Result Summary

| Feature Set | Matched | Clean Source-Necessary | Control Clean Union |
|---|---:|---:|---:|
| last layer | 8/32 | 0/6 | 1 |
| all layers | 8/32 | 0/6 | 2 |

## Artifacts

| Path | SHA256 |
|---|---|
| `results/qwen25math_svamp32_source_latent_probe_20260426/last_layer_ridge_probe.json` | `82b9690cb6c993312ec92044cd7de03f534eaefbabb8bbee6a964b3c5a9101ae` |
| `results/qwen25math_svamp32_source_latent_probe_20260426/last_layer_ridge_probe.md` | `ecdcbf1246b28bbb966cc8d49fb4af79b270f3a2fc60f647070d4a2f3ea0566c` |
| `results/qwen25math_svamp32_source_latent_probe_20260426/all_layers_ridge_probe.json` | `d643ac24b6ec1f0ee9a66073f192ab873443e736e5497cde9029147d7b66c90a` |
| `results/qwen25math_svamp32_source_latent_probe_20260426/all_layers_ridge_probe.md` | `f0c441c534ec0a14a1eaba98dee9cd7d195ca4de941be2d170f9093a70fe0392` |

## Decision

Kill summary-level source-hidden ridge residue readout on this Qwen-Math
surface. A future source-latent branch needs a token/layer-local objective or a
different source-derived supervision signal.

