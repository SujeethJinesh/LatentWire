# Qwen2.5-Math -> Qwen3 Source-Contrastive Sidecar Manifest

- date: `2026-04-26`
- git commit at run time: `b3c589361812037390e1b2d5829db2537257572e`
- status: `source_only_sidecar_router_clears_gate`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- surface: SVAMP32 chat-template exact-ID surface
- method: target/text agreement guard plus 1-byte source residue sidecar

## Result Summary

- target-alone: `8/32`
- source-alone: `6/32`
- text relay: `8/32`
- guarded sidecar best row: `11/32`
- clean source-only target set: `4`
- clean source-necessary recovered: `3`
- control clean union: `0`
- source numeric coverage: `26/32`

## Artifacts

| Path | SHA256 |
|---|---|
| `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json` | `088f0e1651f95ea04a89ec0931276a943ff104a355fe69f434182f68e778ea96` |
| `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.md` | `22cea2201a3b60491c6e38b30e3173582d14c51f583730c92ee3092a67568919` |
| `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_only_sidecar_router_t2t_guard.json` | `c5434aeead9e55f5494ca583533fe863f36ee719e8a5bb75ae6fdb2f6f373306` |
| `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_only_sidecar_router_t2t_guard.md` | `8cb94c6b0ba5d07c428cebcbf54e2b6a0b9b21e2475a9a9413601acdb46e9831` |

## Decision

Promote to SVAMP70 medium confirmation with full artifact and systems
accounting. Do not claim ICLR readiness from this 32-example result alone.

