# SVAMP32 Source-Only Sidecar Router Manifest

- date: `2026-04-26`
- scale-up rung: strict small exact-ID gate
- status: `fails_gate`
- code commit: `8b3d7924` plus local script addition
- target rows: `results/svamp_exactid_baselines32_20260423/target_alone.jsonl`
- source rows: `results/svamp32_stronger_source_baselines_20260424/source_alone.jsonl`
- target set: `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`

## Gate

Source-side numeric predictions form a compact residue sidecar. Target-side
candidate rows are used only as decoder side information. Controls:
`zero_source`, `shuffled_source`, `label_shuffle`, `same_norm_noise`,
`target_only`, and `slots_only`.

Promotion rule:

- matched `>=14/32`
- target-self preserve `3/3`
- clean source-necessary `>=2/6`
- clean control union `0/6`
- source numeric coverage `>=31/32`

## Artifacts

- JSON: `results/svamp32_source_only_sidecar_router_20260426/source_only_router_gate.json`
  - sha256: `6f92482c8b2b500eb4cb3d29a228e0797dea59e5f2fa4c78935c739413addce2`
- Markdown: `results/svamp32_source_only_sidecar_router_20260426/source_only_router_gate.md`
  - sha256: `bbd7e47d55dbeee118b4812ef2b3ac5a305290eb7e947f3e663765510e755b95`
- Log: `.debug/svamp32_anti_memory_perceiver_20260426/logs/source_only_sidecar_router_gate.log`
  - sha256: `8f3645a99c1fde7e266e5a160e76b923a90fa0a39d4777d2507d9f706f06e5ee`

## Evidence

| Moduli | Bytes | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `2,3,5,7` | 1 | 4/32 | 0/3 | 0/6 | 0/6 | 0/6 | fail |
| `97` | 1 | 4/32 | 0/3 | 0/6 | 0/6 | 0/6 | fail |

## Decision

Kill the simple source-generated-numeric sidecar/router branch. It has clean
controls, but it recovers no clean source-necessary IDs and does not preserve
target-self rows.

Next branch should use source latent features or token/layer-level C2C residual
targets, not raw source generated numeric answers.
