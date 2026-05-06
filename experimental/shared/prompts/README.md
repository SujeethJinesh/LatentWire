# Shared Prompt Manifests

These manifests define small, frozen Mac-gate surfaces for SSQ-LR, HORN, and
HBSM smoke runs. They are not benchmark claims; they are deterministic prompt
sets for cache/state/activation packet production before any 5090 validation.

## `hybrid_reasoning_smoke_12_20260506.jsonl`

- Scope: Mac-local smoke surface for the first real SSQ-LR/HORN/HBSM packets.
- Rows: 12 handwritten reasoning prompts spanning arithmetic, algebra, logic,
  commonsense, multi-step, and comparison tasks.
- SHA-256: `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.
- Required use: record this file path in `config.json` as `prompt_source` and
  record its SHA-256 as `prompt_ids_hash` in the form `sha256:<hex>`.
- Exclusions: do not use this manifest as evidence of benchmark accuracy,
  reasoning-model generality, or task performance.
