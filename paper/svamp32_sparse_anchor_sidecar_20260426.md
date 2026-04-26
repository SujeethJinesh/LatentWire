# SVAMP32 Sparse-Anchor Sidecar Smoke - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable positive method plus medium, seed-repeat,
  uncertainty, source-control, systems, and cross-family gates
- current story: C2C still exposes clean same-family headroom, but shallow
  source-derived interfaces are not recovering the clean C2C-only IDs
- blocker: sparse/anchor sidecars did not predict the clean C2C residue targets
  on the strict SVAMP32 headroom slice

## Gate

This cycle implemented a narrow real-model smoke for the quotient/GPA toy
inspiration: a rate-capped sparse anchor projection of source hidden summaries,
plus tokenizer-boundary alignment sidecar features, decoded through the same
SVAMP32 C2C-residue candidate-pool gate.

The smoke used:

- source model: `Qwen/Qwen2.5-Math-1.5B`
- target tokenizer/model: `Qwen/Qwen3-0.6B`
- exact-ID surface:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- target set:
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`
- teacher:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl`
- controls: zero-source, shuffled-source, label-shuffled, target-only, and
  slots-only

Promotion rule:

- matched correct at least `10/32`
- target floor preserved
- at least `2/6` clean source-necessary recoveries
- destructive controls recover `0/6` clean IDs
- sidecar budget reported, with a constrained variant at or under `16`
  bytes/example

## Evidence

| Variant | Bytes/example | Candidates | Matched | Clean Matched | Target-only | Control Clean Union | Decision |
|---|---:|---|---:|---:|---:|---:|---|
| `probe` | 34 | target + C2C | 9/32 | 0/6 | 8/32 | 1/6 | fail |
| `probe_budget16_seed3` | 14 | target + source + text + C2C | 7/32 | 0/6 | 8/32 | 0/6 | fail |

The first variant weakly improved over target-only by one example, but that
example was source/text-explained rather than a clean C2C-headroom ID. The
constrained variant satisfied the byte budget and avoided clean control
recovery, but fell below the target floor and still recovered `0/6` clean IDs.

## Decision

Weaken the current sparse-anchor sidecar implementation. It is not worth
scaling as implemented because two adjacent variants both recover `0/6` clean
C2C-headroom IDs.

Do not kill the broader quotient/GPA sparse-dictionary idea solely from this
smoke: this implementation uses a random sparse anchor projection over source
hidden summaries, not fold-local spherical k-means over token/span anchors.
However, the next attempt must materially change the feature extractor, not
only tune projection seed, top-k, or byte budget.

## Next Gate

Use the new analyzer as a regression harness, then either:

1. implement the stricter token/span dictionary version from the audit
   recommendation, with fold-local atoms, same-norm-noise and boundary-only
   controls; or
2. pivot to an existing real sparse shared-code adapter lane such as
   `bridge_ridge_qk_sae_adapter`, but only if it is evaluated on the same
   clean SVAMP32 C2C-headroom target set with source-destroying controls.

Promotion still requires at least `2/6` clean source-necessary recoveries and
zero clean destructive-control recovery before any SVAMP70 or seed scale-up.

## Artifacts

- analyzer:
  `scripts/analyze_svamp32_sparse_anchor_sidecar_probe.py`
- tests:
  `tests/test_analyze_svamp32_sparse_anchor_sidecar_probe.py`
- first run:
  `results/svamp32_sparse_anchor_sidecar_20260426/probe.json`
  - sha256:
    `a9945e7a3679e382aa83f98a7c486c8091bceee0005ec799dcac5cca4cb81896`
- first run readout:
  `results/svamp32_sparse_anchor_sidecar_20260426/probe.md`
  - sha256:
    `8000a7caf4f040a5b9e436ca41b7571c60573839f85765b3b5b5cb7c5511b7a6`
- constrained run:
  `results/svamp32_sparse_anchor_sidecar_20260426/probe_budget16_seed3.json`
  - sha256:
    `b27aacd16907a933a271183542003a9e8450617b9394389e3cdeaa6d992ad26e`
- constrained run readout:
  `results/svamp32_sparse_anchor_sidecar_20260426/probe_budget16_seed3.md`
  - sha256:
    `bd8726a233f76c41aa55b4e045aef158045233afc1a68b0e05eed613183ce8a3`

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_sparse_anchor_sidecar_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_sparse_anchor_sidecar_probe.py`
