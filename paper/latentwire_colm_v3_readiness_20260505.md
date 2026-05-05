# LatentWire COLM_v3 Readiness

- date: 2026-05-05
- purpose: convert the COLM_v1/COLM_v2 material into one reviewer-ready
  workshop package without importing unsupported ICLR-scale claims.

## Current Readiness

COLM_v3 is plausible but not yet submission-ready. The safe path is a unified
workshop paper that combines the original packet intuition with the later
control hardening, systems boundary accounting, source-choice audits, and C2C
claim boundaries.

ICLR remains blocked by the lack of a broad learned/source-causal receiver that
survives strict destructive controls.

## Safe Main Claim

LatentWire provides a practical protocol and evaluation framework for
byte-scale, source-private model-to-model packet communication, with controlled
evidence of narrow packet utility, explicit utility-per-byte accounting, and
destructive controls that expose answer-choice, target-cache, and leakage
shortcuts.

## Do Not Claim

- solved latent model-to-model communication
- broad latent language
- broad cross-family generality
- raw accuracy superiority over C2C or dense KV/cache-transfer systems
- GPU latency, HBM, energy, or throughput wins without native measurement
- semantic-anchor interpretation where random/shared-coordinate anchors explain
  the evidence equally well

## Evidence Inventory For COLM_v3

| Artifact | Current read | Paper use |
|---|---|---|
| `colm_final/evidence/results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed_b2000/` | ARC 8-byte packet reaches 0.344 over 10 seeds on 1172 test examples; target 0.265; same-budget text 0.300; source-choice preservation is about 0.995 | main narrow positive plus source-choice boundary |
| `colm_final/evidence/results/source_private_colm_acceptance_baselines_20260502/` | explicit source-index/source-rank audit; packet does not beat source index | main destructive/baseline table, not appendix-only |
| `colm_final/evidence/results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/` | OpenBookQA 3-byte packet positive in original package, later v2 hardening weakens broad interpretation | secondary evidence with tightened caveat |
| `colm_final/evidence/results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000/` | Phi-3 source-family substitution fails | cross-family failure boundary |
| `colm_final/evidence/results/source_private_systems_boundary_figure_table_20260502/` | 6-11 framed bytes versus a 768B one-token 1-bit-per-KV-element accounting floor; no native latency/HBM claim | systems boundary table/figure |
| `paper/iclr_colm_v2_live_branch_triage_20260504.md` | C2C replay/headroom revived but packet distillation branches failed or are leakage upper bounds; pre-answer C2C gate is implemented but not complete | C2C boundary and future-work framing |
| `paper/svamp32_c2c_generated_answer_packet_audit_20260505.md` | generated-answer value/index packet is answer leakage, matched by same-byte visible answer and nearly by same-source-choice wrong-row | negative/leakage boundary |
| `experimental/status_20260505.md` | three hard-systems side experiments scaffolded but not evidence | optional future systems appendix only |

## Required COLM_v3 Package

| Required item | Status | Exact next action |
|---|---|---|
| Unified abstract and intro | missing | write one claim-disciplined v3 abstract/intro from v1 motivation plus v2 controls |
| Method/protocol definition | partially present in COLM package | consolidate packet object, source-private interface, receiver, and byte fields |
| Threat model | partially present | make source exposure, same-byte text, source-choice, and leakage shortcuts explicit |
| Strict control suite table | present but split | merge ARC/OBQA/C2C controls into one paper-visible table |
| Main positive result table | present but split | make ARC primary; mark OBQA as caveated if using it |
| Utility-per-byte table | present but needs v3 wording | include packet, text, source-index/rank, and dense-KV byte floors |
| Systems boundary table/figure | present for accounting | label measured-vs-estimated and native-missing rows explicitly |
| Related-work/baseline matrix | present in v2 packet | update to v3 and point C2C/KV/cache methods to source-exposure regimes |
| Negative/failure-boundary table | present but split | include cross-family, source-choice, C2C leakage, and failed learned receivers |
| Claim audit table | present for COLM package | update to v3 claim wording and add artifact paths |
| Reproducibility checklist | present for COLM package | update with v3 artifact manifest and expected commands |
| Artifact manifest | present for v1/v2 pieces | build a v3 manifest that references only claim-supporting artifacts |
| Limitations | present but needs v3 consistency | keep strong limits visible in main text |

## Systems State

| Systems artifact | Current status | Claim allowed |
|---|---|---|
| raw/framed packet bytes | measured/accounted in current artifacts | packet object size and utility-per-byte accounting |
| cacheline-rounded bytes | accounted | conservative transfer object size comparisons |
| dense KV/cache byte floors | analytical estimates | design-space boundary only |
| local packet encode/decode cost | partial/local, not native serving | optional local implementation cost if regenerated cleanly |
| native C2C/KV throughput | missing | no claim |
| GPU latency/HBM/energy | missing | no claim |
| NVIDIA kernel win | missing | runbook/future work only |

## Status Of The Three New Systems Experiments

| Experiment | Status | Recommendation |
|---|---|---|
| HybridKernel | scaffolded, `.venv` created, no evidence | keep as future systems spinout; not a COLM_v3 dependency |
| SinkAware | scaffolded, `.venv` created, no evidence | cheapest Phase 1 audit if spare systems time exists |
| ThoughtFlow-FP8 | scaffolded, `.venv` created, no evidence | high-upside but crowded; use only if pivoting to a separate systems paper |

## Highest-Priority Next Gate

Build the COLM_v3 review packet from existing evidence:

1. v3 abstract/introduction/contribution bullets,
2. v3 claim audit table,
3. v3 artifact manifest,
4. one systems table with measured-vs-estimated labels,
5. one reviewer-facing "what we claim / what we do not claim" section.

Do not run speculative new method experiments until that package exposes a
specific missing table cell or unsupported claim.
