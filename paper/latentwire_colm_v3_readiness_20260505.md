# LatentWire COLM_v3 Readiness

- date: 2026-05-05
- purpose: convert the COLM_v1/COLM_v2 material into one reviewer-ready
  workshop package without importing unsupported ICLR-scale claims.

## Current Readiness

COLM_v3 now has a reproducible review-packet artifact, an integrated COLM-style
LaTeX draft in `colm_final/paper/latentwire_colm2026.tex`, and a ten-reviewer
stress panel in `colm_final/audits/colm_v3_10_reviewer_panel_20260505.md`.
It is close to reviewer circulation, but not yet final submission-ready. The
remaining blocker is a human copyedit/page-budget pass plus final PDF/table
placement review.

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
| `results/latentwire_colm_v3_review_packet_20260505/` | generated v3 claim audit, artifact manifest, systems measured-vs-estimated table, table/figure inventory, side-experiment scoping, and NVIDIA native runbook | reviewer packet and source of truth for paper integration |

## Required COLM_v3 Package

| Required item | Status | Exact next action |
|---|---|---|
| Unified abstract and intro | draft integrated and reviewer-hardened | human copyedit and page-budget review |
| Method/protocol definition | draft integrated | verify notation consistency after copyedit |
| Threat model | draft integrated | check against reviewer claim audit |
| Strict control suite table | draft integrated | validate placement in PDF |
| Main positive result table | present and source-index bounded | keep ARC/OBQA caveats visible |
| Uncertainty summary | integrated | verify placement after PDF build |
| Utility-per-byte table | present with v3 wording | keep packet, text, source-index/rank, and dense-KV byte floors scoped |
| Systems boundary table/figure | draft integrated for accounting | validate measured-vs-estimated labels in PDF |
| Related-work/baseline matrix | draft integrated | check page budget; move overflow to appendix if needed |
| Negative/failure-boundary table | present but split | include cross-family, source-choice, C2C leakage, and failed learned receivers |
| Claim audit table | draft integrated | keep appendix or move to internal audit depending on page limit |
| Reproducibility checklist | partial | convert `artifact_manifest.csv`, `manifest.json`, and expected build commands into the workshop checklist |
| Artifact manifest | generated for v3 | use `results/latentwire_colm_v3_review_packet_20260505/artifact_manifest.csv` and `manifest.json` |
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

## Generated Review Packet

The COLM_v3 review packet now exists at
`results/latentwire_colm_v3_review_packet_20260505/`, with a mirrored memo at
`paper/latentwire_colm_v3_review_packet_20260505.md`. It contains:

1. v3 contribution table,
2. reviewer claim audit,
3. table/figure inventory,
4. measured-vs-estimated systems table,
5. side-experiment scoping,
6. submission checklist,
7. artifact and input manifest,
8. NVIDIA native benchmark runbook.

## Highest-Priority Next Gate

Review and compress the integrated COLM_v3 paper draft:

1. human copyedit for claim discipline and readability,
2. PDF table/figure placement and page-budget check,
3. consistency check between paper, review packet, and artifact manifest,
4. final reproducibility checklist wording,
5. optional move of claim-audit/baseline matrix from main/appendix depending on workshop page limit.

Do not run speculative new method experiments until that package exposes a
specific missing table cell or unsupported claim.

## Ten-Reviewer Panel Read

The internal panel mean score is 6.4/10 with median 6.5/10. The paper is
plausible for a workshop weak accept if titled and framed as source-private
candidate-transfer packets under destructive controls. The same panel would
likely reject any stronger claim that the current method solves latent
communication, beats source-index transfer, or beats C2C.
