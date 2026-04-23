# Paper Workspace

- `iclr2026/` is copied from the official ICLR master template repository.
- `iclr2026.zip` is the matching Overleaf-uploadable archive.
- `story_tracks.md` is the current workshop-vs-full-paper outline for this project.
- `experiment_ledger_20260421.md` is the anti-loop ledger: saturated lanes,
  active positive clues, and the next stack to test.
- `matched_competitor_matrix_20260421.md` is the explicit GSM70 comparison
  matrix with missing competitor rows preserved.
- `gsm8k_contract_artifact_manifest_20260422.md` is the tracked provenance
  index for the current GSM contract evidence chain: smoke gate, live
  `dynalign_module_replace_residrank16` row, matched control, larger frozen
  slice, and the seed-collapse checkpoint-health artifact.
- `gsm8k32_anchor_tail_seed1_20260422.md` is the first bad-seed falsification
  of the new `V`-only anchor-tail runtime wrapper. It shows the wrapper alone
  does not remove the repeated layer-8 `V` checkpoint collapse.
- `gsm8k32_conditioned_bad_seed_controls_20260423.md` is the next robustness
  follow-up on the live dynalign residual lane. It shows that source+target
  whitening removes the catastrophic bad-seed checkpoint collapse on GSM8K32,
  but currently trades away the seed-0 ceiling.
- `gsm8k32_selective_conditioning_l8_v_20260423.md` is the first selective
  conditioning follow-up. It shows that exact layer-8 `V`-only conditioning is
  finite but not promotable: seed `1` only ties target and seed `0` falls below
  target.
- `gsm8k32_selective_conditioning_v_all_layers_20260423.md` is the broader
  `V`-only all-layer conditioning follow-up. It is also finite but not
  promotable: seed `1` ties target and seed `0` again falls below target.
- `gsm8k32_selective_conditioning_l8_kv_20260423.md` is the last simple
  whitening screen. Layer-8 `K/V` conditioning gives the best conditioned
  bad-seed row so far, but seed `0` still falls below target, so whitening is
  no longer the next method branch.
- `ablation_evidence_ladder_20260421.md` is the telemetry-driven stack
  decision table for toy positives, controls, and blockers.
- `../results/query_pool_toy_20260421/hub_router_frontier_sweep_20260421.md`
  is the route-conditioned hub interface sweep. It shows that oracle routing
  helps the hub base, but the current frontier and stop heuristics do not
  transfer additively.
- `../results/query_pool_toy_20260421/route_class_frontier_sweep_20260421.md`
  is the follow-up showing that route-/class-calibrated local protection still
  does not rescue the current frontier family.
