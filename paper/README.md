# Paper Workspace

- `iclr2026/` is copied from the official ICLR master template repository.
- `iclr2026.zip` is the matching Overleaf-uploadable archive.
- `story_tracks.md` is the current workshop-vs-full-paper outline for this project.
- `experiment_ledger_20260421.md` is the anti-loop ledger: saturated lanes,
  active positive clues, and the next stack to test.
- `repo_readiness_review_20260426.md` is the latest repo-wide ICLR readiness
  audit: what has been tried, what is saturated, what remains live, and the next
  exact gate.
- `qwen25math_svamp32_headroom_and_source_probes_20260426.md` records the new
  Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom target set and the negative
  source-only numeric sidecar / source-hidden ridge screens on that surface.
- `qwen25math_svamp32_token_layer_c2c_residual_20260426.md` records the
  negative token/layer-local C2C residual query-bottleneck diagnostic on the
  Qwen2.5-Math -> Qwen3 SVAMP32 surface.
- `svamp32_target_safe_oracle_replay_20260426.md` records the negative
  target-safe dynalign/query-pool oracle replay: even an oracle selector over
  existing candidates recovers only `1/6` clean residual IDs and is matched by
  the source-destroying control oracle.
- `qwen25math_svamp32_perceiver_c2c_residual_20260426.md` records the
  negative Qwen2.5-Math -> Qwen3 Perceiver/query-innovation C2C-residual gate:
  all fixed gates recover `0/6` matched-only clean IDs before generation.
- `qwen25math_svamp32_target_query_source_bottleneck_20260426.md` records the
  implemented target-query-conditioned source bottleneck gate; it reaches only
  `7/32` with `0/6` clean residual IDs and is killed before generation.
- `qwen25math_svamp32_source_soft_prefix_logprob_20260426.md` records the
  calibrated source-only soft-prefix logprob gate; it recovers only `1/6`
  clean source-communication candidate IDs, has `4/6` clean control leaks, and
  is killed as a summary-prefix branch before generation.
- `qwen25math_svamp32_source_cross_attention_logprob_20260426.md` records the
  first token-local source cross-attention logprob gate; it recovers `0/6`
  clean candidate IDs with `4/6` clean control leaks and is not worth scaling
  as the current tiny prefix-emitting implementation.
- `source_surface_reselection_and_svamp70_cross_attention_20260426.md` records
  the consolidated surface reselection after the SVAMP32 connector failures and
  the negative SVAMP70 live cross-attention rescue: `0/6` matched-only clean
  IDs with `3/6` clean control leaks.
- `qwen25math_svamp70_holdout_finalish_guard_20260426.md` records the holdout
  falsification of the fixed `finalish_short_numeric` source-sidecar guard:
  `0/2` clean source-necessary IDs with `2/2` control leakage.
- `qwen25math_svamp70_source_trace_router_20260426.md` records the
  source-trace self-consistency router falsification: live CV recovers only
  `1` clean source-necessary ID with `2` accepted harms, and the single
  holdout clean win survives equation-result permutation.
- `source_generation_diagnostics_artifact_20260426.md` records the tooling
  smoke for source-internal generation confidence artifacts: source-only greedy
  logprob, entropy, top-1 probability, and top-1/top-2 margin are now
  collectable in a sidecar JSONL without mutating baseline prediction files.
- `qwen25math_svamp70_source_confidence_router_20260426.md` records the
  source-internal confidence-router falsification: live CV is clean but weak
  (`24/70`, `2` clean source-necessary), and frozen holdout drops to `7/70`
  with zero clean source-necessary IDs.
- `qwen25math_svamp32_source_contrastive_sidecar_20260426.md` records the
  first positive strict-small Qwen2.5-Math -> Qwen3 source-contrastive sidecar
  stack: target/text agreement preservation plus a 1-byte source residue
  sidecar.
- `qwen25math_svamp70_source_contrastive_sidecar_20260426.md` records the
  medium SVAMP70 confirmation: the sidecar beats target/text but remains below
  C2C and has bootstrap intervals crossing zero.
- `qwen25math_svamp70_holdout_finalish_guard_20260426.md` records the holdout
  failure of the `finalish_short_numeric` source-quality guard: best row
  `9/70`, clean source-necessary `0/2`, and control clean union `2/2`,
  pruning fixed source-quality guarded sidecars.
- `qwen25math_svamp70_chal241_sidecar_gate_20260426.md` records the
  `chal241-310` source-sidecar pruning gate: the sidecar touches clean
  source-only IDs, but total accuracy stays too low and controls leak clean
  IDs, so the slice should not receive C2C or connector spend.
- `svamp70_process_repair_source_controls_20260426.md` records the
  source-control falsification of the old SVAMP70 process-repair row: matched
  repair beats target self-repair by `3/70`, but zero/shuffled-source controls
  recover all route-specific wins, killing it as a communication method.
- `svamp32_c2c_mechanism_syndrome_probe_20260426.md` records the negative
  strict small-gate C2C prefill scalar/residual trace syndrome-distillation
  probe.
- `svamp32_perceiver_answer_teacher_contrastive_20260426.md` records the
  negative teacher-forced pre-gate for the Perceiver answer-teacher plus
  source-control contrastive connector.
- `svamp70_perceiver_answer_teacher_contrastive_20260426.md` records the
  negative SVAMP70 headroom-surface pre-gate for the same connector family.
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
