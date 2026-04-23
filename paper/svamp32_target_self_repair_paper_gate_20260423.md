# SVAMP32 Target-Self-Repair Paper Gate

Date: 2026-04-23

## Paper Status

Not ICLR-ready. SVAMP32 has a clean positive teacher surface, but the current
learned/rotalign rows have not shown source-specific communication beyond
target-side repair and zero/shuffled-source controls.

Estimated distance to ICLR readiness: medium-high. The project needs one
controlled same-pair positive row, seed stability, and one strict cross-family
falsification before widening benchmark scope.

## Current Story

The strongest surviving story is not "latent transfer works" yet. It is:

- target-alone on frozen SVAMP32 exact IDs: `8/32`
- C2C teacher on the same IDs: `16/32`
- target_self_repair: `14/32`, with `3/10` C2C-only recoveries and no target
  losses
- query_pool_transport matched row: `9/32`, with `1/10` C2C-only recovery and
  one target loss

That makes target_self_repair the hard comparator for any claimed positive
method on this slice.

## Exact Blocking Gap

A candidate must recover C2C-only teacher wins using matched source information
that target_self_repair, zero-source, and shuffled-source do not recover. The
current query-pool row does not clear that bar.

## What Changed

Added `scripts/analyze_svamp32_paper_gate.py`, a reusable gate script that
consumes a C2C teacher-probe JSON artifact and applies the current SVAMP32
paper threshold:

- minimum `16/32` correct
- at least `+1` versus target_self_repair
- at least `5/10` C2C-only recoveries
- at least `2` C2C-only recoveries unique versus target_self_repair
- at most `1` target-correct loss
- at most `1` matched C2C-only recovery retained by any source control

Missing target_self_repair or missing source-control overlap is treated as a
gate failure, not as a zero-count control.

Updated gate behavior:

- if `--target-set-json` is provided, the gate also requires clean residual
  C2C-only recoveries from the target set
- by default the minimum clean residual and clean source-necessary counts are
  read from `required_clean_residual_to_clear_gate_if_preserving_self`

Applied it to:

- `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.json`

Gate result:

- `no_candidate_passes_target_self_repair_gate`
- clean-target gate result:
  `no_candidate_passes_target_self_repair_gate`

## Evidence

| Candidate | Correct | Delta vs self-repair | C2C-only | Unique vs self-repair | Max source-control retained | Target losses | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| query_pool_matched | `9/32` | `-5` | `1/10` | `1` | `1` | `1` | fail |

The only matched C2C-only recovered ID is `575d7e83d84c1e67`. It is unique
versus target_self_repair but retained by both zero-source and shuffled-source
controls, so it is not sufficient evidence of source-specific communication.

With the clean residual target set enabled, query_pool_matched recovers `0/6`
clean residual C2C-only IDs and `0/6` clean source-necessary IDs. It fails
`min_clean_residual_recovered` and `min_clean_source_necessary`.

## Top 3 Next Moves

1. C2C-distilled conditional innovation fuser. Train a small target-conditioned
   fuser only on the C2C-over-target_self_repair residual, not on full target
   generation. This matters because it attacks the exact blocker. It might fail
   because there are only `10` C2C-only IDs on SVAMP32. Minimal experiment:
   train on source/target/C2C traces and promote only if it clears the gate
   above.

2. Q-Former/Perceiver-style query bottleneck connector. Replace static KV
   transport with a frozen-backbone learned query interface, following the
   local connector memos in `references/443_query_resampler_connector_refs.md`
   and `references/301_multimodal_diffusion_projector_refs.md`. The closest
   primary-source analogues are BLIP-2 / Q-Former, Flamingo-style gated
   cross-attention, and Perceiver IO:
   https://arxiv.org/abs/2301.12597,
   https://arxiv.org/abs/2204.14198,
   https://arxiv.org/abs/2107.14795.

3. Cross-family falsification design after same-pair gate clears. Keep the
   exact same artifact contract, then run one strict different-family pair only
   after a same-pair candidate beats target_self_repair. This helps prevent
   early benchmark drift; it might fail by showing the method is Qwen-family
   specific, which would be useful falsification.

## Decision

Alive:

- C2C-distilled conditional innovation fuser
- learned query-bottleneck connector with target-conditioned source querying

Saturated:

- selector/runtime swaps on the current query-innovation-resampler checkpoint
- query_pool_transport as a paper method on SVAMP32

Blocked:

- any claim that does not beat target_self_repair on the frozen exact IDs
- any row whose C2C-only wins are retained by zero or shuffled source controls

## Next Exact Gate

Implement one C2C-distilled conditional innovation fuser or learned
query-bottleneck connector, then run matched / zero-source / shuffled-source /
target_self_repair on the frozen SVAMP32 exact-ID surface. Promote only if the
new `scripts/analyze_svamp32_paper_gate.py` verdict is
`candidate_passes_target_self_repair_gate`.
