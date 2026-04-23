# SVAMP32 Innovation Target Set

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The current paper story is still a gated positive-method story:
SVAMP32 exposes C2C teacher headroom, but a publishable method must recover
source-specific C2C-only wins beyond target_self_repair and source controls.

## Current Story

The active comparator stack is:

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`, with `3/10` C2C-only recoveries
- query_pool_transport matched row: `9/32`, with `1/10` C2C-only recovery
  retained by zero-source and shuffled-source

The new target-set artifact asks whether there is still enough residual C2C
headroom after removing IDs already explained by target_self_repair,
source-alone/text relay, and source controls.

## Evidence

Result:

- `residual_headroom_available`

Clean C2C-only residual target IDs:

- `13cb77b698eeadb5`
- `1d50b408c8f5cd2c`
- `2de1549556000830`
- `6e9745b37ab6fc45`
- `aee922049c757331`
- `e3ab8666238a289e`

Excluded from the residual target set:

- target_self_repair already recovers `4c84ebf42812703b`,
  `4d780f825bb8541c`, and `de1bf4d142544e5b`
- source-alone, zero-source, shuffled-source, and current query_pool matched
  all explain `575d7e83d84c1e67`

If a connector preserves the `14/32` target_self_repair row, it only needs `2`
clean residual C2C-only wins to satisfy the current hard gate, and the oracle
target_self_repair plus C2C-teacher ceiling is `21/32`.

The paper gate now accepts this target-set artifact directly via
`--target-set-json` and requires clean residual recovery when scoring a
candidate. On the existing query_pool_transport row, the stricter clean-target
gate reports:

- clean residual recovered: `0/6`
- clean source-necessary recovered: `0/6`
- added failing criteria: `min_clean_residual_recovered`,
  `min_clean_source_necessary`
- verdict: `no_candidate_passes_target_self_repair_gate`

The target set was regenerated from a strict exact-32 probe. The
target_self_repair and selected_route_no_repair controls are no longer scored
by implicitly subsetting the original 70-row repair artifact at promotion time.

## Top 3 Next Moves

1. C2C-distilled conditional innovation fuser. Train only on the clean residual
   target set, with target-correct examples weighted as no-op preservation
   controls. This directly attacks the blocker. It might fail by memorizing the
   six IDs or by becoming another target-self-repair prior.

2. Wyner-Ziv-style query bottleneck connector. Treat the target cache as decoder
   side information and make the source emit only the missing innovation code.
   This is the strongest lateral idea from the subagents, but it is a larger
   implementation branch.

3. Source-necessity replay ablation. For any candidate row, run matched source,
   post-bridge zero, zero-source, deranged shuffled-source with at least two
   salts, and wrong-source controls. This prevents aggregate accuracy from
   hiding target-cache effects.

## Decision

Alive:

- conditional innovation fuser
- learned query-bottleneck connector
- source-necessity replay as the first ablation for either branch

Saturated:

- selector/runtime changes on the current query-innovation-resampler checkpoint
- any claim that counts `575d7e83d84c1e67` as evidence of source
  communication

Blocked:

- implementation of a candidate that preserves target_self_repair while adding
  clean residual C2C-only wins

## Next Exact Gate

Build the smallest conditional innovation connector using the six clean
residual target IDs as the positive supervision surface. Run matched,
post-bridge-zero if available, zero-source, shuffled-source with at least two
salts, and target_self_repair. Promote only if
`scripts/analyze_svamp32_paper_gate.py` returns
`candidate_passes_target_self_repair_gate`.
