# Conditional PQ ICLR / COLM_v2 Status

- COLM_v2 readiness: `scoped_positive_method_ready_for_writeup`
- ICLR readiness: `blocked_by_cross_family_or_broader_benchmark_positive_gate`
- same-family n500 pass rows: `16/16`
- budget-2 pass rows: `4/4`
- less-diagnostic pass rows: `8/8`
- original cross-family pass rows: `0/2`
- schema-grid cross-family pass rows: `0/28`

## Story

Conditional PQ innovation is the current strongest LatentWire_v2 branch: a source-private conditional innovation packet is product-quantized into 2-4 payload bytes and decoded against target/public candidate side information. The defensible claim is shared-schema source-private communication with utility-per-byte accounting, not broad unseen-family latent transfer.

## Evidence

- min decisive source accuracy: `0.996`
- max decisive best-control accuracy: `0.302`
- min decisive CI95 low vs best control: `0.6579999923706055`
- cross-family max source-minus-control: `0.007812000000000041`
- cross-family max CI95 low vs control: `0.0`
- method record bytes range: `[5, 7]`
- method payload bytes: `[2, 4]`
- min KV floor record bytes: `21504`
- native GPU claim allowed: `False`

## Contributions

- `source_private_conditional_innovation_packet`: alive_positive_shared_schema; needs show broader benchmark or held-out-family transfer.
- `strict_destructive_controls`: strong_for_current_synthetic_surface; needs add paper-facing source-index/rank/score and same-byte-text comparators where meaningful.
- `utility_per_byte_systems_accounting`: ready_for_mac_local_packet_boundary_claim; needs native GPU/C2C/KV measurements before throughput, HBM, PCIe, NVLink, energy, or serving claims.

## Submission Gap

COLM_v2 needs the scoped conditional-PQ table, systems waterfall, and limitations integrated. ICLR needs one more positive gate: either held-out-family/public-conditioned residual codebooks, a less synthetic benchmark, or a learned receiver that keeps the same byte and exposure advantages.

## Next Exact Gate

- name: `public_conditioned_conditional_pq_resurrection_gate`
- decision surface: n256 bidirectional held-out family first, then n500/remap repeat if positive
- method: replace static public bases with target-public conditioned residual/codebook decoding while keeping the same source-private conditional innovation packet interface
- pass bar: source minus best destructive/shortcut control >= +0.10 with positive paired CI95 low
- required controls: `target_only`, `answer_masked_source`, `constrained_wrong_row_source`, `same_source_choice_wrong_row`, `candidate_roll_or_deranged_public_basis`, `permuted_codes`, `random_same_byte`, `opaque_slot_or_deranged_basis`, `source_index_rank_score_comparators_when_not_answer_oracles`, `same_byte_visible_text`

## Claim Boundaries

- Do not claim unseen-family transfer: both original cross-family rows and the 28-row basis/schema grid fail.
- Do not claim product quantization, rotations, or side-information coding as standalone novelty.
- Do not claim GPU throughput, HBM savings, latency, energy, PCIe, or NVLink wins from Mac-local packet accounting.
- Do frame C2C/KV methods as dense or cache-sharing baselines with different exposure and byte regimes.

## References

- `references/543_conditional_pq_innovation_refs_20260430.md`
- `references/544_conditional_pq_semantic_schema_systems_refs_20260430.md`
- `references/727_srp_competitor_basis_quant_benchmark_lateral_refresh_20260504.md`
- `references/728_event_triggered_defer_syndrome_packet_refs_20260504.md`
- `references/739_conditional_pq_iclr_colm_v2_status_refs_20260504.md`
