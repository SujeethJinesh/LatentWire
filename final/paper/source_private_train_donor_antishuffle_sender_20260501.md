# Source-Private Train-Donor Anti-Shuffle Sender, 2026-05-01

## Status

- paper readiness: materially stronger ICLR evidence, but still not a
  comfortable full-paper submission.
- current story: a train-only sender can select byte-scale source-private
  packets that help a train-only receiver cross-family while destructive
  controls stay near the target-only band.
- exact blocker: the method now needs a locked validation-based byte-selection
  rule, a public benchmark bridge, and native GPU/vLLM systems rows before the
  ICLR claim is complete.

## Method

This branch replaces the eval-donor anti-shuffle diagnostic with sampled
training donors stored in the packet-builder state. For each eval example, the
sender samples nonoverlap train donors and scores candidate atoms by matched
receiver utility minus donor/null/common-source penalties:

```text
score(atom) =
  carrier(atom) * positive_receiver_margin(atom)
  - lambda * train_donor_strength(atom) * positive_receiver_margin(atom)
  - mu * permuted_null_margin(atom)
  - gamma * train_generic_mass(atom)
```

The promoted setting uses `12` train donors, donor weight `0.50`, null weight
`0.75`, generic weight `0.10`, source identity weight `0.75`, and the sum
carrier. The sender still does not use the gold answer label at eval time; it
selects atoms against the receiver's proposed target candidate.

Lay explanation: the sender no longer learns by looking at the wrong-source
example used in evaluation. Instead, it asks: "does this clue help the real
source more than clues from several unrelated training sources, and does it
stop helping when the receiver is permuted?"

## Evidence

Artifacts:

- `.debug/iclr_20260501_train_donor_antishuffle_seed47_n128_dw0p50_gw0p10_budget14/`
- `.debug/iclr_20260501_train_donor_antishuffle_seed53_n128_sum_budget12/`
- `.debug/iclr_20260501_train_donor_antishuffle_seed59_n128_sum_budget12/`
- `.debug/iclr_20260501_train_donor_antishuffle_seed47_n512_budget14/`
- `.debug/iclr_20260501_train_donor_antishuffle_seed53_n512_budget12_14/`
- `.debug/iclr_20260501_train_donor_antishuffle_seed59_n512_budget12_14/`
- `.debug/iclr_20260501_mac_packet_ring_transport_microbench/`
- `.debug/iclr_20260501_serving_slo_envelope/`
- `.debug/iclr_20260501_systems_rate_assumption_frontier/`

Train-donor anti-shuffle passes the n128 cross-family seed-repeat gate, with a
12-14 byte frontier:

| Seed | Budget | Direction | N | Candidate | Base | Target | Best control | Pass |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 47 | 14 | core_to_holdout | 128 | 0.750 | 0.625 | 0.250 | 0.273 | yes |
| 47 | 14 | holdout_to_core | 128 | 0.648 | 0.500 | 0.250 | 0.258 | yes |
| 53 | 12 | core_to_holdout | 128 | 0.750 | 0.625 | 0.250 | 0.258 | yes |
| 53 | 12 | holdout_to_core | 128 | 0.648 | 0.500 | 0.250 | 0.258 | yes |
| 59 | 12 | core_to_holdout | 128 | 0.750 | 0.625 | 0.250 | 0.250 | yes |
| 59 | 12 | holdout_to_core | 128 | 0.648 | 0.500 | 0.250 | 0.266 | yes |

The larger n512 gate now passes across seeds `47/53/59` under the 12-14B
frontier:

| Seed | Budget | Direction | N | Candidate | Base | Target | Best control | CI95 low vs base | Pass |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 47 | 14 | core_to_holdout | 512 | 0.750 | 0.625 | 0.250 | 0.273 | 0.100 | yes |
| 47 | 14 | holdout_to_core | 512 | 0.652 | 0.500 | 0.250 | 0.254 | 0.123 | yes |
| 53 | 12 | core_to_holdout | 512 | 0.750 | 0.625 | 0.250 | 0.260 | 0.098 | yes |
| 53 | 12 | holdout_to_core | 512 | 0.652 | 0.500 | 0.250 | 0.256 | 0.119 | yes |
| 53 | 14 | holdout_to_core | 512 | 0.652 | 0.500 | 0.250 | 0.254 | 0.119 | yes |
| 59 | 12 | core_to_holdout | 512 | 0.750 | 0.625 | 0.250 | 0.266 | 0.098 | yes |
| 59 | 12 | holdout_to_core | 512 | 0.652 | 0.500 | 0.250 | 0.252 | 0.119 | yes |
| 59 | 14 | core_to_holdout | 512 | 0.750 | 0.625 | 0.250 | 0.268 | 0.096 | yes |
| 59 | 14 | holdout_to_core | 512 | 0.652 | 0.500 | 0.250 | 0.264 | 0.123 | yes |

Seed 53 at 14B fails only `core_to_holdout` because the private-random control
reaches `0.287`, just above the strict target band. This is why the result
should be framed as a 12-14B frontier with a predeclared byte-selection rule,
not as a single fixed-budget claim.

Same-family remains unpromoted because `structured_text_matched` reaches
`0.3125` against target-only `0.250`. The cross-family story is now stronger
than the eval-donor diagnostic because the passing sender contrast is trained
only from training donors.

## Weakened Branches

- eval-donor anti-shuffle: useful diagnostic, not final headline.
- train-mean anti-shuffle: too conservative in `holdout_to_core`.
- intersection/min carrier: suppresses random-source controls in
  `core_to_holdout`, but loses hard-direction lift.
- sum carrier at a single fixed budget: seed-dependent; the defensible claim is
  a 12-14 byte frontier, not one magic byte count.

## Systems Readout

Mac-local systems artifacts pass, with explicit non-claims:

- packet-ring microbench:
  `packet_batch64_p95_ns_per_request=0.674`, `line_bytes/request=5.0`,
  `DMA_bytes/request=6.0`, full private log `9.26x` packet p50, KV byte floor
  `644.9x` packet p50.
- serving SLO envelope: passes as an accounting envelope, while GPU TPOT,
  goodput, and accelerator counters remain unmeasured.
- systems rate frontier: passes as an assumption-aware comparison, not as a
  native win over C2C, KVComm, TurboQuant, QJL, KIVI, or KVQuant.

## Novelty Boundary

Safe claim: this is a train-only source-private packet-selection method for
cross-model candidate disambiguation with destructive controls and byte-level
systems accounting.

Unsafe claims: new source-coding theory, general latent transfer, native KV
cache compression superiority, or production GPU serving speedups.

Primary-source anchors are collected in
`references/564_train_donor_antishuffle_refs_20260501.md`.

## Locked Frontier Follow-Up

Artifact:
`results/source_private_train_donor_antishuffle_locked_rate_frontier_20260501/`.

The locked validation readout passes under the per-seed policy: seed `47`
selects `14B`, seeds `53/59` select `12B`, and all `6/6` selected n512
cross-family rows pass. The weakest selected CI95 lower bound versus the base
packet is `0.098`, and the largest selected best control is `0.273`.

This weakens the hand-picked-budget concern, but does not fully close it: the
global fixed-budget policy still fails. The final ICLR version should either
make one budget pass across seeds or use an example-level train-only diagnostic
to choose `12B` versus `14B` without seeing eval labels.

## Next Gate

Run the larger train-only validation rule, then bridge to one public benchmark
slice and prepare GPU/vLLM TTFT/TPOT/goodput plus HBM/PCIe/NVLink counters.
