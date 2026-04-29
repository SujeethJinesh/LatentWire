# Source-Private CPU Systems Frontier

- date: `2026-04-29`
- artifact: `results/source_private_cpu_systems_frontier_20260429/`
- script: `scripts/build_source_private_cpu_systems_frontier.py`
- test: `tests/test_build_source_private_cpu_systems_frontier.py`
- scale rung: medium confirmation / systems aggregation

## Purpose

This memo consolidates the source-private packet evidence into one
non-cherry-picked CPU artifact. The table includes positive rows, near misses,
and failed rows so the paper story can claim only what the evidence supports.

## Headline

The aggregate now has `63` rows after adding learned Wyner-Ziv packet evidence,
bidirectional cross-family falsification rows, protected residual codec
ablation rows, and progress-enabled target-decoder receiver rows. The
strongest systems result
remains the byte-rate frontier: a `2` byte diagnostic packet reaches oracle
accuracy on the frozen core and holdout surfaces, while structured
JSON/free-text relays need `21`/`17` bytes, query-aware diagnostic-span
compression needs `14` bytes, and full hidden-log relay is `183.2x-186.7x`
larger. Matched-byte text at the packet rate stays at target-only accuracy.

The learned packet story remains positive in scoped settings:

- `6` byte slot/no-intercept scalar packets pass the 5-seed same-codebook gate
  at `1.000` accuracy with clean controls.
- Learned Wyner-Ziv/source-side-information scalar packets pass `9/9` remapped
  slot-codebook rows across `2/4/6` bytes, with accuracy `0.418-0.508` versus
  target-only `0.250`.
- Protected rotated residual packets are source-control positive on `9/9`
  remapped rows and improve the 2-byte scalar row on remaps `101` and `107`,
  but fail strict promotion because p50 decode latency is `3.56-7.33 ms` rather
  than `<2 ms`, and two 6-byte rows trail scalar WZ by more than `0.02`.
- Remapped slot codebooks remain positive but weaker: scalar remap rows are
  `0.463-0.508` accuracy versus target-only `0.250`.
- Canonical RASP gives a `4` byte candidate-relative transport. It passes the
  larger worst-remap slice (`0.442` vs scalar `0.361` and target `0.250`) but
  the seven-remap bootstrap remains a near miss.
- Model-emitted source packets pass on Qwen3.5 small models, Gemma 4 E2B, and
  Granite 3.3 2B strict-prompt rows, with Granite exposing a lower packet-valid
  floor (`0.537`).
- Qwen3 target-decoder CPU n64 rows are positive (`0.656` core, `0.719`
  holdout), but those older rows did not have the progress instrumentation.
- Progress-enabled Qwen3 target-decoder all-control rows now pass at `n=32` on
  both core (`0.688` vs target/control `0.250`) and holdout (`0.750` vs target
  `0.250`, best control `0.281`). This is the strongest current answer to the
  hand-coded-decoder objection: a frozen target model reads the 2-byte packet
  while shuffled/random/matched-byte structured text controls stay near the
  target prior.

## Failures Kept In The Artifact

The aggregate explicitly keeps the main failed rows:

- Canonical RASP core-to-holdout fails (`0.207` vs target `0.250`) and controls
  are not clean.
- Canonical RASP holdout-to-core passes (`0.492` vs target `0.250`), proving the
  cross-family result is asymmetric rather than absent.
- Learned WZ cross-family also fails bidirectionally under the strict all-budget
  rule: `core_to_holdout` is below target at every budget and explained by
  controls; `holdout_to_core` only passes at `6` bytes.
- Protected residual packets are kept as near-miss/fail rows rather than a
  promoted codec method: they beat source controls but miss the strict latency
  and high-budget scalar-preservation thresholds.
- A short-decode target-decoder diagnostic is kept as a failed harness row:
  with `max_new_tokens=8`, Qwen emits only a shared candidate-label prefix and
  valid prediction rate collapses to `0.000`. The valid all-control receiver
  rows therefore use the 24-token cap needed to emit complete labels.
- A direct MPS probe still fails before prediction with the known Apple MPS
  matmul shape error, so the receiver evidence remains CPU-only on this Mac.
- The consistency-posterior packet is pruned as a cross-family fix: the larger
  core-to-holdout row reaches only `0.354` and is matched by an order-mismatch
  control (`0.355`).
- Granite raw-log/no-trace emits no valid packets and stays at target-only.

## Paper Implication

This supports three defensible contributions:

1. A source-private packet benchmark and control protocol that distinguishes
   source evidence from target priors and matched-byte text.
2. A compact packet method family with strong same-family/remap evidence,
   principled codec ablations, frozen target-decoder rows, and model-emitted
   packet rows on current small local models.
3. A systems byte-rate frontier showing large communication savings over
   structured text and hidden-log relay.

It does not support a full bidirectional cross-family latent-transfer claim.
Endpoint TTFT/throughput remains unmeasured, so the paper should currently claim
byte-rate and local CPU decode-cost evidence, not serving-latency superiority.

## Next Gate

The highest-priority reviewer-facing gate is now either (1) an endpoint
TTFT/E2E latency frontier for packet versus structured/full-log relay, or (2) a
new anchor-relative sparse innovation packet that directly targets the failed
bidirectional cross-family rows. The receiver objection is weakened by the n32
all-control frozen-model pass, but not fully closed until n160/n256 all-control
rows are available.
