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

The aggregate has `32` rows: `22` pass rows and `10` fail or near-miss rows.
The strongest systems result remains the byte-rate frontier: a `2` byte
diagnostic packet reaches oracle accuracy on the frozen core and holdout
surfaces, while structured JSON/free-text relays need `21`/`17` bytes,
query-aware diagnostic-span compression needs `14` bytes, and full hidden-log
relay is `183.2x-186.7x` larger. Matched-byte text at the packet rate stays at
target-only accuracy.

The learned packet story remains positive in scoped settings:

- `6` byte slot/no-intercept scalar packets pass the 5-seed same-codebook gate
  at `1.000` accuracy with clean controls.
- Remapped slot codebooks remain positive but weaker: scalar remap rows are
  `0.463-0.508` accuracy versus target-only `0.250`.
- Canonical RASP gives a `4` byte candidate-relative transport. It passes the
  larger worst-remap slice (`0.442` vs scalar `0.361` and target `0.250`) but
  the seven-remap bootstrap remains a near miss.
- Model-emitted source packets pass on Qwen3.5 small models, Gemma 4 E2B, and
  Granite 3.3 2B strict-prompt rows, with Granite exposing a lower packet-valid
  floor (`0.537`).
- Qwen3 target-decoder CPU n64 rows are positive (`0.656` core, `0.719`
  holdout), but this is still too small to close the hand-coded-decoder
  reviewer objection.

## Failures Kept In The Artifact

The aggregate explicitly keeps the main failed rows:

- Canonical RASP core-to-holdout fails (`0.207` vs target `0.250`) and controls
  are not clean.
- Canonical RASP holdout-to-core passes (`0.492` vs target `0.250`), proving the
  cross-family result is asymmetric rather than absent.
- The consistency-posterior packet is pruned as a cross-family fix: the larger
  core-to-holdout row reaches only `0.354` and is matched by an order-mismatch
  control (`0.355`).
- Granite raw-log/no-trace emits no valid packets and stays at target-only.

## Paper Implication

This supports three defensible contributions:

1. A source-private packet benchmark and control protocol that distinguishes
   source evidence from target priors and matched-byte text.
2. A compact packet method family with strong same-family/remap evidence and
   model-emitted packet rows on current small local models.
3. A systems byte-rate frontier showing large communication savings over
   structured text and hidden-log relay.

It does not support a full bidirectional cross-family latent-transfer claim.
Endpoint TTFT/throughput remains unmeasured, so the paper should currently claim
byte-rate and local decode-cost evidence, not serving-latency superiority.

## Next Gate

The highest-priority reviewer-facing gate is a CPU/MPS target-decoder
replication at `n=256` or larger, followed by a diagnostic-code
remap/paraphrase stress table. These address the two strongest objections:
whether the receiver is hand-coded, and whether the method is only a brittle
coded-label protocol.
