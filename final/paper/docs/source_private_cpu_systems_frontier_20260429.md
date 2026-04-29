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

The aggregate now has `101` rows after adding learned Wyner-Ziv packet evidence,
bidirectional cross-family falsification rows, protected residual codec
ablation rows, progress-enabled target-decoder receiver rows, and the static
anchor-relative sparse packet smoke, a learned target-preserving receiver smoke,
Mac endpoint-proxy rows, and paired endpoint uncertainty. The
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
- Static anchor-relative sparse innovation packets pass holdout-to-core at some
  budgets (`0.496` at 2 bytes and `0.373` at 8 bytes with clean controls), but
  fail core-to-holdout. This prunes the shallow hypothesis that relative sparse
  coordinates alone solve bidirectional cross-family transfer.
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
- Mac endpoint-proxy rows now pass on core and holdout at `n=8` and `n=16`
  under the canonical prompt, and at `n=16`/`n=32` under a second `audit`
  prompt paraphrase. The strongest prompt-robust rows are `n=32` audit: core
  packet `0.719` versus target `0.250` and matched-byte text `0.281`; holdout
  packet `0.844` versus target and matched-byte text `0.312`. Query-aware text
  is `7.0x` larger, full hidden-log relay is `183.2x-186.7x` larger, and
  full-log p50 TTFT is `+157.4 ms` to `+163.4 ms` relative to the packet at
  `n=32` audit. A deliberately under-specified `terse` prompt fails on core
  (`0.250` packet, equal to target), so prompt contract specificity remains a
  real limitation.
- Payload-gated rescoring demotes the audit strict-control rows from pass to
  near-miss because the receiver sometimes emits unrelated diagnostic codes.
  The method signal remains strong: core `n=64` audit has packet `0.750`,
  target `0.250`, best source-destroying control `0.203`, and full-log p50 TTFT
  `+260.2 ms`, but packet valid rate is only `0.781`.
- A new label-strict receiver prompt passes both frozen surfaces at `n=16`,
  `n=32`, and `n=64` with exact candidate-label outputs. At `n=64`, core
  reaches packet `0.703`, target `0.250`, best source-destroying control
  `0.250`, valid rate `1.000`, and full-log p50 TTFT `+217.2 ms`; holdout
  reaches packet `0.672`, target `0.250`, best control `0.250`, valid rate
  `1.000`, and full-log p50 TTFT `+192.7 ms`. This is the current strongest
  endpoint receiver row because it avoids the parser-risk objection.
- The paired uncertainty artifact promotes the `n=64` label-strict endpoint
  rows from point estimates to a reviewer-facing statistical gate. Across core
  and holdout, the minimum packet-vs-target lower CI is `+0.297`, the minimum
  packet-vs-best-source-destroying-control lower CI is `+0.297`, and the
  minimum strict-label packet-vs-target lower CI is `+0.281`, with valid rate
  `1.000`. Query-aware diagnostic text is treated as a rate/quality comparator:
  it is accuracy-comparable but costs `14` bytes versus the packet's `2` bytes.
- The `n=160` label-strict scale-up now passes on both core and holdout. Core
  reaches packet `0.675`, strict-label packet `0.662`, target/matched-byte text
  `0.250`, and best source-destroying control `0.250`. Holdout reaches packet
  `0.688`, strict-label packet `0.675`, target/matched-byte text `0.250`, and
  best source-destroying control `0.250`. The combined paired uncertainty row
  gives minimum lower CIs of `+0.350` versus target and best control, and
  `+0.338` for strict-label packet versus target. This clears the local medium
  endpoint rung, with the same caveat that it is still CPU endpoint-proxy
  timing rather than server throughput.
- The learned target-preserving receiver is now seed-stable on the
  same-distribution all-family surface at `8` bytes: three seeds pass with
  matched mean `0.749`, matched minimum `0.514`, max destructive control
  `0.283`, and minimum matched-vs-control delta `+0.230`. The `4` byte version
  is promising but unstable (`2/3` seeds pass; one seed falls to `0.328`).
  Core-to-holdout transfer at `8` bytes fails (`0.453` matched, `0.250`
  target, `0.311` best destructive control, `0.809` oracle), and removing raw
  candidate feature coordinates worsens the held-out result. This directly
  addresses the hand-designed-decoder objection in-distribution but is not yet
  a cross-family method claim.

## Failures Kept In The Artifact

The aggregate explicitly keeps the main failed rows:

- Canonical RASP core-to-holdout fails (`0.207` vs target `0.250`) and controls
  are not clean.
- Canonical RASP holdout-to-core passes (`0.492` vs target `0.250`), proving the
  cross-family result is asymmetric rather than absent.
- Learned WZ cross-family also fails bidirectionally under the strict all-budget
  rule: `core_to_holdout` is below target at every budget and explained by
  controls; `holdout_to_core` only passes at `6` bytes.
- Static AR-SIP repeats the same asymmetry: holdout-to-core has clean positive
  rows, but core-to-holdout stays at or below target-only and some anchor/random
  controls dominate the matched packet.
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
- The audit endpoint prompt is now kept as a near-miss/failure after payload-
  gated rescoring: accuracy and controls are strong, but the receiver does not
  meet the `0.95` valid-output gate because it sometimes emits diagnostic codes
  not present in the transmitted payload.
- The learned receiver core-to-holdout row is kept as a failed row. It remains
  above target, but it misses strict promotion because the best destructive
  control reaches `0.311`, the matched-control gap is only `+0.143`, and the
  diagnostic oracle falls below `0.95`.

## Paper Implication

This supports three defensible contributions:

1. A source-private packet benchmark and control protocol that distinguishes
   source evidence from target priors and matched-byte text.
2. A compact packet method family with strong same-family/remap evidence,
   principled codec ablations, frozen target-decoder rows, and model-emitted
   packet rows on current small local models.
3. A systems byte-rate frontier showing large communication savings over
   structured text and hidden-log relay.
4. A learned target-preserving receiver that converts compact source packets
   into candidate scores and is seed-stable at `8` bytes on the all-family
   surface, with an explicit held-out-family failure documenting the current
   boundary.

It does not support a full bidirectional cross-family latent-transfer claim.
Endpoint-proxy TTFT/E2E telemetry is now measured locally on CPU, including one
passing prompt paraphrase, one failed under-specified prompt, and strict
source-destroying endpoint controls. The paper can claim a Mac-local byte/TTFT
frontier with prompt-contract stress, but it still should not claim serving-
throughput superiority until a real vLLM/OpenAI-compatible endpoint run exists.

## Next Gate

The highest-priority reviewer-facing gate is now a true server-side
TTFT/throughput run when NVIDIA GPUs are available. On the Mac, the next
highest-value technical branch is a family-invariant learned receiver:
anchor-relative/codebook features or fold-heldout calibration at `8` bytes,
because the raw candidate-coordinate receiver is same-distribution stable but
does not clear held-out-family transfer.
