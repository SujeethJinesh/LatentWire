# HellaSwag Non-Science Fixed-Packet Gate

Date: 2026-05-01

## Status

This branch is now the strongest non-science public-benchmark positive slice,
but it is not yet an ICLR-final headline until we run full validation and
HellaSwag-specific controls.

2026-05-01 amendment: the HellaSwag-specific control suite now weakens this
framing. The packet clears metadata/activity controls, destructive controls,
and the original same-byte choice-prefix text relay, but it does not beat a
stricter one-byte source-label-copy control. Treat HellaSwag as a live
method-improvement surface until a non-label-copy packet clears that gate.

## Why We Ran This

CommonsenseQA showed real non-science source signal, but the same-byte text
control could cheaply copy the selected short answer. HellaSwag is a
four-choice adversarial sentence-completion benchmark with longer endings, so
it is a better test of whether the source packet communicates useful
continuation evidence rather than only a top answer label.

In plain terms: one model reads the context and four possible endings, then
sends a tiny two-byte hint. The receiving side uses only that hint plus the
public answer choices. We check whether that tiny hint helps more than a tiny
text message of the same size or broken/shuffled packets.

## Bridge Contract

Artifact:
`results/source_private_hellaswag_bridge_contract_20260501/`

- labeled train/validation rows: `39905/10042`
- public test labels: unavailable/empty, so test is not used
- train/validation overlap: `0`
- forbidden metadata/source fields include `label`, `activity_label`,
  `source_id`, `split`, `split_type`, and `ind`
- frozen validation slice: `official_splits/hellaswag_validation_first1024.jsonl`

## Runs

### 12B frozen-512 smoke

Artifact:
`results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation512_12b/`

- matched/target/same-byte text: `0.482/0.252/0.475`
- matched minus target: `+0.230`
- matched minus same-byte text: `+0.008`
- best destructive control: `shuffled_source_packet` at `0.273`
- pass gate: `false`

Interpretation: the source signal is strong, but at `12B` the text relay is
too close. This is exactly the failure mode reviewers would object to, so the
rate frontier had to move downward.

### 2B/3B/4B/6B frozen-512 rate frontier

Cached seed-stability runs reused the same source choices and varied only the
packet budget.

- `2B`: `3/3` seeds pass; matched mean `0.482`; same-byte text `0.385`;
  minimum text lift `+0.096`
- `3B`: `3/3` seeds pass; same-byte text `0.383`; minimum text lift `+0.098`
- `4B`: `3/3` seeds pass; same-byte text `0.408`; minimum text lift `+0.074`
- `6B`: `3/3` seeds pass; same-byte text `0.443`; minimum text lift `+0.039`

Interpretation: lower byte budgets are better here because they starve the
visible text baseline more than the source-private packet.

### 2B frozen-1024 promoted slice

Artifacts:

- `results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/`
- `results/source_private_hellaswag_seed_stability_20260501_qwen05_hashed_validation1024_2b_5seed/`

Fixed run:

- matched/target/same-byte text: `0.460/0.233/0.386`
- matched minus target: `+0.227`
- matched minus same-byte text: `+0.074`
- matched minus best destructive: `+0.203`
- paired CI95 low versus target: `+0.189`
- pass gate: `true`

Five-seed stability:

- pass count: `5/5`
- matched mean/min/max: `0.461/0.460/0.462`
- target: `0.233`
- same-byte text: `0.386`
- minimum lift over same-byte text: `+0.074`
- minimum lift over best destructive control: `+0.188`
- maximum candidate-derangement accuracy: `0.175`
- minimum paired CI95 lower bound versus target: `+0.186`

## Systems Readout

The promoted `1024`-row run records:

- raw payload: `2B`
- framed record with header/CRC: `5B`
- single-request cacheline/DMA accounting: `64B/128B`
- batch-64 cacheline/DMA accounting: `5B/6B` per request
- source scoring: about `500s` on this Mac CPU for `1024` validation rows
- peak process RSS: `6859.98 MiB`
- source text exposed: `false`
- source KV exposed: `false`

This is a strong byte-accounting result, not a native serving claim. The ICLR
systems row still needs NVIDIA/vLLM TTFT, TPOT, goodput, peak GPU memory, and
HBM/cache traffic measurements.

## Decision

Promote as:

- non-science diagnostic evidence
- strict separation from the original same-byte choice-prefix text relay at
  `2B`
- systems/rate-frontier evidence for very small packets
- metadata/activity leakage falsification after the follow-up control suite

Do not yet claim:

- full HellaSwag validation result
- non-label-copy HellaSwag communication
- native GPU systems win
- universal latent communication
- superiority to KV-cache communication methods such as C2C/KVComm

## Next Gate

Run full HellaSwag validation (`10042` rows) with the `2B` packet and then add
HellaSwag-specific controls:

- activity-label-only packet
- context-only packet
- ending-only packet
- same-activity shuffled source packet
- context-shuffled source packet
- candidate-prefix text relay

If the full validation row preserves a `>=0.02` same-byte-text gap and clean
destructive controls, HellaSwag becomes the best non-science ICLR headline
benchmark.

Updated gate after label-copy controls: before full validation, first build a
source-error repair packet that beats the one-byte source-label-copy control by
at least `0.02` on the frozen `1024` validation slice. The score/margin
headroom probe found a large source top-2 oracle (`0.734` heldout) but no
promotion from margin alone, so the next method should learn when to distrust
the source top choice rather than only quantizing top-vs-runner-up confidence.
