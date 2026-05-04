# Source-Private Systems Boundary Split

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible; ICLR full is still
blocked by either a positive learned method that beats the fixed packet, or
native NVIDIA systems rows that show real TTFT/TPOT/HBM/goodput wins.

Current story: LatentWire is strongest as a source-private fixed-byte packet
protocol with unusually strict control gates. The systems contribution is an
interface boundary: count the object that crosses the wire separately from the
source-side work needed to compute it.

Exact blocking gap: the Mac-local artifacts now make the boundary explicit, but
they do not prove native serving speedups against C2C, KVComm, DroidSpeak, QJL,
TurboQuant, vLLM, or SGLang.

## Gate

Implemented and rebuilt the systems-boundary split in:

- `scripts/build_source_private_byte_amplification_ablation.py`
- `scripts/build_source_private_systems_boundary_figure_table.py`

Artifacts:

- `results/source_private_byte_amplification_ablation_split_20260504/`
- `results/source_private_systems_boundary_figure_table_split_20260504/`

The gate adds two explicit LatentWire rows:

- `latentwire_packet_cached_source`: the communication object only.
- `latentwire_packet_end_to_end_source_scoring`: same packet bytes, with
  source scoring disclosed separately when phase traces exist.

It also adds non-private fp16 source-score and source-logit vector floors so
reviewers cannot collapse byte-small source-state relays into the packet claim.

## Evidence

The byte-amplification artifact passes. It covers 4 benchmark rows and emits 48
interface rows. For each benchmark, the cached-source packet row and
end-to-end source-scoring row are paired and prediction-equivalent. The packet
range remains `4-11B` framed. The fp16 source-score and source-logit floors are
`8B` and explicitly non-private.

The systems-boundary figure/table artifact passes. It emits 8 LatentWire packet
rows: 4 cached-source communication-object rows and 4 end-to-end disclosure
rows. All packet rows remain source-private and set both `native_measured=false`
and `native_claim_allowed=false`. The minimum dense source-state/KV floor is the
QJL-style `1`-bit KV row at `768B`, which is `69.8x` the largest `11B` packet
and `12.0x` a single `64B` padded packet.

The only comparator row with complete Mac phase timing is HellaSwag
`validation_first1024`: source scoring is `488.631 ms/question`, receiver
decode p50 is `28.959 us`, and receiver decode p95 is `30.375 us`. ARC,
OpenBookQA, and full HellaSwag compaction rows are correctly marked as missing
phase traces rather than silently inheriting a speed claim.

## Interpretation

This strengthens the paper by separating two questions:

1. What tiny thing is communicated?
2. What did it cost to compute that tiny thing?

The first question supports the source-private packet contribution. The second
question is still open until native serving rows exist. This is the right
boundary for systems reviewers: the packet is a small communication object, not
yet a measured serving-speed win.

## Contribution Status

Keep:

- Source-private packet protocol: `1B/4B`, `2B/5B`, `3B/6B`, and `8B/11B`
  fixed-byte packet rows with no source text, raw hidden vector, score/logit
  vector, or KV cache exposed.
- Evaluation/falsification suite: source-index controls, candidate-roll and
  wrong-remap controls, score/hidden cache audits, paired confidence intervals,
  full-validation compaction, and physical candidate-text permutation tests.
- Systems boundary artifact: explicit cached-source versus end-to-end rows,
  plus score/logit/hidden/KV floors and native-claim guardrails.

Cut or demote:

- Learned sparse SAE/common-basis as a positive method.
- Universal latent language, latent reasoning, and common-basis claims.
- Any throughput, HBM, PCIe/NVLink, C2C/KVComm/TurboQuant/vLLM/SGLang win until
  native rows are measured.

## Lay Explanation

We made the table distinguish between the tiny message and the cost of making
the message. The tiny message can be just a few bytes. But the source model may
still spend real time deciding what message to send. Reviewers need both facts,
and now the artifacts do not mix them together.

## Next Gate

The next exact method gate should be a rate-distortion / denoising syndrome
packet on HellaSwag Qwen-to-Phi, where oracle headroom remains large. Required
pass condition: beat fixed hybrid and candidate-only with paired CI, while
failing under source-row shuffle, code permutation, candidate-roll,
target-derived-code, and label-permutation controls.

The next systems gate, once NVIDIA is available, is native vLLM/SGLang rows for
`latentwire_packet_cached_source`, `latentwire_packet_end_to_end_source_scoring`,
target-only, same-byte visible text, C2C, KVComm, QJL, and TurboQuant-style
source-state baselines under the existing native metric schema.
