# Source-Private Tool-Trace Compression Baselines

Date: 2026-04-29

## Status

The matched-byte compression gate changed the live method ranking.

The learned random-hyperplane syndrome remains positive against no-source controls, but it is not the best 6-byte transport. A scalar-quantized learned source projection is stronger on the same source-private tool-trace setup.

## Method

The source still observes the private hidden-test/tool-trace log. The target still sees only the public issue and public candidate pool.

Compared methods:

- `matched_learned_syndrome`: ridge source-to-target encoder, random hyperplane signs, Hamming decode.
- `scalar_quantized_source`: same learned source-to-target encoder, random low-dimensional projection, 8-bit scalar quantization per coordinate, L2 decode against target-side candidate projections.
- `raw_source_sign_sketch`: direct sign sketch of private source features with no learned source-to-target bridge.
- controls: target-only, zero-source, random same-byte, scalar shuffled source, scalar answer-masked source.

The scalar row is a TurboQuant/QJL/RaBitQ-inspired baseline: preserve target-side similarity under a tiny byte budget rather than sending text. It is also now a candidate method because it is more accurate and faster than the bit-syndrome packet at equal bytes.

## Primary Sources Added

- TurboQuant: random rotation plus scalar quantization for near-optimal vector quantization and inner-product preservation: https://arxiv.org/abs/2504.19874
- QJL: JL transform followed by sign-bit quantization for KV-cache compression: https://arxiv.org/abs/2406.03482
- RaBitQ: randomized binary quantization with error bounds for high-dimensional nearest-neighbor search: https://arxiv.org/abs/2405.12497
- KVQuant: KV-cache quantization systems baseline: https://arxiv.org/abs/2401.18079
- KIVI: asymmetric 2-bit KV-cache quantization baseline: https://arxiv.org/abs/2402.02750
- Distributed indirect source coding with decoder side information: https://arxiv.org/abs/2405.13483
- Distributed deep JSCC with decoder-only side information: https://arxiv.org/abs/2310.04311

## Evidence

Artifacts:

- `results/source_private_tool_trace_compression_baselines_20260429/`
- `results/source_private_tool_trace_compression_baselines_20260429_seed31/`
- `results/source_private_tool_trace_compression_baselines_20260429_n512/`

### Seed 29/30, 256 Eval

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Raw source sign | Scalar shuffled | Scalar masked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | false | false | 0.934 | 0.973 | 0.250 | 0.316 | 0.305 | 0.172 |
| 12 | false | false | 0.992 | 1.000 | 0.250 | 0.184 | 0.355 | 0.133 |

This slice shows the scalar packet beats the learned syndrome, but strict scalar control fails by a small margin at 6 bytes because shuffled source reaches 0.305 versus the 0.300 cutoff.

### Seed 31/32, 256 Eval

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Raw source sign | Scalar shuffled | Scalar masked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | false | true | 0.910 | 0.945 | 0.250 | 0.172 | 0.227 | 0.207 |

This seed pair passes the strict scalar source-packet rule.

### Seed 29/30, 512 Eval

| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Raw source sign | Scalar shuffled | Scalar masked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | false | true | 0.953 | 0.979 | 0.250 | 0.307 | 0.166 | 0.293 |

The larger frozen slice resolves the seed 29/30 shuffled-source excess and passes the scalar source-packet rule.

### Five-Seed 6-Byte Stability Probe

The scalar packet has stable high matched accuracy, but strict controls are not yet stable enough for a final headline claim.

| Seed pair | Eval n | Strict scalar pass | Scalar | Syndrome | Target | Raw sign | Scalar shuffled | Scalar masked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 -> 30 | 512 | true | 0.979 | 0.953 | 0.250 | 0.307 | 0.166 | 0.293 |
| 31 -> 32 | 256 | true | 0.945 | 0.910 | 0.250 | 0.172 | 0.227 | 0.207 |
| 33 -> 34 | 512 | false | 0.957 | 0.924 | 0.250 | 0.207 | 0.219 | 0.344 |
| 35 -> 36 | 512 | true | 0.992 | 0.955 | 0.250 | 0.182 | 0.201 | 0.131 |
| 37 -> 38 | 512 | false | 0.986 | 0.945 | 0.250 | 0.201 | 0.373 | 0.283 |

Mean scalar accuracy is `0.972` with range `0.945-0.992`. The strict scalar gate passes `3/5` seed pairs. The two failures are not accuracy failures; they are source-destroying-control failures, so the live branch needs control-stabilizing work before promotion.

### Pruned Innovation Variant

I tested a source-innovation variant that subtracts the answer-masked source features before encoding. A no-bias version fixes the answer-masked control on the hard seed `33 -> 34` (`0.215`) but collapses matched innovation accuracy to `0.389`, so it is pruned for now. The useful lesson is that invariant-template suppression must be learned as a contrastive objective, not applied as a naive feature subtraction.

## Interpretation

What passed:

- The 6-byte scalar packet beats target-only by +72.9 points on the 512-example slice.
- It beats the learned sign-syndrome packet by +2.5 points on the 512-example slice.
- Raw source sign sketch is weak, so the result is not explained by generic same-byte source sketching.
- Scalar answer-masked and shuffled-source controls stay within the strict band on the 512-example slice.
- The scalar packet has lower median latency than the bit-syndrome packet in this CPU implementation.
- Five-seed scalar accuracy is high and stable: mean `0.972`, minimum `0.945`.

What failed:

- The original learned syndrome does not beat the scalar quantized transport baseline, so it should not be claimed as the strongest method.
- The 256-example seed 29/30 scalar control was a near miss due to shuffled-source accuracy of 0.305.
- 12-byte scalar is too saturated and has a high shuffled-source control, so 6 bytes is the cleaner claim point.
- The scalar packet passes strict source-destroying controls on only `3/5` seed pairs.
- Naive no-bias source-innovation coding is pruned because it stabilizes controls by destroying most of the useful signal.

## Next Gate

Promote `scalar_quantized_source` to the live method only after:

1. 5-seed 6-byte scalar repeat on 256 or 512 examples.
2. Core-to-holdout family split.
3. Candidate-codebook remap.
4. Candidate-side masking of diagnostic/family tokens.
5. Paired bootstrap confidence intervals versus target-only, learned syndrome, raw sign sketch, and matched-byte text.

If these pass, the paper contribution becomes a systems-friendly learned source-private posterior packet: a tiny quantized vector message that the target decodes with public candidate side information.
