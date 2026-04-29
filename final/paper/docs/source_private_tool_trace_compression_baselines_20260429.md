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

### Control-Stabilized Slot Packet

The control-stabilizing variant removes explicit candidate diagnostic fields and candidate semantics from the target-side candidate representation. The target-side codebook contains only public candidate slots; the source packet must select the correct slot from private evidence. Ridge is fit without an intercept so an answer-masked source cannot emit a learned global prior packet.

Settings: `candidate_view=slot`, `--no-intercept`, 6-byte scalar packet, train/eval `768/512`, all families unless noted.

| Seed pair | Strict scalar pass | Scalar | Target | Constrained shuffle | Answer-masked | Label-shuffled ridge | Raw sign |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 -> 30 | true | 1.000 | 0.250 | 0.000 | 0.250 | 0.242 | 0.307 |
| 31 -> 32 | true | 1.000 | 0.250 | 0.000 | 0.250 | 0.258 | 0.188 |
| 33 -> 34 | true | 1.000 | 0.250 | 0.000 | 0.250 | 0.229 | 0.207 |
| 35 -> 36 | true | 1.000 | 0.250 | 0.000 | 0.250 | 0.254 | 0.182 |
| 37 -> 38 | true | 1.000 | 0.250 | 0.000 | 0.250 | 0.207 | 0.201 |

This is the first learned-packet gate in this branch to pass all five seed pairs with clean source-destroying controls. It is narrower than the semantic candidate-text setting: the target has public candidate slots rather than rich candidate text. That makes the claim cleaner but more scoped.

Cross-family falsification is mixed:

| Train -> Eval | Strict scalar pass | Scalar | Target | Constrained shuffle | Answer-masked | Label-shuffled ridge |
|---|---:|---:|---:|---:|---:|---:|
| core -> holdout | false | 0.125 | 0.250 | 0.625 | 0.250 | 0.275 |
| holdout -> core | true | 0.625 | 0.250 | 0.250 | 0.250 | 0.193 |

So the slot packet is same-family/all-family seed-stable but not yet a symmetric cross-family method.

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
- The slot/no-intercept scalar packet passes `5/5` seeds with perfect matched accuracy and clean controls.

What failed:

- The original learned syndrome does not beat the scalar quantized transport baseline, so it should not be claimed as the strongest method.
- The 256-example seed 29/30 scalar control was a near miss due to shuffled-source accuracy of 0.305.
- 12-byte scalar is too saturated and has a high shuffled-source control, so 6 bytes is the cleaner claim point.
- The scalar packet passes strict source-destroying controls on only `3/5` seed pairs.
- Naive no-bias source-innovation coding is pruned because it stabilizes controls by destroying most of the useful signal.
- Rich candidate text remains leaky: `no_diag` and `semantic` views can preserve very high scalar accuracy while answer-masked or label-shuffled controls rise above the target floor.
- Cross-family slot transfer is asymmetric: holdout-to-core passes, but core-to-holdout fails.

## Next Gate

Promote `scalar_quantized_source` to the live method only after:

1. Codebook remap for the `slot/no-intercept` packet.
2. Paired bootstrap confidence intervals over the 5 seed pairs.
3. A harder cross-family split or a clear same-family claim boundary.
4. A candidate-side ambiguity benchmark where public candidate text is not already target-solvable.
5. A model-emitted version of the slot packet or a learned neural encoder trained on real traces.

## 2026-04-29 Addendum: Slot Remap and Bootstrap

The `slot/no-intercept` scalar packet now has a codebook-remap and paired-bootstrap readout.

Artifacts:

- `results/source_private_tool_trace_slot_remap_20260429_seed101/`
- `results/source_private_tool_trace_slot_remap_20260429_seed103/`
- `results/source_private_tool_trace_slot_remap_20260429_seed107/`
- `results/source_private_slot_packet_bootstrap_20260429/`

Three remapped slot codebooks pass the strict scalar gate:

| Remap seed | Scalar | Target | Best strict control | Raw sign | Delta target CI95 | Delta raw CI95 |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.463 | 0.250 | 0.264 | 0.332 | [0.156, 0.270] | [0.072, 0.189] |
| 103 | 0.508 | 0.250 | 0.266 | 0.316 | [0.199, 0.314] | [0.127, 0.252] |
| 107 | 0.492 | 0.250 | 0.250 | 0.330 | [0.186, 0.303] | [0.104, 0.221] |

Combined with the five same-codebook seed rows, the bootstrap summary reports:

- mean scalar accuracy: `0.808`
- mean scalar minus best strict control: `0.552`
- minimum paired CI95 lower bound versus target-only: `0.156`
- minimum paired CI95 lower bound versus raw source sign sketch: `0.072`

Interpretation:

- Same-codebook slot communication is now strong and clean.
- Remapped slot codebooks remain positive versus target-only and strict controls, so the method is not only one fixed slot convention.
- Remapped rows are weaker and closer to raw sign sketch; the paper should report this as a limitation and avoid overclaiming broad codebook-invariant communication.

If these pass, the paper contribution becomes a systems-friendly learned source-private posterior packet: a tiny quantized vector message that the target decodes with public candidate side information.

## 2026-04-29 Addendum: QJL/TurboQuant Residual Comparator

I added an opt-in `qjl_residual` comparator inspired by QJL/TurboQuant-style scalar-plus-sign residual coding. The packet keeps the same 6-byte budget: a coarse scalar quantized prefix plus QJL-style sign bits over residual directions orthogonalized against the scalar projection. The historical scalar pass gate is intentionally unchanged; QJL residual rows are reported as comparator rows only.

Artifacts:

- `results/source_private_tool_trace_qjl_residual_20260429_seed29_30/`
- `results/source_private_tool_trace_qjl_residual_20260429_remap101/`
- `results/source_private_tool_trace_qjl_residual_20260429_remap103/`
- `results/source_private_tool_trace_qjl_residual_20260429_remap107/`

Results:

| Surface | Scalar | QJL residual | Target | QJL controls clean | QJL - scalar |
|---|---:|---:|---:|---:|---:|
| same-codebook seed29->30 | 1.000 | 1.000 | 0.250 | true | 0.000 |
| remap 101 | 0.463 | 0.447 | 0.250 | true | -0.016 |
| remap 103 | 0.508 | 0.484 | 0.250 | true | -0.023 |
| remap 107 | 0.492 | 0.457 | 0.250 | true | -0.035 |

Interpretation:

- The residual comparator is clean under matched source-destroying controls and remains positive versus target-only.
- It does not improve the decisive remap frontier; scalar-only is still the stronger 6-byte packet.
- QJL/TurboQuant should be framed as a principled matched-byte comparator and systems inspiration, not promoted as the current method.
- The next higher-value branch is remap-invariant relative-anchor transport or a model-emitted packet, not further tuning this residual variant without a new design reason.

## 2026-04-29 Addendum: RASP Relative-Score Packet

I added an opt-in `relative_scores` packet, a Relative-Anchor Stitch Packet
(RASP) variant. Instead of transmitting an absolute predicted target vector, the
source computes the predicted source posterior against the public candidate
anchors and sends one quantized score byte per candidate. With four candidates
this is a 4-byte packet, even under the 6-byte cap. The decoder selects the
candidate with the highest dequantized score.

This variant has a different but realistic assumption: the source may see the
public candidate set/order before sending its packet. The private evidence is
still necessary, checked by answer-masked, constrained-shuffled, label-shuffled,
random-byte, score-permutation, and target-order-mismatch controls.

Artifacts:

- `results/source_private_tool_trace_relative_scores_20260429_seed29_30_budget4/`
- `results/source_private_tool_trace_relative_scores_20260429_remap101_budget4/`
- `results/source_private_tool_trace_relative_scores_20260429_remap103_budget4/`
- `results/source_private_tool_trace_relative_scores_20260429_remap107_budget4/`
- `results/source_private_relative_score_bootstrap_20260429/`

Equal-actual-byte results:

| Surface | Relative | Scalar | Target | Raw sign | Relative - scalar CI95 | Bytes rel/scalar |
|---|---:|---:|---:|---:|---:|---:|
| same-codebook seed29->30 | 1.000 | 1.000 | 0.250 | 0.316 | [0.000, 0.000] | 4/4 |
| remap 101 | 0.494 | 0.426 | 0.250 | 0.326 | [0.033, 0.105] | 4/4 |
| remap 103 | 0.520 | 0.496 | 0.250 | 0.328 | [-0.012, 0.061] | 4/4 |
| remap 107 | 0.506 | 0.502 | 0.250 | 0.326 | [-0.035, 0.043] | 4/4 |

Bootstrap summary:

- mean relative accuracy: `0.630`
- mean relative minus scalar: `0.024`
- mean remap relative minus scalar: `0.032`
- minimum relative-vs-target paired CI95 lower bound: `0.189`
- minimum remap relative-vs-scalar paired CI95 lower bound: `-0.035`

Interpretation:

- RASP is a promising secondary contribution: it is 4-byte, control-clean, and
  improves mean remap accuracy over equal-byte scalar.
- The improvement over scalar is not uniformly significant; only remap `101`
  has a positive paired lower bound versus scalar.
- It should be claimed as a systems/robustness extension, not as a replacement
  for the scalar packet until it survives more seeds or harder cross-family
  splits.

## 2026-04-29 Addendum: RASP Expanded Remap and Cross-Family Falsification

I expanded RASP to seven remap seeds and added bidirectional core/holdout
cross-family rows.

New artifacts:

- `results/source_private_tool_trace_relative_scores_20260429_remap109_budget4/`
- `results/source_private_tool_trace_relative_scores_20260429_remap113_budget4/`
- `results/source_private_tool_trace_relative_scores_20260429_remap127_budget4/`
- `results/source_private_tool_trace_relative_scores_20260429_remap131_budget4/`
- `results/source_private_relative_score_bootstrap_remap8_20260429/`
- `results/source_private_tool_trace_relative_scores_cross_family_20260429_core_to_holdout_budget4/`
- `results/source_private_tool_trace_relative_scores_cross_family_20260429_holdout_to_core_budget4/`

Expanded remap summary:

| Remap | Relative | Scalar | Target | Relative - scalar CI95 | Relative controls clean |
|---:|---:|---:|---:|---:|---:|
| 101 | 0.494 | 0.426 | 0.250 | [0.033, 0.105] | true |
| 103 | 0.520 | 0.496 | 0.250 | [-0.014, 0.061] | true |
| 107 | 0.506 | 0.502 | 0.250 | [-0.035, 0.043] | true |
| 109 | 0.477 | 0.451 | 0.250 | [-0.010, 0.061] | true |
| 113 | 0.473 | 0.436 | 0.250 | [0.000, 0.072] | true |
| 127 | 0.453 | 0.428 | 0.250 | [-0.010, 0.061] | true |
| 131 | 0.506 | 0.434 | 0.250 | [0.035, 0.109] | false |

The expanded bootstrap reports mean remap relative-minus-scalar `+0.037`, but
the pass gate is false: the minimum relative-vs-target CI95 lower bound is
`+0.146`, just under the `+0.15` rule, and remap `131` has a near-threshold
random same-byte control failure (`0.301` versus target `0.250` and the `0.300`
cutoff).

Cross-family:

| Train -> Eval | Relative | Scalar | Target | Relative controls clean | Interpretation |
|---|---:|---:|---:|---:|---|
| core -> holdout | 0.207 | 0.225 | 0.250 | false | fail |
| holdout -> core | 0.492 | 0.375 | 0.250 | true | one-direction pass |

Interpretation:

- RASP remains useful as a systems/remap robustness extension: it improves mean
  remap accuracy over equal-byte scalar and uses candidate-relative 4-byte
  packets.
- It is not yet a headline cross-family method. Core-to-holdout fails and
  holdout-to-core passes, mirroring the asymmetric transfer risk seen in scalar
  rows.
- The next RASP-specific improvement should canonicalize candidate order by
  public candidate ID/hash and retest order-mismatch controls. Otherwise,
  further remap tuning risks looking like a candidate-order side channel.

## 2026-04-29 Addendum: Canonical-Order RASP

Canonical RASP is now added as `relative_scores_canonical`. The source still
sends one quantized score byte per public candidate, but the packet is
serialized by stable public candidate identity and decoded back into the
target's current display order. This removes the strongest display-order
side-channel objection to the first RASP implementation.

New artifacts:

- `results/source_private_relative_canonical_remap101_20260429/`
- `results/source_private_relative_canonical_remap103_20260429/`
- `results/source_private_relative_canonical_remap107_20260429/`
- `results/source_private_relative_canonical_remap109_20260429/`
- `results/source_private_relative_canonical_remap113_20260429/`
- `results/source_private_relative_canonical_remap127_20260429/`
- `results/source_private_relative_canonical_remap131_20260429/`
- `results/source_private_relative_canonical_bootstrap_remap7_20260429/`
- `results/source_private_relative_canonical_remap127_large_20260429/`
- `results/source_private_relative_canonical_remap127_large_bootstrap_20260429/`
- `results/source_private_relative_canonical_core_to_holdout_20260429/`
- `results/source_private_relative_canonical_holdout_to_core_20260429/`

Expanded remap summary:

| Remap | Canonical RASP | Scalar | Target | RASP - scalar CI95 | RASP - target CI95 |
|---:|---:|---:|---:|---:|---:|
| 101 | 0.494 | 0.426 | 0.250 | [0.029, 0.107] | [0.184, 0.303] |
| 103 | 0.520 | 0.496 | 0.250 | [-0.014, 0.061] | [0.213, 0.328] |
| 107 | 0.506 | 0.502 | 0.250 | [-0.035, 0.043] | [0.199, 0.311] |
| 109 | 0.477 | 0.451 | 0.250 | [-0.008, 0.061] | [0.170, 0.281] |
| 113 | 0.473 | 0.436 | 0.250 | [0.002, 0.072] | [0.164, 0.279] |
| 127 | 0.453 | 0.428 | 0.250 | [-0.010, 0.061] | [0.146, 0.262] |
| 131 | 0.506 | 0.434 | 0.250 | [0.035, 0.109] | [0.197, 0.311] |

The seven-remap bootstrap is still strict-fail because the minimum
canonical-vs-target CI95 lower bound is `+0.146`, just below the `+0.150` rule.
However, the larger frozen rerun of the worst remap `127` passes:
canonical RASP `0.442`, scalar `0.361`, target `0.250`, target CI95 low
`+0.152`, scalar CI95 low `+0.053`, controls clean.

Cross-family remains asymmetric:

| Train -> Eval | Canonical RASP | Scalar | Target | Controls clean | Interpretation |
|---|---:|---:|---:|---:|---|
| core -> holdout | 0.207 | 0.225 | 0.250 | false | fail |
| holdout -> core | 0.492 | 0.375 | 0.250 | true | one-direction pass |

Interpretation:

- Canonical RASP is a stronger secondary systems/robustness contribution than
  display-order RASP because it removes display-order serialization while
  preserving a 4-byte payload.
- It should be reported as same-family/remap robustness and as an ablation
  against display-order side-channel concerns.
- It is not a cross-family solution. The next live method branch for that gap
  is a consistency-distilled canonical posterior packet trained under
  candidate-order and source-feature perturbations.
