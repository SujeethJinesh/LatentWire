# When Byte-Scale Model Communication Should Be Text

## Abstract

LatentWire asks whether a source model can send a small no-text packet that improves a receiver model beyond ordinary score and text baselines. Across the completed campaign, the answer is negative for the discrete, byte-scale regimes that are currently claim-ready. One-way score and residual packets do not beat source-index/confidence controls on the held-out aggregate, and a powered dev/gate ladder shows that most apparent source information is already redundant with the receiver: the source signal drops from 2.098527 bits to 0.281933 bits after receiver conditioning. The strongest discrete-evidence escape also fails in the opposite direction: when the same exact symbolic evidence is exposed as a visible 2-byte public code, it reaches 1.000 accuracy versus 0.775 for the learned model packet, a +0.225 advantage with 95% CI [0.192188, 0.257812]. The last cheap privacy-bottleneck escape, L-IB1, also fails: the current cached high-utility packet has 0.875 utility with 1.000 leakage, while the best adversarial bottleneck reaches only 0.602339 proxy utility and still leaks source/evidence identity. The paper therefore contributes a bounded negative: for discrete low-byte evidence, no-text latent packets are dominated by score controls or equal-byte visible evidence unless the method moves to continuous dense state or a genuinely privacy-preserving bottleneck.

## 1. Introduction

Small model-to-model messages are attractive because they promise communication without exposing full reasoning traces. The hard question is not whether a helper model knows something useful; it is whether a byte-scale packet conveys information that the receiver cannot already recover from its own scores, from the source identity, or from an equal-byte visible text code.

The campaign originally sought a positive LatentWire method. The surviving evidence instead supports a sharper claim: the discrete evidence that makes the packet useful is text-capturable under the same byte budget, while the score-like residual signal mostly disappears after conditioning on the receiver. We treat this as a bounded negative rather than as a failed search. It identifies the regimes where no-text packet claims are not yet justified and points to the only remaining credible escape hatches.

This draft makes four contributions.

1. We give a controlled negative result for one-way score/residual packets. On the held-out aggregate with 381 rows, the deployable score packet is -0.023622 below the source-index/confidence baseline with CI [-0.060367, 0.013123]. On the powered dev/gate ladder with 1500 scored rows and 359 gate rows, the current deployable packet is -0.013928 below the best score baseline with CI [-0.050139, 0.022284].

2. We localize the mechanism. The source has information in isolation, but receiver conditioning collapses it: source information drops from 2.098527 bits to 0.281933 bits after the receiver's own evidence is available. The non-deployable source+target-at-encoder upper bound remains positive, so the campaign was not measuring pure noise.

3. We test the discrete-evidence escape directly. On 640 model-helper rows over 160 unique examples, a matched model packet reaches 0.775 accuracy, but the exact same symbolic evidence serialized as a visible 2-byte public signature reaches 1.000 accuracy. Random, shuffled, answer-only, and target-only controls collapse to 0.250.

4. We close the last cheap privacy escape. L-IB1, an adversarial cached privacy-bottleneck selector, cannot preserve current-packet utility without preserving source/evidence identity. Its verdict is `KILL_UTILITY_IS_IDENTITY`.

## 2. Claim Boundary

The claim is deliberately narrow. This paper does not say that all latent communication is impossible, or that continuous hidden-state transfer cannot work. It says that the current byte-scale discrete packets do not establish a no-text communication advantage under the hard baselines required for a paper claim.

The excluded regimes are important. Continuous cache/state transfer methods such as C2C, Interlat, and Latent Cache Flow occupy a different lane because they send high-dimensional hidden state rather than a tiny discrete evidence packet. Privacy-preserving semantic communication and adaptive text anonymization are also separate directions because their objective is a utility/leakage frontier, not raw accuracy under equal-byte text. The local L-IB1 result only kills the cheap cached proxy; it does not rule out a reviewed build-scale neural information-bottleneck method.

The positive method claim remains unavailable until a method beats:

- target-only;
- source-index/confidence;
- source-rank/source-score controls;
- same-byte visible exact evidence;
- adaptive/anonymized same-byte text;
- random same-byte and wrong-row destructive controls.

## 3. Experimental Design

All screening and escape tests used existing non-confirm cached artifacts. The runners refuse confirm-looking paths. Aggregates, split hashes, and source manifests are preserved, while raw held-out examples are not used for screening or prompt-driven iteration.

The main LatentWire artifacts are:

- the one-way deployable score packet aggregate;
- the powered score ladder over scored dev/gate rows;
- the receiver-query packet aggregate;
- the deterministic C2C/KVComm packet smokes used only as anchors;
- the CPU cached escape tests for privacy and exact discrete evidence;
- the L-IB1 cached privacy-bottleneck test.

Uncertainty is reported as paired bootstrap confidence intervals where the campaign runner emitted paired row outcomes. For the discrete-evidence theorem test, the decisive comparison is visible exact public code minus matched model packet over 640 model-helper rows.

## 4. Results

### 4.1 Score Packets Are Redundant With Receiver Evidence

| Branch | n | Result | Verdict |
| --- | ---: | --- | --- |
| One-way deployable score packet | 381 | delta vs source-index/confidence -0.023622, CI [-0.060367, 0.013123] | bounded negative |
| Powered dev/gate score ladder | 359 gate rows | current packet delta vs best score baseline -0.013928, CI [-0.050139, 0.022284] | not deployable |
| Source+target-at-encoder upper bound | 359 gate rows | +0.125348, CI [0.089136, 0.161560] | non-deployable ceiling |
| Receiver-query packet | 359 gate rows | L-Q1 delta vs source-index/confidence -0.022284, CI [-0.055710, 0.011142] | killed |

The non-deployable upper bound matters because it prevents an overstrong interpretation. There is useful source information if the encoder can condition on the target receiver's evidence. The deployable packet fails because the source-only byte-scale message mostly communicates information that the receiver or the named source baseline already supplies.

### 4.2 Receiver Conditioning Collapses the Apparent Source Signal

The powered ladder estimates 2.098527 bits of source information before receiver conditioning and 0.281933 bits after conditioning on receiver evidence. This is the central mechanism for the negative result. Same-family models on shared multiple-choice tasks often fail and succeed for overlapping reasons, so a source packet can look useful until the receiver's own scores and the source identity are included as baselines.

### 4.3 Exact Discrete Evidence Is Better Sent As Text

| Condition | Accuracy | Correct / n |
| --- | ---: | ---: |
| target_only | 0.250000 | 160 / 640 |
| answer_only | 0.250000 | 160 / 640 |
| random_same_byte | 0.250000 | 160 / 640 |
| shuffled_model_packet | 0.250000 | 160 / 640 |
| matched_model_packet | 0.775000 | 496 / 640 |
| full_signature_oracle | 1.000000 | 640 / 640 |

The visible exact signature beats the learned packet by +0.225000 with CI [0.192188, 0.257812]. This is the discrete-evidence theorem in operational form: if the useful packet content is a finite symbolic evidence signature and the receiver may receive the same number of bytes as visible public code, the visible code captures the evidence at least as directly as the learned opaque packet. In this cache, it strictly dominates.

The destructive controls are important. Answer-only, random, shuffled, and target-only variants remain at chance. The win is not an artifact of row frequency, answer prior, or any arbitrary byte string; it is the exact symbolic evidence itself.

### 4.4 L-IB1 Does Not Produce A Privacy-Positive Packet

L-IB1 was the last cheap LatentWire escape. It asked whether a small encoder/feature selector could keep utility while hiding source/evidence identity on existing cached rows.

| Variant | Bytes | Utility | Max leakage | Notes |
| --- | ---: | ---: | ---: | --- |
| target_only | 0 | 0.250000 cached / 0.222222 proxy | 0.385965 | receiver baseline |
| current_high_utility_packet | 8 | 0.875000 cached / 1.000000 proxy | 1.000000 | identity-preserving |
| same_byte_visible_exact_public_code | 8 | 0.875000 cached / 1.000000 proxy | 1.000000 | equal-byte text/code |
| adaptive_anonymized_text_coarse_atoms | 8 | 0.754386 proxy | 0.900585 | text baseline |
| best L-IB1 bottleneck | 4 | 0.602339 proxy | 0.602339 | selected `default`, `integer` |

The verdict is `KILL_UTILITY_IS_IDENTITY`: no cached adversarial bottleneck preserved current-level utility while hiding the source/evidence structure. This does not rule out a full neural IB encoder, but it removes the cheap cached route as a paper-strength positive.

## 5. Failure Modes Found By The Audit

Several apparent positives were invalidated or reinterpreted.

Gold-aware oracles measure gap-to-perfect rather than transmissible information. The campaign therefore treats any verifier/fuser that can see the gold answer as a leakage artifact, not a communication result.

Tool-output packets can be useful, but L-PC2 showed that the equal-byte visible-tool control ties the private packet. Private computation is valuable; an opaque latent packet is not automatically better than sending the tool result as text.

Continuous cache/state transfer remains unsettled. Diagnostic smokes are useful anchors but do not establish a byte-scale no-text packet method. Any future claim must include equal-budget text/source-index controls and destructive controls from the start.

## 6. Related Work

C2C, Interlat, and Latent Cache Flow study model communication through hidden states or caches. Those systems motivate the continuous-state escape hatch but do not directly rescue a discrete byte-scale packet claim.

Privacy and semantic communication work, including IBAL-style information bottlenecks and adaptive text anonymization, motivates a possible utility/leakage frontier. The L-IB1 result here is a cheap local gate against existing packet/evidence rows, not a general impossibility theorem for privacy-preserving communication.

The closest baseline family for this paper is not another latent method but the trivial equal-byte surface: source identity, source confidence, source rank, and visible exact evidence. The paper's central standard is therefore `delta_beyond_score` or `delta_beyond_text`, not raw improvement over target-only.

## 7. Limitations

The exact-discrete evidence test has 640 model-helper rows but only 160 unique examples. It is decisive for the local cache but should be described as an operational theorem plus empirical support, not as a mathematical proof over all communication protocols.

The L-IB1 privacy-bottleneck result is CPU cached and uses proxy classifiers over existing packet metadata. It does not train a live neural encoder, does not run new generation, and does not test continuous dense state. A future privacy-positive method would need a preregistered card, planted tests, leakage reviewers, and matched visible/anonymized text baselines.

The held-out one-way score packet result is aggregate-only in this draft to avoid exposing raw confirmation rows. The paper should retain the aggregate numbers and the access-manifest audit rather than include row payloads.

## 8. Conclusion

The campaign did not find a deployable no-text byte-scale LatentWire method. It found something more precise: score-like source packets are mostly redundant after receiver conditioning, and exact discrete evidence is better transmitted as equal-byte visible public code. The only credible future positives are outside the killed regime: continuous dense state transfer, or a real privacy-preserving bottleneck that beats adaptive same-byte text while hiding source/evidence identity.
