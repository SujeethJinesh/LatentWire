# LatentWire Escape Test Report

RUN_STATUS: CPU_CACHED_ANALYSIS_ONLY

## Guardrails

- No GPU, no MPS, no new generation, no confirm paths, no queue changes.
- Inputs are existing non-confirm cached artifacts; the script refuses confirm-looking input paths.
- This report is not a promotion record. It is a cheap gate for whether an escape deserves a full build.

## Scoop Check

| Axis | Verdict | Evidence |
|---|---|---|
| Privacy-preserving latent inter-LLM packet vs text on a privacy-utility frontier | PARTIAL/OPEN | Adjacent privacy-utility and IB semantic-communication work exists, but I found no direct inter-LLM latent-packet frontier against equal-byte text. Relevant anchors: IBAL semantic communication against model inversion ([arXiv:2312.03252](https://arxiv.org/abs/2312.03252)), adaptive text anonymization privacy-utility frontier ([arXiv:2602.20743](https://arxiv.org/abs/2602.20743)), and embedding inversion leakage ([arXiv:2305.03010](https://arxiv.org/abs/2305.03010)). |
| Continuous state / cache communication accuracy escape | CROWDED/PARTIAL | C2C ([arXiv:2510.03215](https://arxiv.org/abs/2510.03215)), Interlat ([arXiv:2511.09149](https://arxiv.org/abs/2511.09149)), and LCF ([arXiv:2605.22863](https://arxiv.org/abs/2605.22863)) occupy the no-text latent/cache communication lane; a byte-level, destructive-control distillation remains a narrower possible opening. |
| Discrete evidence packet dominated by equal-byte visible evidence | OPEN AS A NEGATIVE/THEORY CLAIM | Search did not find a direct theorem/result for the exact equal-byte discrete-evidence claim. The local cached test below supports it empirically for exact visible signatures. |

## Escape A: Privacy Proxy

- artifact: `results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59/core_to_holdout/predictions_budget8.jsonl`
- achieved rows per condition: `512` / floor `500`
- verdict: `NO_PRIVACY_POSITIVE_ON_CACHED_PROXY`

| Condition | Utility accuracy | Payload bytes | Family leakage acc | Leakage majority baseline |
|---|---:|---:|---:|---:|
| `target_only` | 0.250000 | 0.00 | 0.111111 | 0.099415 |
| `candidate_conditioned_packet_builder` | 0.875000 | 8.00 | 1.000000 | 0.099415 |
| `structured_text_matched` | 0.250000 | 8.00 | 0.152047 | 0.099415 |
| `answer_only_text` | 0.250000 | 8.00 | 0.099415 | 0.099415 |
| `random_same_byte` | 0.251953 | 8.00 | 0.146199 | 0.099415 |

Decision: no privacy positive from the cached proxy. The high-utility latent packet is not compared to a matched-utility visible-text baseline, and the existing visible text control has target-only utility. This parks privacy until a real IB encoder and matched-utility text/anonymization comparator exist.

## Theory Test: Exact Discrete Evidence vs Visible Evidence

- artifacts: `4` helper files under `results/source_private_testlog_packet_cross_model_20260428/`
- model-helper rows: `640` / floor `500`
- unique examples: `160`
- verdict: `VISIBLE_EXACT_DISCRETE_EVIDENCE_DOMINATES_PACKET`

| Condition | Accuracy | Correct / n |
|---|---:|---:|
| `target_only` | 0.250000 | 160 / 640 |
| `answer_only` | 0.250000 | 160 / 640 |
| `random_same_byte` | 0.250000 | 160 / 640 |
| `shuffled_model_packet` | 0.250000 | 160 / 640 |
| `matched_model_packet` | 0.775000 | 496 / 640 |
| `full_signature_oracle` | 1.000000 | 640 / 640 |

- visible exact signature minus matched model packet: `0.225000` CI95 `[0.192188, 0.257812]` over `640` model-helper rows.
- The visible exact signature is the same 2-byte discrete payload and dominates/ties the model packet; this supports the discrete-evidence-is-text-capturable claim rather than a no-text positive.

## Candidate Text-Control Audit

- packet accuracy: `0.875000`
- stored `structured_text_matched` accuracy: `0.250000`
- stored `structured_text_matched` top payload prefixes: `[('7079746573742073', 512)]`
- atom vocab size: `21`
- rows encodable as public 2-byte atom/value code within 8 bytes: `512` / `512`

Interpretation: the existing `structured_text_matched` row is not a valid identical-evidence text baseline because it truncates the private log prefix (`pytest s...`) and does not expose the same atom evidence. But the atom payload is discrete and public-code serializable in the same byte budget for all rows, so this cache does not establish a no-text advantage.

## Bottom Line

- Privacy remains the only cheap LatentWire escape, but this cached proxy does not pass it.
- The exact-discrete evidence check supports the theory: when the same symbolic evidence is visible, text/public code captures or beats the packet.
- Next real work is still writing, unless a separate IB privacy encoder is intentionally built as a new method card with matched-utility text/anonymization baselines.
