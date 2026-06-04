# L_Q1 Receiver-Query Packet

Status: `KILLED_ON_DEV_GATE`

## Motivation

The powered one-way oracle ladder exposed a receiver-aware upper bound: source+target-at-encoder improves over the best baseline, while full source-only fusion does not. L_Q1 tested whether a deployable two-way protocol could recover that gap: the receiver sends a compact query describing its top-2 ambiguity and uncertainty, the source replies only about that ambiguity, and the receiver combines.

## Scoop Check

Verdict: `adjacent_not_exact_scoop_high_risk`.

- CU-HLM is close because it uses uncertainty-aware opportunistic compressed transmission, but its direction is SLM-to-LLM validation/compression rather than a receiver-query/source-reply packet for cross-model answer ambiguity: https://arxiv.org/abs/2505.11788
- DarkForest is close because it uses controlled belief-state communication and policy-limited evidence, but it coordinates multi-agent candidate beliefs rather than a byte-accounted two-way source/receiver score packet: https://huggingface.co/papers/2605.25188
- C2C is close because it is direct semantic communication through KV-cache projection/fusion, but it is latent cache fusion rather than receiver-query-conditioned score packets: https://github.com/thu-nics/C2C
- KVComm is close because it targets efficient inter-model communication with KV sharing/reuse, but it is cache-level sharing, not a small receiver query with a focused source reply: https://openreview.net/pdf?id=F7rUng23nw and https://arxiv.org/abs/2510.12872

Novelty boundary: L_Q1 would have to be claimed as a query-conditioned two-way packet, not as the original one-way source-private LatentWire claim.

## Protocol Tested

- Query: receiver top-2 candidate IDs plus uncertainty bucket.
- Reply: source preference and confidence bucket on the queried pair.
- Byte accounting: query `2` bytes + reply `1` byte = `3` total bytes.
- Decoder: combine target scores with the focused source reply on the queried pair.
- Controls: wrong-row query, wrong-row reply, derangement, coordinate shuffle, query-only, reply-only, equal-total-byte score sketch, equal-total-byte text proxy, and source-copy leakage MI.

## Result

- Gate rows: `359`
- Best equal-total-byte baseline: `source_index_confidence`
- L_Q1 accuracy: `0.169916`
- Best baseline accuracy: `0.192201`
- Delta: `-0.022284`, CI `[-0.055710, 0.011142]`
- Controls collapse: `False`
- Query-only/reply-only explain: `True`
- Source-copy leakage MI: `0.457419` bits for L_Q1 prediction vs source top-1, `0.594858` bits for reply-only vs source top-1.

Decision: kill L_Q1. It does not beat the equal-total-byte baseline, controls do not collapse, and ablations explain the packet.
