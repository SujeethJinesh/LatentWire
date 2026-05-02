# COLM Acceptance Baseline Audit

- date: `2026-05-02`
- scoped COLM gate: `True`
- strict positive beyond source-index gate: `False`
- interpretation: The packet remains positive versus target-only, same-budget text, entropy-matched random index, and destructive controls. It does not beat explicit source-index/source-rank communication on the current frozen ARC/OBQA surfaces. This resolves the reviewer objection by narrowing the claim: current LatentWire is a fixed-byte source-private candidate-transfer protocol, not a method that compresses beyond the source's selected candidate.
- strict gate interpretation: The stricter ICLR-style gate, positive transfer beyond an explicit source-index/source-rank channel, is not met. Passing this artifact is therefore a scoped-COLM correctness gate, not a claim that the current method beats source-index communication.

## Main Baseline Readout

| Benchmark | Split | Seeds | Packet | Source index | Target | Text | Packet-source CI low | Packet-text CI low |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | test | 10 | 0.344 | 0.346 | 0.265 | 0.300 | -0.008 | 0.025 |
| OpenBookQA | test | 5 | 0.378 | 0.378 | 0.276 | 0.350 | -0.006 | 0.000 |

## Rate Curve

| Benchmark | Bytes | Framed | Packet mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| ARC-Challenge | 2 | 5 | 0.342 | 0.335 | 0.345 |
| ARC-Challenge | 3 | 6 | 0.342 | 0.335 | 0.345 |
| ARC-Challenge | 4 | 7 | 0.344 | 0.340 | 0.346 |
| ARC-Challenge | 8 | 11 | 0.344 | 0.342 | 0.346 |
| OpenBookQA | 2 | 5 | 0.378 | 0.378 | 0.380 |
| OpenBookQA | 3 | 6 | 0.378 | 0.378 | 0.380 |
| OpenBookQA | 4 | 7 | 0.378 | 0.378 | 0.378 |
| OpenBookQA | 8 | 11 | 0.378 | 0.378 | 0.378 |

## Reviewer Implication

This artifact should raise confidence in correctness and scope, but not novelty beyond source-choice transfer. The paper should include these rows and avoid claiming superiority to explicit source-index or source-score communication until a richer receiver-family method beats this audit.

Lay explanation: the new rows ask whether the tiny packet beats simply sending which answer the source picked. On the current frozen surfaces, it does not; the packet is best described as a structured fixed-byte way to carry source candidate evidence with strict controls, not as a method that outperforms an explicit source-index channel.
