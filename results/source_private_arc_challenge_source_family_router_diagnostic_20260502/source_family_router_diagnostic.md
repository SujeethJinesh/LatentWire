# ARC Source-Family Router Diagnostic

- date: `2026-05-02`
- pass gate: `False`
- selected metric: `best_score`
- selected threshold: `0.08043015280622001`
- test disagreement rows: `473`
- source-choice oracle accuracy: `0.455`
- packet oracle accuracy on disagreement rows: `0.586`

| Row | Accuracy | Delta vs Qwen | CI95 low vs Qwen | Alt-rate |
|---|---:|---:|---:|---:|
| TinyLlama packet | 0.269 | -0.048 | - | - |
| Qwen-substituted packet | 0.317 | 0.000 | - | - |
| selected router | 0.315 | -0.002 | -0.023 | 0.039 |
| packet oracle | 0.586 | 0.268 | - | - |

## Interpretation

The diagnostic separates source complementarity from routing quality. TinyLlama and Qwen have substantial oracle headroom on disagreement rows, but receiver-side packet confidence alone does not select the better source reliably. This weakens cheap abstention/routing as an ICLR repair and promotes source-side confidence scores or a learned cross-family connector as the next gate.

Lay description: the two source models often pick different answers. We tried to decide which tiny packet to trust using only the receiver's packet-confidence signals learned on validation. That simple router did not recover the oracle headroom, so the next repair needs source-side confidence scores or a learned connector, not just receiver confidence.
