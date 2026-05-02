# HellaSwag Complementarity Headroom Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible; ICLR still needs a learned
  method that exploits the measured headroom.
- Current story: the HellaSwag compact source-private packet is strong, but the
  next method must learn when Qwen-side evidence should override or complement
  the TinyLlama packet.
- Exact remaining blocker: this artifact is an oracle/headroom gate, not a new
  communication method.

## Lay Explanation

This experiment asks whether TinyLlama and Qwen make different useful mistakes.
If TinyLlama's compact packet is wrong but Qwen's side prediction is right on
many rows, then a future conditional packet or selector has something real to
learn. If their errors fully overlap, another connector would have no room to
help.

## Artifact

`results/source_private_hellaswag_complementarity_headroom_gate_20260502/`

## Result

The gate passes.

| Quantity | Value |
|---|---:|
| eval rows | `10042` |
| source packet accuracy | `0.619199` |
| Qwen target-side accuracy | `0.532464` |
| target-or-source oracle accuracy | `0.686815` |
| oracle lift vs source packet | `+0.067616` |
| oracle CI95 low vs source packet | `+0.062637` |
| target-correct / source-wrong rows | `679` |
| source-correct / target-wrong rows | `1550` |
| disagreement rate | `0.292472` |
| positive blocks | `5/5` |

The source-label-copy oracle also has headroom: target-or-source-label oracle
accuracy is `0.661322`, with lift `+0.102569` over source-label copy and CI95
low `+0.096990`.

## Interpretation

This revives a targeted method branch: not another global alignment or
shared-basis scout, but a conditional syndrome/selector packet that learns the
`679` target-correct/source-wrong cases without giving back too many of the
`1550` source-correct/target-wrong cases.

The result also explains why shallow source-code branches have been brittle:
the decision is not simply "trust source" or "trust target." The useful
surface is a conditional disagreement policy with a strict byte budget and
source-destroying controls.

## Decision

Promote conditional HellaSwag syndrome/selector as the next live method gate.
Required controls: packet-only, target-only, source-label-copy, same-byte text,
candidate-id-only, target-derived selector, random same-rate selector,
row-shuffled source, label-permutation train fit, candidate derangement, and
wrong-example source packet.
