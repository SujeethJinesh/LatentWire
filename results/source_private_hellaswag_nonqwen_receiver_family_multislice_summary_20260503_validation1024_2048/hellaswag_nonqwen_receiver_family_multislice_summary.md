# HellaSwag Non-Qwen Receiver-Family Multi-Slice Summary

- pass gate: `False`
- slices: `2`
- contiguous range: `1024:2048`
- total eval rows: `768`
- weighted Phi target-only accuracy: `0.263021`
- weighted TinyLlama packet-only accuracy: `0.506510`
- weighted receiver accuracy: `0.477865`
- weighted target-or-packet oracle accuracy: `0.619792`
- packet minus target-only: `0.243490`
- receiver minus packet-only: `-0.028646`
- oracle minus packet-only: `0.113281`
- source utility slices: `2/2`
- receiver-improvement slices: `0/2`
- minimum receiver CI95 low vs packet-only: `-0.089909`
- packet contract: `2B` raw / `5B` framed

## Interpretation

Two adjacent HellaSwag slices show stable non-Qwen packet utility: the TinyLlama fixed-byte packet beats Phi-3 target-only by a large margin on both slices. The selected receiver still fails to beat packet-only, so this strengthens the receiver-family packet-utility claim but does not close the ICLR cross-model reasoning or receiver-fusion gate.
