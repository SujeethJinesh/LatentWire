# HellaSwag Receiver-Family Multi-Slice Summary

- pass gate: `False`
- source -> target: `Qwen2.5 -> Phi-3-mini`
- slices: `2`
- range: `1024:2048`
- total eval rows: `768`
- weighted target-only accuracy: `0.263021`
- weighted packet-only accuracy: `0.455729`
- weighted receiver accuracy: `0.401042`
- weighted oracle accuracy: `0.593750`
- packet minus target-only: `0.192708`
- receiver minus packet-only: `-0.054688`
- oracle minus packet-only: `0.138021`
- receiver-improvement slices: `0/2`
- min receiver CI95 low vs packet-only: `-0.158854`
- packet: `2B` raw / `5B` framed

## Interpretation

This aggregate asks whether a source-family packet can be consumed by a Phi-3 target receiver on adjacent HellaSwag slices. Packet-only utility over Phi target-only is useful transfer evidence, but receiver improvement over packet-only is required for a learned cross-family receiver claim.
