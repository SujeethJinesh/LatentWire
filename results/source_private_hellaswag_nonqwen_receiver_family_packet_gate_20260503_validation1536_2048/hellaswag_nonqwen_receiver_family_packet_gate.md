# HellaSwag Non-Qwen Receiver-Family Packet Gate

- pass gate: `False`
- target-family transfer gate: `True`
- receiver-improvement gate: `False`
- eval slice: `1536:2048`
- rows: `512`
- train/eval rows: `128/384`
- target family: `Phi-3-mini`
- target-only eval accuracy: `0.255208`
- packet-only eval accuracy: `0.523438`
- receiver eval accuracy: `0.473958`
- packet minus target-only: `0.268229`
- receiver minus packet-only: `-0.049479`
- target score cache hit: `False`

## Interpretation

This is a strict receiver-family scout: the source packet comes from TinyLlama, while the target-side scores come from a non-Qwen model family. A pass would show that the same fixed-byte packet has utility for a different target family and that a target-aware receiver beats the packet-only baseline. A fail still helps the paper by narrowing the claim to source-private packet utility rather than learned cross-family latent reasoning.
