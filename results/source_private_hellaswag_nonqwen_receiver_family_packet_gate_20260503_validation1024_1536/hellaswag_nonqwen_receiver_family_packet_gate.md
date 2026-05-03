# HellaSwag Non-Qwen Receiver-Family Packet Gate

- pass gate: `False`
- target-family transfer gate: `True`
- receiver-improvement gate: `False`
- eval slice: `1024:1536`
- rows: `512`
- train/eval rows: `128/384`
- target family: `Phi-3-mini`
- target-only eval accuracy: `0.270833`
- packet-only eval accuracy: `0.489583`
- receiver eval accuracy: `0.481771`
- packet minus target-only: `0.218750`
- receiver minus packet-only: `-0.007812`
- target score cache hit: `False`

## Interpretation

This is a strict receiver-family scout: the source packet comes from TinyLlama, while the target-side scores come from a non-Qwen model family. A pass would show that the same fixed-byte packet has utility for a different target family and that a target-aware receiver beats the packet-only baseline. A fail still helps the paper by narrowing the claim to source-private packet utility rather than learned cross-family latent reasoning.
