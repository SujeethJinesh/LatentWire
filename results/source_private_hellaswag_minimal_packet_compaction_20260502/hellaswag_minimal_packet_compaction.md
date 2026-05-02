# HellaSwag Minimal Packet Compaction

- pass gate: `True`
- rows covered: `30126`
- original packet: `2B` raw / `5B` framed
- compact packet: `1B` raw / `4B` framed
- theoretical payload bits for four candidates: `2`
- prediction-equivalent rows: `30126/30126`
- qwen mean-zscore compact accuracy: `0.526688`
- qwen hybrid compact accuracy: `0.532464`
- tiny compact accuracy: `0.619199`

## Interpretation

The HellaSwag hidden-innovation decoder only needs the selected candidate id at runtime. The previous second byte was a confidence/debug field and is not used to decode the answer. This compaction preserves the promoted Qwen and TinyLlama packet predictions exactly while reducing the logical packet from 2B raw / 5B framed to 1B raw / 4B framed. It strengthens the systems rate-frontier contribution, but it does not solve the cross-family receiver/common-language gap.
