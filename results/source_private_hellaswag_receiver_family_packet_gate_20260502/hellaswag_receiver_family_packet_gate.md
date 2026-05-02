# HellaSwag Receiver-Family Packet Gate

- pass gate: `False`
- target-family transfer gate: `True`
- receiver-improvement gate: `False`
- source family: `TinyLlama`
- target family: `Qwen2.5`
- train/eval split: validation `0:1024` train, `1024:10042` eval
- selected receiver: `target_margin_accept_packet`
- target-only eval accuracy: `0.483034`
- packet-only eval accuracy: `0.629741`
- receiver eval accuracy: `0.627190`
- delta vs target-only: `0.144156`
- delta vs packet-only: `-0.002550`
- oracle target-or-packet eval accuracy: `0.683744`

## Interpretation

This scout tests whether a different-family target score receiver can use the TinyLlama hidden-innovation packet. It shows target-family utility when the receiver is compared with Qwen target-only scoring, but it does not promote a true receiver-improvement claim unless the target-aware receiver beats packet-only. The remaining ICLR gap is therefore a receiver/common-language method that closes the target-or-packet oracle headroom rather than merely trusting the source-side packet.
