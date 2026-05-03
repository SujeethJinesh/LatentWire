# HellaSwag Receiver-Family Packet Gate

- pass gate: `False`
- target-family transfer gate: `False`
- receiver-improvement gate: `False`
- source family: `Qwen2.5`
- target family: `Phi-3-mini`
- train/eval split: validation `0:128` train, `128:512` eval
- selected receiver: `candidate_ridge_receiver`
- target-only eval accuracy: `0.255208`
- packet-only eval accuracy: `0.437500`
- receiver eval accuracy: `0.330729`
- delta vs target-only: `0.075521`
- delta vs packet-only: `-0.106771`
- oracle target-or-packet eval accuracy: `0.572917`

## Interpretation

This scout tests whether a different-family target score receiver can use the TinyLlama hidden-innovation packet. It shows target-family utility when the receiver is compared with Qwen target-only scoring, but it does not promote a true receiver-improvement claim unless the target-aware receiver beats packet-only. The remaining ICLR gap is therefore a receiver/common-language method that closes the target-or-packet oracle headroom rather than merely trusting the source-side packet.
