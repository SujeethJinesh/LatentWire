# HellaSwag Non-Qwen Score-Simplex Receiver Gate

- pass gate: `False`
- slices: `2`
- range: `1024:2048`
- total eval rows: `768`
- target-only accuracy: `0.263021`
- packet-only accuracy: `0.506510`
- receiver accuracy: `0.442708`
- oracle accuracy: `0.619792`
- receiver minus packet-only: `-0.063802`
- packet-improvement slices: `0/2`
- destructive-control slices: `0/2`

## Interpretation

This cached gate tests the practical common-basis hypothesis suggested by the receiver-family failures: TinyLlama and Phi already share the four HellaSwag candidate slots, so row-centered score vectors are projected into a shared candidate-contrast basis and used by a target-side receiver. A pass requires beating packet-only, not only target-only.
