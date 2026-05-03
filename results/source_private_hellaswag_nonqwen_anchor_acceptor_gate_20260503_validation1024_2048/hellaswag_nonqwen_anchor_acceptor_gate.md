# HellaSwag Non-Qwen Anchor Acceptor Gate

- pass gate: `False`
- preservation gate: `True`
- slices: `2`
- range: `1024:2048`
- total eval rows: `768`
- target-only accuracy: `0.263021`
- packet-only accuracy: `0.506510`
- anchor acceptor accuracy: `0.506510`
- oracle accuracy: `0.619792`
- acceptor minus packet-only: `0.000000`
- max selected raw bytes: `1`
- selected configs: `candidate_only_no_override, candidate_only_no_override`

## Interpretation

This gate tests a packet-preserving acceptor. Unlike the failed score-simplex receiver, the receiver only sees a discrete source code plus the packet choice; if the fit/select split cannot justify overrides, the selected policy falls back to candidate-only packet.
