# Source-Private Public-Only Receiver Ablation

- examples: `500`
- pass gate: `True`
- candidate view: `diag_only`
- train/eval ID overlap: `0`
- public-only accuracy: `0.178`
- target-only accuracy: `0.250`
- public minus target: `-0.072`
- CI95 vs target: `[-0.118, -0.022]`
- p50 latency ms: `0.0627`

Pass rule: Public-only candidate classifier should stay within +0.05 accuracy of target-only to rule out public candidate semantics as a sufficient explanation for packet gains.
