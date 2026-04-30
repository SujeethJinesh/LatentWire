# Source-Private Public-Only Receiver Ablation

- examples: `500`
- pass gate: `True`
- candidate view: `diag_only`
- train/eval ID overlap: `0`
- public-only accuracy: `0.142`
- target-only accuracy: `0.250`
- public minus target: `-0.108`
- CI95 vs target: `[-0.156, -0.060]`
- p50 latency ms: `0.0633`

Pass rule: Public-only candidate classifier should stay within +0.05 accuracy of target-only to rule out public candidate semantics as a sufficient explanation for packet gains.
