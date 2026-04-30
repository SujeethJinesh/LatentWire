# Source-Private Anchor-Relative Crosscoder Receiver Gate

- pass gate: `False`
- train/eval: `core:512` / `holdout:256`
- candidate view: `diag_only`
- diagnostic table mode: `plausible_decoys`
- receiver kind: `ridge`
- packet feature mode: `anchor_relative`
- packet dim: `128`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best control | Text relay | Top knockout | Oracle | CI95 low target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.277 | 0.250 | 0.266 | 0.258 | 0.277 | 0.641 | -0.039 |
| 8 | `False` | 0.270 | 0.250 | 0.258 | 0.207 | 0.270 | 0.742 | -0.043 |

## Pass Rule

Matched anchor-relative packet receiver must beat target-only and best source-destroying control by >=0.15, keep every source-destroying control within target+0.02, avoid being explained by matched-byte structured text, keep full diagnostic oracle >=0.95, have paired CI95 lower bounds >=0.10 versus target and best control, and preserve exact ordered-ID parity.

## Lay Summary

The source sees a private clue and sends a tiny sign-code fingerprint. The target sees only the public question and candidate pool plus that fingerprint. Controls replace, mask, shuffle, or publicly derive the fingerprint to check that any gain comes from private evidence.
