# Source-Private Anchor-Relative Crosscoder Receiver Gate

- pass gate: `False`
- train/eval: `holdout:512` / `core:256`
- candidate view: `diag_only`
- diagnostic table mode: `plausible_decoys`
- receiver kind: `ridge`
- packet feature mode: `anchor_relative`
- packet dim: `128`
- exact ID parity: `True`

| Budget bytes | Pass | Matched | Target | Best control | Text relay | Top knockout | Oracle | CI95 low target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `False` | 0.309 | 0.250 | 0.277 | 0.281 | 0.309 | 0.762 | -0.012 |
| 8 | `False` | 0.301 | 0.250 | 0.266 | 0.262 | 0.301 | 0.828 | -0.020 |

## Pass Rule

Matched anchor-relative packet receiver must beat target-only and best source-destroying control by >=0.15, keep every source-destroying control within target+0.02, avoid being explained by matched-byte structured text, keep full diagnostic oracle >=0.95, have paired CI95 lower bounds >=0.10 versus target and best control, and preserve exact ordered-ID parity.

## Lay Summary

The source sees a private clue and sends a tiny sign-code fingerprint. The target sees only the public question and candidate pool plus that fingerprint. Controls replace, mask, shuffle, or publicly derive the fingerprint to check that any gain comes from private evidence.
