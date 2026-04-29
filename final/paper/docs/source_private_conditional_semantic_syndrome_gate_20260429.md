# Source-Private Conditional Semantic Syndrome Gate

- date: `2026-04-29`
- status: failed as an ontology-robust learned-method gate
- result root: `results/source_private_conditional_semantic_syndrome_gate_20260429/`
- scale rung: smoke / strict-small falsification

## Current Readiness

This gate was designed to attack the main weakness of the shared sparse atom
packet: ontology coupling. The learned conditional semantic syndrome sends a
tiny sign-syndrome packet from source-private evidence and decodes it against
target candidate side information under the synonym-stress candidate view.

It does not promote. The result is still valuable because it prevents an
overclaim: a simple ridge-learned semantic residual does not yet replace the
hand-auditable sparse dictionary under ontology shift.

## Result

- pass gate: `false`
- train/eval: `128/64`
- candidate view: `synonym_stress`
- budgets: `2`, `4`, `8` bytes
- cross-family pass: `false`
- direction pass: all `false`
- max syndrome accuracy: `0.797`
- max lift vs target: `+0.547`
- pass rows: `0`
- oracle candidate residual: `1.000`

## Rows

| Direction | Budget | Syndrome | Target | Best control | Best control name | Delta target | CI95 low | Oracle | Pass |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| core -> holdout | 2 | 0.172 | 0.250 | 0.328 | wrong projection | -0.078 | -0.234 | 1.000 | `false` |
| core -> holdout | 4 | 0.203 | 0.250 | 0.250 | zero source | -0.047 | -0.203 | 1.000 | `false` |
| core -> holdout | 8 | 0.281 | 0.250 | 0.297 | random same-byte | +0.031 | -0.156 | 1.000 | `false` |
| holdout -> core | 2 | 0.641 | 0.250 | 0.375 | answer-masked source | +0.391 | +0.281 | 1.000 | `false` |
| holdout -> core | 4 | 0.703 | 0.250 | 0.500 | answer-masked source | +0.453 | +0.344 | 1.000 | `false` |
| holdout -> core | 8 | 0.500 | 0.250 | 0.500 | answer-masked source | +0.250 | +0.063 | 1.000 | `false` |
| same-family all | 2 | 0.766 | 0.250 | 0.438 | public-only source | +0.516 | +0.328 | 1.000 | `false` |
| same-family all | 4 | 0.797 | 0.250 | 0.438 | public-only source | +0.547 | +0.344 | 1.000 | `false` |
| same-family all | 8 | 0.797 | 0.250 | 0.500 | public-only source | +0.547 | +0.344 | 1.000 | `false` |

## Interpretation

The smoke shows that the target-side synonym surface has enough oracle
headroom, but the learned syndrome is not clean. Same-family and one
cross-family direction have positive matched rows, yet source-destroying
controls also rise. Core -> holdout mostly fails outright.

Do not promote this branch. The next learned method must explicitly train
synonym-invariant dictionaries or consistency constraints, and it must preserve
controls before accuracy matters.

## Next Gate

Either:

1. Train a learned shared dictionary/crosscoder with blocked synonym clusters
   and atom/code derangement controls, or
2. Add a target-preserving abstention gate that rejects packets when
   source-destroying controls resemble matched packets.

Promotion requires bidirectional cross-family synonym-stress pass with all
source-destroying controls within target + `0.03`.
