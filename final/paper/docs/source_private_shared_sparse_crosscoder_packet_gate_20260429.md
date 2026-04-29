# Source-Private Shared Sparse Crosscoder Packet Gate

- date: `2026-04-29`
- status: passed as a strict-small learned-method successor gate
- result root: `results/source_private_shared_sparse_crosscoder_packet_gate_20260429/`
- scale rung: strict-small bidirectional cross-family

## Current Readiness

This is the strongest new method contribution after the endpoint evidence
packet. It moves beyond the public diagnostic-table protocol by sending a
source-private shared sparse atom packet and decoding through target-side
candidate atom overlap. It is still a controlled synthetic benchmark, not a
claim about arbitrary LLM hidden-state transfer.

The method is crosscoder-inspired rather than a trained neural crosscoder: it
uses a shared/private atom dictionary over source-private hidden-test evidence
and target candidate descriptions, transmits atom IDs plus quantized
magnitudes, and verifies causal relevance by atom knockout.

## Result

- pass gate: `true`
- train/eval: `256/128`
- directions: core -> holdout, holdout -> core, same-family all
- budgets: `4`, `8` bytes
- cross-family pass: `true`
- max shared sparse accuracy: `1.000`
- max shared-target delta: `+0.750`
- best source-destroying control: `0.250`
- minimum passing paired-bootstrap CI95 lower bound vs target: `+0.539`
- top-atom knockout lift reduction: `1.000`

## Rows

| Direction | Budget | N | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 128 | `true` | 1.000 | 0.250 | 0.250 | +0.750 | +0.672 | 1.000 |
| core -> holdout | 8 | 128 | `true` | 1.000 | 0.250 | 0.250 | +0.750 | +0.672 | 1.000 |
| holdout -> core | 4 | 128 | `false` | 0.875 | 0.250 | 0.250 | +0.625 | +0.539 | 1.000 |
| holdout -> core | 8 | 128 | `true` | 0.875 | 0.250 | 0.250 | +0.625 | +0.539 | 1.000 |
| same-family all | 4 | 128 | `true` | 0.938 | 0.250 | 0.250 | +0.688 | +0.602 | 1.000 |
| same-family all | 8 | 128 | `true` | 0.938 | 0.250 | 0.250 | +0.688 | +0.609 | 1.000 |

The `4` byte holdout -> core row misses only the oracle threshold. The `8` byte
row passes, so the bidirectional cross-family gate passes under the stated
rule.

## Seed Repeat

The seed-repeat confirmation at seed `31` also passes:

- result root:
  `results/source_private_shared_sparse_crosscoder_packet_gate_20260429_seed_repeat/`
- cross-family pass: `true`
- direction pass: core -> holdout `true`, holdout -> core `true`,
  same-family all `true`
- max shared sparse accuracy: `1.000`
- max shared-target delta: `+0.750`
- minimum passing paired-bootstrap CI95 lower bound vs target: `+0.539`
- top-atom knockout lift reduction: `1.000`
- source-destroying controls: remain at target `0.250`

This is a seed/remap stability check on the same controlled family generator,
not a new benchmark family. It supports stability of the sparse packet result
but does not replace larger frozen-slice or learned-dictionary confirmation.

## Controls

Source-destroying controls include zero-source, shuffled-source,
answer-masked-source, public-only sidecar, target-derived sidecar, random
same-byte packet, answer-only text, structured text truncated to the same byte
budget, and atom-ID derangement. All stay at target accuracy `0.250` in the
passing rows.

The interpretability control is causal: top-atom knockout collapses the method
to target in every passing row, removing `100%` of the matched-minus-target
lift.

## Interpretation

This revives the learned-method story in a bounded way. Previous learned WZ,
RASP, masked innovation, candidate-embedding, and static anchor-relative
interfaces failed or became asymmetric under cross-family controls. This gate
shows that a source-private shared sparse atom packet can generalize
bidirectionally on the controlled repair benchmark while preserving strict
source-destroying controls and causal atom diagnostics.

The claim should remain narrow. The current implementation uses a compact
hand-auditable atom dictionary, not a trained crosscoder over LLM activations.
That makes it interpretable and reproducible on a Mac, but reviewers may still
ask whether the result scales beyond the synthetic repair families. The next
promotion gate is a seed repeat plus a larger frozen slice; the next method
generalization gate is a learned shared dictionary/crosscoder with the same
source-destroying controls.

## Next Gate

Run a larger frozen-slice confirmation and then a learned shared-dictionary
variant:

```bash
./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py \
  --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_large \
  --budgets 4 8 \
  --train-examples 512 \
  --eval-examples 512 \
  --seed 37
```

Promotion beyond strict-small requires preserving paired CI lower bounds above
zero, keeping all source-destroying controls within target + `0.03`, and
showing the same atom-knockout behavior on the larger slice.
