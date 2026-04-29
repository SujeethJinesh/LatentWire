# Source-Private Shared Sparse N512 Boundary

- date: `2026-04-29`
- status: larger frozen-slice confirmation plus ontology-boundary falsification
- native result root:
  `results/source_private_shared_sparse_crosscoder_packet_gate_20260429_n512/`
- synonym-stress result root:
  `results/source_private_shared_sparse_crosscoder_packet_gate_20260429_n512_synonym_stress/`
- scale rung: large frozen slice for the shared sparse packet branch

## Current Readiness

The paper is stronger as a scoped source-private communication paper, but still
not a broad latent-transfer paper. The current strongest positive method is an
agreed-dictionary shared sparse packet: it communicates source-private evidence
through 4-8 byte sparse atom packets and decodes with target-side candidate
side information.

The exact blocker to a broader ICLR claim is unchanged: the same packet fails
when candidate-side ontology terms are paraphrased outside the hand-auditable
atom rules. That makes the current claim an interpretable source-private
protocol, not robust semantic latent transfer.

## Native N512 Confirmation

Command:

```bash
./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py \
  --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_n512 \
  --budgets 4 8 \
  --train-examples 512 \
  --eval-examples 512 \
  --seed 41
```

Outcome: pass.

| Direction | Budget | N | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 512 | `true` | 1.000 | 0.250 | 0.250 | +0.750 | +0.711 | 1.000 |
| core -> holdout | 8 | 512 | `true` | 1.000 | 0.250 | 0.256 | +0.750 | +0.713 | 1.000 |
| holdout -> core | 4 | 512 | `false` | 0.875 | 0.250 | 0.250 | +0.625 | +0.580 | 1.000 |
| holdout -> core | 8 | 512 | `true` | 0.875 | 0.250 | 0.252 | +0.625 | +0.582 | 1.000 |
| same-family all | 4 | 512 | `true` | 0.938 | 0.250 | 0.250 | +0.688 | +0.645 | 1.000 |
| same-family all | 8 | 512 | `true` | 0.938 | 0.250 | 0.250 | +0.688 | +0.648 | 1.000 |

Headline:

- cross-family pass: `true`
- direction pass: all `true`
- pass rows: `5`
- max shared sparse accuracy: `1.000`
- max shared-target delta: `+0.750`
- minimum passing CI95 lower bound vs target: `+0.582`
- top-atom knockout removes `100%` of matched-minus-target lift in passing rows

This materially strengthens the agreed-dictionary packet contribution: the
method is not only a strict-small artifact, and the atom-knockout diagnostic
remains causal on a larger frozen slice.

## N512 Synonym/Ontology Stress

Command:

```bash
./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py \
  --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_n512_synonym_stress \
  --budgets 4 8 \
  --train-examples 512 \
  --eval-examples 512 \
  --seed 41 \
  --candidate-atom-view synonym_stress
```

Outcome: fail.

| Direction | Budget | N | Pass | Shared sparse | Target | Best control | Delta target | CI95 low | Knockout reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4 | 512 | `false` | 0.375 | 0.250 | 0.250 | +0.125 | +0.098 | 1.000 |
| core -> holdout | 8 | 512 | `false` | 0.375 | 0.250 | 0.250 | +0.125 | +0.098 | 1.000 |
| holdout -> core | 4 | 512 | `false` | 0.250 | 0.250 | 0.250 | +0.000 | +0.000 | 0.000 |
| holdout -> core | 8 | 512 | `false` | 0.125 | 0.250 | 0.250 | -0.125 | -0.154 | 0.000 |
| same-family all | 4 | 512 | `false` | 0.312 | 0.250 | 0.250 | +0.062 | +0.043 | 1.000 |
| same-family all | 8 | 512 | `false` | 0.250 | 0.250 | 0.250 | +0.000 | -0.031 | 0.000 |

Headline:

- cross-family pass: `false`
- direction pass: all `false`
- pass rows: `0`
- max shared sparse accuracy: `0.375`
- max shared-target delta: `+0.125`

The failure is useful. It rules out the overclaim that the current sparse atom
packet is synonym-invariant semantic transfer. The method requires an agreed
source/target atom ontology. The next learned branch must either induce a
synonym-invariant shared dictionary or use target-preserving abstention when
the candidate ontology is out of distribution.

## Technical Contribution Status

The strongest three paper contributions after this gate are:

1. **Source-private evidence-packet benchmark and controls.** The benchmark
   isolates target priors, matched-byte relays, source-destroying controls,
   answer masking, target-derived sidecars, and exact-ID parity.
2. **Extreme-rate packet systems frontier.** The 2-byte packet remains the
   far-left systems row; query-aware text catches up only at higher bytes, and
   KV/cache baselines belong in a byte-floor/caveat table rather than as direct
   same-task competitors.
3. **Interpretable agreed-dictionary sparse packet.** The N512 native pass plus
   atom-knockout diagnostics show a causal source-private signal at 4-8 bytes,
   but only under an agreed ontology.

We should not promote the learned semantic-syndrome branch yet. The simple
learned residual failed synonym stress, and the larger sparse result says the
right next method is a learned synonym-invariant dictionary/crosscoder, not more
surface tuning of the sign-syndrome row.

## Reviewer Implication

This evidence helps answer:

- “Does the positive sparse packet survive beyond a tiny slice?” Yes, on native
  agreed-ontology `n=512`.
- “Is the gain causal?” Yes, top-atom knockout removes the lift.
- “Is this robust semantic latent communication?” No, synonym stress fails.
- “What should the paper claim?” Source-private extreme-rate communication with
  an interpretable agreed sparse dictionary, plus explicit negative boundaries.

## Next Exact Gate

Implement a learned synonym-invariant shared dictionary packet:

- train/eval split: `512/256` or `512/512`;
- candidate views: native and synonym-stress;
- packet budgets: `2`, `4`, `8` bytes;
- controls: zero-source, shuffled-source, public-only, answer-masked,
  target-derived, random same-byte, atom/dictionary ID derangement, top-feature
  knockout, matched-byte text, query-aware text;
- pass rule: native passes, synonym-stress preserves at least half of native
  lift or beats target by `>= +0.15`, controls remain within target + `0.03`,
  and causal knockout removes at least `50%` of lift.

If that gate fails, the final paper should freeze the method claim at
source-private agreed-dictionary packets and use the learned failures as honest
negative-boundary evidence.
