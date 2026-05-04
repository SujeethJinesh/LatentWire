# Source-Private Conditional PQ Public-SVD Whitening Gate

- date: `2026-05-04`
- artifact: `results/source_private_conditional_pq_public_svd_whiten_gate_20260504/`
- code: `scripts/run_source_private_conditional_pq_innovation_gate.py`
- tests: `tests/test_run_source_private_conditional_pq_innovation_gate.py`
- references: `references/743_public_svd_whiten_conditional_pq_refs_20260504.md`
- status: failed held-out-family resurrection gate; keep as a ruled-out
  deterministic public-conditioning diagnostic, not an ICLR-positive method.

## Purpose

The previous public-zscore conditional-PQ gate failed because corrupted or
label-shuffled packets stayed too competitive with matched source packets. This
follow-up tested a slightly richer deterministic receiver-side geometry:
`--conditioning-mode public_svd_whiten`.

For each row, the receiver builds a public candidate innovation matrix,
subtracts the row mean, fits an SVD in that public candidate subspace, projects
both source-predicted innovations and candidate innovations through the same
whitened public subspace, and then applies the existing rate-capped
product-quantized packet decoder.

Layman version: the source still sends a tiny repair code. The receiver now
tries to build a better per-question decoder ring from the visible answer
choices by finding their strongest directions of variation first.

## Commands

Core to holdout:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_public_svd_whiten_gate_20260504/core_to_holdout_semantic_public_svd_whiten_n256 \
  --train-examples 768 --eval-examples 256 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set core --eval-family-set holdout \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view semantic \
  --source-topk 64 --target-topk 32 \
  --conditioning-mode public_svd_whiten \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

Holdout to core:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_public_svd_whiten_gate_20260504/holdout_to_core_semantic_public_svd_whiten_n256 \
  --train-examples 768 --eval-examples 256 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set holdout --eval-family-set core \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view semantic \
  --source-topk 64 --target-topk 32 \
  --conditioning-mode public_svd_whiten \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

## Results

| Direction | Pass | Source | Target | Best control | Source-control | CI95 low vs best | Oracle | Decision |
|---|---:|---:|---:|---|---:|---:|---:|---|
| core -> holdout | `false` | 0.375000 | 0.250000 | permuted codes, 0.613281 | -0.238281 | -0.320410 | 1.000000 | fail |
| holdout -> core | `false` | 0.250000 | 0.250000 | permuted codes, 0.750000 | -0.500000 | -0.558594 | 1.000000 | fail |

Important controls:

- `permuted_codes` reached `0.613281` core-to-holdout and `0.750000`
  holdout-to-core;
- `random_same_byte` reached `0.515625` core-to-holdout and `0.441406`
  holdout-to-core;
- `public_condition_only` tied source at `0.375000` core-to-holdout and tied
  target at `0.250000` holdout-to-core;
- `deranged_public_basis` and `opaque_slot_basis` stayed near target, so public
  candidate geometry matters, but not in a source-causal way;
- unquantized predicted accuracy stayed at target (`0.250000`) in both
  directions, while the target innovation oracle remained `1.000000`.

## Interpretation

This is a stronger negative than public-zscore. Projecting through the row's
public SVD subspace makes the code identities themselves act like an accidental
answer-slot transform. That is exactly what the `permuted_codes` and
`random_same_byte` controls are supposed to reveal.

The gate does not show source-private communication. It shows that a richer
deterministic public basis can amplify packet-code artifacts unless the receiver
has a learned integrity/no-op mechanism or a different target-native objective.

## Branch Decision

Do not widen `public_svd_whiten` to n500, remap repeats, or additional
benchmarks. The deterministic public-conditioning PQ path is weakened. The
next highest-value branch should be one of:

- a learned conditional codebook/receiver closer to QINCo-style implicit
  codebooks, with explicit corruption-to-no-op training and the same destructive
  controls;
- or a target-self-resonance encoder preflight that first proves compact packets
  can recreate target-native behavior before adding cross-model source
  conditioning.

For COLM_v2, this branch should be cited only as a failure analysis supporting
the need for destructive packet controls.
