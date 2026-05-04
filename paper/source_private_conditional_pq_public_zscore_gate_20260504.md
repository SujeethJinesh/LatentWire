# Source-Private Conditional PQ Public-Zscore Gate

- date: `2026-05-04`
- artifact: `results/source_private_conditional_pq_public_zscore_gate_20260504/`
- code: `scripts/run_source_private_conditional_pq_innovation_gate.py`
- tests: `tests/test_run_source_private_conditional_pq_innovation_gate.py`
- references: `references/740_public_conditioned_conditional_pq_gate_refs_20260504.md`
- status: failed held-out-family resurrection gate; keep harness, do not
  promote as ICLR-positive method.

## Purpose

The conditional-PQ status memo promoted the next exact gate: keep the
source-private conditional innovation packet fixed, but replace the static
public basis with target-public conditioning. This first bounded attempt adds
`--conditioning-mode public_zscore`, which normalizes both predicted source
innovations and public candidate innovations by each row's public candidate
mean and scale before product quantization and decoding.

Layman version: the source still sends the same tiny repair code. The target
now adjusts its decoder ring for each question using only the public answer
choices, then tries to read the code.

## Commands

Core to holdout:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_public_zscore_gate_20260504/core_to_holdout_semantic_public_zscore_n256 \
  --train-examples 768 --eval-examples 256 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set core --eval-family-set holdout \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view semantic \
  --source-topk 64 --target-topk 32 \
  --conditioning-mode public_zscore \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

Holdout to core:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_public_zscore_gate_20260504/holdout_to_core_semantic_public_zscore_n256 \
  --train-examples 768 --eval-examples 256 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set holdout --eval-family-set core \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view semantic \
  --source-topk 64 --target-topk 32 \
  --conditioning-mode public_zscore \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

## Results

| Direction | Pass | Source | Target | Best control | CI95 low vs best | Oracle | Decision |
|---|---:|---:|---:|---|---:|---:|---|
| core -> holdout | `false` | 0.335938 | 0.250000 | permuted codes, 0.375000 | -0.125000 | 1.000000 | fail |
| holdout -> core | `false` | 0.453125 | 0.250000 | label-shuffled encoder, 0.441406 | -0.085938 | 1.000000 | fail |

Important controls:

- `public_condition_only` stayed at `0.250000` in both directions;
- `answer_masked_source` stayed at `0.250000` in both directions;
- `same_answer_slot_wrong_row_source` reached `0.320312` on core-to-holdout
  and `0.335938` on holdout-to-core, below matched but still nontrivial;
- `deranged_public_basis` and `opaque_slot_basis` stayed below target in both
  directions;
- but `permuted_codes` and `random_same_byte` reached `0.375000` on
  core-to-holdout, and `label_shuffled_encoder` reached `0.441406` on
  holdout-to-core.

## Interpretation

This weakens simple row-public normalization as the cross-family fix. The good
news is that the public conditioner itself did not answer the task: public-only
and answer-masked controls stayed at target-only. The failure is instead that
the conditioned space makes corrupted or label-shuffled packets too competitive
with matched packets.

The oracle remains `1.000000`, so target-side headroom exists. The missing
piece is source-causal packet decoding, not candidate-side separability.

## Branch Decision

Do not promote public-zscore conditional PQ to n500/remap repeats. Keep the
conditioning harness because it is reusable, but the next conditional-PQ branch
needs a stronger public-conditioned residual codebook or receiver trained to
decode corrupted packets to no-op. A pass must beat label-shuffled, permuted,
random same-byte, wrong-row, public-only, answer-masked, and deranged/opaque
controls with paired positive uncertainty before it can update the ICLR story.
