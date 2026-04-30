# Source-Private Conditional PQ Innovation Gate

- date: `2026-04-30`
- artifact: `results/source_private_conditional_pq_innovation_gate_20260430/`
- code:
  - `scripts/run_source_private_conditional_pq_innovation_gate.py`
  - `scripts/summarize_source_private_conditional_pq_innovation_gate.py`
- tests:
  - `tests/test_run_source_private_conditional_pq_innovation_gate.py`
  - `tests/test_summarize_source_private_conditional_pq_innovation_gate.py`
- references: `references/543_conditional_pq_innovation_refs_20260430.md`
- status: promoted as a bounded positive method branch; not yet an ICLR-final
  unseen-family result

## Purpose

The previous PQ branch failed under disjoint train/eval IDs because the source
packet encoded a marginal projected source vector. This gate instead sends a
conditional innovation:

```text
source innovation = source(matched private log) - source(answer-masked private log)
target-side innovation = candidate representation - target-prior candidate representation
```

A ridge encoder maps the source innovation into the target/public innovation
basis, then a protected-Hadamard product codebook sends a 2- or 4-byte packet.
The target decodes against public candidate innovations. The design is a
Wyner-Ziv-style side-information protocol: send only what the target does not
already know, and require destructive packets to collapse back to target-only.

Layman version: the source does not send the whole private log. It sends a tiny
code for "what changed after seeing the hidden test." The target compares that
tiny code against public candidate fixes and should use it only when the public
basis matches.

## Commands

Main n500 shared-text row:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_shared_text_remap101_uph_constrained_label \
  --train-examples 768 --eval-examples 500 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set all --eval-family-set all \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --source-topk 64 --target-topk 32 \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

Main n500 anchor-relative row:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_anchor_relative_remap101_budget2_uph_constrained_label \
  --train-examples 768 --eval-examples 500 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set all --eval-family-set all \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --anchor-count 128 --basis-view anchor_relative \
  --source-topk 64 --target-topk 32 \
  --budget-bytes 2 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

Aggregate:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_conditional_pq_innovation_gate.py \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n256_shared_text_remap101_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_shared_text_remap101_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_shared_text_remap103_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_shared_text_remap107_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n256_anchor_relative_remap101_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_anchor_relative_remap101_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_anchor_relative_remap103_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_anchor_relative_remap107_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_anchor_relative_remap101_budget2_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_shared_text_remap101_budget2_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n256_core_to_holdout_shared_text_remap101_uph_constrained_label \
  --run-dir results/source_private_conditional_pq_innovation_gate_20260430/n256_holdout_to_core_shared_text_remap101_uph_constrained_label \
  --output-dir results/source_private_conditional_pq_innovation_gate_20260430/summary
```

## Results

Aggregate pass gate is `true`: all same-family disjoint n500 rows pass, and the
held-out-family rows fail cleanly rather than leaking through controls.

| Surface | Basis | Bytes | Remap | Pass | Source | Target | Best control | CI95 low vs control | Unique ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n256 all->all | shared text | 4 | 101 | `true` | 1.000 | 0.250 | 0.277 | +0.672 | 0.895 |
| n500 all->all | shared text | 4 | 101 | `true` | 1.000 | 0.250 | 0.274 | +0.684 | 0.872 |
| n500 all->all | shared text | 4 | 103 | `true` | 1.000 | 0.250 | 0.272 | +0.686 | 0.872 |
| n500 all->all | shared text | 4 | 107 | `true` | 1.000 | 0.250 | 0.274 | +0.684 | 0.872 |
| n256 all->all | anchor-relative | 4 | 101 | `true` | 0.996 | 0.250 | 0.258 | +0.684 | 0.973 |
| n500 all->all | anchor-relative | 4 | 101 | `true` | 0.996 | 0.250 | 0.260 | +0.694 | 0.950 |
| n500 all->all | anchor-relative | 4 | 103 | `true` | 0.996 | 0.250 | 0.262 | +0.692 | 0.942 |
| n500 all->all | anchor-relative | 4 | 107 | `true` | 0.998 | 0.250 | 0.256 | +0.702 | 0.940 |
| n500 all->all | anchor-relative | 2 | 101 | `true` | 1.000 | 0.250 | 0.276 | +0.682 | 0.594 |
| n500 all->all | shared text | 2 | 101 | `true` | 1.000 | 0.250 | 0.288 | +0.668 | 0.532 |
| n256 core->holdout | shared text | 4 | 101 | `false` | 0.281 | 0.250 | 0.273 | -0.012 | 0.625 |
| n256 holdout->core | shared text | 4 | 101 | `false` | 0.297 | 0.250 | 0.277 | -0.012 | 0.500 |

Important controls:

- train/eval ID intersection is `0` for all rows
- deranged public basis collapses to `0.000` on the positive n500 rows
- opaque slot basis stays at target-only `0.250`
- answer-masked source stays at target-only `0.250`
- constrained shuffled source stays near target (`0.254-0.266` on key rows)
- budget-2 rows reduce payload uniqueness substantially while preserving
  source accuracy

## Interpretation

This is the strongest positive-method row so far because it directly fixes the
disjoint-ID failure from the previous PQ receiver diagnostic. The fix is not a
larger codebook or a looser receiver. It is a change in what is transmitted:
conditional source innovation instead of marginal source state.

Promote this as contribution:

1. A Wyner-Ziv-style source-private conditional innovation packet.
2. A common-basis test showing shared-text and anchor-relative public bases pass,
   while opaque slot and deranged public bases collapse.
3. A rate/lookup-risk improvement: 2-byte n500 rows remain perfect while unique
   payload ratio drops to `0.532-0.594`.

Do not overclaim:

- This is not unseen-family transfer. Bidirectional core/holdout rows fail.
- This is not protocol-free latent communication. The target relies on a public
  side-information basis.
- This is not a native GPU/vLLM systems result.
- The task remains synthetic tool-trace repair; widening remains required.

## Readiness Impact

COLM workshop: stronger. We now have a positive method row with disjoint IDs,
strict destructive controls, a clear systems packet interface, and a precise
negative boundary.

ICLR full paper: improved but not comfortable. The paper still needs at least
one of:

- held-out-family or less synthetic benchmark success,
- a model-mediated receiver that consumes the same conditional packet,
- larger seed repeats beyond the current remap stability,
- native NVIDIA/vLLM telemetry for end-to-end serving claims.

Next exact gate: run a less synthetic or held-out-schema variant of conditional
innovation. If it fails, write the paper around shared-schema source-private
side-information communication rather than broad cross-family latent transfer.
