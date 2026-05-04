# Conditional PQ Integrity Threshold Gate

Date: `2026-05-04`

## Status

- COLM_v2: still viable as a scoped source-private packet paper using the
  same-family conditional-PQ evidence and honest negative controls.
- ICLR: still blocked. This gate does not provide the missing held-out-family
  positive method.
- Current story: conditional PQ can carry useful byte-scale source-private
  packets in shared-schema settings, but the held-out-family receiver still
  cannot tell valid packets from packet-shaped artifacts.

## Method

Implemented:

- `scripts/build_source_private_conditional_pq_integrity_threshold_gate.py`
- `tests/test_build_source_private_conditional_pq_integrity_threshold_gate.py`
- artifacts under
  `results/source_private_conditional_pq_integrity_threshold_gate_20260504/`
- references:
  `references/746_conditional_pq_integrity_threshold_refs_20260504.md`

The gate reuses the existing conditional-PQ packet and corruption/no-op
candidate receiver, but adds an explicit scalar integrity layer. The receiver
first decodes what the packet wants to do. Then an integrity score decides
whether to accept the packet or no-op to the target prior.

Candidate trust scores were selected on held-out train rows:

- `score_margin`
- `chosen_score`
- `max_similarity`
- `negative_min_l2`
- `margin_plus_similarity`
- `margin_minus_min_l2_001`

The eval controls remain the strict held-out-family controls: target-only,
label-shuffled encoder, constrained wrong-row source, same-answer-slot
wrong-row source, answer-masked source, public-condition-only, permuted codes,
random same-byte, deranged public basis, candidate roll, and opaque-slot basis.

## Command

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/build_source_private_conditional_pq_integrity_threshold_gate.py \
  --output-dir results/source_private_conditional_pq_integrity_threshold_gate_20260504/core_to_holdout_semantic_public_zscore_n256_w001 \
  --train-examples 768 --integrity-select-examples 256 --eval-examples 256 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set core --eval-family-set holdout \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view semantic \
  --source-topk 64 --target-topk 32 \
  --conditioning-mode public_zscore \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 \
  --receiver-noop-weight 0.01 \
  --bootstrap-samples 1000
```

## Result

The first decision surface fails, so the reverse direction was not run.

| Split | Fit | Select | Eval |
|---|---:|---:|---:|
| core -> holdout | 512 | 256 | 256 |

| Metric | Value |
|---|---:|
| selected score | `negative_min_l2` |
| selected threshold | `-1.563480` |
| target-only accuracy | `0.250000` |
| source accuracy | `0.425781` |
| best control | `label_shuffled_encoder` |
| best control accuracy | `0.457031` |
| source minus best control | `-0.031250` |
| CI95 low vs best control | `-0.097656` |
| source accept rate | `0.773438` |
| max corrupt accept rate | `1.000000` |
| unquantized predicted accuracy | `0.511719` |
| target innovation oracle accuracy | `1.000000` |

Condition table:

| Condition | Accuracy | Accept rate |
|---|---:|---:|
| `target_only` | `0.250000` | `0.000000` |
| `source` | `0.425781` | `0.773438` |
| `label_shuffled_encoder` | `0.457031` | `0.605469` |
| `constrained_shuffled_source` | `0.343750` | `0.722656` |
| `same_answer_slot_wrong_row_source` | `0.390625` | `0.746094` |
| `answer_masked_source` | `0.250000` | `1.000000` |
| `public_condition_only` | `0.250000` | `1.000000` |
| `permuted_codes` | `0.250000` | `0.183594` |
| `random_same_byte` | `0.261719` | `0.074219` |
| `deranged_public_basis` | `0.214844` | `0.773438` |
| `candidate_roll` | `0.222656` | `0.773438` |
| `opaque_slot_basis` | `0.269531` | `0.699219` |

## Interpretation

This gate shows that simple scalar integrity does not solve the held-out-family
conditional-PQ failure. The source packet is useful relative to target-only,
but the selected integrity rule still accepts corrupted packet families at
source-scale rates and loses to the label-shuffled encoder control.

The headroom is still visible: unquantized predicted accuracy is `0.511719`
and the target innovation oracle is `1.000000`. The failure is therefore not
"no possible signal"; it is that this byte packet plus scalar integrity rule
does not preserve source-causal signal cleanly enough.

## Decision

Do not continue conditional-PQ held-out-family rescue work by adding only:

- deterministic public transforms;
- no-op weight sweeps;
- scalar integrity thresholds over the same receiver scores.

The next ICLR branch needs a qualitatively different source-causal interface
or a benchmark where source quality and complementarity are easier to separate
from packet artifacts. For COLM_v2, this is a useful negative row showing why
source-private utility must be reported with label-shuffle, wrong-row,
candidate-roll, random same-byte, and packet-integrity controls.

## Lay Explanation

We taught the receiver to ask, "does this tiny packet look trustworthy?" before
using it. It did use many real packets, but it also trusted fake packet
patterns too often. So this trust rule is not good enough for ICLR.
