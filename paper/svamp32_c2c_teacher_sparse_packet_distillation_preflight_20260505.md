# SVAMP32 C2C Teacher Sparse-Packet Distillation Preflight

- date: `2026-05-05`
- status: `c2c_teacher_sparse_packet_distillation_preflight_fails_deployable_method_oracle_bound_alive`
- COLM_v2 readiness impact: strengthens the C2C comparison and claim-boundary table
- ICLR readiness impact: still blocked; the oracle sparse packet exists, but no deployable source-derived predictor has cleared controls

## Why This Gate

The ARC tokenwise and candidate-local source-evidence gates failed, so the live
ICLR branch returned to the frozen SVAMP32 C2C surface where dense C2C has real
headroom: target-alone is `8/32`, C2C teacher is `16/32`, and C2C-only
target-complementary wins are `10`.

This gate asks whether the existing evidence already supports a deployable
sparse-packet distillation claim, or only an oracle bound.

## Implementation

Added `scripts/build_svamp32_c2c_teacher_sparse_packet_distillation_preflight.py`.

The script aggregates and validates:

- frozen SVAMP32 target/source/text/C2C generation rows;
- the C2C teacher innovation probe;
- the clean residual target-set split;
- the 1-byte oracle C2C-derived syndrome sidecar;
- failed source-hidden, source-token query, and C2C-prefill trace predictors.

Added `tests/test_build_svamp32_c2c_teacher_sparse_packet_distillation_preflight.py`.

## Artifacts

- `results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_20260505/summary.json`
- `results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_20260505/summary.md`
- `results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_20260505/manifest.json`

## Results

| Row | Kind | Correct | Teacher-only | Clean residual | Source-necessary clean | Control clean | Bytes | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `target_only` | observed baseline | 8/32 | 0 | 0 | 0 | 0 | 0 | baseline |
| `source_alone` | observed baseline | 5/32 | 1 | 0 | 0 | 0 | 0 | weak source |
| `same_byte_text_to_text` | observed baseline | 2/32 | 0 | 0 | 0 | 0 | 0 | weak text |
| `dense_c2c_teacher` | observed teacher | 16/32 | 10 | 6 | 0 | 0 | 0 | high-bandwidth teacher |
| `target_source_text_oracle_union` | oracle over deployable outputs | 12/32 | 1 | 0 | 0 | 0 | 0 | source/text outputs lack clean C2C residuals |
| `oracle_c2c_syndrome_targetpool` | oracle sparse packet bound | 14/32 | 6 | 2 | 2 | 0 | 1 | clears as bound, not method |
| `oracle_c2c_syndrome_augmentedpool` | oracle sparse packet bound | 15/32 | 7 | 3 | 3 | 0 | 1 | clears as bound, not method |
| `source_latent_syndrome` | deployable source-hidden probe | 9/32 | 2-3 | 0 | 0 | 0 | 1 | fails |
| `learned_query_syndrome` | deployable source-token probe | 9-10/32 | 2 | 0 | 0 | 0 | 1 | fails |
| `c2c_prefill_trace_syndrome` | deployable C2C-trace probe | 11-12/32 | 0 | 0 | 0 | 0 | 1 | fails |

## Decision

The branch is not dead, but the current deployable evidence is insufficient.
The dense C2C teacher remains the only strong complementary signal. The 1-byte
C2C-derived syndrome sidecar proves a compact packet can decode useful answers
when the packet already contains the teacher residue, but current source-final,
source-hidden, learned query-bottleneck, and prefill-trace predictors do not
produce that residue.

For COLM_v2, this is a useful baseline and claim-boundary artifact. For ICLR,
it rules out claiming that the current sparse packets solve C2C-style transfer.

## Hypothesis Update

- alive: sparse C2C-residue packets as an oracle target and as a hardware-friendly story if we can make the residue source-causal;
- weakened: source-final/text outputs, pooled source hidden states, source-token query bottlenecks, and prefill C2C traces as sufficient predictors;
- promoted: richer generation-time dense-teacher traces or a new source-causal residual objective;
- ruled out: claiming Sparse Resonance Packets beat C2C from the current SVAMP32 evidence.

## Next Exact Gate

Collect or generate richer generation-time dense-teacher traces, then train a
source-causal sparse residual packet that must recover at least `2/6` clean C2C
residual IDs while preserving target-self wins and passing zero-source,
source-shuffle, label-shuffle, target-only, and slots-only controls.

Lay explanation: C2C knows how to fix some target mistakes, and a 1-byte
oracle hint can sometimes point the receiver to the right answer. But our
current ways of predicting that hint from the source model do not work yet.
