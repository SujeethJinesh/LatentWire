# SVAMP32 C2C Teacher Sparse-Packet Distillation Preflight

- date: `2026-05-05`
- status: `c2c_teacher_sparse_packet_distillation_preflight_fails_deployable_method_oracle_bound_alive`
- deployable distillation pass: `False`
- oracle sparse sidecar alive: `True`
- target: `8/32`
- C2C teacher: `16/32`
- clean residual targets: `6`

## Evidence Table

| Row | Kind | Correct | Teacher-only | Clean residual | Source-necessary clean | Control clean | Bytes | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `target_only` | `observed_generation_baseline` | 8/32 | 0 | 0 | 0 | 0 | 0 | `` |
| `source_alone` | `observed_generation_baseline` | 5/32 | 1 | 0 | 0 | 0 | 0 | `` |
| `same_byte_text_to_text` | `observed_generation_baseline` | 2/32 | 0 | 0 | 0 | 0 | 0 | `` |
| `dense_c2c_teacher` | `observed_generation_baseline` | 16/32 | 10 | 6 | 0 | 0 | 0 | `` |
| `target_source_text_oracle_union` | `oracle_candidate_union_bound_not_deployable` | 12/32 | 1 | 0 | 0 | 0 | 0 | `` |
| `oracle_c2c_syndrome_targetpool` | `oracle_sparse_packet_bound_not_deployable` | 14/32 | 6 | 2 | 2 | 0 | 1 | `syndrome_sidecar_bound_clears_gate_not_method` |
| `oracle_c2c_syndrome_augmentedpool` | `oracle_sparse_packet_bound_not_deployable` | 15/32 | 7 | 3 | 3 | 0 | 1 | `syndrome_sidecar_bound_clears_gate_not_method` |
| `source_latent_syndrome_0` | `deployable_source_hidden_probe` | 9/32 | 2 | 0 | 0 | 0 | 1 | `source_latent_syndrome_probe_fails_gate` |
| `source_latent_syndrome_1` | `deployable_source_hidden_probe` | 9/32 | 3 | 0 | 0 | 0 | 1 | `source_latent_syndrome_probe_fails_gate` |
| `learned_query_syndrome_0` | `deployable_source_token_probe` | 10/32 | 2 | 0 | 0 | 0 | 1 | `learned_syndrome_probe_fails_gate` |
| `learned_query_syndrome_1` | `deployable_source_token_probe` | 9/32 | 2 | 0 | 0 | 0 | 1 | `learned_syndrome_probe_fails_gate` |
| `c2c_prefill_trace_syndrome_0` | `deployable_c2c_trace_probe` | 11/32 | 0 | 0 | 0 | 0 | 1 | `c2c_mechanism_syndrome_probe_fails_gate` |
| `c2c_prefill_trace_syndrome_1` | `deployable_c2c_trace_probe` | 12/32 | 0 | 0 | 0 | 0 | 1 | `c2c_mechanism_syndrome_probe_fails_gate` |

## Decision

- The dense C2C teacher remains the only strong complementary surface on this frozen slice.
- The 1-byte C2C-derived syndrome sidecar remains useful as an oracle bound, not as a deployable method.
- Existing deployable predictors from source final answers, source hidden summaries, source-token query bottlenecks, and C2C prefill traces do not recover the clean C2C residual IDs.
- Do not claim sparse packets beat or solve C2C from this evidence. The next method must predict the C2C residual from a genuinely source-causal signal or collect a richer dense-teacher trace.

## Next Gate

Collect or generate richer generation-time dense-teacher traces, then train a source-causal sparse residual packet that must recover at least 2/6 clean C2C residual IDs while preserving target-self wins and passing zero/shuffle/label-shuffle/target-only/slots-only controls.
