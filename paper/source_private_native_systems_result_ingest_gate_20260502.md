# Native Systems Result Ingest Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets, public-basis ARC/OpenBookQA
  evidence, byte/exposure systems accounting, and a falsification ladder are
  the defensible core.
- Exact gap: native systems claims cannot be made until NVIDIA/vLLM/SGLang and
  cache-communication baselines are ingested with complete quality, latency,
  memory, traffic, byte, and source-exposure metrics.

## Gate

New script:
`scripts/validate_source_private_native_systems_results.py`

Artifact:
`results/source_private_native_systems_result_ingest_gate_20260502/`

Inputs:

- `results/source_private_native_systems_benchmark_plan_20260501/native_systems_metric_schema.csv`
- `results/source_private_native_systems_benchmark_plan_20260501/native_systems_baseline_rows.csv`
- optional future native measurement CSV/JSON/JSONL rows via `--measurement`

The validator turns the existing native systems runbook into an enforceable
gate. It checks that every required native row includes all required fields,
type-valid values, source-exposure flags, payload bytes, transferred source
state bytes, and row IDs tied to the required baseline set.

## Result

- validator pass: `True`
- native systems complete: `False`
- paper native win allowed: `False`
- measurement rows ingested: `0`
- required baseline rows: `11`
- missing required rows: `11`
- invalid measurement rows: `0`
- claim-boundary matrix:
  `results/source_private_native_systems_result_ingest_gate_20260502/native_systems_claim_boundary_matrix.csv`

Missing rows:

- `latentwire_packet_cached_source`
- `latentwire_packet_end_to_end_source_scoring`
- `target_only_vllm`
- `target_only_sglang`
- `same_byte_visible_text`
- `source_label_copy_control`
- `c2c_cache_to_cache`
- `kvcomm_selective_kv`
- `kvcomm_online_cross_context`
- `qjl_1bit_source_state`
- `turboquant_lowbit_source_state`

## Decision

Promote this as the systems-side reviewer guardrail. It gives the paper a
credible systems contribution now: strict boundary accounting plus a native
ingest gate that refuses premature GPU/HBM/throughput claims.

The new claim-boundary matrix makes the guardrail paper-facing: each required
row now states whether throughput, latency, and memory-traffic claims are
allowed. In the current no-measurement state, LatentWire rows are allowed only
as Mac-local packet byte/exposure accounting; C2C/KVComm/QJL/TurboQuant rows
are marked as required source-state comparators not yet measured; and every
native throughput/HBM claim remains forbidden.

Do not claim native systems superiority yet. The gate correctly marks native
systems incomplete until the required NVIDIA rows are measured and ingested.

## Lay Explanation

This checker is a checklist with teeth. It will not let us say the system is
faster or more memory efficient on GPUs until every required method is measured
with the same accuracy, latency, memory, traffic, byte, and privacy fields.

## Llama-8B Source Scout Blocker

I also added
`scripts/build_source_private_arc_challenge_llama8b_disagreement_source_scout.py`
as the bounded next source branch. A two-row preflight showed that the locally
cached Meta-Llama-3.1-8B-Instruct model can load on MPS with `float16`, but the
full frozen-disagreement run hit an Apple MPS attention-shape failure:

`LLVM ERROR: Failed to infer result type(s): "mps.matmul"`

This is an environment/hardware blocker, not a scientific negative result. The
script is ready to rerun with a fixed attention path, CPU fallback, or NVIDIA.

## Next Gate

If NVIDIA access arrives, run either:

- the full Llama-8B source-family gate against ARC validation/test; or
- a learned query/cache connector with matched source/target activations.

For Mac-local work, the next highest-value step is to ingest real native rows
once the user provides them, then regenerate the systems boundary table with
`native_systems_complete` still guarded by this validator.
