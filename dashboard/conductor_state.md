# Conductor State

- stage: `stage0_cache_freeze_complete`
- branch: `codex-campaign`
- cache_units: `1087`
- row_entities: `8725`
- scan_mode: `metadata_file_level`
- split_counts: `{'confirm': 1793, 'dev': 5167, 'gate': 1765}`
- partition_status_counts: `{'named_dev_gate_no_confirm': 318, 'no_named_partition_detected': 557, 'prior_eval_rows_present_no_clean_confirm': 212}`
- planted_confirm_path_guard: `pass`
- planted_dev_path_allowed: `pass`

## Readiness

Stage 0 is complete for local cache handling. Existing cache rows/entities are frozen into dev/gate/confirm assignments in `splits/stage0_cache_rows_hash.jsonl`, and confirm-bearing source paths are listed in `splits/stage0_confirm_quarantine_hashes.txt`.

This run used `metadata_file_level`. With `metadata_file_level`, each discovered cache/result file is treated as the split entity; method-specific Stage-1 runners may do finer row parsing later, but they must still consume this manifest and preserve the confirm quarantine.

All Stage-1/2 screens must fit only on dev/gate entities. Any positive from these historical caches remains **PROVISIONAL** until held-out confirmation or fresh data exists.

## Saturation

- Alive locally: cache-only LatentWire and Channel-Set screening that consumes `splits/stage0_cache_rows_hash.jsonl`.
- Parked: CUDA, W4A16, 30B, long 20K-token decode, native vLLM/Nsight profiler, fresh confirmation data.
- Highest-priority next branch: implement the Stage-1 method runners against the frozen dev/gate manifests, beginning with LatentWire L-ScoreComp and Channel-Set C-A1/C-F offline recovery.
