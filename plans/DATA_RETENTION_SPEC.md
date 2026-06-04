# plans/DATA_RETENTION_SPEC.md — write-once data + provenance (binding)

Aggregate JSON alone is not evidence. **Raw per-example / per-trace data must be kept**, or a good-looking morning brief can sit on unrecoverable evidence. These rules are binding for every runner.

## Write-once rule
All experiment outputs are **write-once. Never overwrite a result path.** Re-runs create a new `run_id`.

Every run creates:
```text
results/<stage>/<method_id>/<run_id>/result.json
results/<stage>/<method_id>/<run_id>/per_example.jsonl   # or per_trace.jsonl (Channel-Set)
results/<stage>/<method_id>/<run_id>/stdout.log
results/<stage>/<method_id>/<run_id>/stderr.log
results/<stage>/<method_id>/<run_id>/access_manifest.json # wrapped open() calls (leakage guard)
results/<stage>/<method_id>/<run_id>/env.json
results/<stage>/<method_id>/<run_id>/command.txt
```
**Aggregates are disposable; per-example/per-trace files are the source of truth.** The aggregator may be re-run from raw files at any time.

## Cache provenance (every cache carries these)
```text
cache_schema_version · dataset name + version · split name + split hash ·
model HF ID / local path / snapshot hash · quantization path · prompt/template hash ·
runner git commit · created_utc · sha256 of the cache payload
```

## Result-schema additions (numeric; the aggregator gates on numbers)
Add to the common envelope:
```jsonc
"provenance": {
  "run_id": "...",
  "parent_run_ids": ["..."],
  "data_schema_version": "v1",
  "cache_schema_version": "v1",
  "method_card_hash": "...",
  "runner_git_commit": "...",
  "models_lock_hash": "...",
  "baselines_lock_hash": "..."
}
```
**Channel-Set** — make the denominator explicit so a "ratio win" cannot be an artifact of no-gap traces (enforces the CE21 no-gap filter):
```jsonc
"denominators": {
  "n_traces_total": 12,
  "n_traces_positive_gap": 8,
  "n_traces_no_gap": 4,
  "n_layers": 40,
  "n_positions": 6
}
```
**LatentWire** — prove the packet is not repeating the source-copy failure (it currently follows the source choice at 0.995–0.999):
```jsonc
"leakage_audit": {
  "packet_predicts_source_top1_acc": 0.0,
  "mutual_info_packet_source_top1_bits": 0.0,
  "candidate_id_decodable_from_packet_acc": 0.0
}
```

## Audit-agent checks (every job, before GPU and before promotion)
- runner-contract compliance + numeric-only fields (no prose control labels);
- `split_hash`/`cache_hash`/`method_card_hash` present and matching the frozen manifest;
- `access_manifest.json` contains **no** `*_confirm*` path during Stage 1/2;
- baseline parity (re-derives a known baseline number);
- denominators present (Channel-Set) and `leakage_audit` present (LatentWire);
- null-method sentinels did not clear any promote gate.
Any violation → `KILLED`/`PARKED`, never silently used.
