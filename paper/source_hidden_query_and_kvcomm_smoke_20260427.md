# Source-Hidden Query and KVComm Smoke

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; no deployable method has
   passed live plus holdout source controls, seed stability, systems metrics,
   and cross-family falsification.
2. Current paper story: source-derived side information remains plausible, but
   shallow routers and direct source-hidden readouts are not stable enough.
3. Exact blocker to submission: MPS remains blocked by orphaned PID `31103`,
   and no reusable offline activation tensors exist for a full latent-injection
   smoke.
4. Current live branch or top candidates: no live method branch. Top next
   executable branch after MPS clears is fixed-budget KV/cache communication
   baseline, followed by zero-init gated latent side-information only if the
   baseline/surface gate justifies it.
5. Highest-priority gate for this cycle: run the cheapest CPU latent-sideinfo
   diagnostic and verify KVComm baseline tooling.
6. Scale-up rung: smoke / branch preparation.

## Source-Hidden Query Bottleneck Smoke

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_latent_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --probe-model query_bottleneck \
  --query-epochs 2 \
  --query-slots 4 \
  --moduli 2,3,5,7 \
  --feature-layers last \
  --device cpu \
  --dtype float32 \
  --min-numeric-coverage 31 \
  --output-json .debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.json \
  --output-md .debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.md
```

Result: `source_latent_syndrome_probe_fails_gate`.

- Matched: `11/32`.
- Zero-source: `14/32`.
- Shuffled-source: `14/32`.
- Label-shuffled: `14/32`.
- Target-only: `14/32`.
- Slots-only: `8/32`.
- Clean source-necessary IDs: `0`.

Artifact hashes:

- `.debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.json`:
  `033c9cff44bba273dc71a1fff39e626afdb8da8be05a118317f3264576db881c`
- `.debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.md`:
  `7ecdb0140d00d84d398a838c782325d50905e208439edde4fc0d4777dbaa4575`

Decision: direct source-hidden query-bottleneck syndrome readout is weakened
again. It does not justify MPS scale-up by itself.

## KVComm Tooling Smoke

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 1 \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl
```

Result: tooling smoke passed, method quality not evaluated.

- Accuracy: `0/1`.
- Generated tokens: `8`.
- CPU latency: `0.9505s`.
- Selected layers: `[1, 6, 2, 8, 7, 5, 4]`.

Artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl`:
  `ddfa80b562ebcda86e0e2578e33d7d010f18cb003b9f1bb326e0c6f9940eb64e`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl.meta.json`:
  `b051921a3089b8af7f8f2c3ef89aed8ffaf6c6edb3b563313374ce3e75abed40`

Direct script invocation initially failed with:

```text
ModuleNotFoundError: No module named 'latent_bridge'
```

Fix: `latent_bridge/kvcomm_eval.py` now bootstraps the repo root onto
`sys.path`, so both direct script invocation and module invocation work.
Verified:

```bash
./venv_arm64/bin/python latent_bridge/kvcomm_eval.py --help
./venv_arm64/bin/python -m py_compile latent_bridge/kvcomm_eval.py
```

## KVComm Source-Control Harness Smoke

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 4 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 2 \
  --source-control-modes all \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_cpu_smoke_predictions.jsonl
```

Result: source-control harness smoke passed, method quality not evaluated.

- Matched, zero-source, shuffled-source, and target-only final modes all reused
  the same matched-selected layers: `[1, 6, 2, 8, 7, 5, 4]`.
- Accuracy is `0/2` for every mode at `max_new_tokens=4`, so this is only a
  provenance/tooling check.
- Shuffled-source records correctly use nonmatching source IDs on the two-row
  smoke.
- Target-only bypasses KVComm generation; zero-source zeroes the matched source
  cache while preserving source-cache shape/device/dtype.
- MPS remained blocked by orphaned PID `31103`; this run was CPU-only.

Artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_cpu_smoke_predictions.jsonl`:
  `ce1dd54cb3e96056e821cc9397b61151ebe054e345c8bb8ba1347d45cd519ea6`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_cpu_smoke_predictions.jsonl.meta.json`:
  `0ae7d5a9f6f38a8fa51a36dca7de9828fbc4ff2881bf1c1b60b0b0f8187238d4`

## Literature Update

Added `references/470_kv_cache_latent_communication_baselines_refs.md`.

Primary-source update:

- C2C is now an ICLR 2026 baseline for KV-cache communication.
- KVComm is an ICLR 2026 selective-KV baseline with fixed communication budget.
- Vector translation motivates conservative fixed-blend cross-family latent
  ablations, but only after the C2C/KVComm baselines are logged.

## Decision

Do not promote direct source-hidden query bottlenecks. Fixed-budget KV/cache
communication now has a source-control harness contract, but only CPU smoke
evidence. A new zero-init gated latent side-information method should only be
run after this baseline contract is exercised on a real decision slice or a
stronger source surface is available.

## Next Exact Gate

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source scout from
`paper/postkill_historical_cpu_audit_20260427.md` or a one-example MPS KVComm
smoke, then scale KVComm to `svamp70_live` only if exact ID/numeric coverage is
preserved.
