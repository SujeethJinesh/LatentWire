# Channel-Set Backfill Readiness

- foreground GPU: `0`
- C_A1 readiness: `BACKFILL_READY`
- C_F readiness: `NEEDS_OFFLINE_CONTROL_CLEANUP`
- CE13 readiness: `NEEDS_TINY_CACHE`
- C_A2 readiness: `TESTS_ONLY`

## C_A1 Tail/CVaR

Smallest useful GPU job is a backfill parity card, not foreground confirmation: replay exact cached C_A1 gate IDs with native W4A16/ParoQuant forwards and keep `promotion_allowed: false`.

Offline headroom is limited but nonzero: Stage-1 triage reports 6 parsed gate rows, 5 positive-median and 1 nonpositive. A native replay should decide whether tail/CVaR headroom is real or a cache artifact.

Command template:

```bash
python experimental/outlier_migrate/phase9/run_om_driftrot_clip_subset.py \
  --run-id <new_write_once_c_a1_granite_gate_replay> \
  --candidate-id clip_tight \
  --split-name diagnostic \
  --prompt-indices 7,9 \
  --base-run-dir experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z \
  --scale-clip-min 0.5 \
  --scale-clip-max 2.0 \
  --batch-size 1 \
  --dtype bfloat16
```

Estimated GPU time: 2-4 hours. No paper promotion until paired native rows beat matched-budget controls.

## C_F Control Contamination

Do not promote C_F. The direct M26 `random_matched_core` control does not explain the M26 artifact by itself, but the broader C_F family remains contaminated: Stage-1 triage is mixed, and M10 has a random-control-beats-method kill marker.

Minimal fix before any GPU foreground:

1. Build an identical-row denominator for static top-K, EMA/stable-core, and random matched controls.
2. Require positive method-minus-control separation on the same rows.
3. Only then write a native-forward packet.

Until then, C_F is backfill-only and may become killed if random/control rows explain the cleaned denominator.

## CE13 Warmup Policy

No parseable warmup-policy cache exists. Required first step is a tiny write-once dev/gate cache with locked row IDs and no confirm paths. GPU time now: 0 hours. Consider 2-4 GPU hours only after the cache exists.

## C_A2 And Sidecars

C_A2 remains tests-only: orthogonality, full-precision equivalence, and KV-cache basis consistency tests are prerequisite evidence, and no method screen is live. Sidecars remain parked/no-GPU unless revived by strict bytes/gain criteria.
