# Target-Only Sampling Source Sidecar Smoke

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: smoke
- live branch after this cycle: sampled target candidate pool plus compact
  source-derived candidate-score sidecar

## Question

After target-only stochastic sampling makes one remaining clean source-only ID
reachable in the target candidate pool, can a compact source-derived sidecar
select that candidate while source-destroying controls miss it?

## Commands

```bash
./venv_arm64/bin/python scripts/extend_target_set_candidate_labels.py \
  --base-target-set results/no_source_candidate_surface_20260427/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --candidate target_sample_s0=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s0 \
  --candidate target_sample_s1=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s1 \
  --candidate target_sample_s2=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s2 \
  --candidate target_sample_s3=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s3 \
  --candidate target_sample_s4=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s4 \
  --candidate target_sample_s5=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s5 \
  --candidate target_sample_s6=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s6 \
  --candidate target_sample_s7=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s7 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --output-md results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.md \
  --manifest-json results/target_only_sampling_clean3_20260427/sampled_clean3_target_set_manifest.json
```

```bash
./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set sampled_clean3=path=results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json,role=sampled_target_pool,note=zero_source_pool_plus_target_samples_clean3 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.json \
  --output-md results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.md
```

```bash
./venv_arm64/bin/python scripts/materialize_svamp_source_candidate_sidecars.py \
  --live-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --output-dir results/target_only_sampling_clean3_20260427/source_candidate_sidecars \
  --sidecar-bits 8 \
  --label-prior 0.0 \
  --date 2026-04-27
```

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --holdout-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --live-sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --holdout-sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --outer-folds 3 \
  --accept-penalty 0.05 \
  --harm-weight 4.0 \
  --min-live-correct 1 \
  --min-live-clean-source-necessary 1 \
  --min-holdout-correct 1 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke \
  --output-predictions-jsonl results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke/predictions.jsonl
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/top_sidecar_selector.json \
  --output-md results/target_only_sampling_clean3_20260427/top_sidecar_selector.md
```

## Evidence

The sampled clean3 target pool has target `0/3`, source `3/3`,
target-side oracle `1/3`, and clean-in-pool `1/3`. The reachable clean ID is
`14bfbfc94f2c2e7b`; its gold answer is `3` and the target candidate values
include `1`, `16`, `19`, `2`, `3`, `5`, and `8`.

The learned semantic-predicate decoder fails this tiny smoke because its
erasure rule accepts nothing:

- matched correct: `0/3`
- accepted: `0`
- clean source-necessary: `0`
- accepted harm: `0`

The deterministic top-candidate selector passes the smoke after fixing the
random-sidecar control to preserve score distribution while destroying the
candidate-value mapping:

| Condition | Correct | Accepted | Clean Correct | Accepted Harm |
|---|---:|---:|---:|---:|
| matched | `1/3` | `1` | `1` | `0` |
| shuffled source | `0/3` | `1` | `0` | `0` |
| random sidecar | `0/3` | `1` | `0` | `0` |
| target only | `0/3` | `0` | `0` | `0` |
| slots only | `0/3` | `0` | `0` | `0` |

Matched source selects value `3` for `14bfbfc94f2c2e7b` with confidence
`3.75`; randomized same-byte sidecar selects `5`; shuffled source either
abstains or selects a wrong value on non-clean IDs.

## Decision

Promote the branch only from source-surface discovery to smoke-positive. This is
not paper evidence yet: it is one clean ID on a three-example slice, with a
handwritten top selector. It is worth the next strict small gate because it
separates source-derived candidate scoring from target-only pool construction
and passes zero-source-equivalent controls on the reachable ID.

The branch should be killed if the next exact-ID expansion cannot produce more
reachable clean IDs without source leakage, or if repeated/randomized sidecar
controls explain the matched gains.

## Artifacts

- `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.md`
- `results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.md`
- `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/manifest.md`
- `results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke/semantic_predicate_decoder.md`
- `results/target_only_sampling_clean3_20260427/top_sidecar_selector.md`

Hashes:

- `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`:
  `786bc1a4c336483633e74a64d3d309602d0bcb79d758b8fc0190f311832d52ff`
- `results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.json`:
  `22ccec02fa0ee77d990511743ac3dd766b04f033695cb2865e0235fdb19e62fb`
- `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl`:
  `003adb88d1424f8b7b444d0972c85141d2cc889f29e27a0487ba48605ea7e66f`
- `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/manifest.json`:
  `b51345d0b2efab9d7bfb4a91ba032790aeacfbcb2040220f5593add014f0b050`
- `results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke/semantic_predicate_decoder.json`:
  `dd42726fb336d7df8423a5158eb0e5e5f392865f95a98d9d961b363939b9faa5`
- `results/target_only_sampling_clean3_20260427/top_sidecar_selector.json`:
  `58c79810a3a8bb19c2431f0a19f973af3a46a05d1094ee2172de62a915879aeb`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_candidate_score_sidecar_top_select.py -q
```

Result: `3 passed`.

## Next Gate

Run the cheapest strict small gate that can stay CPU/artifact-only while PID
`31103` remains stuck:

1. Materialize additional exact-ID target-only sampled rows if existing
   artifacts contain more clean source-only candidate values.
2. Add repeated random-sidecar controls or multiple deterministic salts so the
   same-byte control is not a lucky single draw.
3. Promote only if matched source keeps at least one more source-necessary clean
   win, control clean union remains `0`, accepted target-correct harm remains
   `0`, and the candidate pool provenance is target-only/no-source.

If no existing artifact expands reachable clean IDs, wait for MPS cleanup before
sampling a wider exact SVAMP32/GSM32 target-only candidate pool.
