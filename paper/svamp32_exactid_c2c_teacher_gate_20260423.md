# SVAMP32 Exact-ID C2C Teacher Gate

Date: 2026-04-23

## Status

This run clears a stronger reproducibility gate for SVAMP, but it does not yet
clear a direct latent-source positive-method gate. The fresh exact-ID `N=32`
surface shows that C2C has substantial target-complementary headroom; raw
source-alone and text-to-text do not.

## Commands

Materialization:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/svamp_exactid_baselines32_20260423 \
  --limit 32 \
  --methods target source t2t c2c \
  --device mps \
  --max-new-tokens 64 \
  --continue-on-error
```

After hardening the materializer, the command was rerun in resume mode. All four
rows validated with `existing_output_validation=ok`, exact ordered ID parity,
unique IDs, sidecar config parity, and manifest-level SHA256 provenance.

Headroom scan:

```bash
./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface fresh_svamp32_source=target_path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,source_path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=fresh_svamp32_source \
  --surface fresh_svamp32_t2t=target_path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,source_path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,target_method=target_alone,source_method=text_to_text,note=fresh_svamp32_text \
  --surface fresh_svamp32_c2c=target_path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,source_path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,target_method=target_alone,source_method=c2c_generate,note=fresh_svamp32_c2c \
  --min-source-only 5 \
  --output-json results/svamp_exactid_baselines32_20260423/headroom_surfaces.json \
  --output-md results/svamp_exactid_baselines32_20260423/headroom_surfaces.md
```

## Readout

| Row | Correct | Source-only vs target | Oracle vs target | Status |
|---|---:|---:|---:|---|
| `target_alone` | 8/32 | n/a | n/a | baseline |
| `source_alone` | 5/32 | 3 | 11/32 | weak source-complementary surface |
| `text_to_text` | 2/32 | 1 | 9/32 | weak source-complementary surface |
| `c2c_generate` | 16/32 | 10 | 18/32 | strong source-complementary teacher surface |

All rows have exact ordered ID parity and unique IDs. Numeric extraction
coverage is complete except `source_alone`, which is `31/32`.

## Decision

The raw source-alone gate fails the predeclared direct-source threshold
(`3/32` source-only wins, below `5/32`). Do not train a connector that claims
direct latent-source transfer from this source-alone surface.

The C2C row is strong enough to justify a teacher/competitor branch: train a
small, rate-limited innovation connector against C2C-only wins, but require
zero-source and shuffled-source controls to collapse. If controls retain the
teacher-overlap wins, kill the branch as target-side repair/cache behavior.

Next exact gate: C2C-teacher innovation probe on the same SVAMP32 IDs, with
matched-source, zero-source, shuffled-source, and target-only rows, plus
win-ID provenance against the `10` C2C-only IDs.
