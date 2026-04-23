# SVAMP Exact-ID Baseline Materialization

Date: 2026-04-23

## Status

This is a dev-smoke reproducibility gate, not paper evidence. It validates that
fresh SVAMP generation baselines can be materialized as independent, resumable
method rows with strict ordered `example_id` parity before scaling to the
proposed `N=32`/`N=70` decision surface.

## Command

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/svamp_exactid_baselines_20260423 \
  --limit 5 \
  --methods target source t2t c2c \
  --device mps \
  --max-new-tokens 64 \
  --continue-on-error
```

The command was rerun after completion to validate resume behavior. All four
rows were skipped with `existing_output_validation=ok`.

## Result

Artifacts:

- `results/svamp_exactid_baselines_20260423/manifest.json`
- `results/svamp_exactid_baselines_20260423/manifest.md`
- `results/svamp_exactid_baselines_20260423/target_alone.jsonl`
- `results/svamp_exactid_baselines_20260423/source_alone.jsonl`
- `results/svamp_exactid_baselines_20260423/text_to_text.jsonl`
- `results/svamp_exactid_baselines_20260423/c2c_generate.jsonl`

Fresh `limit=5` readout:

| Method | Correct | Exact ordered IDs | Numeric coverage | Source-only vs target |
|---|---:|---:|---:|---:|
| `target` | 2/5 | true | 5/5 | n/a |
| `source` | 2/5 | true | 5/5 | 1 |
| `t2t` | 0/5 | true | 5/5 | 0 |
| `c2c` | 1/5 | true | 5/5 | 0 |

The source-only win ID for `source` is `d64f6e35083ffe8c`. The source/target
oracle is `3/5`.

## Decision

The runner is now adequate for the next gate, but the `limit=5` surface is too
small to decide method viability. The next exact gate is a fresh SVAMP32
materialization with `target`, `source`, `t2t`, and `c2c` rows. Pass condition:
`source` has at least `5/32` source-only wins with exact ordered ID parity and
near-complete numeric coverage; otherwise treat SVAMP primarily as a teacher or
upper-bound surface for C2C/text innovations rather than a direct latent-source
claim.
