# Source-Surface Answer-Masking Audit

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: source-surface discovery
- decision: no existing stored target-set surface supports an answer-masked
  source-sidecar gate

## Question

Do existing `results/` target-set surfaces contain clean source-necessary IDs
where the gold answer is present in the target-side receiver pool but is not
explained by the source final or source verified numeric answer?

## Command

```bash
./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root results \
  --date 2026-04-27 \
  --output-json results/source_surface_answer_masking_audit_20260427/audit.json \
  --output-md results/source_surface_answer_masking_audit_20260427/audit.md
```

## Evidence

The audit loaded `12` existing target-set surfaces with clean IDs and skipped
`3` non-loadable candidate JSONs. No loaded surface had an
answer-unexplained clean ID in the receiver pool.

Top surfaces by clean target-side pool headroom:

| Surface | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |
|---|---:|---:|---:|
| `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json` | `2` | `2` | `0` |
| `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json` | `1` | `1` | `0` |
| `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json` | `4` | `1` | `0` |
| `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json` | `10` | `1` | `0` |
| `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json` | `3` | `1` | `0` |

This confirms the clean3 failure generalizes across stored surfaces: whenever a
clean source-needed answer is already in the receiver pool, it is explained by
the source final or verified numeric answer.

## Decision

No current stored surface is suitable for the next answer-masked source-sidecar
gate. The project should not spend more CPU cycles tuning candidate-score
sidecars on these artifacts.

The next live direction is a new source/surface mechanism that either:

- masks final answer tokens before source-sidecar formation and tests whether
  non-answer reasoning cues still help, or
- generates a richer target-side candidate surface after MPS cleanup and
  requires answer-masked source controls from the first gate.

## Artifacts

- `results/source_surface_answer_masking_audit_20260427/audit.md`
- `results/source_surface_answer_masking_audit_20260427/audit.json`

Hash:

- `results/source_surface_answer_masking_audit_20260427/audit.json`:
  `7e6a3acf0cda9e0fb033695d0ab09496d83c0e9ca9f9f4b6013fbd10aeb6e816`

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_audit_source_surface_answer_masking.py \
  tests/test_materialize_sidecar_counterfactuals.py \
  tests/test_analyze_candidate_score_sidecar_top_select.py -q
```

Result: `7 passed`.

## Next Gate

If PID `31103` remains stuck, continue CPU-only literature and harness work for
an answer-masked communication interface. Once MPS is clear, generate a fresh
same-family SVAMP/GSM strict small surface with source final-answer masking
built in from the first pass.
