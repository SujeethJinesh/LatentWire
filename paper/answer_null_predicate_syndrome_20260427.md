# Answer-Null Predicate Syndrome

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: smoke
- decision: kill this structured predicate-syndrome heuristic

## Question

Can a source sidecar transmit only non-answer predicates, decoded against
target-side candidate context, and recover clean source-needed examples without
candidate values, candidate IDs, source final numbers, verified answer numbers,
or residue hashes?

## Method

`scripts/analyze_answer_null_predicate_syndrome.py` extracts answer-masked
source predicates:

- operation class: add/subtract/multiply/divide
- relation class: difference, total, each/per, ratio
- unit tokens from a fixed SVAMP-relevant vocabulary
- equation shape with all numeric values replaced by `N`

The receiver computes predicate overlap against target-side candidate text. The
transmitted syndrome does not include candidate values or source final numbers.

Controls:

- matched source predicates
- shuffled-source predicates
- random same-size predicate syndrome
- target-only
- slots-only

## Commands

```bash
./venv_arm64/bin/python scripts/analyze_answer_null_predicate_syndrome.py \
  --target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --min-confidence 0.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.json \
  --output-md results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.md
```

```bash
./venv_arm64/bin/python scripts/analyze_answer_null_predicate_syndrome.py \
  --target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --min-confidence 0.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.json \
  --output-md results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.md
```

## Evidence

At confidence `0.0`:

| Surface | Status | Matched Correct | Accepted | Clean Correct | Accepted Harm | Control Clean Union |
|---|---|---:|---:|---:|---:|---:|
| live | `answer_null_predicate_syndrome_fails_smoke` | `16/70` | `45` | `0` | `12` | `0` |
| holdout | `answer_null_predicate_syndrome_fails_smoke` | `13/70` | `46` | `1` | `5` | `1` |

The one holdout clean ID, `ab1e71e8928661d0`, is not source-necessary because
the random same-size syndrome and shuffled-source controls also recover it.
Threshold sweeps at `0.1`, `0.5`, and `1.0` on both live and holdout recover no
source-necessary clean IDs.

## Decision

Kill this heuristic predicate-syndrome branch. The predicates are answer-null,
but they are too generic: they produce many accepted harms and the one clean
holdout recovery is explained by controls.

This narrows the next viable direction to either learned answer-null verifier
features or fresh target/source surface generation. Existing stored artifacts
do not support another cheap positive gate.

## Artifacts

- `results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.md`
- `results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.md`

Hashes:

- `results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.json`:
  `70f093a89fb99d485ce86b038fe327ec2cdbbdd9c847a65e04261cb089c562fe`
- `results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.json`:
  `a5d6bf035c641f301d2f497f6b33219687ce76ca894ab45e4abb279417737f00`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_answer_null_predicate_syndrome.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_answer_null_predicate_syndrome.py
```

Result: `2 passed`; compile passed.

## Next Gate

The next meaningful gate requires a fresh surface or learned answer-null
features. Because PID `31103` remains stuck in `STAT=UE`, do not start MPS
generation. Resume with:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If clear, run a fresh strict small same-family target/source surface with source
final-answer masking from the first pass.
