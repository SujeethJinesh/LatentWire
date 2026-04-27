# Masked Process-Verifier Sidecar

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: smoke
- decision: kill this heuristic masked process-verifier sidecar

## Question

Can source reasoning process features help choose target-side candidate values
after masking source final and verified numeric answer values?

## Method

`scripts/materialize_masked_process_verifier_sidecars.py` builds an
answer-masked process-overlap sidecar. For each example it:

- extracts source operation, equation-shape, non-answer numeric, and lexical
  features after replacing source final and verified numeric answers with
  `<ANS>`;
- extracts the same process features from each target-side candidate row;
- scores target-side candidate values by source/candidate process overlap;
- never adds source-only candidate values to the receiver pool.

## Commands

```bash
./venv_arm64/bin/python scripts/materialize_masked_process_verifier_sidecars.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --output-dir results/masked_process_verifier_sidecars_20260427 \
  --date 2026-04-27
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --sidecar-jsonl results/masked_process_verifier_sidecars_20260427/live_masked_process_sidecars.jsonl \
  --min-confidence 0.5 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/masked_process_verifier_sidecars_20260427/live_top_selector.json \
  --output-md results/masked_process_verifier_sidecars_20260427/live_top_selector.md
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --sidecar-jsonl results/masked_process_verifier_sidecars_20260427/holdout_masked_process_sidecars.jsonl \
  --min-confidence 0.5 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/masked_process_verifier_sidecars_20260427/holdout_top_selector.json \
  --output-md results/masked_process_verifier_sidecars_20260427/holdout_top_selector.md
```

## Evidence

Materialization:

- live: `70` examples, clean `6`, answer-excluded top `55`
- holdout: `70` examples, clean `2`, answer-excluded top `62`

Top-selector gate at confidence `0.5`:

| Surface | Status | Matched Correct | Accepted | Clean Correct | Accepted Harm | Control Clean Union |
|---|---|---:|---:|---:|---:|---:|
| live | `top_sidecar_selector_fails_smoke` | `21/70` | `0` | `0` | `0` | `0` |
| holdout | `top_sidecar_selector_fails_smoke` | `8/70` | `0` | `0` | `0` | `0` |

A threshold sweep on holdout at `0`, `0.1`, `0.25`, `0.5`, and `1.0` also
failed with no source-necessary clean IDs. This is not only a confidence cutoff
issue.

## Decision

Kill the heuristic process-overlap sidecar. It removes direct answer copying,
but the remaining operation/equation/lexical overlap is too weak to rank
source-needed target-side candidates. The next answer-null branch should use
more structured non-answer predicates or learned verifier features rather than
simple text overlap.

## Artifacts

- `results/masked_process_verifier_sidecars_20260427/manifest.md`
- `results/masked_process_verifier_sidecars_20260427/live_top_selector.md`
- `results/masked_process_verifier_sidecars_20260427/holdout_top_selector.md`

Hashes:

- `results/masked_process_verifier_sidecars_20260427/manifest.json`:
  `d7ff1fd198fcd0a2fc3200d0fc332c234f372d06133c6827e316823c1f3a10d3`
- `results/masked_process_verifier_sidecars_20260427/live_masked_process_sidecars.jsonl`:
  `03047945d55036b8269331ed2286b09ac9a83e1dae06d264ab32c5bc96d2a0d6`
- `results/masked_process_verifier_sidecars_20260427/holdout_masked_process_sidecars.jsonl`:
  `99d8f7fe9821125c2321a1e0f6ae6bbd354ef54016af7f36c1e2f72f7f3868a3`
- `results/masked_process_verifier_sidecars_20260427/live_top_selector.json`:
  `f132337726df48abd7f4f6051b7c56e07085edb94dad0410a975416fc398ef64`
- `results/masked_process_verifier_sidecars_20260427/holdout_top_selector.json`:
  `01967b1885a2536b29f573b7ae4e12c55d57f49293ef91551d4c73e48b1fc8cf`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_masked_process_verifier_sidecars.py -q
./venv_arm64/bin/python -m py_compile scripts/materialize_masked_process_verifier_sidecars.py
```

Result: `3 passed`; compile passed.

## Next Gate

Try the more structured answer-null predicate syndrome branch. Encode operation
sequence, quantity roles, equation-shape buckets, unit relation, and sign/order
predicates without candidate IDs, candidate values, source final numbers, or
residue hashes. Kill it if masked predicates cannot recover clean IDs on the
holdout surfaces where target-side pools already contain clean answers.
