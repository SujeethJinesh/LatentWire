# Condition Likelihood Receiver Live Feasibility

- date: `2026-04-27`
- status: `condition_likelihood_receiver_live_matched_feasibility_fails`
- scale-up rung: smoke / branch kill
- git commit at run: `2479a1898d6bd1ccd21166acb5bcf400f77556ad`

## Question

Before spending CPU on condition-specific receiver controls and holdout, can the
matched Qwen3 receiver likelihood sketch satisfy the live gate by itself?

## Result

Matched-only live CV fails before controls:

- live status: `fails`
- failing criteria: `min_correct`, `min_clean_source_necessary`,
  `max_accepted_harm`
- matched correct: `15/70`
- matched accepted: `14`
- clean source-necessary IDs: `4d780f825bb8541c`
- duplicate-answer clean IDs: none
- accepted target-correct harm: `7`

This kills the condition-specific target-likelihood receiver on the current
SVAMP70 live surface. Fair controls are still useful infrastructure, but there
is no reason to collect the remaining receiver sketches for this failed source
surface.

## Harness Updates

- Added `scripts/build_condition_likelihood_candidate_pools.py`.
- Updated `scripts/analyze_condition_likelihood_receiver_gate.py` to track
  canonical duplicate answers and subtract duplicate clean IDs from
  source-necessary counts.
- Added `tests/test_build_condition_likelihood_candidate_pools.py`.
- Extended `tests/test_analyze_condition_likelihood_receiver_gate.py`.

Control pool definitions now preserve same-slot conditions where appropriate:

- `matched`: target/text/source unchanged.
- `zero_source`: target/text unchanged, source blanked.
- `shuffled_source`: target/text unchanged, source replaced by deterministic
  off-example source answer with correctness recomputed.
- `label_shuffle`: labels are permuted before scoring; source content occupies
  the target-labeled slot and target content occupies the source-labeled slot.
- `target_only`: three slots remain, but non-target slots are filled with the
  target output.
- `slots_only`: target remains and non-target slots are blanked.

## Artifacts

- `results/condition_likelihood_receiver_live_feasibility_20260427/live_matched_feasibility.json`
  - sha256: `06dafa60c62724f440965d90d22af799dfacb4f3f8704904202c2984bf7b7fe7`
- `results/condition_likelihood_receiver_live_feasibility_20260427/live_matched_feasibility.md`
  - sha256: `dfa9b37a27ebd5ce2154c8440b98128e83a9f43478040c4d7e03d2e47afc623a`
- `results/qwen3_condition_likelihood_receiver_20260427/live_candidates/manifest.json`
- `results/qwen3_condition_likelihood_receiver_20260427/live_candidates/manifest.md`
- `results/qwen3_condition_likelihood_receiver_20260427/holdout_candidates/manifest.json`
- `results/qwen3_condition_likelihood_receiver_20260427/holdout_candidates/manifest.md`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_build_condition_likelihood_candidate_pools.py tests/test_analyze_condition_likelihood_receiver_gate.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py tests/test_collect_source_likelihood_sketch.py -q
./venv_arm64/bin/python -m py_compile scripts/build_condition_likelihood_candidate_pools.py scripts/analyze_condition_likelihood_receiver_gate.py
```

Result: `14 passed in 0.12s`; compile passed.

## Next Gate

Do not collect more Qwen3 target-likelihood receiver controls on this SVAMP70
surface. If MPS clears, resume source-surface/interface reset. If MPS remains
blocked, the next CPU-only work should use the new control harness only on a
stronger source candidate surface, not on this killed receiver branch.
