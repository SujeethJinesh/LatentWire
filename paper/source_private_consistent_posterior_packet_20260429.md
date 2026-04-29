# Source-Private Consistent Posterior Packet Gate

- date: `2026-04-29`
- rung: cross-family falsification smoke plus larger frozen slice
- status: pruned as a cross-family fix; useful negative method ablation
- live branch after gate: scalar packet plus canonical RASP for same-family/remap
  evidence; systems-rate frontier is now the highest-priority strengthening gate

## Method

`consistent_posterior_packet` is an opt-in packet variant in the tool-trace
compression runner. It trains a NumPy ridge encoder to predict a smoothed
candidate posterior centroid rather than a single gold candidate vector. During
training, each example contributes multiple consistency views with source
feature dropout and negative-candidate dropout while always retaining the gold
candidate. At evaluation, the source emits one quantized canonical candidate
score byte per candidate, and the target decodes with the same canonical
identity mapping as canonical RASP.

This branch was motivated by JEPA/consistency-style objectives: predict an
abstract target-side posterior state rather than reconstructing the private
tool trace.

## Controls

The new row uses the same strict control family as canonical RASP:

- label-shuffled ridge
- constrained shuffled source with answer-label mismatch
- answer-masked source
- random same-byte
- order-mismatch bytes
- permuted-score bytes

The candidate row does not affect the historical scalar `pass_gate`.

## Artifacts

- `results/source_private_consistent_posterior_core_to_holdout_20260429/`
- `results/source_private_consistent_posterior_holdout_to_core_20260429/`
- `results/source_private_consistent_posterior_core_to_holdout_large_20260429/`
- `results/source_private_consistent_posterior_holdout_to_core_large_20260429/`

## Medium Cross-Family Result

At `768/512`, `feature_dim=512`, `budget=4`, `candidate_view=slot`,
`fit_intercept=False`:

| Train -> Eval | Consistent posterior | Canonical RASP | Scalar | Target | Controls clean | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| core -> holdout | 0.381 | 0.207 | 0.225 | 0.250 | true | improves failed direction but misses +15 rule |
| holdout -> core | 0.391 | 0.492 | 0.375 | 0.250 | true | positive vs target but worse than canonical RASP |

The medium slice showed a real gain in the previously failed core-to-holdout
direction, but it was only `+0.115` over target, below the `+0.150` rule.

## Larger Cross-Family Result

At `1536/1024`, same configuration:

| Train -> Eval | Consistent posterior | Canonical RASP | Scalar | Target | Controls clean | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| core -> holdout | 0.354 | 0.146 | 0.370 | 0.250 | false | fail: order-mismatch control matches source row |
| holdout -> core | 0.495 | 0.502 | 0.368 | 0.250 | true | pass, but not better than canonical RASP |

The large slice prunes the method as a cross-family solution. In the hard
core-to-holdout direction, the source row is below scalar and the
order-mismatch control is effectively identical to the source row
(`0.355` vs `0.354`), so the apparent signal is not a clean source-communication
effect.

## Interpretation

This branch should be reported as a serious negative/ablation:

- consistency distillation helped the previously failed direction on the
  medium slice;
- it did not survive the larger strict control check;
- it does not justify a cross-family claim;
- it strengthens the argument that the next full-paper work should prioritize
  systems-rate frontier and stronger external validity rather than more
  same-surface posterior tuning.

## Exact Commands

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_compression_baselines.py \
  tests/test_summarize_source_private_relative_score_bootstrap.py -q
```

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_compression_baselines.py \
  --output-dir results/source_private_consistent_posterior_core_to_holdout_large_20260429 \
  --train-examples 1536 --eval-examples 1024 \
  --train-family-set core --eval-family-set holdout \
  --candidates 4 --feature-dim 512 --budgets 4 \
  --train-seed 29 --eval-seed 30 --ridge 1e-2 \
  --candidate-view slot --no-intercept \
  --packet-variants relative_scores_canonical consistent_posterior_packet
```

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_compression_baselines.py \
  --output-dir results/source_private_consistent_posterior_holdout_to_core_large_20260429 \
  --train-examples 1536 --eval-examples 1024 \
  --train-family-set holdout --eval-family-set core \
  --candidates 4 --feature-dim 512 --budgets 4 \
  --train-seed 29 --eval-seed 30 --ridge 1e-2 \
  --candidate-view slot --no-intercept \
  --packet-variants relative_scores_canonical consistent_posterior_packet
```
