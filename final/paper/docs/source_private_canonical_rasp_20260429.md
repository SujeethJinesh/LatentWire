# Source-Private Canonical RASP Gate

- date: `2026-04-29`
- rung: medium confirmation plus one larger frozen worst-remap slice
- live branch: source-private scalar packet with canonical RASP as a secondary
  robustness/systems contribution
- status: strengthened same-family/remap evidence; not a cross-family fix

## Method

Canonical RASP is an opt-in relative-score packet variant for the tool-trace
compression runner. The source computes one learned candidate-relative score per
public candidate, serializes the quantized scores in stable public candidate
identity order, and the target maps those bytes back to the displayed candidate
order before selecting the argmax.

This directly tests whether the relative-score packet is communicating source
evidence keyed to candidate identity rather than exploiting candidate display
position.

## Code Changes

- `scripts/run_source_private_tool_trace_compression_baselines.py`
  - adds `relative_scores_canonical`
  - adds canonical candidate-order serialization/decoding metadata
  - adds canonical source-destroying controls:
    - label-shuffled ridge
    - constrained shuffled source with answer-label mismatch
    - answer-masked source
    - random same-byte
    - order-mismatch bytes
    - permuted score bytes
- `scripts/summarize_source_private_relative_score_bootstrap.py`
  - adds `--method-condition relative_canonical_score_source`
- focused tests cover the new opt-in variant and canonical bootstrap summary.

## Artifacts

- `results/source_private_relative_canonical_remap101_20260429/`
- `results/source_private_relative_canonical_remap103_20260429/`
- `results/source_private_relative_canonical_remap107_20260429/`
- `results/source_private_relative_canonical_remap109_20260429/`
- `results/source_private_relative_canonical_remap113_20260429/`
- `results/source_private_relative_canonical_remap127_20260429/`
- `results/source_private_relative_canonical_remap131_20260429/`
- `results/source_private_relative_canonical_bootstrap_remap7_20260429/`
- `results/source_private_relative_canonical_remap127_large_20260429/`
- `results/source_private_relative_canonical_remap127_large_bootstrap_20260429/`
- `results/source_private_relative_canonical_core_to_holdout_20260429/`
- `results/source_private_relative_canonical_holdout_to_core_20260429/`

## Seven-Remap Medium Result

At `768/512`, `feature_dim=512`, `budget=4`, `candidate_view=slot`,
`fit_intercept=False`, canonical RASP is positive in point accuracy on every
remap and improves mean equal-byte accuracy over scalar by `+0.037`.

| Remap | Canonical RASP | Scalar | Target | RASP - scalar CI95 | RASP - target CI95 |
|---:|---:|---:|---:|---:|---:|
| 101 | 0.494 | 0.426 | 0.250 | [0.029, 0.107] | [0.184, 0.303] |
| 103 | 0.520 | 0.496 | 0.250 | [-0.014, 0.061] | [0.213, 0.328] |
| 107 | 0.506 | 0.502 | 0.250 | [-0.035, 0.043] | [0.199, 0.311] |
| 109 | 0.477 | 0.451 | 0.250 | [-0.008, 0.061] | [0.170, 0.281] |
| 113 | 0.473 | 0.436 | 0.250 | [0.002, 0.072] | [0.164, 0.279] |
| 127 | 0.453 | 0.428 | 0.250 | [-0.010, 0.061] | [0.146, 0.262] |
| 131 | 0.506 | 0.434 | 0.250 | [0.035, 0.109] | [0.197, 0.311] |

Bootstrap summary:

- pass gate: `false`
- mean canonical RASP accuracy: `0.490`
- mean canonical RASP minus scalar: `+0.037`
- minimum canonical RASP vs target CI95 low: `+0.146`
- minimum canonical RASP vs scalar CI95 low: `-0.035`

The seven-remap gate is close but not strictly passed because the worst
target-only paired CI lower bound is `+0.146`, under the `+0.150` rule.

## Larger Worst-Remap Slice

The worst remap by target CI, seed `127`, was rerun at `1536/1024`.

| Surface | Canonical RASP | Scalar | Target | RASP - scalar CI95 | RASP - target CI95 | Controls clean |
|---|---:|---:|---:|---:|---:|---:|
| remap 127 large | 0.442 | 0.361 | 0.250 | [0.053, 0.110] | [0.152, 0.233] | true |

This larger slice passes the canonical RASP gate and suggests the seven-remap
miss is likely finite-slice uncertainty, not a source-control failure.

## Cross-Family Falsification

| Train -> Eval | Canonical RASP | Scalar | Target | Controls clean | Interpretation |
|---|---:|---:|---:|---:|---|
| core -> holdout | 0.207 | 0.225 | 0.250 | false | fail |
| holdout -> core | 0.492 | 0.375 | 0.250 | true | one-direction pass |

Canonical ordering does not solve the asymmetric cross-family transfer failure.
Core-to-holdout has failed controls, including label-shuffled and order-mismatch
rows, and should not be represented as communication.

## Interpretation

Promote canonical RASP as a stronger secondary contribution than display-order
RASP:

- It uses the same 4-byte actual payload.
- It decouples packet byte order from candidate display order.
- It improves average remap robustness over scalar.
- It has a larger-slice pass on the previously worst remap.

Do not promote it as a full cross-family method. The next cross-family branch
should be the consistency-distilled posterior packet proposed by the subagent
review: train a canonical posterior packet under candidate-order and feature
drop perturbations, then require bidirectional core/holdout success before any
cross-family claim.

## Exact Commands

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_compression_baselines.py \
  tests/test_summarize_source_private_relative_score_bootstrap.py -q
```

Expanded summary:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_relative_score_bootstrap.py \
  --result-dirs \
  results/source_private_relative_canonical_remap101_20260429 \
  results/source_private_relative_canonical_remap103_20260429 \
  results/source_private_relative_canonical_remap107_20260429 \
  results/source_private_relative_canonical_remap109_20260429 \
  results/source_private_relative_canonical_remap113_20260429 \
  results/source_private_relative_canonical_remap127_20260429 \
  results/source_private_relative_canonical_remap131_20260429 \
  --output-dir results/source_private_relative_canonical_bootstrap_remap7_20260429 \
  --budget 4 --bootstrap-samples 2000 --seed 29 \
  --method-condition relative_canonical_score_source
```

Worst-remap larger slice summary:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_relative_score_bootstrap.py \
  --result-dirs results/source_private_relative_canonical_remap127_large_20260429 \
  --output-dir results/source_private_relative_canonical_remap127_large_bootstrap_20260429 \
  --budget 4 --bootstrap-samples 2000 --seed 29 \
  --method-condition relative_canonical_score_source
```
