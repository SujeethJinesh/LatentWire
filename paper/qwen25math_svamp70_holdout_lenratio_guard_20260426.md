# Qwen2.5-Math -> Qwen3 SVAMP70 Holdout Length-Ratio Guard

- date: `2026-04-26`
- status: holdout replication failed
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval slice: SVAMP `chal-101` through `chal-170`
- scale-up rung: medium confirmation / disjoint holdout falsification

## Start Status

- current ICLR readiness: not ready
- current story: the original SVAMP70 source-sidecar row reached `26/70`
  with a one-byte source residue and no text relay
- exact blocker: determine whether the decoded length-ratio guard generalizes
  beyond the tuned/current SVAMP70 slice

## Baselines

| Method | Correct | Numeric Coverage | Notes |
|---|---:|---:|---|
| source-alone | 8/70 | 64/70 | source signal is weak on this slice |
| target-alone | 8/70 | 70/70 | harder target surface than prior SVAMP70 |
| text relay | 18/70 | 70/70 | strong verbose relay improvement |
| C2C | 37/70 | 70/70 | strong cache-communication headroom |

Source/target oracle is only `14/70`; after excluding text relay, clean
source-only IDs are `2`.

## Length-Ratio Sidecar

Policy: use the 1-byte source residue sidecar only when the source has a
numeric prediction and `source_prediction_chars / target_prediction_chars <=
1.0`; otherwise keep target. This is the parameterized form of the
`shorter_than_target_numeric` guard.

| Moduli | Bytes | Matched | Clean Matched | Clean Necessary | Control Clean Union | Status |
|---|---:|---:|---:|---:|---:|---|
| 2,3 | 1 | 9 | 1 | 0 | 2 | fail |
| 2,3,5 | 1 | 10 | 1 | 0 | 2 | fail |
| 2,3,5,7 | 1 | 10 | 1 | 0 | 2 | fail |
| 97 | 1 | 10 | 1 | 0 | 2 | fail |

Paired comparisons for the selected best row:

| Comparison | Delta | Candidate-only | Baseline-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|
| sidecar vs target | +0.0286 | 3 | 1 | [-0.0286, +0.0857] | 0.6171 |
| sidecar vs text | -0.1143 | 4 | 12 | [-0.2286, +0.0000] | 0.0801 |
| sidecar vs C2C | -0.3857 | 3 | 30 | [-0.5143, -0.2571] | 0.0000 |

## Decision

Weaken the hand-built length-ratio guard as a live paper method. It still has a
small target improvement on this holdout (`10/70` versus `8/70`), but it fails
the source-necessity gate: clean source-necessary recovery is `0/2` because
controls recover the clean IDs. It is also far below text relay and C2C.

This does not kill the broader source-sidecar idea, because the slice has only
two clean source-only IDs after text exclusion. It does kill direct scale-up of
the fixed `source_target_len_ratio <= 1.0` guard without a learned or
cross-validated router and stronger source surface.

## Next Gate

Do not widen this fixed guard to 500 examples. The next highest-value branch is
one of:

1. cross-validated or learned source router over source/target features,
   trained on one slice and evaluated on disjoint IDs, with the same
   source-destroying controls;
2. a stronger source-surface discovery gate where source-alone has enough clean
   target-complementary IDs for a sidecar to learn from;
3. a stack that keeps the source-derived sidecar but adds target-preservation
   without text relay, with component ablations.

Promotion requires clean source-necessary IDs under matched source and zero
clean control union on a disjoint holdout.

## Artifacts

- eval slice:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170.jsonl`
  - sha256: `448ca8505c2abad84fbfbc03f8a50c56cd5617b9be1c038f3ed49a1a96da3b64`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/manifest.md`
  - sha256: `2f3080cfa4cfdd2c9455585c30931671f39c4e908ddc419a003f735390394854`
- source-contrastive target set:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`
  - sha256: `dd11d8f33b24757222d310342bbf12ce27c115cb091c2f44a8287c8d126721d3`
- sidecar analysis:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_lenratio_guard_sidecar.json`
  - sha256: `cc5b5f2d64f3521b4ab11cd11ea96ac04c848d3c00b163f22823671bec1cfe81`
- sidecar predictions:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_lenratio_guard_predictions.jsonl`
  - sha256: `0ddb79fdd203615c4978b5b7b9d47dbaf72166a65d44d6ddd51be5a5ad0ad267`
- paired comparisons:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/lenratio_guard_paired_vs_target.md`
  - sha256: `a93747b47df37abc595f738f14a8513b82adf95e204b3658ed4ef22c2b54641f`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/lenratio_guard_paired_vs_text.md`
  - sha256: `0738a885f739670266da9d7bc528f9e9d808a6263588b85f3ba6cc90e046363f`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/lenratio_guard_paired_vs_c2c.md`
  - sha256: `44f6951d1f5c77930bd3e3f6610510f8b4fe933adfec2cb3f7ea69ef8a485dd3`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py
```
