# HellaSwag Terminal-Tail Stability Rerun

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible, but ICLR full is
still blocked.

Current story: the HellaSwag dense bagged hidden-innovation packet is still the
strongest hard-surface source-private result, and it clears validation
`0:9216`. It does not clear full validation under the strict robustness rule.

Exact blocking gap: the terminal validation tail `9216:10042` still fails the
jackknife robustness requirement, even though its aggregate paired margin is
positive.

## Gate

Reran the frozen three-train-sample bagged hidden-innovation method on the
terminal HellaSwag validation tail without changing the method.

- script:
  `scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py`
- artifact:
  `results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260504_qwen05_train512_validation9216_10042_terminal_tail_stability/`
- refreshed evidence bundle:
  `results/source_private_iclr_evidence_bundle_20260504_terminal_tail_stability/`
- reused score cache:
  `results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/source_eval_score_cache.json`
- reused hidden cache:
  `results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/source_eval_hidden_cache.npz`
- bootstrap samples: `5000`
- terminal rows: `826`
- packet: `2B` raw / `5B` framed

This is the predeclared no-method-change stability check from the earlier
full-validation tail memo. It asks whether the short-tail failure was just
bootstrap noise or whether at least one train-sample jackknife subbag remains
too unstable.

## Result

The gate still fails.

| Metric | Value |
|---|---:|
| selected accuracy | `0.539952` |
| best label-copy accuracy | `0.498789` |
| source-rank/index-only bagged control | `0.497579` |
| score-only bagged control | `0.497579` |
| zero-hidden control | `0.497579` |
| wrong-example hidden control | `0.484262` |
| candidate-roll hidden control | `0.433414` |
| score-channel-roll hidden control | `0.266344` |
| delta vs best label-copy | `+0.041162` |
| CI95 low vs best label-copy | `+0.014528` |
| CI95 low vs source-rank/index-only | `+0.020581` |
| CI95 low vs score-only | `+0.020581` |
| jackknife pass count | `1/3` |
| jackknife min delta vs best label-copy | `+0.020581` |
| jackknife min CI95 low vs best label-copy | `-0.007264` |

Jackknife rows:

| Held-out train sample | Selected | Best label-copy | Delta | CI95 low | Pass |
|---|---:|---:|---:|---:|---|
| `1729` | `0.519370` | `0.497579` | `+0.021792` | `+0.001211` | `false` |
| `2027` | `0.519370` | `0.498789` | `+0.020581` | `-0.007264` | `false` |
| `2039` | `0.555690` | `0.498789` | `+0.056901` | `+0.025424` | `true` |

The first jackknife row has a positive CI against best label-copy, but it fails
because the CI low against the source-rank/index-only control is exactly `0.0`.
The second jackknife row fails the best-label-copy CI directly. The third row
passes cleanly.

## Interpretation

Promote:

- The terminal tail is still not a collapse. The aggregate result beats
  label-copy, source-rank/index-only, score-only, and zero-hidden controls with
  positive paired confidence intervals.
- Corrupted hidden controls stay lower than the selected packet.
- The `2B` raw / `5B` framed source-private packet boundary remains intact.

Weaken:

- HellaSwag full-validation is not a strict pass.
- Repeating the uncertainty estimate with more bootstrap samples does not
  rescue the terminal-tail jackknife requirement.
- The current dense bagged method should not be widened or tuned only to make
  the same terminal tail pass.

## Decision

Freeze the HellaSwag claim as validation `0:9216` plus a positive but
jackknife-unstable terminal-tail diagnostic. Do not claim full-validation
HellaSwag success in an ICLR paper from the current method.

The next exact method gate should change the mechanism, not the confidence
estimator. The most defensible next branches are:

1. a source-conditioned top-2 / rival repair method that attacks the remaining
   HellaSwag oracle headroom while keeping source-index, source-rank,
   source-score, zero, wrong-row, and candidate-roll controls;
2. a learned receiver interface that first clears target/self-resonance
   controls and then uses source-conditioned residual slots;
3. native systems rows if NVIDIA hardware arrives before the next method branch
   is ready.

## Lay Explanation

We reran the final HellaSwag slice with the exact same tiny hidden clue and a
more precise uncertainty check. The clue still helps on average, but one
held-out version of the training recipe is too uncertain. So this is useful
evidence, but not strong enough to say the method passes the whole benchmark.
