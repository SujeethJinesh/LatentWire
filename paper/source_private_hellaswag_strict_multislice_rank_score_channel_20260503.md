# HellaSwag Strict Rank/Score-Channel Multi-Slice Gate

Date: 2026-05-03

## Status

This is the strongest current positive-method evidence in the repo. It promotes
the HellaSwag hidden-innovation branch from a first-slice result to a strict
`0:9216` frozen-validation result, while also recording a full-validation tail
stress failure.

It is still not ICLR-ready. The remaining blocker is receiver-family or
cross-family falsification under the same controls, plus systems measurements
on NVIDIA for any throughput or HBM claim.

## Artifacts

- strict `0:9216` aggregate:
  `results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260503_rank_score_channel_qwen05_validation0_9216/`
- full `0:10042` stress aggregate:
  `results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260503_rank_score_channel_qwen05_validation0_10042/`
- held-out strict slice artifacts:
  `results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_rank_score_channel_qwen05_train512_validation*/`
- first-slice strict artifact:
  `results/source_private_hellaswag_hidden_innovation_bagged_gate_20260503_rank_score_channel_controls_qwen05_train512_validation1024/`

All held-out slice reruns reused the existing 2026-05-01 source score and
source hidden caches. No new model inference was required.

## Strict `0:9216` Result

The strict contiguous validation prefix passes:

| Metric | Value |
| --- | ---: |
| pass gate | `true` |
| rows | `9216` |
| slice pass count | `9/9` |
| weighted selected accuracy | `0.525499` |
| weighted best label-copy accuracy | `0.483941` |
| weighted source-rank/index-only accuracy | `0.479384` |
| weighted score-only accuracy | `0.479384` |
| weighted zero-hidden accuracy | `0.479384` |
| min delta vs best label-copy | `+0.034180` |
| min delta vs source-rank/index-only | `+0.037109` |
| min delta vs score-only | `+0.037109` |
| min CI95 low vs best label-copy | `+0.011719` |
| min CI95 low vs source-rank/index-only | `+0.018555` |
| max score-channel-roll hidden control | `0.273438` |
| packet | `2B` raw / `5B` framed |

Each strict 1024-row slice from `0:9216` passes the same gate: selected
hidden-innovation beats best label-copy, source-rank/index-only, score-only,
and zero-hidden by at least `0.02`; paired CI95 lows are positive; wrong-row,
candidate-roll, and score-channel-roll controls stay below label-copy; all
three train-sample jackknife subbags pass.

## Full `0:10042` Stress

The full validation stress does not pass the strict aggregate:

| Metric | Value |
| --- | ---: |
| pass gate | `false` |
| rows | `10042` |
| slice pass count | `9/10` |
| weighted selected accuracy | `0.526688` |
| weighted best label-copy accuracy | `0.485162` |
| weighted source-rank/index-only accuracy | `0.480880` |
| weighted score-only accuracy | `0.480880` |
| rank/score-channel control slices available | `10/10` |

The terminal `9216:10042` tail has positive full-slice lift:

- selected accuracy: `0.539952`;
- best label-copy: `0.498789`;
- source-rank/index-only: `0.497579`;
- selected minus best label-copy: `+0.041162`;
- paired CI95 low vs best label-copy: `+0.013287`;
- selected minus source-rank/index-only: `+0.042373`;
- paired CI95 low vs source-rank/index-only: `+0.020581`.

But it fails the jackknife stability rule: only `1/3` train-sample subbags
pass, with jackknife min CI95 low versus best label-copy `-0.007264`.

## Interpretation

The HellaSwag positive branch is alive and materially stronger: the first 9216
validation examples pass strict rank, score, zero-hidden, and hidden-corruption
controls. The effect is not explained by sending the source answer, source
rank/index, or source score metadata.

The terminal tail weakens the branch. It does not show an average-effect
collapse, but it shows support sensitivity: the full-slice mean is positive,
while one jackknife subbag is not stable enough. This should be written as
"strict `0:9216` passes; full-tail stress remains unresolved," not as a full
validation pass.

## Lay Explanation

We split HellaSwag into chunks. On the first nine chunks, the tiny hidden
message consistently helps more than simple shortcuts like "copy the source
answer" or "copy the source score." On the final smaller chunk, the average
still looks good, but one version of the trained helper is unstable. That means
the method is promising, but we should not pretend the whole validation set is
fully solved.

## Decision

Promote the HellaSwag hidden-innovation branch to the strongest current ICLR
candidate, with the strict claim limited to `0:9216`.

Next gate:

1. Run a receiver-family or cross-family falsification under the same strict
   controls.
2. If that fails, implement the ARC sparse common-feature innovation packet
   branch with feature-id shuffle, target-derived packet, same-byte text, and
   candidate-score-roll controls.
3. Keep native systems claims limited to byte/privacy accounting until NVIDIA
   vLLM/SGLang/LMCache/C2C/KVComm/TurboQuant rows are measured.
