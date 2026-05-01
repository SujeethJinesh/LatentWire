# Source-Private CommonsenseQA Non-Science Probe, 2026-05-01

## Status

- bridge contract:
  `results/source_private_commonsenseqa_bridge_contract_20260501/`
- validation source-cache run:
  `results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b/`
- strict text-margin seed-stability probe:
  `results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b/`
- relaxed text-margin seed-stability probe:
  `results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b_gap001/`
- references:
  `references/575_commonsenseqa_non_science_probe_refs_20260501.md`

## Contract

CommonsenseQA was added because OpenBookQA is useful but still science-adjacent.
The local contract materializes the labeled train/validation splits as
`9741/1221` canonical rows with no cross-split content overlap. Source-visible
fields are only `question` and `choices`; `answerKey`, `answer_index`,
`answer_label`, and `question_concept` are forbidden. The Hugging Face test
split has empty labels, so this Mac-feasible gate is validation-only.

The dataset contains repeated candidate strings within some rows
(`146` duplicate-choice rows across train/validation). These are logged as
dataset warnings rather than removed, because the source/target/control
surfaces all see the same canonical rows.

## Results

| Probe | Split | Budget | Seeds | Pass | Matched mean/min/max | Target | Same-byte text | Min lift vs target | Min lift vs text | Min CI95 low |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed source cache | validation | 12B | 1 | no strict text win | 0.440 / 0.440 / 0.440 | 0.206 | 0.440 | 0.234 | 0.000 | 0.199 |
| seed stability, strict text margin | validation | 2B | 5 | 0/5 | 0.438 / 0.437 / 0.439 | 0.206 | 0.424 | 0.232 | 0.013 | 0.195 |
| seed stability, relaxed text margin | validation | 2B | 5 | 5/5 | 0.438 / 0.437 / 0.439 | 0.206 | 0.424 | 0.232 | 0.013 | 0.195 |

## Interpretation

CommonsenseQA confirms the source model signal is not science-only: matched
packet accuracy is roughly `0.44` versus a `0.206` target-only baseline, with
destructive controls near target. However, same-byte text is too strong here.
At `12B` it exactly catches the packet, and at `2B` the packet beats text by
only `0.013`, below the stricter `0.02` margin used for promotion.

This is therefore a live diagnostic, not a headline ICLR row. The next method
branch should target non-science tasks by improving the packet/text separation,
not by weakening the text comparator. Candidate directions are stricter
source-private packet formats that do not devolve into answer-label relay,
learned common-basis denoising, or a benchmark where same-byte text cannot
encode the source decision so directly.

## Next Gate

Do not promote CommonsenseQA until it passes a predeclared strict text-margin
gate. The next exact experiment is a 2-4B non-label packet variant that preserves
the source-choice lift while widening the same-byte-text gap, followed by
validation seed repeats under the same target-only, source-shuffle, random,
target-derived, label-permutation, and candidate-derangement controls.
