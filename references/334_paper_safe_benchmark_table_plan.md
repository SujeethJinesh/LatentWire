# Paper-Safe Held-Out Benchmark Table Plan

Web check: 2026-04-21.

Goal: define the minimum benchmark table that is safe to put in the paper after the selector failures.
The table must separate real held-out gains from selector artifacts, and every row must be
pairwise-comparable to the same target-alone control on the exact same slice.

## What must be in the table

### Direct competitors

- `C2C`
- `KVPress` `none`
- `KVPress` `ExpectedAttentionPress`
- `KVzip` only if the harness is clean and the slice is identical
- `KVComm` only on its native supported task set unless the replay path is validated on a clean clone

### Internal controls

- `target-alone`
- best current internal bridge proxy
- strict stochastic selector
- confidence-gated route expansion only if it is reported as a compute-policy ablation, not as the main claim

### Exact slices

- `gsm8k_eval_70`
- `gsm8k_100`
- `svamp_eval_70`
- `gsm8k_gate_search_30` only as calibration/dev evidence, not as the main paper table

## Telemetry fields required per row

- `accuracy`
- `paired_delta_vs_target_alone`
- `paired_bootstrap_ci_95`
- `paired_p_value`
- `n_examples`
- `avg_bytes`
- `avg_tokens`
- `latency_ms_per_example`
- `decode_budget`
- `prompt_template_id`
- `answer_parser_version`
- `selector_seed`
- `candidate_order_seed`
- `selector_fallback_rate`
- `target_selection_rate`
- `seed_selection_rate`
- `route_collapse`
- `unique_position_frac`
- `prefix_frac`
- `suffix_frac`
- `full_trace_frac`
- `mean_score_entropy`
- `pool_entropy`
- `pool_top_weight`
- `raw_response_hash`

## Reporting rules

- Use the same source-target pair, decoding budget, prompt template, and answer extractor for every comparable row.
- Report paired bootstrap significance on exact matched examples, not unpaired aggregate accuracy only.
- Keep calibration rows separate from held-out rows.
- If a selector is involved, report its selection rate and fallback rate in the same row.
- Do not promote any row to the main paper unless it survives the held-out slices and the paired CI excludes zero versus `target-alone`.

## What counts as a positive method

- On at least one held-out slice, the method must beat `target-alone` with a positive paired delta and a paired bootstrap CI whose lower bound is above zero.
- It must also stay competitive against the strongest direct comparator on that same slice, ideally `C2C`.
- The gain must survive the same exact prompt, parser, and decoding budget, with no calibration on the test slice.
- Selector improvements alone are not sufficient; the row must show a real accuracy gain, not just better route statistics.

## What is paper-safe to say now

- The current selector stack is still a mechanism story, not yet a paper claim.
- The latest failures show that listwise and pairwise verifiers can be dominated by position bias or weak competence.
- The benchmark table therefore needs direct competitors, same-model controls, and paired reporting side by side so the paper can distinguish semantic transfer from selector artifacts.
- A `C2C` refresh on `gsm8k_eval_70` was attempted on 2026-04-21 with the exact Qwen pair, but did not return within the interactive window and was stopped before any prediction file was written. Treat the existing tracked C2C rows as the current paper rows until this can run as a background/overnight job with progress logging.

## Sources to anchor the reporting choice

- Koehn, *Statistical Significance Tests for Machine Translation Evaluation*: https://people.csail.mit.edu/people/koehn/publications/bootstrap2004.pdf
- Paired evaluation of machine-learning models: https://pubmed.ncbi.nlm.nih.gov/37602225/
