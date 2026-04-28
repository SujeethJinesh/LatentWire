# Source-Private Tool-Trace Baseline Pack

- gate: `source_private_tool_trace_baseline_pack_20260429`
- pass gate: `True`
- claim: Explicit source-private tool-trace packets communicate hidden execution evidence to a target-side candidate decoder.
- boundary: The method is not raw-log repair inference and not unstructured latent transfer; the explicit private REPAIR_DIAG trace field is the communication interface.

## Model Rows

| Surface | Family | Seed | Model | Mode | Matched | Target | Best control | Valid | Bytes | Delta target 95% CI |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.252 | 0.776 | 1.55 | [0.516, 0.600] |
| core_seed29 | core | 29 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | 2.00 | [0.714, 0.788] |
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | 0.00 | [0.000, 0.000] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.256 | 0.776 | 1.55 | [0.516, 0.602] |
| core_seed31 | core | 31 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.256 | 1.000 | 2.00 | [0.710, 0.786] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.256 | 0.000 | 0.00 | [0.000, 0.000] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | trace_no_hint | 0.922 | 0.250 | 0.258 | 0.864 | 1.73 | [0.632, 0.712] |
| holdout_seed30 | holdout | 30 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.258 | 1.000 | 2.00 | [0.710, 0.788] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.258 | 0.000 | 0.00 | [0.000, 0.000] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | trace_no_hint | 0.924 | 0.250 | 0.252 | 0.860 | 1.72 | [0.634, 0.716] |
| holdout_seed32 | holdout | 32 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | 2.00 | [0.710, 0.786] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | 0.00 | [0.000, 0.000] |

## Deterministic Baselines

| Surface | Kind | Label | Accuracy | Mean bytes | Mean tokens |
|---|---|---|---:|---:|---:|
| core_medium_seed29 | no-source | target-only | 0.250 | 0.00 | 0.00 |
| core_medium_seed29 | no-source | target wrapper/no-source | 0.250 | 0.00 | 0.00 |
| core_medium_seed29 | method-oracle | matched deterministic trace packet (32 bytes) | 1.000 | 2.00 | 1.00 |
| core_medium_seed29 | source-destroying control | zero-source | 0.250 | 0.00 | 0.00 |
| core_medium_seed29 | source-destroying control | shuffled-source | 0.250 | 2.00 | 1.00 |
| core_medium_seed29 | source-destroying control | random same-byte | 0.250 | 2.00 | 1.00 |
| core_medium_seed29 | leakage control | answer-only sidecar | 0.250 | 32.00 | 1.00 |
| core_medium_seed29 | leakage control | answer-masked | 0.250 | 0.00 | 0.00 |
| core_medium_seed29 | target-prior control | target-derived sidecar | 0.250 | 2.00 | 1.00 |
| core_medium_seed29 | matched-byte text baseline | matched-byte hidden-log text | 0.250 | 32.00 | 4.00 |
| core_medium_seed29 | oracle/text relay | full hidden-log relay | 1.000 | 366.45 | 33.87 |
| core_medium_seed29 | oracle | full diagnostic text | 1.000 | 14.00 | 1.00 |
| holdout_seed30 | no-source | target-only | 0.250 | 0.00 | 0.00 |
| holdout_seed30 | no-source | target wrapper/no-source | 0.250 | 0.00 | 0.00 |
| holdout_seed30 | method-oracle | matched deterministic trace packet (4 bytes) | 1.000 | 2.00 | 1.00 |
| holdout_seed30 | source-destroying control | zero-source | 0.250 | 0.00 | 0.00 |
| holdout_seed30 | source-destroying control | shuffled-source | 0.250 | 2.00 | 1.00 |
| holdout_seed30 | source-destroying control | random same-byte | 0.250 | 2.00 | 1.00 |
| holdout_seed30 | leakage control | answer-only sidecar | 0.250 | 4.00 | 1.00 |
| holdout_seed30 | leakage control | answer-masked | 0.250 | 0.00 | 0.00 |
| holdout_seed30 | target-prior control | target-derived sidecar | 0.250 | 2.00 | 1.00 |
| holdout_seed30 | matched-byte text baseline | matched-byte hidden-log text | 0.250 | 4.00 | 1.00 |
| holdout_seed30 | oracle/text relay | full hidden-log relay | 1.000 | 373.50 | 34.75 |
| holdout_seed30 | oracle | full diagnostic text | 1.000 | 14.00 | 1.00 |

## Systems

- primary packet mean bytes range: `1.55-2.00`
- primary packet validity range: `0.776-1.000`
- deterministic matched-byte text mean accuracy: `0.250`
- deterministic full hidden-log relay mean accuracy: `1.000`

## Threat Model

| Risk | Control | Result |
|---|---|---|
| target prior or wrapper solves the task | target-only and target-wrapper/no-source rows | target rows stay at 0.250 on all 500-example surfaces |
| packet works without matched private source evidence | zero-source, shuffled-source, and random same-byte rows | best controls remain 0.252-0.258 while matched rows are 0.808-1.000 |
| answer-label leakage explains the gain | answer-only and answer-masked sidecars | answer controls stay at target-only |
| target-derived metadata explains the gain | target-derived sidecar | target-derived sidecar stays at target-only |
| matched-byte text relay explains the gain | matched-byte hidden-log text baseline | matched-byte text stays at 0.250 while 2-byte trace packets reach 1.000 deterministically |
| raw hidden logs are enough without the trace protocol | raw_log_no_trace model rows | trace removal returns Qwen3 to 0.250 with 0 valid packets |
| template-family overfitting | disjoint held-out repair families | Qwen3 reaches 0.922/0.924 and Phi-3 reaches 1.000 on held-out seeds |
| seed instability | four frozen 500-example surfaces | 8/8 primary rows pass; min paired lower bound over target-only is 0.516 |

## Remaining Reviewer Gaps

| Gap | Status |
|---|---|
| matched-byte structured JSON/free-text relay | partially covered by truncated hidden-log text; JSON relay still needed |
| target helper-only/no-log oracle | target-only and target-wrapper covered; stronger helper-only no-log baselines still needed |
| masked trace component ablations | raw_log_no_trace is covered; expected/actual, line-number, test-name masking remain future reviewer-risk rows |
| candidate/selector separation | candidate pool recall is deterministic 1.0; paper table should still separate pool recall and selector accuracy |
| second target-family pair | source emitters are cross-family; target decoder is deterministic protocol decoder, so a learned/LLM target-family row is not yet claimed |

## Next Reviewer-Risk Gate

`source_private_tool_trace_paper_claim_draft_20260429`: convert this pack into method, benchmark, baseline, and limitation sections with exact claim language.
