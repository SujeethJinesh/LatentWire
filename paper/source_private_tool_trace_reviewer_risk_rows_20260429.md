# Source-Private Tool-Trace Reviewer-Risk Rows

- date: `2026-04-29`
- status: reviewer-risk gate passed for deterministic rows
- live branch: explicit source-private tool-trace packet handoff
- scale rung: reviewer-risk confirmation over representative `500`-example core and held-out surfaces

## Question

Do reviewer-risk deterministic baselines explain the source-private packet gain?

This gate targets four objections left open by the baseline pack:

- matched-byte structured JSON relay
- matched-byte concise free-text relay
- helper-template/no-log target oracle
- trace-component masking and candidate-pool versus selector reporting

## Setup

Two representative frozen surfaces were rerun with the expanded deterministic
harness:

- core seed `29`, `500` examples
- held-out seed `30`, `500` examples

Budgets: `2,4,8,16,32` bytes.

New negative controls:

- `structured_json_matched`: truncated `{"repair_diag": "<code>"}` relay
- `structured_free_text_matched`: truncated `repair diag is <code>` relay
- `helper_only_no_log`: helper template without private log or diagnostic value
- `diag_masked_full_log`: full hidden log with `REPAIR_DIAG` masked

New positive oracles:

- `expected_actual_masked_full_log`: full log with expected/actual values masked
- `test_name_masked_full_log`: full log with test name masked
- full hidden log
- full diagnostic text

The target candidate pool contains the gold candidate for every example, so the
reported matched-packet accuracy is selector accuracy, not pool recall.

## Results

At the compact `2`-byte packet budget:

| Surface | Matched packet | Target | Best source-destroying control | Best reviewer negative | Min reviewer oracle | Candidate-pool recall |
|---|---:|---:|---:|---:|---:|---:|
| core seed 29 | `1.000` | `0.250` | `0.254` | `0.250` | `1.000` | `1.000` |
| held-out seed 30 | `1.000` | `0.250` | `0.254` | `0.250` | `1.000` | `1.000` |

Detailed `2`-byte rows:

| Surface | Matched text | JSON | Free text | Helper/no-log | Diag masked | Full log | Expected/actual masked | Test-name masked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core seed 29 | `0.250` | `0.250` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` | `1.000` |
| held-out seed 30 | `0.250` | `0.250` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` | `1.000` |

Budget sweep:

- passing budgets: `2,4,8,16` on both surfaces
- best budget bytes: `2` on both surfaces
- at `32` bytes, structured JSON and free text become oracles (`1.000`), so
  the `32`-byte row intentionally fails the compact-packet pass rule

## Interpretation

The new rows strengthen the scoped claim:

- Matched-byte structured JSON and free-text relays do not explain the
  `2`-byte packet gain.
- A helper template without the private log or diagnostic value does not help
  the target.
- Masking the diagnostic in the full log kills the gain, while masking
  expected/actual values or test name does not.
- The benchmark is selector-limited only after the gold candidate is present;
  candidate-pool recall is `1.000` in these deterministic surfaces.

The `32`-byte structured relay result is not a failure of the paper story. It
shows the expected rate tradeoff: if text relay is allowed enough bytes to
carry the diagnostic in a parseable format, it becomes an oracle. The method
claim should therefore stay tied to compact source-private packets and report
the structured-relay curve.

## Artifacts

- `results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/`
- `results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/`
- `scripts/run_source_private_hidden_repair_packet_smoke.py`
- `tests/test_run_source_private_hidden_repair_packet_smoke.py`

## Decision

Promote reviewer-risk rows as passed for the deterministic packet protocol.
The remaining ICLR readiness work is final table integration and, if time
permits, an optional learned/LLM-mediated target-family row. Do not broaden the
claim beyond explicit private tool-trace packet communication.

## Next Gate

`source_private_tool_trace_final_table_20260429`:

- integrate model rows, deterministic control rows, reviewer-risk rows, bytes,
  token counts, validity, and candidate-pool/selector separation into one
  paper-ready table
- decide whether the optional learned target-family row is necessary or should
  be listed as a limitation
