# Source-Private Tool-Trace Paper Skeleton Manifest

- gate: `source_private_tool_trace_paper_skeleton_20260429`
- date: `2026-04-29`
- status: drafted

## Inputs

- `paper/source_private_tool_trace_final_table_20260429.md`
- `paper/source_private_tool_trace_reviewer_risk_rows_20260429.md`
- `paper/source_private_tool_trace_paper_claim_draft_20260429.md`
- `paper/repo_readiness_review_20260426.md`
- `paper/experiment_ledger_20260421.md`

## Output

- `paper/source_private_tool_trace_paper_skeleton_20260429.md`

## Decision

The skeleton should frame the contribution as a rate-capped source-private
communication protocol with explicit tool-trace packets and strict
source-destroying controls. It should not frame the result as learned latent
transfer or raw-log repair inference.

## Next Gate

`source_private_tool_trace_target_decoder_smoke_20260429`: replace the
deterministic protocol lookup with an LLM-mediated or learned target-side
selector on a small frozen slice. If it fails, keep the protocol decoder as the
scoped claim and list learned target decoders as a limitation.
