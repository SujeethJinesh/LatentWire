# Source-Private Tool-Trace Paper Claim Draft Manifest

- gate: `source_private_tool_trace_paper_claim_draft_20260429`
- date: `2026-04-29`
- status: documentation / claim-boundary gate
- pass gate: `True`

## Inputs

- `paper/source_private_tool_trace_baseline_pack_20260429.md`
- `results/source_private_tool_trace_baseline_pack_20260429/baseline_pack.md`
- `paper/source_private_hidden_repair_packet_seed_repeat_20260429.md`
- `paper/repo_readiness_review_20260426.md`
- `paper/experiment_ledger_20260421.md`

## Output

- `paper/source_private_tool_trace_paper_claim_draft_20260429.md`

## Decision

Promote the live branch only as a scoped positive-method story:
explicit source-private tool-trace packets communicate hidden execution evidence
to a target-side candidate decoder.

Do not claim raw-log repair inference, unstructured latent transfer, universal
cross-model communication, or a learned target-side bridge from the current
evidence.

## Evidence Snapshot

- primary `trace_no_hint` rows passing: `8/8`
- Qwen3 matched accuracy range: `0.808-0.924`
- Phi-3 matched accuracy range: `1.000`
- target-only accuracy: `0.250`
- best source-destroying controls: `0.252-0.258`
- minimum paired-bootstrap lower bound over target-only: `0.516`
- model-produced packet bytes: `1.55-2.00`
- full hidden-log relay: roughly `366-374` bytes and `34` tokens per example
- `raw_log_no_trace`: returns Qwen3 to `0.250` with `0` valid packets

## Next Gate

`source_private_tool_trace_reviewer_risk_rows_20260429`:

- matched-byte structured JSON/free-text relay
- helper-only/no-log target oracle
- trace-component masking ablations
- candidate-pool recall separated from selector accuracy
