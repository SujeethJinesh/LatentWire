# Source-Private Tool-Trace Reviewer-Risk Rows Manifest

- gate: `source_private_tool_trace_reviewer_risk_rows_20260429`
- date: `2026-04-29`
- status: passed

## Commands

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 \
  --candidates 4 \
  --seed 29 \
  --budgets 2,4,8,16,32 \
  --family-set core \
  --output-dir results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29
```

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 \
  --candidates 4 \
  --seed 30 \
  --budgets 2,4,8,16,32 \
  --family-set holdout \
  --output-dir results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30
```

## Outcome

Both representative `500`-example surfaces pass at budgets `2,4,8,16`; both
intentionally fail at `32` bytes because structured JSON/free-text relays have
enough budget to expose the diagnostic and become oracles.

At the `2`-byte paper packet budget:

| Surface | Matched packet | Target | Best source-destroying control | Best reviewer negative | Min reviewer oracle | Candidate-pool recall |
|---|---:|---:|---:|---:|---:|---:|
| core seed 29 | `1.000` | `0.250` | `0.254` | `0.250` | `1.000` | `1.000` |
| held-out seed 30 | `1.000` | `0.250` | `0.254` | `0.250` | `1.000` | `1.000` |

## Artifacts

- `core_seed29/benchmark.jsonl`
- `core_seed29/sweep_summary.json`
- `core_seed29/sweep_summary.md`
- `core_seed29/manifest.json`
- `core_seed29/manifest.md`
- `holdout_seed30/benchmark.jsonl`
- `holdout_seed30/sweep_summary.json`
- `holdout_seed30/sweep_summary.md`
- `holdout_seed30/manifest.json`
- `holdout_seed30/manifest.md`

## Interpretation

The deterministic reviewer-risk rows strengthen the scoped claim: compact
private tool-trace packets beat target/no-source and matched-byte structured
relay controls, the helper template alone does not help, and masking the
diagnostic kills the full-log oracle while masking expected/actual values or
test names does not.
