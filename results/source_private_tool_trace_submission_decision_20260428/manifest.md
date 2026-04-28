# Source-Private Tool-Trace Submission Decision Manifest

- date: `2026-04-28`
- status: scoped-submit decision reached
- live branch: explicit source-private tool-trace packet handoff
- scale rung: large frozen slice plus submission-polish gate
- source commit before this decision patch:
  `aa9c311cdeeddfc5ba2dfd094c5383da01e0e1ab`

## Decision

Proceed with the scoped protocol-method submission path. Do not run the
optional target-decoder `n=160` scale-up unless the paper claim expands to make
an LLM target receiver a main result.

## Updated Paper Artifacts

| Artifact | Size bytes | SHA256 |
|---|---:|---|
| `paper/iclr2026/source_private_tool_trace.tex` | `19313` | `cf3e31fe5cd63c37753be5808e9de49082a6061343714c63ae6316226cee257b` |
| `paper/iclr2026/source_private_tool_trace.pdf` | `226859` | `a2ac3bbd4c7c0a8b386345222cbcea0e09e67f1e6c69e61daa3a75f57e512078` |
| `paper/iclr2026/source_private_tool_trace.log` | `19940` | `15ecd4940bde76aa870fddf42c8a850900d732ac1494b15bfb0efe3b960f9326` |
| `paper/source_private_tool_trace_submission_decision_20260428.md` | `3827` | `99413deb5bb898330104f55a4087b3d31f181f26b1030bd0aa4e4c0e78eec4be` |

## Decisive Evidence Artifacts

| Artifact | SHA256 |
|---|---|
| `results/source_private_tool_trace_baseline_pack_20260429/baseline_pack.md` | `24ebd0dcd66f74ca82e87d769b3417e1ea6234b8d69bb2c68ae7247e1a480ca4` |
| `paper/source_private_tool_trace_reviewer_risk_rows_20260429.md` | `ff741998b94c612e32f438a4aae27356f5ab25bd3fb3d6b08bb5bc0045825cc2` |
| `paper/source_private_tool_trace_target_decoder_smoke_20260429.md` | `b8d0d38e7bb113024097c96c7c7c4be2507d036ab6051f2e08c8ec39cb6e1c3b` |

## Figure Artifacts

| Artifact | SHA256 |
|---|---|
| `results/source_private_tool_trace_latex_or_figures_20260430/source_private_setup.svg` | `20cd5f56ff78234093693d53a3e92a3519eb816d23b945433d38ddd0fd51d37d` |
| `results/source_private_tool_trace_latex_or_figures_20260430/source_private_setup.pdf` | `e4eb38679dd063abea244dc3f2e634be0dd9ff0a4cf0f30c4e047b7a142a5afc` |
| `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.csv` | `b4f7a24257c8e9b087f52af66917bade8be544d82750c6e03ea862d8c25c6d0e` |
| `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.svg` | `c389b31dc1d7a0216dd846613446f25eaa97fe6b22e7106beb20c574c2f5f8c4` |
| `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.pdf` | `6f8220b8216218fd24c55f49bd262778565bb9cdc1ed31bda2c110be5f880a2c` |

## Reproducibility Note

The repo tracks paper-facing summaries and manifests. Some detailed raw
JSON/JSONL prediction artifacts remain under ignored `results/` or `.debug/`
paths from the original runs. The next human-read gate should either accept the
tracked summary/manifests as paper-supporting artifacts or archive the decisive
raw JSON/JSONL files before external release.
