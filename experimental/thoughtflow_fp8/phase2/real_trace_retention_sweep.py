"""Sweep real generated traces across keep fractions for ThoughtFlow-FP8."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .run_real_trace_retention import ROOT, OUT_DIR, DEFAULT_TRACES, _run
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_real_trace_retention import ROOT, OUT_DIR, DEFAULT_TRACES, _run


KEEP_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def _status(rows: list[dict[str, object]]) -> str:
    best_phase_margin = max(float(row["phase_margin_vs_best_other"]) for row in rows)
    best_math_margin = max(float(row["math_margin_vs_best_other"]) for row in rows)
    if best_phase_margin >= 0.05 and best_math_margin >= -0.05:
        return "ALIVE on a retention-rate band; next gate is KV/cache telemetry."
    return "WEAKENED; no keep-rate band beats the strongest proxy on real generated traces."


def _summarize_fraction(result: dict[str, object]) -> dict[str, object]:
    summary = result["summary"]
    thought = summary["thoughtflow"]
    others = [metrics for policy, metrics in summary.items() if policy != "thoughtflow"]
    return {
        "keep_fraction": result["keep_fraction"],
        "thoughtflow_phase": thought["phase_recall"],
        "best_other_phase": max(float(metrics["phase_recall"]) for metrics in others),
        "phase_margin_vs_best_other": float(thought["phase_recall"])
        - max(float(metrics["phase_recall"]) for metrics in others),
        "thoughtflow_math": thought["math_state_recall"],
        "best_other_math": max(float(metrics["math_state_recall"]) for metrics in others),
        "math_margin_vs_best_other": float(thought["math_state_recall"])
        - max(float(metrics["math_state_recall"]) for metrics in others),
        "thoughtflow_anchor": thought["anchor_recall"],
    }


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 Real-Trace Retention Sweep",
        "",
        f"Status: **{result['status']}**",
        "",
        "This sweeps saved generation traces across matched keep fractions.",
        "It is still a text-proxy gate, not KV-cache telemetry and not a GPU result.",
        "",
        "| Keep fraction | ThoughtFlow phase | Best other phase | Phase margin | ThoughtFlow math | Best other math | Math margin |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {keep_fraction:.2f} | {thoughtflow_phase:.3f} | {best_other_phase:.3f} | {phase_margin_vs_best_other:.3f} | {thoughtflow_math:.3f} | {best_other_math:.3f} | {math_margin_vs_best_other:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The current protected-token policy should advance only if it has a keep-rate band where it beats the strongest proxy rather than tying it.",
            "If this sweep is weakened, the branch needs hidden/KV saliency before GPU work.",
        ]
    )
    (OUT_DIR / "real_trace_retention_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [_summarize_fraction(_run(DEFAULT_TRACES, keep_fraction)) for keep_fraction in KEEP_FRACTIONS]
    result = {"rows": rows, "status": _status(rows)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_trace_retention_sweep.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
