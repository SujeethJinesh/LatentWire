"""Run ThoughtFlow retention policies on existing generated reasoning traces.

This is still a Mac-local proxy gate: it uses saved text generations as traces,
not real KV-cache tensors or GPU timing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from .simulate_phase_retention import (
        Token,
        _recall,
        longflow_like,
        rkv_like,
        thin_kv_like,
        thoughtflow,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from simulate_phase_retention import (
        Token,
        _recall,
        longflow_like,
        rkv_like,
        thin_kv_like,
        thoughtflow,
    )


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/thoughtflow_fp8/phase2"

DEFAULT_TRACES = [
    ROOT / "results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_alone.jsonl",
    ROOT / "results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/text_to_text.jsonl",
    ROOT / "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
]

PHASE_WORDS = {
    "answer",
    "because",
    "check",
    "finally",
    "first",
    "given",
    "hence",
    "identify",
    "initially",
    "next",
    "now",
    "second",
    "so",
    "step",
    "subtract",
    "therefore",
    "then",
    "thus",
    "verify",
    "we",
}

MATH_RE = re.compile(r"(?:\d|[=+\-*/^$\\[\]{}])")


def _clean(token: str) -> str:
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", token.lower())


def _label_trace(text: str) -> list[Token]:
    raw_tokens = text.replace("\n", " ").split()
    trace: list[Token] = []
    for idx, raw in enumerate(raw_tokens):
        clean = _clean(raw)
        if not clean:
            continue
        if len(trace) < 4:
            label = "anchor"
            importance = 1.0 - 0.05 * len(trace)
        elif clean in PHASE_WORDS or clean.startswith("step") or re.fullmatch(r"\d+[\).:]", raw.strip()):
            label = "phase"
            importance = 0.82
        elif MATH_RE.search(raw):
            label = "math_state"
            importance = 0.72
        else:
            label = "reason"
            importance = 0.48
        trace.append(Token(raw, label, importance))
    return trace


def _dataset_name(path: Path) -> str:
    parent = path.parent.name
    return f"{parent}/{path.stem}"


def _load_traces(paths: list[Path]) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("prediction") or row.get("generated") or row.get("text") or "")
                trace = _label_trace(text)
                if len(trace) < 8:
                    continue
                traces.append(
                    {
                        "dataset": _dataset_name(path),
                        "example_id": row.get("example_id", row.get("index")),
                        "trace": trace,
                    }
                )
    return traces


def _summarize_policies(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)

    summary: dict[str, object] = {}
    for policy, policy_rows in sorted(grouped.items()):
        summary[policy] = {
            "n_traces": len(policy_rows),
            "tokens": mean(float(row["tokens"]) for row in policy_rows),
            "keep_rate": mean(float(row["keep_rate"]) for row in policy_rows),
            "anchor_recall": mean(float(row["anchor_recall"]) for row in policy_rows),
            "phase_recall": mean(float(row["phase_recall"]) for row in policy_rows),
            "math_state_recall": mean(float(row["math_state_recall"]) for row in policy_rows),
        }
    return summary


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary = _summarize_policies(rows)
    by_dataset: dict[str, object] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = _summarize_policies(dataset_rows)

    return {"summary": summary, "by_dataset": by_dataset}


def _run(paths: list[Path], keep_fraction: float) -> dict[str, object]:
    policies = {
        "longflow_like": longflow_like,
        "thin_kv_like": thin_kv_like,
        "rkv_like": rkv_like,
        "thoughtflow": thoughtflow,
    }
    traces = _load_traces(paths)
    rows: list[dict[str, object]] = []
    for item in traces:
        trace = item["trace"]
        assert isinstance(trace, list)
        budget = max(1, math.ceil(len(trace) * keep_fraction))
        for name, policy in policies.items():
            kept = policy(trace, budget)
            rows.append(
                {
                    "dataset": item["dataset"],
                    "example_id": item["example_id"],
                    "policy": name,
                    "tokens": len(trace),
                    "budget": budget,
                    "keep_rate": len(kept) / len(trace),
                    "anchor_recall": _recall(trace, kept, "anchor"),
                    "phase_recall": _recall(trace, kept, "phase"),
                    "math_state_recall": _recall(trace, kept, "math_state"),
                }
            )
    result = _summarize(rows)
    result.update(
        {
            "input_paths": [str(path.relative_to(ROOT)) for path in paths if path.exists()],
            "keep_fraction": keep_fraction,
            "n_traces": len(traces),
            "rows": rows,
            "status": _status(result["summary"]),
        }
    )
    return result


def _status(summary: dict[str, object]) -> str:
    thought = summary["thoughtflow"]
    others = [metrics for policy, metrics in summary.items() if policy != "thoughtflow"]
    best_other_phase = max(float(metrics["phase_recall"]) for metrics in others)
    best_other_anchor = max(float(metrics["anchor_recall"]) for metrics in others)
    phase_margin = float(thought["phase_recall"]) - best_other_phase
    anchor_ok = float(thought["anchor_recall"]) >= best_other_anchor - 0.05
    if phase_margin >= 0.10 and anchor_ok:
        return "ALIVE on real generated traces; next gate needs real KV/cache telemetry."
    return "WEAKENED on real generated traces; do not advance without a better protected-token signal."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# ThoughtFlow-FP8 Real-Trace Retention Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        "This gate reuses saved generation traces already in the LatentWire repo.",
        "It is not model accuracy evidence, not KV-cache telemetry, and not a GPU systems result.",
        "",
        f"- traces: {result['n_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        "",
        "## Overall",
        "",
        "| Policy | Traces | Avg tokens | Keep rate | Anchor recall | Phase recall | Math-state recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, metrics in result["summary"].items():
        lines.append(
            "| {policy} | {n_traces:d} | {tokens:.1f} | {keep_rate:.3f} | {anchor_recall:.3f} | {phase_recall:.3f} | {math_state_recall:.3f} |".format(
                policy=policy, **metrics
            )
        )
    lines.extend(["", "## By Dataset", ""])
    for dataset, summary in result["by_dataset"].items():
        lines.extend(
            [
                f"### `{dataset}`",
                "",
                "| Policy | Anchor recall | Phase recall | Math-state recall |",
                "|---|---:|---:|---:|",
            ]
        )
        for policy, metrics in summary.items():
            lines.append(
                "| {policy} | {anchor_recall:.3f} | {phase_recall:.3f} | {math_state_recall:.3f} |".format(
                    policy=policy, **metrics
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "The branch remains useful only if the protected-token policy keeps phase/control markers at matched budget on real traces.",
            "If advanced, the next gate must move from text heuristics to real KV/cache telemetry and accuracy or perplexity impact.",
        ]
    )
    (OUT_DIR / "real_trace_retention_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-fraction", type=float, default=0.35)
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[])
    args = parser.parse_args()

    paths = args.input_jsonl or DEFAULT_TRACES
    result = _run(paths, args.keep_fraction)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_trace_retention_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
