from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any


METHOD_ORDER = (
    "target_only",
    "kvcomm_matched",
    "kvcomm_zero_source",
    "kvcomm_shuffled_source",
)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _prediction(row: dict[str, Any]) -> str:
    return str(row.get("normalized_prediction") or row.get("prediction") or "").strip().lower()


def _method_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        by_method[str(row["method"])].append(row)

    stats: dict[str, dict[str, Any]] = {}
    for method, rows in by_method.items():
        n = len(rows)
        counts = collections.Counter(_prediction(row) for row in rows)
        stats[method] = {
            "n": n,
            "accuracy": sum(bool(row.get("correct")) for row in rows) / max(n, 1),
            "answer_distribution": dict(sorted(counts.items())),
            "mean_communicated_cache_bytes": (
                sum(float(row.get("communicated_cache_bytes", 0.0)) for row in rows) / max(n, 1)
            ),
            "selected_layers": rows[0].get("selected_layers", []),
        }
    return stats


def _paired_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_example: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in records:
        by_example[str(row["example_id"])][str(row["method"])] = row

    complete = [
        methods
        for methods in by_example.values()
        if "target_only" in methods and "kvcomm_matched" in methods and "kvcomm_zero_source" in methods
    ]
    n = len(complete)

    def rate(predicate) -> float:
        return sum(1 for methods in complete if predicate(methods)) / max(n, 1)

    diagnostics = {
        "paired_examples": n,
        "matched_zero_prediction_agreement": rate(
            lambda methods: _prediction(methods["kvcomm_matched"]) == _prediction(methods["kvcomm_zero_source"])
        ),
        "matched_target_prediction_agreement": rate(
            lambda methods: _prediction(methods["kvcomm_matched"]) == _prediction(methods["target_only"])
        ),
        "zero_target_prediction_agreement": rate(
            lambda methods: _prediction(methods["kvcomm_zero_source"]) == _prediction(methods["target_only"])
        ),
        "matched_damages_target": rate(
            lambda methods: bool(methods["target_only"].get("correct"))
            and not bool(methods["kvcomm_matched"].get("correct"))
        ),
        "matched_repairs_target": rate(
            lambda methods: not bool(methods["target_only"].get("correct"))
            and bool(methods["kvcomm_matched"].get("correct"))
        ),
        "zero_damages_target": rate(
            lambda methods: bool(methods["target_only"].get("correct"))
            and not bool(methods["kvcomm_zero_source"].get("correct"))
        ),
        "zero_repairs_target": rate(
            lambda methods: not bool(methods["target_only"].get("correct"))
            and bool(methods["kvcomm_zero_source"].get("correct"))
        ),
    }
    if all("kvcomm_shuffled_source" in methods for methods in complete):
        diagnostics.update(
            {
                "shuffled_target_prediction_agreement": rate(
                    lambda methods: _prediction(methods["kvcomm_shuffled_source"])
                    == _prediction(methods["target_only"])
                ),
                "shuffled_damages_target": rate(
                    lambda methods: bool(methods["target_only"].get("correct"))
                    and not bool(methods["kvcomm_shuffled_source"].get("correct"))
                ),
                "shuffled_repairs_target": rate(
                    lambda methods: not bool(methods["target_only"].get("correct"))
                    and bool(methods["kvcomm_shuffled_source"].get("correct"))
                ),
            }
        )
    return diagnostics


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KVComm Damage Diagnostic",
        "",
        "This diagnostic asks whether local KVComm failures come from source content or from the cache-injection mechanism itself. High matched-vs-zero agreement means the transferred source values are not the main driver; the cache prefix/position path is.",
        "",
    ]
    for task in payload["tasks"]:
        lines.extend(
            [
                f"## {task['task']}",
                "",
                f"- prediction file: `{task['prediction_file']}`",
                f"- paired examples: `{task['paired']['paired_examples']}`",
                f"- matched vs zero-source prediction agreement: `{task['paired']['matched_zero_prediction_agreement']:.3f}`",
                f"- matched damages target-only: `{task['paired']['matched_damages_target']:.3f}`",
                f"- matched repairs target-only: `{task['paired']['matched_repairs_target']:.3f}`",
                "",
                "| method | n | accuracy | mean communicated bytes | answer distribution |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for method in METHOD_ORDER:
            stats = task["methods"].get(method)
            if not stats:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        str(stats["n"]),
                        _format_float(stats["accuracy"]),
                        _format_float(stats["mean_communicated_cache_bytes"]),
                        json.dumps(stats["answer_distribution"], sort_keys=True),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "| paired diagnostic | value |",
                "|---|---:|",
            ]
        )
        for key, value in task["paired"].items():
            if key == "paired_examples":
                continue
            lines.append(f"| {key} | {_format_float(value)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("TASK", "PREDICTIONS"),
        required=True,
        help="Task label plus KVComm prediction JSONL.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    tasks = []
    for task, path_raw in args.input:
        path = pathlib.Path(path_raw)
        records = _read_jsonl(path)
        tasks.append(
            {
                "task": task,
                "prediction_file": str(path),
                "methods": _method_stats(records),
                "paired": _paired_diagnostics(records),
            }
        )
    payload = {"tasks": tasks}
    pathlib.Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pathlib.Path(args.output_md).write_text(_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
