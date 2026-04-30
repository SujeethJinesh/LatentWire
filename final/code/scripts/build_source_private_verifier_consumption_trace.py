from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

CONDITION_ROLE = {
    "target_only": "no_source",
    "matched_packet": "source_packet",
    "deranged_candidate_diag_table": "source_destroying_control",
    "shuffled_packet": "source_destroying_control",
    "random_same_byte": "source_destroying_control",
    "random_noncandidate_same_byte": "source_destroying_control",
    "structured_json_2byte": "matched_byte_text_control",
    "structured_free_text_2byte": "matched_byte_text_control",
}

CSV_COLUMNS = (
    "result_dir",
    "condition",
    "role",
    "examples",
    "accuracy",
    "valid_prediction_rate",
    "mean_payload_bytes",
    "mean_packet_record_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "batch64_line_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "mean_payload_tokens",
    "mean_generated_tokens",
    "mean_binary_forward_passes",
    "p50_latency_ms",
    "p95_latency_ms",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "source_signal_destroyed",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _prediction_path(result_dir: pathlib.Path) -> pathlib.Path:
    resolved = _resolve(result_dir)
    final_path = resolved / "target_predictions.jsonl"
    partial_path = resolved / "target_predictions.partial.jsonl"
    if final_path.exists():
        return final_path
    if partial_path.exists():
        return partial_path
    raise FileNotFoundError(f"missing target predictions in {resolved}")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ceil_quantum(value: float, quantum: int) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil(value / quantum) * quantum)


def _batch_quantum_per_request(value: float, *, batch_size: int, quantum: int) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil((value * batch_size) / quantum) * quantum) / batch_size


def _condition_flags(condition: str) -> dict[str, bool]:
    role = CONDITION_ROLE.get(condition, "other_control")
    source_private = role in {
        "source_packet",
        "source_destroying_control",
        "matched_byte_text_control",
    }
    return {
        "source_private": source_private,
        "source_text_exposed": condition in {"structured_json_2byte", "structured_free_text_2byte"},
        "source_kv_exposed": False,
        "source_signal_destroyed": role in {"source_destroying_control", "matched_byte_text_control"},
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _condition_metrics(
    *,
    result_dir: pathlib.Path,
    condition: str,
    rows: list[dict[str, Any]],
    line_size: int,
    dma_burst: int,
    batch_size: int,
    record_overhead_bytes: int,
) -> dict[str, Any]:
    payload_bytes = [float(row.get("payload_bytes", 0.0)) for row in rows]
    mean_payload_bytes = statistics.fmean(payload_bytes) if payload_bytes else 0.0
    mean_packet_record_bytes = 0.0 if mean_payload_bytes <= 0 else mean_payload_bytes + record_overhead_bytes
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    binary_forward_passes = [
        len(row.get("candidate_binary_logprobs") or [])
        for row in rows
    ]
    flags = _condition_flags(condition)
    return {
        "result_dir": str(result_dir),
        "condition": condition,
        "role": CONDITION_ROLE.get(condition, "other_control"),
        "examples": len(rows),
        "accuracy": statistics.fmean(1.0 if row.get("correct") else 0.0 for row in rows),
        "valid_prediction_rate": statistics.fmean(1.0 if row.get("valid_prediction") else 0.0 for row in rows),
        "mean_payload_bytes": mean_payload_bytes,
        "mean_packet_record_bytes": mean_packet_record_bytes,
        "single_request_cacheline_bytes": _ceil_quantum(mean_packet_record_bytes, line_size),
        "single_request_dma_bytes": _ceil_quantum(mean_packet_record_bytes, dma_burst),
        "batch64_line_bytes_per_request": _batch_quantum_per_request(
            mean_packet_record_bytes, batch_size=batch_size, quantum=line_size
        ),
        "batch64_dma_bytes_per_request": _batch_quantum_per_request(
            mean_packet_record_bytes, batch_size=batch_size, quantum=dma_burst
        ),
        "mean_payload_tokens": statistics.fmean(float(row.get("payload_tokens", 0.0)) for row in rows),
        "mean_generated_tokens": statistics.fmean(float(row.get("generated_tokens", 0.0)) for row in rows),
        "mean_binary_forward_passes": statistics.fmean(binary_forward_passes),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        **flags,
    }


def _group_by_condition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    return grouped


def build_verifier_consumption_trace(
    *,
    result_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    line_size: int = 64,
    dma_burst: int = 128,
    batch_size: int = 64,
    record_overhead_bytes: int = 3,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        prediction_path = _prediction_path(result_dir)
        rows = _read_jsonl(prediction_path)
        sources.append(
            {
                "result_dir": str(result_dir),
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256_file(prediction_path),
                "rows": len(rows),
            }
        )
        for condition, condition_rows in sorted(_group_by_condition(rows).items()):
            all_rows.append(
                _condition_metrics(
                    result_dir=result_dir,
                    condition=condition,
                    rows=condition_rows,
                    line_size=line_size,
                    dma_burst=dma_burst,
                    batch_size=batch_size,
                    record_overhead_bytes=record_overhead_bytes,
                )
            )

    matched_rows = [row for row in all_rows if row["condition"] == "matched_packet"]
    target_rows = [row for row in all_rows if row["condition"] == "target_only"]
    control_rows = [
        row
        for row in all_rows
        if row["role"] in {"source_destroying_control", "matched_byte_text_control"}
    ]
    if not matched_rows or not target_rows or not control_rows:
        raise ValueError("trace requires matched_packet, target_only, and at least one control row")

    per_result: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        rows = [row for row in all_rows if row["result_dir"] == str(result_dir)]
        by_condition = {row["condition"]: row for row in rows}
        matched = by_condition["matched_packet"]
        target = by_condition["target_only"]
        controls = [row for row in rows if row["role"] in {"source_destroying_control", "matched_byte_text_control"}]
        best_control = max(controls, key=lambda row: row["accuracy"])
        per_result.append(
            {
                "result_dir": str(result_dir),
                "matched_accuracy": matched["accuracy"],
                "target_only_accuracy": target["accuracy"],
                "best_control": best_control["condition"],
                "best_control_accuracy": best_control["accuracy"],
                "matched_minus_target": matched["accuracy"] - target["accuracy"],
                "matched_minus_best_control": matched["accuracy"] - best_control["accuracy"],
                "matched_p50_latency_ms": matched["p50_latency_ms"],
                "matched_p95_latency_ms": matched["p95_latency_ms"],
                "matched_mean_binary_forward_passes": matched["mean_binary_forward_passes"],
                "matched_mean_payload_bytes": matched["mean_payload_bytes"],
                "matched_mean_packet_record_bytes": matched["mean_packet_record_bytes"],
                "matched_single_request_cacheline_bytes": matched["single_request_cacheline_bytes"],
                "matched_single_request_dma_bytes": matched["single_request_dma_bytes"],
                "matched_batch64_line_bytes_per_request": matched["batch64_line_bytes_per_request"],
                "matched_batch64_dma_bytes_per_request": matched["batch64_dma_bytes_per_request"],
            }
        )

    headline = {
        "pass_gate": all(row["matched_minus_best_control"] >= 0.15 for row in per_result),
        "result_dirs": len(result_dirs),
        "rows": len(all_rows),
        "min_matched_accuracy": min(row["matched_accuracy"] for row in per_result),
        "max_target_only_accuracy": max(row["target_only_accuracy"] for row in per_result),
        "max_best_control_accuracy": max(row["best_control_accuracy"] for row in per_result),
        "min_matched_minus_target": min(row["matched_minus_target"] for row in per_result),
        "min_matched_minus_best_control": min(row["matched_minus_best_control"] for row in per_result),
        "max_matched_p50_latency_ms": max(row["matched_p50_latency_ms"] for row in per_result),
        "max_matched_p95_latency_ms": max(row["matched_p95_latency_ms"] for row in per_result),
        "max_matched_mean_binary_forward_passes": max(row["matched_mean_binary_forward_passes"] for row in per_result),
        "max_matched_mean_payload_bytes": max(row["matched_mean_payload_bytes"] for row in per_result),
        "max_matched_mean_packet_record_bytes": max(row["matched_mean_packet_record_bytes"] for row in per_result),
        "single_request_line_floor_bytes": line_size,
        "single_request_dma_floor_bytes": dma_burst,
        "batch_size": batch_size,
        "record_overhead_bytes": record_overhead_bytes,
    }
    payload = {
        "gate": "source_private_verifier_consumption_trace",
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "per_result": per_result,
        "rows": all_rows,
        "sources": sources,
        "interpretation": (
            "This trace measures the cost of consuming the source-private packet with the frozen target-side "
            "binary verifier. It separates boundary payload bytes from target-side compute: the source sends "
            "a 2-byte payload inside a small packet record, while the current verifier spends one target "
            "forward pass per candidate."
        ),
        "non_claims": [
            "This is Mac CPU receiver telemetry, not production GPU/vLLM throughput.",
            "Cache-line and DMA values are deterministic accounting proxies.",
            "The current binary verifier has a target-side compute cost that must be reduced or amortized for a systems headline.",
        ],
    }

    resolved_output = _resolve(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    (resolved_output / "verifier_consumption_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (resolved_output / "verifier_consumption_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({column: _fmt(row[column]) for column in CSV_COLUMNS})
    md_lines = [
        "# Source-Private Verifier Consumption Trace",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- result dirs: `{headline['result_dirs']}`",
        f"- min matched accuracy: `{headline['min_matched_accuracy']:.3f}`",
        f"- max target-only accuracy: `{headline['max_target_only_accuracy']:.3f}`",
        f"- min matched minus best control: `{headline['min_matched_minus_best_control']:.3f}`",
        f"- max matched p50 latency ms: `{headline['max_matched_p50_latency_ms']:.2f}`",
        f"- max matched binary forward passes/example: `{headline['max_matched_mean_binary_forward_passes']:.2f}`",
        f"- matched source-boundary payload bytes: `{headline['max_matched_mean_payload_bytes']:.2f}`",
        f"- matched packet record bytes: `{headline['max_matched_mean_packet_record_bytes']:.2f}`",
        "",
        "| result | condition | role | acc | payload B | record B | line B | DMA B | fwd/ex | p50 ms | p95 ms | exposure |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in all_rows:
        exposure = []
        if row["source_private"]:
            exposure.append("source-private")
        if row["source_text_exposed"]:
            exposure.append("text")
        if row["source_kv_exposed"]:
            exposure.append("KV")
        if row["source_signal_destroyed"]:
            exposure.append("destroyed")
        md_lines.append(
            f"| {pathlib.Path(row['result_dir']).name} | {row['condition']} | {row['role']} | "
            f"{row['accuracy']:.3f} | {row['mean_payload_bytes']:.2f} | "
            f"{row['mean_packet_record_bytes']:.2f} | "
            f"{row['single_request_cacheline_bytes']:.0f} | {row['single_request_dma_bytes']:.0f} | "
            f"{row['mean_binary_forward_passes']:.2f} | {row['p50_latency_ms']:.2f} | "
            f"{row['p95_latency_ms']:.2f} | {', '.join(exposure)} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Non-Claims",
            "",
            *[f"- {claim}" for claim in payload["non_claims"]],
            "",
        ]
    )
    (resolved_output / "verifier_consumption_trace.md").write_text("\n".join(md_lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [
            "verifier_consumption_trace.json",
            "verifier_consumption_trace.csv",
            "verifier_consumption_trace.md",
            "manifest.json",
            "manifest.md",
        ],
    }
    (resolved_output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (resolved_output / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Verifier Consumption Trace Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- result dirs: `{headline['result_dirs']}`",
                f"- min matched minus target: `{headline['min_matched_minus_target']:.3f}`",
                f"- min matched minus best control: `{headline['min_matched_minus_best_control']:.3f}`",
                f"- max matched p50 latency ms: `{headline['max_matched_p50_latency_ms']:.2f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--line-size", type=int, default=64)
    parser.add_argument("--dma-burst", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--record-overhead-bytes", type=int, default=3)
    args = parser.parse_args()
    payload = build_verifier_consumption_trace(
        result_dirs=args.result_dirs,
        output_dir=args.output_dir,
        line_size=args.line_size,
        dma_burst=args.dma_burst,
        batch_size=args.batch_size,
        record_overhead_bytes=args.record_overhead_bytes,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2))


if __name__ == "__main__":
    main()
