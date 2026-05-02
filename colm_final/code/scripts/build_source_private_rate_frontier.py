from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

INTERFACES = {
    "target_only": ("target-only", "no source"),
    "matched_repair_packet": ("diagnostic packet", "method"),
    "structured_text_matched": ("hidden-log truncation", "matched-byte text"),
    "structured_json_matched": ("JSON relay", "matched-byte text"),
    "structured_free_text_matched": ("free-text relay", "matched-byte text"),
    "query_aware_diag_span": ("query-aware diagnostic span", "query-aware compressed text"),
    "query_aware_masked_span": ("query-aware masked diagnostic span", "query-aware text control"),
    "full_hidden_log": ("full hidden-log relay", "oracle text relay"),
    "full_diag_text": ("full diagnostic text", "oracle diagnostic text"),
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate_rows(surface: str, sweep: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget_row in sweep["budget_summaries"]:
        budget = budget_row["budget_bytes"]
        metrics = budget_row["metrics"]
        for condition, (interface, kind) in INTERFACES.items():
            if condition not in metrics:
                continue
            metric = metrics[condition]
            rows.append(
                {
                    "surface": surface,
                    "budget_bytes": budget,
                    "condition": condition,
                    "interface": interface,
                    "kind": kind,
                    "accuracy": metric["accuracy"],
                    "mean_payload_bytes": metric["mean_payload_bytes"],
                    "mean_payload_tokens": metric["mean_payload_tokens"],
                    "p50_latency_ms": metric["p50_latency_ms"],
                    "p95_latency_ms": metric.get("p95_latency_ms", metric["p50_latency_ms"]),
                }
            )
    return rows


def _target_prediction(example: dict[str, Any]) -> str:
    return max(example["candidates"], key=lambda row: float(row["prior_score"]))["label"]


def _extract_diag(payload: str) -> str | None:
    match = re.search(r"REPAIR_DIAG=([A-Z][0-9])", payload)
    return match.group(1) if match else None


def _decode_diag_payload(example: dict[str, Any], payload: str) -> str:
    diag = _extract_diag(payload)
    if not diag:
        return _target_prediction(example)
    matches = [candidate for candidate in example["candidates"] if candidate["handles_diagnostic"] == diag]
    if not matches:
        return _target_prediction(example)
    return max(matches, key=lambda row: float(row["prior_score"]))["label"]


def _diag_span_payload(example: dict[str, Any], *, budget_bytes: int, mask_diag: bool) -> str:
    full = f"REPAIR_DIAG={example['diagnostic_code']}"
    if mask_diag:
        full = "REPAIR_DIAG=??"
    return full.encode("utf-8")[:budget_bytes].decode("utf-8", errors="ignore")


def _query_aware_rows(surface: str, benchmark_path: pathlib.Path, *, budgets: list[int]) -> list[dict[str, Any]]:
    examples = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for condition, mask_diag in [("query_aware_diag_span", False), ("query_aware_masked_span", True)]:
            correct = 0
            payload_bytes: list[int] = []
            payload_tokens: list[int] = []
            latencies: list[float] = []
            for example in examples:
                start = time.perf_counter()
                payload = _diag_span_payload(example, budget_bytes=budget, mask_diag=mask_diag)
                prediction = _decode_diag_payload(example, payload)
                latencies.append((time.perf_counter() - start) * 1000.0)
                correct += int(prediction == example["answer_label"])
                payload_bytes.append(len(payload.encode("utf-8")))
                payload_tokens.append(len(re.findall(r"\S+", payload)))
            interface, kind = INTERFACES[condition]
            rows.append(
                {
                    "surface": surface,
                    "budget_bytes": budget,
                    "condition": condition,
                    "interface": interface,
                    "kind": kind,
                    "accuracy": correct / len(examples),
                    "mean_payload_bytes": statistics.fmean(payload_bytes),
                    "mean_payload_tokens": statistics.fmean(payload_tokens),
                    "p50_latency_ms": statistics.median(latencies),
                    "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                }
            )
    return rows


def _first_oracle_byte(rows: list[dict[str, Any]], *, condition: str, oracle_accuracy: float) -> float | None:
    candidates = [
        row["mean_payload_bytes"]
        for row in rows
        if row["condition"] == condition and row["accuracy"] >= oracle_accuracy
    ]
    return min(candidates) if candidates else None


def build_rate_frontier(
    *,
    sweep_paths: list[tuple[str, pathlib.Path]],
    output_dir: pathlib.Path,
    query_benchmark_paths: dict[str, pathlib.Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweeps = [(surface, _read_json(path)) for surface, path in sweep_paths]
    rows = [row for surface, sweep in sweeps for row in _rate_rows(surface, sweep)]
    if query_benchmark_paths:
        for surface, path in query_benchmark_paths.items():
            budgets = sorted(
                {
                    row["budget_bytes"]
                    for row in rows
                    if row["surface"] == surface and row["condition"] == "matched_repair_packet"
                }
            )
            rows.extend(_query_aware_rows(surface, path, budgets=budgets))
    surfaces = sorted({row["surface"] for row in rows})
    per_surface: list[dict[str, Any]] = []
    for surface in surfaces:
        surface_rows = [row for row in rows if row["surface"] == surface]
        target_accuracy = max(row["accuracy"] for row in surface_rows if row["condition"] == "target_only")
        packet_rows = [row for row in surface_rows if row["condition"] == "matched_repair_packet"]
        packet_oracle_bytes = _first_oracle_byte(surface_rows, condition="matched_repair_packet", oracle_accuracy=1.0)
        json_oracle_bytes = _first_oracle_byte(surface_rows, condition="structured_json_matched", oracle_accuracy=1.0)
        free_text_oracle_bytes = _first_oracle_byte(surface_rows, condition="structured_free_text_matched", oracle_accuracy=1.0)
        query_aware_oracle_bytes = _first_oracle_byte(surface_rows, condition="query_aware_diag_span", oracle_accuracy=1.0)
        full_log_bytes = min(row["mean_payload_bytes"] for row in surface_rows if row["condition"] == "full_hidden_log")
        full_diag_bytes = min(row["mean_payload_bytes"] for row in surface_rows if row["condition"] == "full_diag_text")
        matched_byte_text_at_packet = [
            row
            for row in surface_rows
            if row["kind"] == "matched-byte text" and packet_oracle_bytes is not None and row["mean_payload_bytes"] <= packet_oracle_bytes
        ]
        per_surface.append(
            {
                "surface": surface,
                "target_accuracy": target_accuracy,
                "packet_oracle_bytes": packet_oracle_bytes,
                "json_oracle_bytes": json_oracle_bytes,
                "free_text_oracle_bytes": free_text_oracle_bytes,
                "query_aware_oracle_bytes": query_aware_oracle_bytes,
                "full_log_bytes": full_log_bytes,
                "full_diag_bytes": full_diag_bytes,
                "packet_vs_json_oracle_compression": None
                if packet_oracle_bytes is None or json_oracle_bytes is None
                else json_oracle_bytes / packet_oracle_bytes,
                "packet_vs_free_text_oracle_compression": None
                if packet_oracle_bytes is None or free_text_oracle_bytes is None
                else free_text_oracle_bytes / packet_oracle_bytes,
                "packet_vs_query_aware_oracle_compression": None
                if packet_oracle_bytes is None or query_aware_oracle_bytes is None
                else query_aware_oracle_bytes / packet_oracle_bytes,
                "packet_vs_full_log_compression": None if packet_oracle_bytes is None else full_log_bytes / packet_oracle_bytes,
                "packet_vs_full_diag_compression": None if packet_oracle_bytes is None else full_diag_bytes / packet_oracle_bytes,
                "packet_p50_latency_ms": min(row["p50_latency_ms"] for row in packet_rows),
                "matched_byte_text_at_packet_accuracy_max": max(
                    (row["accuracy"] for row in matched_byte_text_at_packet),
                    default=None,
                ),
            }
        )
    payload = {
        "gate": "source_private_rate_frontier",
        "source_sweeps": [str(path) for _, path in sweep_paths],
        "rows": rows,
        "per_surface": per_surface,
        "headline": {
            "surfaces": len(per_surface),
            "packet_oracle_bytes_max": max(row["packet_oracle_bytes"] for row in per_surface if row["packet_oracle_bytes"] is not None),
            "json_oracle_bytes_min": min(row["json_oracle_bytes"] for row in per_surface if row["json_oracle_bytes"] is not None),
            "free_text_oracle_bytes_min": min(row["free_text_oracle_bytes"] for row in per_surface if row["free_text_oracle_bytes"] is not None),
            "query_aware_oracle_bytes_min": min(
                row["query_aware_oracle_bytes"] for row in per_surface if row["query_aware_oracle_bytes"] is not None
            ),
            "packet_vs_json_oracle_compression_min": min(
                row["packet_vs_json_oracle_compression"]
                for row in per_surface
                if row["packet_vs_json_oracle_compression"] is not None
            ),
            "packet_vs_full_log_compression_min": min(
                row["packet_vs_full_log_compression"]
                for row in per_surface
                if row["packet_vs_full_log_compression"] is not None
            ),
            "packet_vs_query_aware_oracle_compression_min": min(
                row["packet_vs_query_aware_oracle_compression"]
                for row in per_surface
                if row["packet_vs_query_aware_oracle_compression"] is not None
            ),
            "matched_byte_text_at_packet_accuracy_max": max(
                row["matched_byte_text_at_packet_accuracy_max"]
                for row in per_surface
                if row["matched_byte_text_at_packet_accuracy_max"] is not None
            ),
        },
        "pass_gate": all(
            row["packet_oracle_bytes"] is not None
            and row["json_oracle_bytes"] is not None
            and row["free_text_oracle_bytes"] is not None
            and row["packet_oracle_bytes"] < row["json_oracle_bytes"]
            and row["packet_oracle_bytes"] < row["free_text_oracle_bytes"]
            and row["packet_oracle_bytes"] < row["query_aware_oracle_bytes"]
            and row["matched_byte_text_at_packet_accuracy_max"] == row["target_accuracy"]
            for row in per_surface
        ),
        "pass_rule": (
            "On every surface, the source-private packet must reach oracle accuracy at fewer bytes than "
            "structured JSON/free-text/query-aware relays, and matched-byte text at the packet byte point must stay at target-only accuracy."
        ),
        "caveat": "Latency is local Python single-request timing; this artifact proves rate frontier, not endpoint TTFT.",
    }
    (output_dir / "rate_frontier.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "rate_frontier.md", payload)
    manifest = {
        "artifacts": ["rate_frontier.json", "rate_frontier.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "rate_frontier.json": _sha256_file(output_dir / "rate_frontier.json"),
            "rate_frontier.md": _sha256_file(output_dir / "rate_frontier.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Rate Frontier Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- packet bytes max: `{payload['headline']['packet_oracle_bytes_max']:.1f}`",
                f"- JSON oracle bytes min: `{payload['headline']['json_oracle_bytes_min']:.1f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Rate Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- packet oracle bytes max: `{h['packet_oracle_bytes_max']:.1f}`",
        f"- JSON/free-text oracle bytes min: `{h['json_oracle_bytes_min']:.1f}` / `{h['free_text_oracle_bytes_min']:.1f}`",
        f"- query-aware diagnostic-span oracle bytes min: `{h['query_aware_oracle_bytes_min']:.1f}`",
        f"- packet vs JSON oracle compression min: `{h['packet_vs_json_oracle_compression_min']:.1f}x`",
        f"- packet vs query-aware oracle compression min: `{h['packet_vs_query_aware_oracle_compression_min']:.1f}x`",
        f"- packet vs full hidden-log compression min: `{h['packet_vs_full_log_compression_min']:.1f}x`",
        f"- matched-byte text at packet accuracy max: `{h['matched_byte_text_at_packet_accuracy_max']:.3f}`",
        "",
        "## Per-Surface Frontier",
        "",
        "| Surface | Target | Packet oracle bytes | JSON oracle bytes | Free-text oracle bytes | Query-aware oracle bytes | Full log bytes | Packet vs JSON | Packet vs query-aware | Packet vs full log | Matched-byte text at packet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_surface"]:
        lines.append(
            f"| {row['surface']} | {row['target_accuracy']:.3f} | {row['packet_oracle_bytes']:.1f} | "
            f"{row['json_oracle_bytes']:.1f} | {row['free_text_oracle_bytes']:.1f} | "
            f"{row['query_aware_oracle_bytes']:.1f} | {row['full_log_bytes']:.1f} | "
            f"{row['packet_vs_json_oracle_compression']:.1f}x | "
            f"{row['packet_vs_query_aware_oracle_compression']:.1f}x | "
            f"{row['packet_vs_full_log_compression']:.1f}x | "
            f"{row['matched_byte_text_at_packet_accuracy_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rate Rows",
            "",
            "| Surface | Budget | Interface | Kind | Accuracy | Bytes | Tokens | p50 ms |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['surface']} | {row['budget_bytes']} | {row['interface']} | {row['kind']} | "
            f"{row['accuracy']:.3f} | {row['mean_payload_bytes']:.1f} | "
            f"{row['mean_payload_tokens']:.1f} | {row['p50_latency_ms']:.3f} |"
        )
    lines.extend(["", "## Caveat", "", payload["caveat"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_rate_frontier_20260429"))
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_rate_frontier(
        sweep_paths=[
            ("core seed29", ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/sweep_summary.json"),
            ("holdout seed30", ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/sweep_summary.json"),
        ],
        output_dir=output_dir,
        query_benchmark_paths={
            "core seed29": ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl",
            "holdout seed30": ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/benchmark.jsonl",
        },
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
