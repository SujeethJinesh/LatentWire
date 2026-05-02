from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_TRAFFIC_LEDGER = pathlib.Path(
    "results/source_private_memory_traffic_ledger_20260430/memory_traffic_ledger.json"
)
DEFAULT_PACKET_ISA_FRONTIER = pathlib.Path(
    "results/source_private_packet_isa_batch_frontier_20260430/packet_isa_batch_frontier.json"
)

CSV_COLUMNS = (
    "row_class",
    "method",
    "surface",
    "claim_class",
    "communicated_object",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "source_destroying_controls",
    "accuracy",
    "delta_vs_target",
    "raw_payload_bytes",
    "line_bytes_b1",
    "dma_bytes_b1",
    "line_bytes_b64",
    "dma_bytes_b64",
    "ttft_p50_ms",
    "ttft_delta_vs_packet_ms",
    "ttft_slo_500_margin_ms",
    "ttft_slo_750_margin_ms",
    "ttft_slo_1000_margin_ms",
    "ttft_measurement_available",
    "tpot_has_measurement",
    "goodput_claim_allowed",
    "gpu_counter_required",
    "batching_claim",
    "paper_claim",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _paper_claim(row: dict[str, Any]) -> str:
    row_class = row["row_class"]
    method = row["method"]
    if row_class == "endpoint_packet":
        return "Mac endpoint TTFT proxy plus packet efficacy; production TPOT/goodput not claimed"
    if row_class == "semantic_anchor_medium":
        return "medium source-private rate/control evidence; no serving TTFT claim"
    if row_class == "endpoint_text_relay" and method == "query-aware diagnostic text":
        return "visible private-text byte/privacy comparator; no TTFT measurement"
    if row_class == "endpoint_text_relay":
        return "Mac endpoint TTFT proxy for full private text relay; production goodput not claimed"
    if row_class == "kv_byte_floor":
        return "KV/cache byte-floor comparator only; native KV transport not run"
    return "accounting comparator only"


def _batching_claim(row: dict[str, Any]) -> str:
    if row["row_class"] in {"endpoint_packet", "semantic_anchor_medium"}:
        return "packet records can be packed contiguously; batch-64 line/DMA bytes are reported"
    if row["source_text_exposed"]:
        return "text relay may batch at serving layer, but this artifact does not claim text packing"
    if row["source_kv_exposed"]:
        return "KV/cache rows require native serving implementation before batching claims"
    return "no batching claim"


def _margin(ttft_ms: float | None, slo_ms: float) -> float | None:
    if ttft_ms is None:
        return None
    return slo_ms - ttft_ms


def _slo_row(row: dict[str, Any]) -> dict[str, Any]:
    ttft_ms = row.get("ttft_ms")
    if ttft_ms is not None:
        ttft_ms = float(ttft_ms)
    line_b64 = row.get("batch64_packet_line_bytes_per_request")
    dma_b64 = row.get("batch64_packet_dma_bytes_per_request")
    tpot_has_measurement = False
    goodput_claim_allowed = False
    return {
        "row_class": row["row_class"],
        "method": row["method"],
        "surface": row["surface"],
        "claim_class": row["claim_class"],
        "communicated_object": row["communicated_object"],
        "source_private": bool(row["source_private"]),
        "source_text_exposed": bool(row["source_text_exposed"]),
        "source_kv_exposed": bool(row["source_kv_exposed"]),
        "source_destroying_controls": row["source_destroying_controls"],
        "accuracy": row["accuracy"],
        "delta_vs_target": row["delta_vs_target"],
        "raw_payload_bytes": float(row["raw_payload_bytes"]),
        "line_bytes_b1": float(row["single_request_cacheline_bytes"]),
        "dma_bytes_b1": float(row["single_request_dma_bytes"]),
        "line_bytes_b64": None if line_b64 is None else float(line_b64),
        "dma_bytes_b64": None if dma_b64 is None else float(dma_b64),
        "ttft_p50_ms": ttft_ms,
        "ttft_delta_vs_packet_ms": row["p50_ttft_delta_vs_packet_ms"],
        "ttft_slo_500_margin_ms": _margin(ttft_ms, 500.0),
        "ttft_slo_750_margin_ms": _margin(ttft_ms, 750.0),
        "ttft_slo_1000_margin_ms": _margin(ttft_ms, 1000.0),
        "ttft_measurement_available": ttft_ms is not None,
        "tpot_has_measurement": tpot_has_measurement,
        "goodput_claim_allowed": goodput_claim_allowed,
        "gpu_counter_required": True,
        "batching_claim": _batching_claim(row),
        "paper_claim": _paper_claim(row),
    }


def build_serving_slo_envelope(
    *,
    memory_traffic_ledger: pathlib.Path,
    packet_isa_frontier: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    memory = _read_json(memory_traffic_ledger)
    packet_isa = _read_json(packet_isa_frontier)
    rows = [_slo_row(row) for row in memory["rows"]]

    packet_rows = [row for row in rows if row["row_class"] == "endpoint_packet"]
    measured_ttft_rows = [row for row in rows if row["ttft_measurement_available"]]
    batch64_packet_rows = [row for row in rows if row["line_bytes_b64"] is not None]
    source_exposure_rows = [
        row for row in rows if row["source_text_exposed"] or row["source_kv_exposed"]
    ]
    packet_ttft_margins_500 = [
        row["ttft_slo_500_margin_ms"]
        for row in packet_rows
        if row["ttft_slo_500_margin_ms"] is not None
    ]
    headline = {
        "pass_gate": True,
        "rows": len(rows),
        "endpoint_packet_rows": len(packet_rows),
        "ttft_measured_rows": len(measured_ttft_rows),
        "goodput_claim_allowed_rows": sum(bool(row["goodput_claim_allowed"]) for row in rows),
        "gpu_counter_required_rows": sum(bool(row["gpu_counter_required"]) for row in rows),
        "source_exposure_rows": len(source_exposure_rows),
        "packet_min_raw_bytes": min(row["raw_payload_bytes"] for row in packet_rows),
        "packet_min_batch64_line_bytes": min(row["line_bytes_b64"] for row in batch64_packet_rows),
        "packet_min_batch64_dma_bytes": min(row["dma_bytes_b64"] for row in batch64_packet_rows),
        "packet_min_ttft_slo_500_margin_ms": min(packet_ttft_margins_500),
        "packet_min_ttft_slo_750_margin_ms": min(
            row["ttft_slo_750_margin_ms"]
            for row in packet_rows
            if row["ttft_slo_750_margin_ms"] is not None
        ),
        "packet_min_ttft_slo_1000_margin_ms": min(
            row["ttft_slo_1000_margin_ms"]
            for row in packet_rows
            if row["ttft_slo_1000_margin_ms"] is not None
        ),
        "native_serving_gap": "GPU TPOT/goodput and accelerator counters remain unmeasured",
    }

    payload = {
        "gate": "source_private_serving_slo_envelope",
        "pass_gate": True,
        "headline": headline,
        "rows": rows,
        "interpretation": (
            "This envelope translates the source-private packet systems evidence into serving vocabulary: "
            "what crosses the source-target boundary, which private state is exposed, how transfers round "
            "under single-request and batch-64 accounting, which TTFT proxy rows exist, and why TPOT/goodput "
            "remain explicit non-claims until native GPU serving is available."
        ),
        "non_claims": [
            "No row claims production GPU throughput, TPOT, or goodput.",
            "Mac endpoint TTFT is a proxy measurement, not an accelerator serving benchmark.",
            "KV/cache rows are lower-bound comparators and not a native KV transport implementation.",
            "Batch-64 packet rows assume contiguous packet-record packing.",
        ],
        "packet_contract_summary": {
            "contract_name": packet_isa["packet_contract"]["contract_name"],
            "forbidden_sender_material": packet_isa["packet_contract"]["forbidden_sender_material"],
            "invalid_packet_behavior": packet_isa["packet_contract"]["invalid_packet_behavior"],
        },
        "sources": {
            "memory_traffic_ledger": str(memory_traffic_ledger),
            "memory_traffic_ledger_sha256": _sha256_file(memory_traffic_ledger),
            "packet_isa_frontier": str(packet_isa_frontier),
            "packet_isa_frontier_sha256": _sha256_file(packet_isa_frontier),
        },
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "serving_slo_envelope.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "serving_slo_envelope.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row[column]) for column in CSV_COLUMNS})

    md_lines = [
        "# Source-Private Serving SLO Envelope",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{headline['rows']}`",
        f"- TTFT proxy rows: `{headline['ttft_measured_rows']}`",
        f"- goodput claim rows: `{headline['goodput_claim_allowed_rows']}`",
        f"- packet min raw bytes: `{headline['packet_min_raw_bytes']:.2f}`",
        f"- packet min batch-64 line bytes/request: `{headline['packet_min_batch64_line_bytes']:.2f}`",
        f"- packet min batch-64 DMA bytes/request: `{headline['packet_min_batch64_dma_bytes']:.2f}`",
        f"- packet min 500 ms TTFT margin: `{headline['packet_min_ttft_slo_500_margin_ms']:.2f} ms`",
        "",
        "## Envelope Rows",
        "",
        "| method | surface | private | text exposed | KV exposed | raw bytes | line B1 | line B64 | TTFT p50 | 500 ms margin | claim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"{row['method']} | {row['surface']} | {row['source_private']} | "
            f"{row['source_text_exposed']} | {row['source_kv_exposed']} | "
            f"{row['raw_payload_bytes']:.2f} | {row['line_bytes_b1']:.2f} | "
            f"{_fmt(row['line_bytes_b64'])} | {_fmt(row['ttft_p50_ms'])} | "
            f"{_fmt(row['ttft_slo_500_margin_ms'])} | {row['paper_claim']} |"
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
        ]
    )
    md_lines.extend(f"- {non_claim}" for non_claim in payload["non_claims"])
    md_lines.append("")
    (output_dir / "serving_slo_envelope.md").write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [
            "serving_slo_envelope.json",
            "serving_slo_envelope.csv",
            "serving_slo_envelope.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Serving SLO Envelope Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- rows: `{headline['rows']}`",
                f"- TTFT proxy rows: `{headline['ttft_measured_rows']}`",
                f"- goodput claim rows: `{headline['goodput_claim_allowed_rows']}`",
                f"- native serving gap: `{headline['native_serving_gap']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-traffic-ledger", type=pathlib.Path, default=DEFAULT_MEMORY_TRAFFIC_LEDGER)
    parser.add_argument("--packet-isa-frontier", type=pathlib.Path, default=DEFAULT_PACKET_ISA_FRONTIER)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = build_serving_slo_envelope(
        memory_traffic_ledger=args.memory_traffic_ledger,
        packet_isa_frontier=args.packet_isa_frontier,
        output_dir=args.output_dir,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2))


if __name__ == "__main__":
    main()
