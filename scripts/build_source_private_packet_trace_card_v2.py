from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_SYSTEMS_FRONTIER = pathlib.Path(
    "results/source_private_systems_rate_assumption_frontier_20260430/systems_rate_assumption_frontier.json"
)
DEFAULT_MEMORY_LEDGER = pathlib.Path("results/source_private_memory_traffic_ledger_20260430/memory_traffic_ledger.json")
DEFAULT_PACKET_ISA = pathlib.Path(
    "results/source_private_packet_isa_batch_frontier_20260430/packet_isa_batch_frontier.json"
)
DEFAULT_QWEN_RECEIVER_UNCERTAINTY = pathlib.Path(
    "results/source_private_balanced_diag_target_decoder_20260430/"
    "paired_uncertainty_qwen3_seed29_core_holdout_n64_binary_logprob_deranged_cpu/"
    "target_decoder_uncertainty.json"
)

TRACE_COLUMNS = (
    "trace_group",
    "method",
    "surface",
    "raw_payload_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "batch64_line_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "accuracy",
    "target_accuracy",
    "delta_vs_target",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "controls",
    "claim_scope",
    "systems_readout",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _packet_rows(memory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in memory["rows"]:
        if row["row_class"] in {"endpoint_packet", "endpoint_text_relay", "semantic_anchor_medium", "kv_byte_floor"}:
            rows.append(
                {
                    "trace_group": row["row_class"],
                    "method": row["method"],
                    "surface": row["surface"],
                    "raw_payload_bytes": row["raw_payload_bytes"],
                    "single_request_cacheline_bytes": row["single_request_cacheline_bytes"],
                    "single_request_dma_bytes": row["single_request_dma_bytes"],
                    "batch64_line_bytes_per_request": row["batch64_packet_line_bytes_per_request"],
                    "batch64_dma_bytes_per_request": row["batch64_packet_dma_bytes_per_request"],
                    "accuracy": row["accuracy"],
                    "target_accuracy": row["target_accuracy"],
                    "delta_vs_target": row["delta_vs_target"],
                    "source_private": row["source_private"],
                    "source_text_exposed": row["source_text_exposed"],
                    "source_kv_exposed": row["source_kv_exposed"],
                    "controls": row["source_destroying_controls"],
                    "claim_scope": row["claim_class"],
                    "systems_readout": row["traffic_conclusion"],
                }
            )
    return rows


def _batch_record(packet_isa: dict[str, Any], *, payload_bytes: int = 2, batch_size: int = 64) -> dict[str, Any]:
    for row in packet_isa["rows"]:
        if int(row["payload_bytes"]) == payload_bytes and int(row["batch_size"]) == batch_size:
            return row
    raise ValueError(f"missing packet ISA row for payload={payload_bytes}, batch={batch_size}")


def _qwen_receiver_summary(qwen: dict[str, Any]) -> dict[str, Any]:
    rows = qwen["rows"]
    return {
        "rows": len(rows),
        "pass_rows": qwen["headline"]["pass_rows"],
        "min_matched_minus_target": qwen["headline"]["min_matched_minus_target"],
        "min_matched_minus_best_control": qwen["headline"]["min_matched_minus_best_control"],
        "min_ci95_low_vs_best_control": qwen["headline"]["min_paired_ci95_low_vs_best_control"],
        "min_valid_prediction_rate": qwen["headline"]["min_valid_prediction_rate"],
        "max_cpu_p50_latency_ms": qwen["headline"]["max_p50_latency_ms"],
        "mean_payload_bytes": min(row["matched_mean_payload_bytes"] for row in rows),
        "scope": "frozen Qwen3-0.6B CPU equality-verifier evidence, not production serving throughput",
    }


def _checklist(
    *,
    systems: dict[str, Any],
    memory: dict[str, Any],
    packet_isa: dict[str, Any],
    qwen: dict[str, Any],
    batch_record: dict[str, Any],
) -> list[dict[str, Any]]:
    systems_h = systems["headline"]
    memory_h = memory["headline"]
    qwen_h = qwen["headline"]
    return [
        {
            "check": "strict_source_controls",
            "pass": systems_h["endpoint_packet_rows"] == systems_h["endpoint_packet_rows_passing"]
            and qwen_h["pass_rows"] == qwen_h["rows"],
            "value": f"endpoint {systems_h['endpoint_packet_rows_passing']}/{systems_h['endpoint_packet_rows']}; qwen {qwen_h['pass_rows']}/{qwen_h['rows']}",
            "reviewer_risk_reduced": "packet lift is not from zero/shuffled/random/text/deranged controls",
        },
        {
            "check": "same_byte_text_negative",
            "pass": systems_h["same_byte_text_accuracy_max"] <= systems_h["contract_failure_packet_accuracy"],
            "value": f"{systems_h['same_byte_text_accuracy_max']:.3f}",
            "reviewer_risk_reduced": "same-byte text relay does not explain packet accuracy",
        },
        {
            "check": "query_aware_text_raw_gap",
            "pass": memory_h["query_aware_text_raw_ratio_min"] >= 7.0,
            "value": f"{memory_h['query_aware_text_raw_ratio_min']:.1f}x raw bytes; {memory_h['query_aware_text_cacheline_ratio_min']:.1f}x cache-line bytes",
            "reviewer_risk_reduced": "strongest visible-text relay needs more semantic payload and exposes private text",
        },
        {
            "check": "full_log_transport_gap",
            "pass": memory_h["full_log_raw_ratio_min"] >= 100.0 and memory_h["full_log_cacheline_ratio_min"] >= 2.0,
            "value": f"{memory_h['full_log_raw_ratio_min']:.1f}x raw; {memory_h['full_log_cacheline_ratio_min']:.1f}x line; +{memory_h['full_log_ttft_delta_ms_min']:.2f} ms TTFT proxy",
            "reviewer_risk_reduced": "full private text relay is not the same operating point",
        },
        {
            "check": "kv_byte_floor_gap",
            "pass": memory_h["kv_raw_ratio_min"] >= 1000.0 and memory_h["kv_cacheline_ratio_min"] >= 100.0,
            "value": f"{memory_h['kv_raw_ratio_min']:.0f}x raw; {memory_h['kv_cacheline_ratio_min']:.0f}x line",
            "reviewer_risk_reduced": "KV/cache movement is a different and much larger transport object",
        },
        {
            "check": "batch_amortization",
            "pass": float(batch_record["line_bytes_per_request_packed"]) <= 8.0
            and float(batch_record["dma_bytes_per_request_packed"]) <= 8.0,
            "value": f"{batch_record['line_bytes_per_request_packed']:.2f} line B/request; {batch_record['dma_bytes_per_request_packed']:.2f} DMA B/request at batch {batch_record['batch_size']}",
            "reviewer_risk_reduced": "the one-line single-request floor can be amortized by packed packet records",
        },
        {
            "check": "production_overclaim_guard",
            "pass": any("not measured accelerator throughput" in claim.lower() for claim in packet_isa["non_claims"]),
            "value": "; ".join(packet_isa["non_claims"]),
            "reviewer_risk_reduced": "Mac trace card does not pretend to be NVIDIA/HBM throughput",
        },
    ]


def build_packet_trace_card_v2(
    *,
    systems_frontier: pathlib.Path,
    memory_ledger: pathlib.Path,
    packet_isa: pathlib.Path,
    qwen_receiver_uncertainty: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    systems = _read_json(systems_frontier)
    memory = _read_json(memory_ledger)
    packet = _read_json(packet_isa)
    qwen = _read_json(qwen_receiver_uncertainty)
    batch64 = _batch_record(packet, payload_bytes=2, batch_size=64)
    trace_rows = _packet_rows(memory)
    checklist = _checklist(
        systems=systems,
        memory=memory,
        packet_isa=packet,
        qwen=qwen,
        batch_record=batch64,
    )
    qwen_summary = _qwen_receiver_summary(qwen)
    headline = {
        "pass_gate": all(item["pass"] for item in checklist),
        "checklist_passed": sum(1 for item in checklist if item["pass"]),
        "checklist_total": len(checklist),
        "packet_raw_bytes_min": memory["headline"]["packet_raw_bytes_min"],
        "packet_single_request_cacheline_bytes_min": memory["headline"]["packet_single_request_cacheline_bytes_min"],
        "packet_batch64_line_bytes_per_request": batch64["line_bytes_per_request_packed"],
        "packet_batch64_dma_bytes_per_request": batch64["dma_bytes_per_request_packed"],
        "query_aware_text_raw_ratio": memory["headline"]["query_aware_text_raw_ratio_min"],
        "query_aware_text_cacheline_ratio": memory["headline"]["query_aware_text_cacheline_ratio_min"],
        "full_log_raw_ratio_min": memory["headline"]["full_log_raw_ratio_min"],
        "kv_raw_ratio_min": memory["headline"]["kv_raw_ratio_min"],
        "qwen_receiver_pass_rows": qwen_summary["pass_rows"],
        "qwen_receiver_rows": qwen_summary["rows"],
        "qwen_receiver_cpu_p50_latency_ms_max": qwen_summary["max_cpu_p50_latency_ms"],
    }
    payload = {
        "gate": "source_private_packet_trace_card_v2",
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "qwen_receiver_summary": qwen_summary,
        "packet_record_layout": {
            "payload_bytes": batch64["payload_bytes"],
            "header_bytes": batch64["header_bytes"],
            "parity_bytes": batch64["parity_bytes"],
            "packet_bytes_with_overhead": batch64["packet_bytes"],
            "requests_per_64b_line": batch64["requests_per_64b_line"],
            "requests_per_128b_burst": batch64["requests_per_128b_burst"],
            "batch64_cache_lines_packed": batch64["cache_lines_64b_packed"],
            "batch64_dma_bursts_packed": batch64["dma_bursts_128b_packed"],
        },
        "claim_checklist": checklist,
        "trace_rows": trace_rows,
        "allowed_claim": (
            "LatentWire packets are a source-private, byte-scale side-information interface with strict controls. "
            "They are smaller than visible text relays in raw payload, avoid private text and source KV exposure, and "
            "amortize below one cache line per request when packed across batches."
        ),
        "non_claims": [
            "This does not prove production GPU serving throughput.",
            "This does not beat KV compression on native KV-cache tasks.",
            "This does not make the exact-table receiver a protocol-free latent-transfer method.",
            "Qwen CPU verifier latency is model-consumption evidence, not a systems win.",
        ],
        "sources": {
            "systems_frontier": str(systems_frontier),
            "systems_frontier_sha256": _sha256_file(systems_frontier),
            "memory_ledger": str(memory_ledger),
            "memory_ledger_sha256": _sha256_file(memory_ledger),
            "packet_isa": str(packet_isa),
            "packet_isa_sha256": _sha256_file(packet_isa),
            "qwen_receiver_uncertainty": str(qwen_receiver_uncertainty),
            "qwen_receiver_uncertainty_sha256": _sha256_file(qwen_receiver_uncertainty),
        },
    }
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "packet_trace_card_v2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "packet_trace_rows_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in trace_rows:
            writer.writerow({column: _fmt(row.get(column)) for column in TRACE_COLUMNS})
    (output_dir / "claim_checklist_v2.json").write_text(
        json.dumps(checklist, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_lines = [
        "# Source-Private Packet Trace Card v2",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- checklist: `{headline['checklist_passed']}/{headline['checklist_total']}`",
        f"- packet raw bytes min: `{headline['packet_raw_bytes_min']:.2f}`",
        f"- single-request cache-line bytes: `{headline['packet_single_request_cacheline_bytes_min']:.2f}`",
        f"- batch-64 line bytes/request: `{headline['packet_batch64_line_bytes_per_request']:.2f}`",
        f"- batch-64 DMA bytes/request: `{headline['packet_batch64_dma_bytes_per_request']:.2f}`",
        f"- query-aware text raw ratio: `{headline['query_aware_text_raw_ratio']:.2f}x`",
        f"- query-aware text cache-line ratio: `{headline['query_aware_text_cacheline_ratio']:.2f}x`",
        f"- full-log raw ratio min: `{headline['full_log_raw_ratio_min']:.2f}x`",
        f"- KV raw ratio min: `{headline['kv_raw_ratio_min']:.2f}x`",
        f"- Qwen receiver pass rows: `{headline['qwen_receiver_pass_rows']}/{headline['qwen_receiver_rows']}`",
        f"- Qwen CPU p50 latency max: `{headline['qwen_receiver_cpu_p50_latency_ms_max']:.2f} ms`",
        "",
        "## Claim Checklist",
        "",
        "| Check | Pass | Value | Reviewer risk reduced |",
        "|---|---:|---|---|",
    ]
    for item in checklist:
        md_lines.append(
            f"| `{item['check']}` | `{item['pass']}` | {item['value']} | {item['reviewer_risk_reduced']} |"
        )
    md_lines.extend(
        [
            "",
            "## Trace Rows",
            "",
            "| Group | Method | Raw B | Line B | Batch64 line B/req | Accuracy | Exposure | Scope |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in trace_rows:
        exposure = "source-private"
        if row["source_text_exposed"]:
            exposure = "private text"
        if row["source_kv_exposed"]:
            exposure = "source KV"
        batch_line = "" if row["batch64_line_bytes_per_request"] is None else f"{row['batch64_line_bytes_per_request']:.2f}"
        md_lines.append(
            f"| `{row['trace_group']}` | {row['method']} | {_fmt_number(row['raw_payload_bytes'])} | "
            f"{_fmt_number(row['single_request_cacheline_bytes'])} | {batch_line} | "
            f"{_fmt_number(row['accuracy'], 3)} | {exposure} | {row['claim_scope']} |"
        )
    md_lines.extend(["", "## Allowed Claim", "", payload["allowed_claim"], "", "## Non-Claims", ""])
    md_lines.extend(f"- {claim}" for claim in payload["non_claims"])
    md_lines.append("")
    (output_dir / "packet_trace_card_v2.md").write_text("\n".join(md_lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [
            "packet_trace_card_v2.json",
            "packet_trace_card_v2.md",
            "packet_trace_rows_v2.csv",
            "claim_checklist_v2.json",
            "manifest.json",
            "manifest.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Packet Trace Card v2 Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- checklist: `{headline['checklist_passed']}/{headline['checklist_total']}`",
                f"- packet raw bytes min: `{headline['packet_raw_bytes_min']:.2f}`",
                f"- batch-64 line bytes/request: `{headline['packet_batch64_line_bytes_per_request']:.2f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-frontier", type=pathlib.Path, default=DEFAULT_SYSTEMS_FRONTIER)
    parser.add_argument("--memory-ledger", type=pathlib.Path, default=DEFAULT_MEMORY_LEDGER)
    parser.add_argument("--packet-isa", type=pathlib.Path, default=DEFAULT_PACKET_ISA)
    parser.add_argument("--qwen-receiver-uncertainty", type=pathlib.Path, default=DEFAULT_QWEN_RECEIVER_UNCERTAINTY)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = build_packet_trace_card_v2(
        systems_frontier=args.systems_frontier,
        memory_ledger=args.memory_ledger,
        packet_isa=args.packet_isa,
        qwen_receiver_uncertainty=args.qwen_receiver_uncertainty,
        output_dir=args.output_dir,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pass_gate": payload["pass_gate"],
                "headline": payload["headline"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
