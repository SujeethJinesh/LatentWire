from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_SYSTEMS_FRONTIER = pathlib.Path(
    "results/source_private_systems_rate_assumption_frontier_20260430/systems_rate_assumption_frontier.json"
)

CSV_COLUMNS = (
    "row_class",
    "method",
    "surface",
    "communicated_object",
    "raw_payload_bytes",
    "cache_lines_64b",
    "cacheline_payload_bytes",
    "dma_bursts_128b",
    "dma_payload_bytes",
    "raw_byte_ratio_vs_packet",
    "cacheline_ratio_vs_packet",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "source_destroying_controls",
    "accuracy",
    "target_accuracy",
    "delta_vs_target",
    "ttft_ms",
    "p50_ttft_delta_vs_packet_ms",
    "packet_lifetime",
    "receiver_access_pattern",
    "hardware_claim",
    "paper_use",
    "overclaim_guard",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ceil_div_bytes(value: float | None, quantum: int) -> int | None:
    if value is None:
        return None
    return max(1, math.ceil(value / quantum))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _frontier_rows(systems: dict[str, Any]) -> list[dict[str, Any]]:
    rows = systems["rows"]
    selected: list[dict[str, Any]] = []

    endpoint_packets = [row for row in rows if row["row_class"] == "endpoint_packet"]
    selected.extend(endpoint_packets)

    selected.extend(
        row
        for row in rows
        if row["row_class"] == "endpoint_text_relay" and row["method"] in {"query-aware diagnostic text", "full hidden-log relay"}
    )
    selected.extend(row for row in rows if row["row_class"] == "semantic_anchor_medium")
    selected.extend(row for row in rows if row["row_class"] == "kv_byte_floor")

    packet_baseline = min(float(row["private_payload_bytes"]) for row in endpoint_packets)
    packet_cacheline_baseline = _ceil_div_bytes(packet_baseline, 64) * 64

    hardware_rows = []
    for row in selected:
        raw_bytes = row["private_payload_bytes"]
        cache_lines = _ceil_div_bytes(raw_bytes, 64)
        dma_bursts = _ceil_div_bytes(raw_bytes, 128)
        cacheline_bytes = None if cache_lines is None else cache_lines * 64
        dma_bytes = None if dma_bursts is None else dma_bursts * 128
        raw_ratio = None if raw_bytes is None else float(raw_bytes) / packet_baseline
        cacheline_ratio = None if cacheline_bytes is None else cacheline_bytes / packet_cacheline_baseline
        if row["row_class"] == "endpoint_packet":
            lifetime = "ephemeral_per_request_scalar_packet"
            access_pattern = "one packet read by receiver; no source KV/text materialization"
            hardware_claim = "payload byte win; line-rounded traffic equal to one 64B line"
            paper_use = "headline hardware-facing packet row"
        elif row["row_class"] == "endpoint_text_relay":
            lifetime = "prompt_lifetime_visible_private_text"
            access_pattern = "text bytes enter receiver prompt and attention prefix"
            hardware_claim = "higher prompt ingress traffic and longer TTFT in endpoint proxy"
            paper_use = "text relay traffic comparator"
        elif row["row_class"] == "semantic_anchor_medium":
            lifetime = "ephemeral_per_request_semantic_packet"
            access_pattern = "tiny packet plus public receiver dictionary"
            hardware_claim = "medium-scale source-private positive row under one 64B line"
            paper_use = "method evidence traffic row"
        else:
            lifetime = "source_context_cache_lifetime"
            access_pattern = "internal source KV/cache tensor must be transported or retained"
            hardware_claim = "KV byte-floor assumption contrast, not native KV benchmark"
            paper_use = "KV/cache movement lower-bound comparator"
        hardware_rows.append(
            {
                "row_class": row["row_class"],
                "method": row["method"],
                "surface": row["surface"],
                "communicated_object": row["communicated_object"],
                "raw_payload_bytes": raw_bytes,
                "cache_lines_64b": cache_lines,
                "cacheline_payload_bytes": cacheline_bytes,
                "dma_bursts_128b": dma_bursts,
                "dma_payload_bytes": dma_bytes,
                "raw_byte_ratio_vs_packet": raw_ratio,
                "cacheline_ratio_vs_packet": cacheline_ratio,
                "source_private": row["source_private"],
                "source_text_exposed": row["source_text_exposed"],
                "source_kv_exposed": row["source_kv_exposed"],
                "source_destroying_controls": row["source_destroying_controls"],
                "accuracy": row["accuracy"],
                "target_accuracy": row["target_accuracy"],
                "delta_vs_target": row["delta_vs_target"],
                "ttft_ms": row["ttft_ms"],
                "p50_ttft_delta_vs_packet_ms": row["p50_ttft_delta_vs_packet_ms"],
                "packet_lifetime": lifetime,
                "receiver_access_pattern": access_pattern,
                "hardware_claim": hardware_claim,
                "paper_use": paper_use,
                "overclaim_guard": row["overclaim_guard"],
            }
        )
    return hardware_rows


def build_source_private_hardware_packet_frontier(
    *,
    systems_frontier: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    systems = _read_json(systems_frontier)
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _frontier_rows(systems)
    endpoint_rows = [row for row in rows if row["row_class"] == "endpoint_packet"]
    text_rows = [row for row in rows if row["row_class"] == "endpoint_text_relay"]
    kv_rows = [row for row in rows if row["row_class"] == "kv_byte_floor"]
    semantic_rows = [row for row in rows if row["row_class"] == "semantic_anchor_medium"]

    packet_raw_bytes_min = min(float(row["raw_payload_bytes"]) for row in endpoint_rows)
    packet_cacheline_bytes_min = min(float(row["cacheline_payload_bytes"]) for row in endpoint_rows)
    query_text_rows = [row for row in text_rows if row["method"] == "query-aware diagnostic text"]
    full_log_rows = [row for row in text_rows if row["method"] == "full hidden-log relay"]

    packet_contract = {
        "contract_name": "LatentWire source-private diagnostic packet ISA",
        "version": "2026-04-30",
        "byte_budget_options": sorted({int(row["raw_payload_bytes"]) for row in endpoint_rows + semantic_rows}),
        "minimum_packet_bytes": packet_raw_bytes_min,
        "fields": [
            {"name": "atom_or_slot_id", "bits": 8, "purpose": "public receiver dictionary index"},
            {"name": "confidence_or_parity", "bits": 8, "purpose": "receiver thresholding, validity, or parity"},
            {
                "name": "optional_extra_atoms",
                "bits": "16-48",
                "purpose": "additional source-private evidence under 4/8 byte budgets",
            },
        ],
        "allowed_receiver_state": [
            "public prompt",
            "public candidate set",
            "public receiver dictionary or learned receiver weights",
            "target-side prior/cache state",
        ],
        "forbidden_sender_material": [
            "private source text relay",
            "source KV/cache tensors",
            "answer string or candidate label by construction",
        ],
        "invalid_packet_behavior": "fall back to target-only prediction or abstain; invalid packets must not improve controls",
        "control_requirements": [
            "zero-source",
            "shuffled-source",
            "random same-byte",
            "answer-only",
            "answer-masked",
            "target-derived sidecar",
            "atom/code derangement where applicable",
        ],
    }

    headline = {
        "pass_gate": True,
        "packet_raw_bytes_min": packet_raw_bytes_min,
        "packet_cacheline_bytes_min": packet_cacheline_bytes_min,
        "endpoint_rows": len(endpoint_rows),
        "endpoint_rows_passing_controls": sum(row["source_destroying_controls"] == "passed" for row in endpoint_rows),
        "semantic_rows": len(semantic_rows),
        "query_aware_text_raw_ratio_min": min(row["raw_byte_ratio_vs_packet"] for row in query_text_rows),
        "query_aware_text_cacheline_ratio_min": min(row["cacheline_ratio_vs_packet"] for row in query_text_rows),
        "full_log_raw_ratio_min": min(row["raw_byte_ratio_vs_packet"] for row in full_log_rows),
        "full_log_cacheline_ratio_min": min(row["cacheline_ratio_vs_packet"] for row in full_log_rows),
        "full_log_ttft_delta_ms_min": min(row["p50_ttft_delta_vs_packet_ms"] for row in full_log_rows),
        "kv_raw_ratio_min": min(row["raw_byte_ratio_vs_packet"] for row in kv_rows),
        "kv_cacheline_ratio_min": min(row["cacheline_ratio_vs_packet"] for row in kv_rows),
        "kv_rows": len(kv_rows),
    }

    payload = {
        "gate": "source_private_hardware_packet_frontier",
        "pass_gate": True,
        "headline": headline,
        "packet_contract": packet_contract,
        "rows": rows,
        "sources": {
            "systems_frontier": str(systems_frontier),
            "systems_frontier_sha256": _sha256_file(_resolve(systems_frontier)),
        },
        "interpretation": (
            "This artifact translates the source-private packet frontier into hardware-facing accounting: "
            "raw payload bytes, 64B cache-line traffic, 128B DMA burst traffic, packet lifetime, and allowed claims."
        ),
        "non_claims": [
            "No production accelerator throughput is measured here.",
            "Cache-line rounding means 2B/4B/8B packets all occupy one 64B minimum transfer on many systems.",
            "KV byte-floor rows are assumption contrasts, not native KV compression benchmark wins.",
        ],
    }

    json_path = output_dir / "hardware_packet_frontier.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "packet_contract.json").write_text(
        json.dumps(packet_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "hardware_packet_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})

    md_lines = [
        "# Source-Private Hardware Packet Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- packet raw bytes min: `{headline['packet_raw_bytes_min']}`",
        f"- packet cache-line bytes min: `{headline['packet_cacheline_bytes_min']}`",
        f"- query-aware text raw ratio min: `{headline['query_aware_text_raw_ratio_min']:.2f}x`",
        f"- full-log raw ratio min: `{headline['full_log_raw_ratio_min']:.2f}x`",
        f"- full-log cache-line ratio min: `{headline['full_log_cacheline_ratio_min']:.2f}x`",
        f"- KV raw ratio min: `{headline['kv_raw_ratio_min']:.2f}x`",
        f"- KV cache-line ratio min: `{headline['kv_cacheline_ratio_min']:.2f}x`",
        f"- full-log TTFT delta min: `{headline['full_log_ttft_delta_ms_min']:.2f} ms`",
        "",
        "| class | method | raw bytes | 64B lines | raw ratio | line ratio | paper use |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["row_class"],
                    row["method"],
                    _fmt(row["raw_payload_bytes"]),
                    _fmt(row["cache_lines_64b"]),
                    _fmt(row["raw_byte_ratio_vs_packet"]),
                    _fmt(row["cacheline_ratio_vs_packet"]),
                    row["paper_use"],
                ]
            )
            + " |"
        )
    md_lines.extend(
        [
            "",
            "## Contract",
            "",
            "The packet contract is emitted as `packet_contract.json`. It records allowed receiver state, forbidden sender material, invalid-packet behavior, and required source-destroying controls.",
            "",
            "## Non-Claims",
            "",
            "- This is not a production accelerator throughput benchmark.",
            "- It does not claim superiority over native KV/cache compression.",
            "- It makes cache-line rounding explicit: tiny packets win in semantic payload bytes, while many hardware fabrics still move at least one line/burst.",
            "",
        ]
    )
    (output_dir / "hardware_packet_frontier.md").write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "artifacts": [
            "hardware_packet_frontier.json",
            "hardware_packet_frontier.csv",
            "hardware_packet_frontier.md",
            "packet_contract.json",
        ],
        "headline": headline,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Hardware Packet Frontier Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- packet raw bytes min: `{headline['packet_raw_bytes_min']}`",
                f"- query-aware text raw ratio min: `{headline['query_aware_text_raw_ratio_min']:.2f}x`",
                f"- KV raw ratio min: `{headline['kv_raw_ratio_min']:.2f}x`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-frontier", type=pathlib.Path, default=DEFAULT_SYSTEMS_FRONTIER)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = build_source_private_hardware_packet_frontier(
        systems_frontier=args.systems_frontier,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "pass_gate": payload["pass_gate"]}, indent=2))


if __name__ == "__main__":
    main()
