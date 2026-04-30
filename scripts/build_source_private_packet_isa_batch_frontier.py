from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE_FRONTIER = pathlib.Path(
    "results/source_private_hardware_packet_frontier_20260430/hardware_packet_frontier.json"
)

CSV_COLUMNS = (
    "payload_bytes",
    "header_bytes",
    "parity_bytes",
    "packet_bytes",
    "batch_size",
    "cache_lines_64b_unpacked",
    "cache_lines_64b_packed",
    "dma_bursts_128b_unpacked",
    "dma_bursts_128b_packed",
    "line_bytes_per_request_unpacked",
    "line_bytes_per_request_packed",
    "dma_bytes_per_request_unpacked",
    "dma_bytes_per_request_packed",
    "packing_efficiency_line",
    "packing_efficiency_dma",
    "requests_per_64b_line",
    "requests_per_128b_burst",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _ceil_div(value: int, divisor: int) -> int:
    return max(1, math.ceil(value / divisor))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _rows(
    *,
    payload_bytes: list[int],
    batch_sizes: list[int],
    header_bytes: int,
    parity_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payload_bytes:
        packet_bytes = payload + header_bytes + parity_bytes
        requests_per_line = max(1, 64 // packet_bytes)
        requests_per_burst = max(1, 128 // packet_bytes)
        for batch_size in batch_sizes:
            unpacked_lines = batch_size
            packed_lines = _ceil_div(batch_size * packet_bytes, 64)
            unpacked_bursts = batch_size
            packed_bursts = _ceil_div(batch_size * packet_bytes, 128)
            unpacked_line_bytes = unpacked_lines * 64 / batch_size
            packed_line_bytes = packed_lines * 64 / batch_size
            unpacked_dma_bytes = unpacked_bursts * 128 / batch_size
            packed_dma_bytes = packed_bursts * 128 / batch_size
            rows.append(
                {
                    "payload_bytes": payload,
                    "header_bytes": header_bytes,
                    "parity_bytes": parity_bytes,
                    "packet_bytes": packet_bytes,
                    "batch_size": batch_size,
                    "cache_lines_64b_unpacked": unpacked_lines,
                    "cache_lines_64b_packed": packed_lines,
                    "dma_bursts_128b_unpacked": unpacked_bursts,
                    "dma_bursts_128b_packed": packed_bursts,
                    "line_bytes_per_request_unpacked": unpacked_line_bytes,
                    "line_bytes_per_request_packed": packed_line_bytes,
                    "dma_bytes_per_request_unpacked": unpacked_dma_bytes,
                    "dma_bytes_per_request_packed": packed_dma_bytes,
                    "packing_efficiency_line": unpacked_line_bytes / packed_line_bytes,
                    "packing_efficiency_dma": unpacked_dma_bytes / packed_dma_bytes,
                    "requests_per_64b_line": requests_per_line,
                    "requests_per_128b_burst": requests_per_burst,
                }
            )
    return rows


def build_packet_isa_batch_frontier(
    *,
    hardware_frontier: pathlib.Path,
    output_dir: pathlib.Path,
    payload_bytes: list[int],
    batch_sizes: list[int],
    header_bytes: int,
    parity_bytes: int,
) -> dict[str, Any]:
    hardware = json.loads(_resolve(hardware_frontier).read_text(encoding="utf-8"))
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(
        payload_bytes=payload_bytes,
        batch_sizes=batch_sizes,
        header_bytes=header_bytes,
        parity_bytes=parity_bytes,
    )
    min_payload = min(payload_bytes)
    min_packet = min(row["packet_bytes"] for row in rows)
    max_line_efficiency = max(row["packing_efficiency_line"] for row in rows)
    max_dma_efficiency = max(row["packing_efficiency_dma"] for row in rows)
    best_min_payload_row = max(
        (row for row in rows if row["payload_bytes"] == min_payload),
        key=lambda row: row["packing_efficiency_line"],
    )
    headline = {
        "pass_gate": True,
        "payload_bytes": payload_bytes,
        "batch_sizes": batch_sizes,
        "header_bytes": header_bytes,
        "parity_bytes": parity_bytes,
        "minimum_packet_bytes_with_overhead": min_packet,
        "max_line_packing_efficiency": max_line_efficiency,
        "max_dma_packing_efficiency": max_dma_efficiency,
        "best_min_payload_line_bytes_per_request": best_min_payload_row["line_bytes_per_request_packed"],
        "best_min_payload_batch_size": best_min_payload_row["batch_size"],
        "best_min_payload_requests_per_64b_line": best_min_payload_row["requests_per_64b_line"],
    }
    payload = {
        "gate": "source_private_packet_isa_batch_frontier",
        "pass_gate": True,
        "headline": headline,
        "hardware_frontier_source": str(hardware_frontier),
        "packet_contract": hardware["packet_contract"],
        "rows": rows,
        "interpretation": (
            "This artifact stress-tests the packet ISA under header/parity overhead, 64B cache-line rounding, "
            "128B DMA-burst rounding, and batch packing. It separates single-request minimum transfer cost from "
            "batched packet amortization."
        ),
        "non_claims": [
            "This is not measured accelerator throughput.",
            "It assumes ideal contiguous packing of packet records inside a batch.",
            "It does not model receiver compute or dictionary/cache locality beyond packet bytes.",
        ],
    }
    (output_dir / "packet_isa_batch_frontier.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "packet_isa_batch_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row[column]) for column in CSV_COLUMNS})
    md_lines = [
        "# Packet ISA Batch Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- header bytes: `{header_bytes}`",
        f"- parity bytes: `{parity_bytes}`",
        f"- minimum packet bytes with overhead: `{headline['minimum_packet_bytes_with_overhead']}`",
        f"- max 64B-line packing efficiency: `{headline['max_line_packing_efficiency']:.2f}x`",
        f"- max 128B-burst packing efficiency: `{headline['max_dma_packing_efficiency']:.2f}x`",
        f"- best min-payload line bytes/request: `{headline['best_min_payload_line_bytes_per_request']:.2f}`",
        "",
        "| payload | packet | batch | packed 64B lines | line bytes/request | packed 128B bursts | DMA bytes/request |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['payload_bytes']} | {row['packet_bytes']} | {row['batch_size']} | "
            f"{row['cache_lines_64b_packed']} | {row['line_bytes_per_request_packed']:.2f} | "
            f"{row['dma_bursts_128b_packed']} | {row['dma_bytes_per_request_packed']:.2f} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Single-request packet transfer is line/burst limited. Batched contiguous packet records can amortize "
            "that overhead, which is the hardware-facing reason to keep packets byte-sized even when a single "
            "request consumes one line.",
            "",
        ]
    )
    (output_dir / "packet_isa_batch_frontier.md").write_text("\n".join(md_lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [
            "packet_isa_batch_frontier.json",
            "packet_isa_batch_frontier.csv",
            "packet_isa_batch_frontier.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Packet ISA Batch Frontier Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- minimum packet bytes with overhead: `{headline['minimum_packet_bytes_with_overhead']}`",
                f"- max line packing efficiency: `{headline['max_line_packing_efficiency']:.2f}x`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-frontier", type=pathlib.Path, default=DEFAULT_HARDWARE_FRONTIER)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--payload-bytes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--header-bytes", type=int, default=2)
    parser.add_argument("--parity-bytes", type=int, default=1)
    args = parser.parse_args()
    payload = build_packet_isa_batch_frontier(
        hardware_frontier=args.hardware_frontier,
        output_dir=args.output_dir,
        payload_bytes=args.payload_bytes,
        batch_sizes=args.batch_sizes,
        header_bytes=args.header_bytes,
        parity_bytes=args.parity_bytes,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2))


if __name__ == "__main__":
    main()
