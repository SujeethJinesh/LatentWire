from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/source_private_packet_ring_transport_microbench.c"

CSV_COLUMNS = (
    "profile",
    "record_bytes",
    "batch_size",
    "line_bytes_per_request",
    "dma_bytes_per_request",
    "p50_ns_per_request",
    "p95_ns_per_request",
    "cv",
    "ratio_p50_vs_packet_same_batch",
    "ratio_p95_vs_packet_same_batch",
    "source_text_exposed",
    "source_kv_exposed",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ceil_quantum(value: int, quantum: int) -> int:
    return max(quantum, math.ceil(value / quantum) * quantum)


def _compile(binary: pathlib.Path, *, cc: str) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [cc, "-O3", "-std=c11", "-Wall", "-Wextra", str(SOURCE), "-o", str(binary)],
        check=True,
        cwd=ROOT,
    )


def _run(binary: pathlib.Path, *, target_bytes: int, repeats: int, min_iterations: int) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), str(target_bytes), str(repeats), str(min_iterations)],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def _augment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packet_by_batch = {
        int(row["batch_size"]): row
        for row in rows
        if row["profile"] == "packet_2b_payload_5b_record"
    }
    augmented: list[dict[str, Any]] = []
    for row in rows:
        record_bytes = int(row["record_bytes"])
        batch_size = int(row["batch_size"])
        packet = packet_by_batch[batch_size]
        line_bytes = _ceil_quantum(record_bytes * batch_size, 64) / batch_size
        dma_bytes = _ceil_quantum(record_bytes * batch_size, 128) / batch_size
        augmented.append(
            {
                **row,
                "line_bytes_per_request": line_bytes,
                "dma_bytes_per_request": dma_bytes,
                "ratio_p50_vs_packet_same_batch": float(row["p50_ns_per_request"])
                / max(1e-9, float(packet["p50_ns_per_request"])),
                "ratio_p95_vs_packet_same_batch": float(row["p95_ns_per_request"])
                / max(1e-9, float(packet["p95_ns_per_request"])),
            }
        )
    return augmented


def build_microbench(
    *,
    output_dir: pathlib.Path,
    binary: pathlib.Path,
    target_bytes: int,
    repeats: int,
    min_iterations: int,
    cc: str,
) -> dict[str, Any]:
    cc_path = shutil.which(cc)
    if cc_path is None:
        raise RuntimeError(f"compiler not found: {cc}")
    output_dir = _resolve(output_dir)
    binary = _resolve(binary)
    output_dir.mkdir(parents=True, exist_ok=True)
    _compile(binary, cc=cc_path)
    raw = _run(binary, target_bytes=target_bytes, repeats=repeats, min_iterations=min_iterations)
    rows = _augment_rows(raw["rows"])
    by_profile_batch = {(row["profile"], int(row["batch_size"])): row for row in rows}
    packet64 = by_profile_batch[("packet_2b_payload_5b_record", 64)]
    query64 = by_profile_batch[("query_aware_text_14b", 64)]
    full64 = by_profile_batch[("full_hidden_log_370b", 64)]
    kv64 = by_profile_batch[("qjl_1bit_kv_floor_21504b", 64)]
    packet_rows = [row for row in rows if row["profile"] == "packet_2b_payload_5b_record"]
    headline = {
        "pass_gate": True,
        "target_bytes_per_repeat": raw["target_bytes_per_repeat"],
        "repeats": raw["repeats"],
        "packet_batch64_record_bytes": packet64["record_bytes"],
        "packet_batch64_line_bytes_per_request": packet64["line_bytes_per_request"],
        "packet_batch64_dma_bytes_per_request": packet64["dma_bytes_per_request"],
        "packet_batch64_p50_ns_per_request": packet64["p50_ns_per_request"],
        "packet_batch64_p95_ns_per_request": packet64["p95_ns_per_request"],
        "packet_max_cv": max(float(row["cv"]) for row in packet_rows),
        "query_text_batch64_p95_ratio_vs_packet": query64["ratio_p95_vs_packet_same_batch"],
        "full_log_batch64_p50_ratio_vs_packet": full64["ratio_p50_vs_packet_same_batch"],
        "kv_floor_batch64_p50_ratio_vs_packet": kv64["ratio_p50_vs_packet_same_batch"],
        "all_packet_p95_under_1us": all(float(row["p95_ns_per_request"]) <= 1000.0 for row in packet_rows),
        "all_packet_cv_under_15pct": all(float(row["cv"]) <= 0.15 for row in packet_rows),
    }
    checks = [
        {
            "check": "packet_batch64_p95_under_1us",
            "pass": packet64["p95_ns_per_request"] <= 1000.0,
            "value": f"{packet64['p95_ns_per_request']:.2f} ns/request",
        },
        {
            "check": "packet_batch64_line_bytes",
            "pass": packet64["line_bytes_per_request"] <= 5.0 and packet64["dma_bytes_per_request"] <= 6.0,
            "value": f"{packet64['line_bytes_per_request']:.2f} line B/request; {packet64['dma_bytes_per_request']:.2f} DMA B/request",
        },
        {
            "check": "query_text_not_much_faster",
            "pass": query64["ratio_p95_vs_packet_same_batch"] >= 0.75,
            "value": f"{query64['ratio_p95_vs_packet_same_batch']:.2f}x packet p95 at batch64",
        },
        {
            "check": "full_log_measured_slower",
            "pass": full64["ratio_p50_vs_packet_same_batch"] >= 2.0,
            "value": f"{full64['ratio_p50_vs_packet_same_batch']:.2f}x packet p50 at batch64",
        },
        {
            "check": "kv_floor_measured_slower",
            "pass": kv64["ratio_p50_vs_packet_same_batch"] >= 20.0,
            "value": f"{kv64['ratio_p50_vs_packet_same_batch']:.2f}x packet p50 at batch64",
        },
        {
            "check": "packet_repeat_stability",
            "pass": headline["all_packet_cv_under_15pct"],
            "value": f"max packet CV {headline['packet_max_cv']:.4f}",
        },
    ]
    headline["pass_gate"] = all(check["pass"] for check in checks)
    payload = {
        "gate": "source_private_mac_packet_ring_transport_microbench",
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "checks": checks,
        "rows": rows,
        "compiler": cc_path,
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256_file(SOURCE),
        "binary_path": str(binary),
        "interpretation": (
            "Local Mac packet-ring microbenchmark for contiguous pack-copy-verify transport. "
            "It measures boundary movement for tiny packet records, query-aware private text, full private logs, "
            "and KV byte-floor buffers across batch sizes. It is not GPU serving throughput."
        ),
        "non_claims": [
            "No SSH or remote accelerator was used.",
            "This is local CPU/unified-memory copy timing, not NVIDIA HBM or vLLM serving.",
            "The benchmark measures transport micro-operations, not model forward-pass latency.",
        ],
    }
    (output_dir / "packet_ring_transport_microbench.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "raw_microbench_output.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "packet_ring_transport_microbench.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})
    lines = [
        "# Mac Packet-Ring Transport Microbench",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- repeats: `{raw['repeats']}`",
        f"- target bytes/repeat: `{raw['target_bytes_per_repeat']}`",
        f"- packet batch64 p50 ns/request: `{packet64['p50_ns_per_request']:.2f}`",
        f"- packet batch64 p95 ns/request: `{packet64['p95_ns_per_request']:.2f}`",
        f"- packet batch64 line bytes/request: `{packet64['line_bytes_per_request']:.2f}`",
        f"- packet batch64 DMA bytes/request: `{packet64['dma_bytes_per_request']:.2f}`",
        f"- query-text p95 ratio vs packet at batch64: `{query64['ratio_p95_vs_packet_same_batch']:.2f}x`",
        f"- full-log p50 ratio vs packet at batch64: `{full64['ratio_p50_vs_packet_same_batch']:.2f}x`",
        f"- KV-floor p50 ratio vs packet at batch64: `{kv64['ratio_p50_vs_packet_same_batch']:.2f}x`",
        "",
        "## Checks",
        "",
        "| Check | Pass | Value |",
        "|---|---:|---|",
    ]
    for check in checks:
        lines.append(f"| `{check['check']}` | `{check['pass']}` | {check['value']} |")
    lines.extend(
        [
            "",
            "## Batch-64 Rows",
            "",
            "| Profile | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | p95 ratio vs packet | Exposure |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in [row for row in rows if int(row["batch_size"]) == 64]:
        exposure = "source-private"
        if row["source_text_exposed"]:
            exposure = "private text"
        if row["source_kv_exposed"]:
            exposure = "source KV"
        lines.append(
            f"| `{row['profile']}` | {row['record_bytes']} | {row['line_bytes_per_request']:.2f} | "
            f"{row['dma_bytes_per_request']:.2f} | {row['p50_ns_per_request']:.2f} | "
            f"{row['p95_ns_per_request']:.2f} | {row['ratio_p95_vs_packet_same_batch']:.2f} | {exposure} |"
        )
    lines.extend(["", payload["interpretation"], ""])
    (output_dir / "packet_ring_transport_microbench.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [
            "packet_ring_transport_microbench.json",
            "packet_ring_transport_microbench.md",
            "packet_ring_transport_microbench.csv",
            "raw_microbench_output.json",
            "manifest.json",
            "manifest.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Mac Packet-Ring Transport Microbench Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- packet batch64 p95 ns/request: `{packet64['p95_ns_per_request']:.2f}`",
                f"- full-log p50 ratio vs packet at batch64: `{full64['ratio_p50_vs_packet_same_batch']:.2f}x`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path(".debug/source_private_packet_ring_transport_microbench"))
    parser.add_argument("--target-bytes", type=int, default=134_217_728)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-iterations", type=int, default=128)
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    payload = build_microbench(
        output_dir=args.output_dir,
        binary=args.binary,
        target_bytes=args.target_bytes,
        repeats=args.repeats,
        min_iterations=args.min_iterations,
        cc=args.cc,
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
