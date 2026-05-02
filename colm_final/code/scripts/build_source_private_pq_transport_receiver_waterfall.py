from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TRANSPORT = pathlib.Path(
    "results/source_private_mac_packet_ring_transport_microbench_pq7_20260430/packet_ring_transport_microbench.json"
)
DEFAULT_RECEIVER = pathlib.Path(
    "results/source_private_pq_receiver_batch_microbench_20260430/pq_receiver_batch_microbench.json"
)

CSV_COLUMNS = (
    "component",
    "profile",
    "batch_size",
    "record_bytes",
    "source_text_exposed",
    "source_kv_exposed",
    "line_bytes_per_request",
    "dma_bytes_per_request",
    "p50_ns_per_request",
    "p95_ns_per_request",
    "p50_ms_per_request",
    "p95_ms_per_request",
    "notes",
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
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _transport_row(transport: dict[str, Any], profile: str, batch_size: int) -> dict[str, Any]:
    for row in transport["rows"]:
        if row["profile"] == profile and int(row["batch_size"]) == batch_size:
            return row
    raise KeyError(f"missing transport row: {profile} batch {batch_size}")


def _max_receiver_metric(receiver: dict[str, Any], key: str) -> float:
    return max(float(row[key]) for row in receiver["rows"])


def _receiver_batch_metric(receiver: dict[str, Any], batch_size: int, key: str) -> float:
    return max(float(row["batch_results"][str(batch_size)][key]) for row in receiver["rows"])


def _rows(transport: dict[str, Any], receiver: dict[str, Any], *, batch_size: int) -> list[dict[str, Any]]:
    profiles = [
        ("packet", "packet_2b_payload_5b_record", "legacy 2B diagnostic packet record"),
        ("transport", "pq_packet_4b_payload_7b_record", "4B PQ payload plus 3B record overhead"),
        ("transport", "query_aware_text_14b", "query-aware private text comparator"),
        ("transport", "full_hidden_log_370b", "full private-log relay comparator"),
        ("transport", "qjl_1bit_kv_floor_21504b", "1-bit KV byte-floor comparator"),
        ("transport", "kivi_2bit_kv_floor_43008b", "2-bit KV byte-floor comparator"),
    ]
    rows: list[dict[str, Any]] = []
    for component, profile, notes in profiles:
        row = _transport_row(transport, profile, batch_size)
        rows.append(
            {
                "component": component,
                "profile": profile,
                "batch_size": batch_size,
                "record_bytes": int(row["record_bytes"]),
                "source_text_exposed": bool(row["source_text_exposed"]),
                "source_kv_exposed": bool(row["source_kv_exposed"]),
                "line_bytes_per_request": float(row["line_bytes_per_request"]),
                "dma_bytes_per_request": float(row["dma_bytes_per_request"]),
                "p50_ns_per_request": float(row["p50_ns_per_request"]),
                "p95_ns_per_request": float(row["p95_ns_per_request"]),
                "p50_ms_per_request": float(row["p50_ns_per_request"]) / 1_000_000.0,
                "p95_ms_per_request": float(row["p95_ns_per_request"]) / 1_000_000.0,
                "notes": notes,
            }
        )
    rows.append(
        {
            "component": "receiver",
            "profile": "pq_resident_table_decode_max",
            "batch_size": 1,
            "record_bytes": int(receiver["headline"]["packet_record_bytes_per_request"]),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "line_bytes_per_request": None,
            "dma_bytes_per_request": None,
            "p50_ns_per_request": None,
            "p95_ns_per_request": None,
            "p50_ms_per_request": _max_receiver_metric(receiver, "resident_table_decode_p50_ms"),
            "p95_ms_per_request": _max_receiver_metric(receiver, "resident_table_decode_p95_ms"),
            "notes": "max resident lookup over all remap/variant rows",
        }
    )
    rows.append(
        {
            "component": "receiver",
            "profile": f"pq_batch{batch_size}_decode_max",
            "batch_size": batch_size,
            "record_bytes": int(receiver["headline"]["packet_record_bytes_per_request"]),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "line_bytes_per_request": receiver["headline"]["batch256_amortized_128b_packet_record_bytes_per_request"]
            if batch_size == 256
            else None,
            "dma_bytes_per_request": receiver["headline"]["batch256_amortized_128b_packet_record_bytes_per_request"]
            if batch_size == 256
            else None,
            "p50_ns_per_request": None,
            "p95_ns_per_request": None,
            "p50_ms_per_request": _receiver_batch_metric(receiver, batch_size, "per_request_p50_ms"),
            "p95_ms_per_request": _receiver_batch_metric(receiver, batch_size, "per_request_p95_ms"),
            "notes": "max batched decode over all remap/variant rows",
        }
    )
    return rows


def build_pq_transport_receiver_waterfall(
    *,
    transport_path: pathlib.Path,
    receiver_path: pathlib.Path,
    output_dir: pathlib.Path,
    batch_size: int,
) -> dict[str, Any]:
    transport = _read_json(transport_path)
    receiver = _read_json(receiver_path)
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows(transport, receiver, batch_size=batch_size)
    pq_transport = next(row for row in rows if row["profile"] == "pq_packet_4b_payload_7b_record")
    query_text = next(row for row in rows if row["profile"] == "query_aware_text_14b")
    full_log = next(row for row in rows if row["profile"] == "full_hidden_log_370b")
    kv_floor = next(row for row in rows if row["profile"] == "qjl_1bit_kv_floor_21504b")
    receiver_batch = next(row for row in rows if row["profile"] == f"pq_batch{batch_size}_decode_max")
    resident = next(row for row in rows if row["profile"] == "pq_resident_table_decode_max")

    pq_transport_p95_ms = float(pq_transport["p95_ms_per_request"])
    receiver_batch_p50_ms = float(receiver_batch["p50_ms_per_request"])
    headline = {
        "pass_gate": True,
        "transport_pass_gate": bool(transport["pass_gate"]),
        "receiver_pass_gate": bool(receiver["pass_gate"]),
        "batch_size": batch_size,
        "pq_record_bytes": pq_transport["record_bytes"],
        "pq_transport_batch64_line_bytes_per_request": pq_transport["line_bytes_per_request"],
        "pq_transport_batch64_dma_bytes_per_request": pq_transport["dma_bytes_per_request"],
        "pq_transport_batch64_p95_ns_per_request": pq_transport["p95_ns_per_request"],
        "pq_receiver_batch64_p50_ms_per_request": receiver_batch_p50_ms,
        "pq_receiver_resident_p50_ms_per_request": resident["p50_ms_per_request"],
        "transport_share_of_receiver_batch64_p50": pq_transport_p95_ms / max(receiver_batch_p50_ms, 1e-12),
        "query_text_record_ratio_vs_pq": float(query_text["record_bytes"]) / float(pq_transport["record_bytes"]),
        "query_text_exposes_private_text": bool(query_text["source_text_exposed"]),
        "query_text_p95_ratio_vs_pq_transport": float(query_text["p95_ns_per_request"])
        / max(float(pq_transport["p95_ns_per_request"]), 1e-12),
        "full_log_record_ratio_vs_pq": float(full_log["record_bytes"]) / float(pq_transport["record_bytes"]),
        "full_log_p50_ratio_vs_pq_transport": float(full_log["p50_ns_per_request"])
        / max(float(pq_transport["p50_ns_per_request"]), 1e-12),
        "kv_floor_record_ratio_vs_pq": float(kv_floor["record_bytes"]) / float(pq_transport["record_bytes"]),
        "kv_floor_p50_ratio_vs_pq_transport": float(kv_floor["p50_ns_per_request"])
        / max(float(pq_transport["p50_ns_per_request"]), 1e-12),
        "max_receiver_mismatch_count": max(
            int(receiver["headline"]["max_table_prediction_mismatch_count"]),
            int(receiver["headline"]["max_batch_prediction_mismatch_count"]),
        ),
    }
    checks = [
        {
            "check": "transport_gate_passes",
            "pass": headline["transport_pass_gate"],
            "value": str(headline["transport_pass_gate"]),
        },
        {
            "check": "receiver_gate_passes",
            "pass": headline["receiver_pass_gate"],
            "value": str(headline["receiver_pass_gate"]),
        },
        {
            "check": "pq_transport_under_1us",
            "pass": float(pq_transport["p95_ns_per_request"]) < 1000.0,
            "value": f"{pq_transport['p95_ns_per_request']:.2f} ns/request",
        },
        {
            "check": "pq_receiver_under_0p25ms",
            "pass": float(receiver_batch["p95_ms_per_request"]) < 0.25 and float(resident["p95_ms_per_request"]) < 0.25,
            "value": f"batch p95 {receiver_batch['p95_ms_per_request']:.5f} ms; resident p95 {resident['p95_ms_per_request']:.5f} ms",
        },
        {
            "check": "pq_receiver_exact",
            "pass": headline["max_receiver_mismatch_count"] == 0,
            "value": str(headline["max_receiver_mismatch_count"]),
        },
        {
            "check": "query_text_larger_and_exposes_text",
            "pass": headline["query_text_record_ratio_vs_pq"] >= 2.0 and headline["query_text_exposes_private_text"],
            "value": f"{headline['query_text_record_ratio_vs_pq']:.2f}x bytes; exposes text={headline['query_text_exposes_private_text']}",
        },
        {
            "check": "full_log_transport_slower",
            "pass": headline["full_log_p50_ratio_vs_pq_transport"] >= 5.0,
            "value": f"{headline['full_log_p50_ratio_vs_pq_transport']:.2f}x PQ p50 transport",
        },
        {
            "check": "kv_floor_transport_slower",
            "pass": headline["kv_floor_p50_ratio_vs_pq_transport"] >= 100.0,
            "value": f"{headline['kv_floor_p50_ratio_vs_pq_transport']:.2f}x PQ p50 transport",
        },
    ]
    headline["pass_gate"] = all(check["pass"] for check in checks)
    payload = {
        "gate": "source_private_pq_transport_receiver_waterfall",
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "checks": checks,
        "rows": rows,
        "interpretation": (
            "This artifact joins measured packet-ring transport with the PQ resident receiver microbench. "
            "It supports a Mac-local boundary-traffic plus receiver-kernel claim for 7-byte PQ packet records, "
            "not a production GPU serving or protocol-free latent-reasoning claim."
        ),
        "non_claims": [
            "No NVIDIA, vLLM, TTFT, TPOT, HBM, PCIe, or NVLink measurement is included.",
            "Query-aware text can have similar copy timing at this size but exposes private text and uses 2x the PQ record bytes.",
            "KV rows are byte-floor transport comparators, not native KVComm/C2C implementations.",
        ],
        "sources": {
            "transport": str(transport_path),
            "transport_sha256": _sha256_file(transport_path),
            "receiver": str(receiver_path),
            "receiver_sha256": _sha256_file(receiver_path),
        },
    }

    json_path = output_dir / "pq_transport_receiver_waterfall.json"
    md_path = output_dir / "pq_transport_receiver_waterfall.md"
    csv_path = output_dir / "pq_transport_receiver_waterfall.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row[column]) for column in CSV_COLUMNS})
    lines = [
        "# PQ Transport + Receiver Waterfall",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- PQ record bytes: `{headline['pq_record_bytes']}`",
        f"- PQ transport batch64 p95: `{headline['pq_transport_batch64_p95_ns_per_request']:.2f} ns/request`",
        f"- PQ receiver batch64 p50: `{headline['pq_receiver_batch64_p50_ms_per_request']:.5f} ms/request`",
        f"- transport share of receiver p50: `{headline['transport_share_of_receiver_batch64_p50']:.6f}`",
        f"- query-aware text record ratio vs PQ: `{headline['query_text_record_ratio_vs_pq']:.2f}x`",
        f"- full-log transport p50 ratio vs PQ: `{headline['full_log_p50_ratio_vs_pq_transport']:.2f}x`",
        f"- KV-floor transport p50 ratio vs PQ: `{headline['kv_floor_p50_ratio_vs_pq_transport']:.2f}x`",
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
            "## Rows",
            "",
            "| Component | Profile | Batch | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | p50 ms/req | p95 ms/req | Exposure | Notes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        exposure = "source-private"
        if row["source_text_exposed"]:
            exposure = "private text"
        if row["source_kv_exposed"]:
            exposure = "source KV"
        lines.append(
            f"| `{row['component']}` | `{row['profile']}` | {row['batch_size']} | {row['record_bytes']} | "
            f"{_fmt(row['line_bytes_per_request'])} | {_fmt(row['dma_bytes_per_request'])} | "
            f"{_fmt(row['p50_ns_per_request'])} | {_fmt(row['p95_ns_per_request'])} | "
            f"{_fmt(row['p50_ms_per_request'])} | {_fmt(row['p95_ms_per_request'])} | "
            f"{exposure} | {row['notes']} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [
            "pq_transport_receiver_waterfall.json",
            "pq_transport_receiver_waterfall.md",
            "pq_transport_receiver_waterfall.csv",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "pq_transport_receiver_waterfall.json": _sha256_file(json_path),
            "pq_transport_receiver_waterfall.md": _sha256_file(md_path),
            "pq_transport_receiver_waterfall.csv": _sha256_file(csv_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# PQ Transport + Receiver Waterfall Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", type=pathlib.Path, default=DEFAULT_TRANSPORT)
    parser.add_argument("--receiver", type=pathlib.Path, default=DEFAULT_RECEIVER)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_pq_transport_receiver_waterfall_20260430"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    payload = build_pq_transport_receiver_waterfall(
        transport_path=args.transport,
        receiver_path=args.receiver,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"], "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
