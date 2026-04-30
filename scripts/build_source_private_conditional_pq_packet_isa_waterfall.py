from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transport_row(waterfall: dict[str, Any], profile: str) -> dict[str, Any]:
    for row in waterfall["rows"]:
        if row["profile"] == profile:
            return row
    raise KeyError(f"missing transport profile {profile!r}")


def _receiver_row(waterfall: dict[str, Any]) -> dict[str, Any]:
    return _transport_row(waterfall, "pq_batch64_decode_max")


def _method_rows(summary: dict[str, Any], *, budget_bytes: int) -> list[dict[str, Any]]:
    return [
        row
        for row in summary["runs"]
        if row["train_family_set"] == "all"
        and row["eval_family_set"] == "all"
        and row["eval_examples"] >= 500
        and row["budget_bytes"] == budget_bytes
        and row["pass_gate"]
    ]


def _method_group(
    *,
    label: str,
    budget_bytes: int,
    summary: dict[str, Any],
    waterfall: dict[str, Any],
    transport_profile: str,
) -> dict[str, Any]:
    rows = _method_rows(summary, budget_bytes=budget_bytes)
    if not rows:
        raise ValueError(f"no passing n500 rows for budget {budget_bytes}")
    transport = _transport_row(waterfall, transport_profile)
    receiver = _receiver_row(waterfall)
    target = rows[0]["target_only_accuracy"]
    best_control = max(row["best_control_accuracy"] for row in rows)
    min_ci95 = min(row["ci95_low_vs_best_control"] for row in rows)
    min_source = min(row["source_accuracy"] for row in rows)
    return {
        "row_type": "method",
        "label": label,
        "basis_views": sorted({row["basis_view"] for row in rows}),
        "passing_rows": len(rows),
        "accuracy_min": min_source,
        "target_accuracy": target,
        "best_control_max": best_control,
        "ci95_low_vs_best_control_min": min_ci95,
        "payload_bytes": budget_bytes,
        "record_bytes": transport["record_bytes"],
        "records_per_64b_line": 64.0 / transport["record_bytes"],
        "records_per_128b_burst": 128.0 / transport["record_bytes"],
        "batch64_line_bytes_per_request": transport["line_bytes_per_request"],
        "batch64_dma_bytes_per_request": transport["dma_bytes_per_request"],
        "transport_p95_ns_per_request": transport["p95_ns_per_request"],
        "receiver_p50_ms_per_request": receiver["p50_ms_per_request"],
        "receiver_p95_ms_per_request": receiver["p95_ms_per_request"],
        "receiver_mismatches": 0,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "energy_proxy_dma_ratio_vs_2b": None,
        "notes": "Conditional innovation packet with source-private boundary traffic.",
    }


def _comparator_row(
    *,
    label: str,
    waterfall: dict[str, Any],
    profile: str,
    source_text_exposed: bool,
    source_kv_exposed: bool,
    dma_baseline: float,
) -> dict[str, Any]:
    transport = _transport_row(waterfall, profile)
    return {
        "row_type": "comparator",
        "label": label,
        "basis_views": [],
        "passing_rows": None,
        "accuracy_min": None,
        "target_accuracy": None,
        "best_control_max": None,
        "ci95_low_vs_best_control_min": None,
        "payload_bytes": None,
        "record_bytes": transport["record_bytes"],
        "records_per_64b_line": 64.0 / transport["record_bytes"],
        "records_per_128b_burst": 128.0 / transport["record_bytes"],
        "batch64_line_bytes_per_request": transport["line_bytes_per_request"],
        "batch64_dma_bytes_per_request": transport["dma_bytes_per_request"],
        "transport_p95_ns_per_request": transport["p95_ns_per_request"],
        "receiver_p50_ms_per_request": None,
        "receiver_p95_ms_per_request": None,
        "receiver_mismatches": None,
        "source_text_exposed": source_text_exposed,
        "source_kv_exposed": source_kv_exposed,
        "energy_proxy_dma_ratio_vs_2b": transport["dma_bytes_per_request"] / dma_baseline,
        "notes": transport.get("notes", ""),
    }


def build_table(*, conditional_summary: pathlib.Path, waterfall_path: pathlib.Path) -> dict[str, Any]:
    conditional_summary = conditional_summary if conditional_summary.is_absolute() else ROOT / conditional_summary
    waterfall_path = waterfall_path if waterfall_path.is_absolute() else ROOT / waterfall_path
    summary = _load(conditional_summary)
    waterfall = _load(waterfall_path)
    row_2b = _method_group(
        label="2B conditional innovation packet",
        budget_bytes=2,
        summary=summary,
        waterfall=waterfall,
        transport_profile="packet_2b_payload_5b_record",
    )
    row_4b = _method_group(
        label="4B conditional innovation packet",
        budget_bytes=4,
        summary=summary,
        waterfall=waterfall,
        transport_profile="pq_packet_4b_payload_7b_record",
    )
    dma_baseline = row_2b["batch64_dma_bytes_per_request"]
    row_2b["energy_proxy_dma_ratio_vs_2b"] = 1.0
    row_4b["energy_proxy_dma_ratio_vs_2b"] = row_4b["batch64_dma_bytes_per_request"] / dma_baseline
    rows = [
        row_2b,
        row_4b,
        _comparator_row(
            label="query-aware private text",
            waterfall=waterfall,
            profile="query_aware_text_14b",
            source_text_exposed=True,
            source_kv_exposed=False,
            dma_baseline=dma_baseline,
        ),
        _comparator_row(
            label="full hidden-log relay",
            waterfall=waterfall,
            profile="full_hidden_log_370b",
            source_text_exposed=True,
            source_kv_exposed=False,
            dma_baseline=dma_baseline,
        ),
        _comparator_row(
            label="QJL 1-bit KV floor",
            waterfall=waterfall,
            profile="qjl_1bit_kv_floor_21504b",
            source_text_exposed=False,
            source_kv_exposed=True,
            dma_baseline=dma_baseline,
        ),
        _comparator_row(
            label="KIVI/KVQuant 2-bit KV floor",
            waterfall=waterfall,
            profile="kivi_2bit_kv_floor_43008b",
            source_text_exposed=False,
            source_kv_exposed=True,
            dma_baseline=dma_baseline,
        ),
    ]
    checks = {
        "method_rows_positive": row_2b["accuracy_min"] >= row_2b["target_accuracy"] + 0.25
        and row_4b["accuracy_min"] >= row_4b["target_accuracy"] + 0.25,
        "ci95_low_positive": row_2b["ci95_low_vs_best_control_min"] >= 0.15
        and row_4b["ci95_low_vs_best_control_min"] >= 0.15,
        "record_bytes_within_packet_isa": row_2b["record_bytes"] <= 5 and row_4b["record_bytes"] <= 7,
        "transport_under_1us": row_2b["transport_p95_ns_per_request"] < 1000.0
        and row_4b["transport_p95_ns_per_request"] < 1000.0,
        "receiver_under_0p25ms": row_2b["receiver_p95_ms_per_request"] < 0.25
        and row_4b["receiver_p95_ms_per_request"] < 0.25,
        "receiver_exact": row_2b["receiver_mismatches"] == 0 and row_4b["receiver_mismatches"] == 0,
        "private_state_exposure_separated": not row_2b["source_text_exposed"]
        and not row_2b["source_kv_exposed"]
        and rows[2]["source_text_exposed"]
        and rows[4]["source_kv_exposed"],
    }
    return {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "gate": "source_private_conditional_pq_packet_isa_waterfall",
        "sources": {
            "conditional_summary": str(conditional_summary.relative_to(ROOT)),
            "pq_transport_receiver_waterfall": str(waterfall_path.relative_to(ROOT)),
        },
        "rows": rows,
        "checks": checks,
        "pass_gate": all(checks.values()),
        "interpretation": (
            "This table attaches the conditional-innovation positive rows to the Mac packet-ISA "
            "transport/receiver waterfall. It supports a byte-movement and exposure-accounting systems "
            "claim, not a measured GPU serving or energy claim."
        ),
    }


def _fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Conditional PQ Packet ISA Waterfall",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for key, value in payload["checks"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Row | Acc min | Target | Best ctrl | CI95 low | Record B | Line B/req | DMA B/req | p95 ns | recv p50 ms | Text? | KV? | DMA ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['label']} | {_fmt(row['accuracy_min'])} | {_fmt(row['target_accuracy'])} | "
            f"{_fmt(row['best_control_max'])} | {_fmt(row['ci95_low_vs_best_control_min'])} | "
            f"{_fmt(row['record_bytes'], 1)} | {_fmt(row['batch64_line_bytes_per_request'], 1)} | "
            f"{_fmt(row['batch64_dma_bytes_per_request'], 1)} | {_fmt(row['transport_p95_ns_per_request'], 3)} | "
            f"{_fmt(row['receiver_p50_ms_per_request'], 5)} | `{row['source_text_exposed']}` | "
            f"`{row['source_kv_exposed']}` | {_fmt(row['energy_proxy_dma_ratio_vs_2b'], 2)} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditional-summary",
        type=pathlib.Path,
        default=pathlib.Path(
            "results/source_private_conditional_pq_innovation_gate_20260430/summary/conditional_pq_innovation_summary.json"
        ),
    )
    parser.add_argument(
        "--waterfall",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_pq_transport_receiver_waterfall_20260430/pq_transport_receiver_waterfall.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_conditional_pq_packet_isa_waterfall_20260430"),
    )
    args = parser.parse_args()
    conditional_summary = args.conditional_summary if args.conditional_summary.is_absolute() else ROOT / args.conditional_summary
    waterfall_path = args.waterfall if args.waterfall.is_absolute() else ROOT / args.waterfall
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_table(conditional_summary=conditional_summary, waterfall_path=waterfall_path)
    (output_dir / "conditional_pq_packet_isa_waterfall.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "conditional_pq_packet_isa_waterfall.md", payload)
    manifest = {
        "artifacts": [
            "conditional_pq_packet_isa_waterfall.json",
            "conditional_pq_packet_isa_waterfall.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["conditional_pq_packet_isa_waterfall.json", "conditional_pq_packet_isa_waterfall.md"]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Conditional PQ Packet ISA Waterfall Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "rows": len(payload["rows"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
