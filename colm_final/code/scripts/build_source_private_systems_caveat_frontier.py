from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_ENDPOINT_SUMMARIES = (
    "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
    "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
)
DEFAULT_UNCERTAINTY_SUMMARIES = (
    "results/source_private_endpoint_uncertainty_20260429/core_label_strict_n160/summary.json",
    "results/source_private_endpoint_uncertainty_20260429/label_strict_n160/summary.json",
)
DEFAULT_TERSE_FAILURE = (
    "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_terse/summary.json"
)
DEFAULT_KV_TABLE = "results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json"

CSV_COLUMNS = (
    "surface",
    "scope",
    "prompt_contract",
    "pass_gate",
    "packet_accuracy",
    "packet_strict_accuracy",
    "target_accuracy",
    "matched_byte_text_accuracy",
    "best_source_destroying_control_accuracy",
    "packet_valid_rate",
    "packet_payload_bytes",
    "query_aware_payload_bytes",
    "full_log_payload_bytes",
    "packet_vs_query_payload_compression",
    "packet_vs_full_log_payload_compression",
    "packet_p50_ttft_ms",
    "full_log_p50_ttft_ms",
    "full_log_ttft_delta_vs_packet_ms",
    "min_packet_vs_target_ci95_low",
    "min_packet_vs_best_control_ci95_low",
    "min_strict_packet_vs_target_ci95_low",
    "qjl_1bit_cache_bytes_vs_packet_min",
    "kivi_2bit_cache_bytes_vs_packet_min",
    "caveat",
)

RELATED_WORK = (
    {
        "method": "LatentWire 2-byte source-private packet",
        "source": "this work",
        "role": "headline systems row",
        "comparison_axis": "source-private residual evidence, 2-byte payload, strict controls",
    },
    {
        "method": "C2C cache-to-cache communication",
        "source": "https://arxiv.org/abs/2510.03215",
        "role": "closest high-rate cross-model internal-state baseline",
        "comparison_axis": "source/target KV-cache projection rather than public-side-info packet",
    },
    {
        "method": "KVComm selective KV sharing",
        "source": "https://openreview.net/forum?id=F7rUng23nw",
        "role": "KV communication baseline/framing",
        "comparison_axis": "selected KV pairs/layers rather than extreme-rate private packet",
    },
    {
        "method": "TurboQuant",
        "source": "https://arxiv.org/abs/2504.19874",
        "role": "low-bit vector/KV byte-floor comparator",
        "comparison_axis": "same-model vector quantization, not source-destroying communication control",
    },
    {
        "method": "QJL",
        "source": "https://arxiv.org/abs/2406.03482",
        "role": "1-bit sign-sketch byte-floor comparator",
        "comparison_axis": "inner-product preserving high-dimensional sketch",
    },
    {
        "method": "vLLM / PagedAttention and DistServe",
        "source": "https://arxiv.org/abs/2309.06180; https://arxiv.org/abs/2401.09670",
        "role": "serving metric convention",
        "comparison_axis": "future production TTFT/TPOT/throughput baseline",
    },
    {
        "method": "Diffusion/JEPA latent prediction",
        "source": "https://arxiv.org/abs/2212.09748; https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf",
        "role": "future learned-interface inspiration",
        "comparison_axis": "latent prediction objective, not current systems baseline",
    },
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _surface_name(path: pathlib.Path, summary: dict[str, Any]) -> str:
    name = path.parent.name
    if name.startswith("core_"):
        return f"core n{summary['n']} {summary.get('prompt_style', 'unknown')}"
    if name.startswith("holdout_"):
        return f"holdout n{summary['n']} {summary.get('prompt_style', 'unknown')}"
    return f"{name} n{summary.get('n', 'unknown')}"


def _metric(summary: dict[str, Any], condition: str, key: str, default: float | None = None) -> float:
    metrics = summary["metrics"].get(condition)
    if metrics is None:
        if default is None:
            raise KeyError(f"missing condition {condition}")
        return default
    value = metrics.get(key, default)
    if value is None:
        raise KeyError(f"missing metric {condition}.{key}")
    return float(value)


def _endpoint_row(
    *,
    summary_path: pathlib.Path,
    summary: dict[str, Any],
    uncertainty: dict[str, Any],
    kv_table: dict[str, Any],
) -> dict[str, Any]:
    packet = summary["metrics"]["matched_packet"]
    full_log = summary["metrics"]["full_hidden_log"]
    query = summary["metrics"]["query_aware_diag_span"]
    row = {
        "surface": _surface_name(summary_path, summary),
        "scope": "Mac CPU endpoint-proxy, not production serving throughput",
        "prompt_contract": summary.get("prompt_style", "unknown"),
        "pass_gate": bool(summary["pass_gate"] and uncertainty["pass_gate"]),
        "packet_accuracy": packet["accuracy"],
        "packet_strict_accuracy": packet.get("strict_accuracy", packet["accuracy"]),
        "target_accuracy": _metric(summary, "target_only", "accuracy"),
        "matched_byte_text_accuracy": _metric(summary, "matched_byte_text_2", "accuracy"),
        "best_source_destroying_control_accuracy": summary["best_source_destroying_control_accuracy"],
        "packet_valid_rate": packet["valid_prediction_rate"],
        "packet_payload_bytes": packet["mean_payload_bytes"],
        "query_aware_payload_bytes": query["mean_payload_bytes"],
        "full_log_payload_bytes": full_log["mean_payload_bytes"],
        "packet_vs_query_payload_compression": summary["packet_vs_query_payload_compression"],
        "packet_vs_full_log_payload_compression": summary["packet_vs_full_log_payload_compression"],
        "packet_p50_ttft_ms": packet["p50_ttft_ms"],
        "full_log_p50_ttft_ms": full_log["p50_ttft_ms"],
        "full_log_ttft_delta_vs_packet_ms": summary["full_log_ttft_delta_vs_packet_ms"],
        "min_packet_vs_target_ci95_low": uncertainty["min_packet_vs_target_ci95_low"],
        "min_packet_vs_best_control_ci95_low": uncertainty["min_packet_vs_best_control_ci95_low"],
        "min_strict_packet_vs_target_ci95_low": uncertainty["min_strict_packet_vs_target_ci95_low"],
        "qjl_1bit_cache_bytes_vs_packet_min": kv_table["headline"]["min_non_packet_qjl_1bit_bytes_vs_packet"],
        "kivi_2bit_cache_bytes_vs_packet_min": kv_table["headline"]["min_non_packet_kivi_2bit_bytes_vs_packet"],
        "caveat": (
            "Packet beats target and source-destroying controls at 2 bytes; query-aware/structured text can match or beat "
            "accuracy at higher byte rates; TTFT is local CPU proxy telemetry."
        ),
    }
    return row


def _terse_failure_row(summary_path: pathlib.Path, summary: dict[str, Any], kv_table: dict[str, Any]) -> dict[str, Any]:
    packet = summary["metrics"]["matched_packet"]
    full_log = summary["metrics"]["full_hidden_log"]
    query = summary["metrics"]["query_aware_diag_span"]
    return {
        "surface": _surface_name(summary_path, summary),
        "scope": "Mac CPU endpoint-proxy prompt-contract failure case",
        "prompt_contract": summary.get("prompt_style", "unknown"),
        "pass_gate": bool(summary["pass_gate"]),
        "packet_accuracy": packet["accuracy"],
        "packet_strict_accuracy": packet.get("strict_accuracy", packet["accuracy"]),
        "target_accuracy": _metric(summary, "target_only", "accuracy"),
        "matched_byte_text_accuracy": _metric(summary, "matched_byte_text_2", "accuracy"),
        "best_source_destroying_control_accuracy": max(
            _metric(summary, "matched_byte_text_2", "accuracy"),
            _metric(summary, "target_only", "accuracy"),
        ),
        "packet_valid_rate": packet["valid_prediction_rate"],
        "packet_payload_bytes": packet["mean_payload_bytes"],
        "query_aware_payload_bytes": query["mean_payload_bytes"],
        "full_log_payload_bytes": full_log["mean_payload_bytes"],
        "packet_vs_query_payload_compression": summary["packet_vs_query_payload_compression"],
        "packet_vs_full_log_payload_compression": summary["packet_vs_full_log_payload_compression"],
        "packet_p50_ttft_ms": packet["p50_ttft_ms"],
        "full_log_p50_ttft_ms": full_log["p50_ttft_ms"],
        "full_log_ttft_delta_vs_packet_ms": summary["full_log_ttft_delta_vs_packet_ms"],
        "min_packet_vs_target_ci95_low": None,
        "min_packet_vs_best_control_ci95_low": None,
        "min_strict_packet_vs_target_ci95_low": None,
        "qjl_1bit_cache_bytes_vs_packet_min": kv_table["headline"]["min_non_packet_qjl_1bit_bytes_vs_packet"],
        "kivi_2bit_cache_bytes_vs_packet_min": kv_table["headline"]["min_non_packet_kivi_2bit_bytes_vs_packet"],
        "caveat": "Under-specified prompt contract: the packet collapses to target accuracy, so the receiver contract is a required part of the method.",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Systems Caveat Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- paper claim scope: {payload['claim_scope']}",
        "",
        "## Headline",
        "",
    ]
    for key, value in payload["headline"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Endpoint Rows",
            "",
            "| Surface | Pass | Packet acc | Target acc | Best control | Packet bytes | Query bytes | Full-log bytes | Full-log TTFT delta ms | Caveat |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['surface']} | `{row['pass_gate']}` | {row['packet_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_source_destroying_control_accuracy']:.3f} | "
            f"{row['packet_payload_bytes']:.1f} | {row['query_aware_payload_bytes']:.1f} | "
            f"{row['full_log_payload_bytes']:.1f} | {row['full_log_ttft_delta_vs_packet_ms']:.1f} | "
            f"{row['caveat']} |"
        )
    lines.extend(
        [
            "",
            "## Related Systems Positioning",
            "",
            "| Method | Source | Role | Comparison axis |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["related_work"]:
        lines.append(f"| {row['method']} | {row['source']} | {row['role']} | {row['comparison_axis']} |")
    lines.extend(
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
    for item in payload["non_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_systems_caveat_frontier(
    *,
    endpoint_summaries: list[pathlib.Path],
    uncertainty_summaries: list[pathlib.Path],
    terse_failure_summary: pathlib.Path,
    kv_table_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_paths = [_resolve(path) for path in endpoint_summaries]
    uncertainty_paths = [_resolve(path) for path in uncertainty_summaries]
    if len(endpoint_paths) != len(uncertainty_paths):
        raise ValueError("endpoint_summaries and uncertainty_summaries must have the same length")
    kv_table = _read_json(_resolve(kv_table_path))
    rows = []
    for endpoint_path, uncertainty_path in zip(endpoint_paths, uncertainty_paths, strict=True):
        rows.append(
            _endpoint_row(
                summary_path=endpoint_path,
                summary=_read_json(endpoint_path),
                uncertainty=_read_json(uncertainty_path),
                kv_table=kv_table,
            )
        )
    terse_path = _resolve(terse_failure_summary)
    rows.append(_terse_failure_row(terse_path, _read_json(terse_path), kv_table))

    pass_rows = [row for row in rows if row["scope"].startswith("Mac CPU endpoint-proxy, not production")]
    headline = {
        "passing_endpoint_rows": sum(1 for row in pass_rows if row["pass_gate"]),
        "endpoint_rows": len(pass_rows),
        "packet_payload_bytes": min(row["packet_payload_bytes"] for row in pass_rows),
        "min_packet_minus_target_accuracy": min(row["packet_accuracy"] - row["target_accuracy"] for row in pass_rows),
        "min_packet_minus_best_control_accuracy": min(
            row["packet_accuracy"] - row["best_source_destroying_control_accuracy"] for row in pass_rows
        ),
        "min_packet_vs_query_payload_compression": min(row["packet_vs_query_payload_compression"] for row in pass_rows),
        "min_packet_vs_full_log_payload_compression": min(
            row["packet_vs_full_log_payload_compression"] for row in pass_rows
        ),
        "min_full_log_ttft_delta_vs_packet_ms": min(row["full_log_ttft_delta_vs_packet_ms"] for row in pass_rows),
        "min_packet_vs_target_ci95_low": min(row["min_packet_vs_target_ci95_low"] for row in pass_rows),
        "min_qjl_1bit_cache_bytes_vs_packet": kv_table["headline"]["min_non_packet_qjl_1bit_bytes_vs_packet"],
        "terse_prompt_pass_gate": rows[-1]["pass_gate"],
    }
    pass_gate = (
        headline["passing_endpoint_rows"] == headline["endpoint_rows"]
        and headline["min_packet_minus_target_accuracy"] >= 0.15
        and headline["min_packet_minus_best_control_accuracy"] >= 0.15
        and headline["min_packet_vs_query_payload_compression"] >= 7.0
        and headline["min_packet_vs_full_log_payload_compression"] >= 100.0
        and headline["min_full_log_ttft_delta_vs_packet_ms"] > 0.0
        and headline["min_packet_vs_target_ci95_low"] > 0.0
        and headline["min_qjl_1bit_cache_bytes_vs_packet"] >= 1000.0
        and headline["terse_prompt_pass_gate"] is False
    )
    payload = {
        "gate": "source_private_systems_caveat_frontier",
        "pass_gate": pass_gate,
        "claim_scope": "Mac-local endpoint proxy plus derived KV/cache byte floors; production serving remains future work.",
        "source_endpoint_summaries": [str(path.relative_to(ROOT)) for path in endpoint_paths],
        "source_uncertainty_summaries": [str(path.relative_to(ROOT)) for path in uncertainty_paths],
        "source_terse_failure_summary": str(terse_path.relative_to(ROOT)),
        "source_kv_table": str(_resolve(kv_table_path).relative_to(ROOT)),
        "headline": headline,
        "rows": rows,
        "related_work": RELATED_WORK,
        "interpretation": (
            "The systems contribution is an extreme-rate communication frontier: a 2-byte source-private packet "
            "passes strict endpoint controls on Mac CPU n160 rows, while visible text/KV-style alternatives occupy "
            "higher byte-rate regimes. This artifact deliberately records that query-aware structured text can match "
            "accuracy at 14 bytes and that the measured TTFT/E2E rows are local CPU proxy telemetry, not production "
            "throughput."
        ),
        "non_claims": [
            "No claim of beating TurboQuant, QJL, KIVI, KVQuant, C2C, or KVComm on their native same-model/KV tasks.",
            "No production GPU serving throughput claim until vLLM/OpenAI-compatible server runs are available.",
            "No broad cross-family latent-transfer claim; the current cross-family appendix is negative-boundary evidence.",
            "No prompt-contract-free receiver claim; the terse prompt failure shows the public receiver contract matters.",
        ],
    }

    json_path = output_dir / "systems_caveat_frontier.json"
    csv_path = output_dir / "systems_caveat_frontier.csv"
    md_path = output_dir / "systems_caveat_frontier.md"
    manifest_path = output_dir / "manifest.json"
    manifest_md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name, manifest_md_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": headline,
        "interpretation": payload["interpretation"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_md_path.write_text(
        "\n".join(
            [
                "# Source-Private Systems Caveat Frontier Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- endpoint rows: `{headline['passing_endpoint_rows']}/{headline['endpoint_rows']}`",
                f"- packet payload bytes: `{headline['packet_payload_bytes']}`",
                f"- min packet-vs-query compression: `{headline['min_packet_vs_query_payload_compression']:.1f}x`",
                f"- min QJL 1-bit cache bytes / packet: `{headline['min_qjl_1bit_cache_bytes_vs_packet']:.1f}x`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-summary", action="append", type=pathlib.Path)
    parser.add_argument("--uncertainty-summary", action="append", type=pathlib.Path)
    parser.add_argument("--terse-failure-summary", type=pathlib.Path, default=pathlib.Path(DEFAULT_TERSE_FAILURE))
    parser.add_argument("--kv-table", type=pathlib.Path, default=pathlib.Path(DEFAULT_KV_TABLE))
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_systems_caveat_frontier_20260429"),
    )
    args = parser.parse_args()
    endpoint_summaries = args.endpoint_summary or [pathlib.Path(path) for path in DEFAULT_ENDPOINT_SUMMARIES]
    uncertainty_summaries = args.uncertainty_summary or [pathlib.Path(path) for path in DEFAULT_UNCERTAINTY_SUMMARIES]
    output_dir = _resolve(args.output_dir)
    payload = build_systems_caveat_frontier(
        endpoint_summaries=endpoint_summaries,
        uncertainty_summaries=uncertainty_summaries,
        terse_failure_summary=args.terse_failure_summary,
        kv_table_path=args.kv_table,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "pass_gate": payload["pass_gate"], "headline": payload["headline"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
