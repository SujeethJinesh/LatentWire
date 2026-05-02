from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import platform
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_ENDPOINT_ROWS = (
    pathlib.Path(
        "results/source_private_mac_endpoint_proxy_frontier_20260429/"
        "core_seed29_qwen3_n160_cpu_label_strict_controls/endpoint_proxy_rows.jsonl"
    ),
    pathlib.Path(
        "results/source_private_mac_endpoint_proxy_frontier_20260429/"
        "holdout_seed30_qwen3_n160_cpu_label_strict_controls/endpoint_proxy_rows.jsonl"
    ),
)
DEFAULT_ENDPOINT_SUMMARIES = (
    pathlib.Path(
        "results/source_private_mac_endpoint_proxy_frontier_20260429/"
        "core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json"
    ),
    pathlib.Path(
        "results/source_private_mac_endpoint_proxy_frontier_20260429/"
        "holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json"
    ),
)
DEFAULT_KV_BASELINE_TABLE = pathlib.Path(
    "results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json"
)
DEFAULT_PACKET_ISA_FRONTIER = pathlib.Path(
    "results/source_private_packet_isa_batch_frontier_20260430/packet_isa_batch_frontier.json"
)

CONDITION_ORDER = (
    "target_only",
    "matched_packet",
    "matched_byte_text_2",
    "random_same_byte_packet",
    "deranged_candidate_diag_table",
    "query_aware_diag_span",
    "structured_json_diag",
    "structured_free_text_diag",
    "full_hidden_log",
)

SOURCE_DESTROYING_CONTROLS = {
    "matched_byte_text_2",
    "random_same_byte_packet",
    "deranged_candidate_diag_table",
}

SOURCE_TEXT_CONDITIONS = {
    "matched_byte_text_2",
    "query_aware_diag_span",
    "structured_json_diag",
    "structured_free_text_diag",
    "full_hidden_log",
}

PRIVATE_TEXT_CONDITIONS = {
    "query_aware_diag_span",
    "structured_json_diag",
    "structured_free_text_diag",
    "full_hidden_log",
}

CSV_COLUMNS = (
    "surface",
    "condition",
    "row_class",
    "claim_class",
    "accuracy",
    "strict_accuracy",
    "target_accuracy",
    "delta_vs_target",
    "payload_bytes",
    "transport_record_bytes",
    "raw_ratio_vs_packet",
    "single_request_line_bytes_64b",
    "line_ratio_vs_packet",
    "single_request_dma_bytes_128b",
    "dma_ratio_vs_packet",
    "batch64_packet_record_line_bytes_per_request",
    "batch64_packet_record_dma_bytes_per_request",
    "prompt_tokens",
    "prompt_token_delta_vs_packet",
    "prompt_bytes",
    "prompt_byte_delta_vs_packet",
    "generated_tokens",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p50_e2e_ms",
    "p95_e2e_ms",
    "qjl_1bit_kv_delta_bytes_vs_packet",
    "kivi_2bit_kv_delta_bytes_vs_packet",
    "turboquant_2p5bit_kv_delta_bytes_vs_packet",
    "turboquant_3p5bit_kv_delta_bytes_vs_packet",
    "fp16_bf16_kv_delta_bytes_vs_packet",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "source_destroying_control",
    "transport_conclusion",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _ceil_bytes(value: float, quantum: int) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil(value / quantum) * quantum)


def _run_sysctl(name: str) -> str | None:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = proc.stdout.strip()
    return value or None


def _host_profile() -> dict[str, Any]:
    memsize = _run_sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "mac_model": _run_sysctl("hw.model"),
        "cpu_brand": _run_sysctl("machdep.cpu.brand_string"),
        "memory_bytes": None if memsize is None else int(memsize),
        "memory_gib": None if memsize is None else round(int(memsize) / (1024**3), 2),
        "execution_note": (
            "Profile generation is CPU-only deterministic artifact accounting over existing endpoint rows; "
            "it does not run an MPS generation job."
        ),
    }


def _surface_name(path: pathlib.Path, summary: dict[str, Any]) -> str:
    folder = path.parent.name
    if folder.startswith("core_"):
        return f"core n{summary['n']} {summary.get('prompt_style', 'canonical')}"
    if folder.startswith("holdout_"):
        return f"holdout n{summary['n']} {summary.get('prompt_style', 'canonical')}"
    return folder


def _group_ids_by_condition(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(str(row["example_id"]))
    return grouped


def _exact_id_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_ids_by_condition(rows)
    if not grouped:
        return {"exact_id_parity": False, "conditions": [], "n": 0}
    first_condition = sorted(grouped)[0]
    reference = grouped[first_condition]
    return {
        "exact_id_parity": all(ids == reference for ids in grouped.values()),
        "conditions": sorted(grouped),
        "n": len(reference),
        "reference_condition": first_condition,
        "exact_id_sha256": hashlib.sha256("\n".join(reference).encode("utf-8")).hexdigest(),
    }


def _packet_batch_lookup(packet_isa: dict[str, Any], *, payload_bytes: int, batch_size: int) -> dict[str, Any] | None:
    for row in packet_isa["rows"]:
        if int(row["payload_bytes"]) == payload_bytes and int(row["batch_size"]) == batch_size:
            return row
    return None


def _row_class(condition: str) -> str:
    if condition == "target_only":
        return "target_baseline"
    if condition == "matched_packet":
        return "source_private_packet"
    if condition in SOURCE_DESTROYING_CONTROLS:
        return "source_destroying_control"
    if condition == "full_hidden_log":
        return "private_log_relay"
    if condition in PRIVATE_TEXT_CONDITIONS:
        return "private_text_relay"
    return "endpoint_condition"


def _claim_class(condition: str) -> str:
    if condition == "matched_packet":
        return "headline_endpoint_proxy"
    if condition == "target_only":
        return "no_source_baseline"
    if condition in SOURCE_DESTROYING_CONTROLS:
        return "destructive_control"
    if condition in PRIVATE_TEXT_CONDITIONS:
        return "source_text_comparator"
    return "endpoint_accounting"


def _transport_conclusion(condition: str, line_ratio: float, raw_ratio: float) -> str:
    if condition == "matched_packet":
        return "2-byte source-private packet; one line when isolated and batch-packable to 5B/request"
    if condition == "target_only":
        return "no source-side transport"
    if condition == "query_aware_diag_span":
        return "short text ties one 64B line but uses 7x raw bytes and exposes private evidence text"
    if condition in {"structured_json_diag", "structured_free_text_diag"}:
        return "structured text uses more raw bytes, exposes private evidence text, and adds prompt/KV tokens"
    if condition == "full_hidden_log":
        return "full private log is hundreds of bytes, multiple lines, larger prompt/KV traffic, and slower TTFT"
    if condition == "matched_byte_text_2":
        return "same-byte text control stays at target accuracy, so raw text budget alone does not explain the gain"
    if condition in SOURCE_DESTROYING_CONTROLS:
        return "source-destroying control; should stay near target-only"
    return f"endpoint accounting row with {raw_ratio:.2f}x raw and {line_ratio:.2f}x line bytes versus packet"


def _kv_delta_bytes(
    *,
    prompt_delta_vs_packet: float,
    bytes_per_token: dict[str, float],
    scheme: str,
) -> float:
    return max(0.0, prompt_delta_vs_packet) * float(bytes_per_token[scheme])


def _build_surface_rows(
    *,
    rows_path: pathlib.Path,
    summary_path: pathlib.Path,
    summary: dict[str, Any],
    kv_baseline: dict[str, Any],
    packet_isa: dict[str, Any],
    amortized_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = _read_jsonl(rows_path)
    parity = _exact_id_parity(raw_rows)
    surface = _surface_name(rows_path, summary)
    metrics_by_condition = summary["metrics"]
    packet_metrics = metrics_by_condition["matched_packet"]
    target_metrics = metrics_by_condition["target_only"]
    packet_payload = float(packet_metrics["mean_payload_bytes"])
    packet_line = _ceil_bytes(packet_payload, 64)
    packet_dma = _ceil_bytes(packet_payload, 128)
    packet_batch = _packet_batch_lookup(
        packet_isa,
        payload_bytes=int(packet_payload),
        batch_size=amortized_batch_size,
    )

    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        metrics = metrics_by_condition[condition]
        payload_bytes = float(metrics["mean_payload_bytes"])
        transport_record_bytes = payload_bytes + 3.0 if condition == "matched_packet" else payload_bytes
        line_bytes = _ceil_bytes(transport_record_bytes, 64)
        dma_bytes = _ceil_bytes(transport_record_bytes, 128)
        prompt_delta = max(0.0, float(metrics["mean_prompt_tokens"]) - float(packet_metrics["mean_prompt_tokens"]))
        prompt_byte_delta = float(metrics["mean_prompt_bytes"]) - float(packet_metrics["mean_prompt_bytes"])
        batch_line = None
        batch_dma = None
        if condition == "matched_packet" and packet_batch is not None:
            batch_line = float(packet_batch["line_bytes_per_request_packed"])
            batch_dma = float(packet_batch["dma_bytes_per_request_packed"])
        line_ratio = 0.0 if condition == "target_only" else line_bytes / packet_line
        dma_ratio = 0.0 if condition == "target_only" else dma_bytes / packet_dma
        raw_ratio = 0.0 if condition == "target_only" else payload_bytes / packet_payload
        row = {
            "surface": surface,
            "condition": condition,
            "row_class": _row_class(condition),
            "claim_class": _claim_class(condition),
            "accuracy": float(metrics["accuracy"]),
            "strict_accuracy": float(metrics["strict_accuracy"]),
            "target_accuracy": float(target_metrics["accuracy"]),
            "delta_vs_target": float(metrics["accuracy"]) - float(target_metrics["accuracy"]),
            "payload_bytes": payload_bytes,
            "transport_record_bytes": transport_record_bytes,
            "raw_ratio_vs_packet": raw_ratio,
            "single_request_line_bytes_64b": line_bytes,
            "line_ratio_vs_packet": line_ratio,
            "single_request_dma_bytes_128b": dma_bytes,
            "dma_ratio_vs_packet": dma_ratio,
            "batch64_packet_record_line_bytes_per_request": batch_line,
            "batch64_packet_record_dma_bytes_per_request": batch_dma,
            "prompt_tokens": float(metrics["mean_prompt_tokens"]),
            "prompt_token_delta_vs_packet": prompt_delta,
            "prompt_bytes": float(metrics["mean_prompt_bytes"]),
            "prompt_byte_delta_vs_packet": prompt_byte_delta,
            "generated_tokens": float(metrics["mean_generated_tokens"]),
            "p50_ttft_ms": float(metrics["p50_ttft_ms"]),
            "p95_ttft_ms": float(metrics["p95_ttft_ms"]),
            "p50_e2e_ms": float(metrics["p50_e2e_ms"]),
            "p95_e2e_ms": float(metrics["p95_e2e_ms"]),
            "qjl_1bit_kv_delta_bytes_vs_packet": _kv_delta_bytes(
                prompt_delta_vs_packet=prompt_delta,
                bytes_per_token=kv_baseline["bytes_per_token"],
                scheme="qjl_1bit_sign_proxy",
            ),
            "kivi_2bit_kv_delta_bytes_vs_packet": _kv_delta_bytes(
                prompt_delta_vs_packet=prompt_delta,
                bytes_per_token=kv_baseline["bytes_per_token"],
                scheme="kivi_2bit_proxy",
            ),
            "turboquant_2p5bit_kv_delta_bytes_vs_packet": _kv_delta_bytes(
                prompt_delta_vs_packet=prompt_delta,
                bytes_per_token=kv_baseline["bytes_per_token"],
                scheme="turboquant_2p5bit_proxy",
            ),
            "turboquant_3p5bit_kv_delta_bytes_vs_packet": _kv_delta_bytes(
                prompt_delta_vs_packet=prompt_delta,
                bytes_per_token=kv_baseline["bytes_per_token"],
                scheme="turboquant_3p5bit_proxy",
            ),
            "fp16_bf16_kv_delta_bytes_vs_packet": _kv_delta_bytes(
                prompt_delta_vs_packet=prompt_delta,
                bytes_per_token=kv_baseline["bytes_per_token"],
                scheme="fp16_bf16",
            ),
            "source_private": condition == "matched_packet",
            "source_text_exposed": condition in SOURCE_TEXT_CONDITIONS,
            "source_kv_exposed": False,
            "source_destroying_control": condition in SOURCE_DESTROYING_CONTROLS,
        }
        row["transport_conclusion"] = _transport_conclusion(condition, line_ratio, raw_ratio)
        rows.append(row)
    return rows, {"surface": surface, **parity, "summary_pass_gate": bool(summary["pass_gate"])}


def build_mac_unified_memory_transport_profile(
    *,
    endpoint_rows: list[pathlib.Path],
    endpoint_summaries: list[pathlib.Path],
    kv_baseline_table: pathlib.Path,
    packet_isa_frontier: pathlib.Path,
    output_dir: pathlib.Path,
    amortized_batch_size: int,
) -> dict[str, Any]:
    if len(endpoint_rows) != len(endpoint_summaries):
        raise ValueError("endpoint_rows and endpoint_summaries must have the same length")

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kv_baseline = _read_json(kv_baseline_table)
    packet_isa = _read_json(packet_isa_frontier)

    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for rows_path, summary_path in zip(endpoint_rows, endpoint_summaries, strict=True):
        summary = _read_json(summary_path)
        surface_rows, parity = _build_surface_rows(
            rows_path=rows_path,
            summary_path=summary_path,
            summary=summary,
            kv_baseline=kv_baseline,
            packet_isa=packet_isa,
            amortized_batch_size=amortized_batch_size,
        )
        rows.extend(surface_rows)
        parity_rows.append(parity)

    packet_rows = [row for row in rows if row["condition"] == "matched_packet"]
    query_rows = [row for row in rows if row["condition"] == "query_aware_diag_span"]
    full_log_rows = [row for row in rows if row["condition"] == "full_hidden_log"]
    text_rows = [row for row in rows if row["condition"] in PRIVATE_TEXT_CONDITIONS]
    control_rows = [row for row in rows if row["source_destroying_control"]]
    positive_rows = [row for row in rows if row["condition"] == "matched_packet"]

    min_packet_payload = min(row["payload_bytes"] for row in packet_rows)
    max_control_delta = max(row["delta_vs_target"] for row in control_rows)
    min_packet_delta = min(row["delta_vs_target"] for row in packet_rows)
    min_full_log_raw_ratio = min(row["raw_ratio_vs_packet"] for row in full_log_rows)
    min_full_log_line_ratio = min(row["line_ratio_vs_packet"] for row in full_log_rows)
    min_query_raw_ratio = min(row["raw_ratio_vs_packet"] for row in query_rows)
    min_query_line_ratio = min(row["line_ratio_vs_packet"] for row in query_rows)
    min_qjl_full_log_ratio = min(
        row["qjl_1bit_kv_delta_bytes_vs_packet"] / min_packet_payload for row in full_log_rows
    )
    min_batch_line = min(
        row["batch64_packet_record_line_bytes_per_request"]
        for row in packet_rows
        if row["batch64_packet_record_line_bytes_per_request"] is not None
    )
    min_batch_dma = min(
        row["batch64_packet_record_dma_bytes_per_request"]
        for row in packet_rows
        if row["batch64_packet_record_dma_bytes_per_request"] is not None
    )

    pass_gate = (
        all(row["exact_id_parity"] for row in parity_rows)
        and all(row["summary_pass_gate"] for row in parity_rows)
        and min_packet_payload == 2.0
        and min_packet_delta >= 0.40
        and max_control_delta <= 0.02
        and min_query_raw_ratio >= 7.0
        and min_query_line_ratio == 1.0
        and min_full_log_raw_ratio > 100.0
        and min_full_log_line_ratio >= 6.0
        and min_qjl_full_log_ratio > 300_000.0
        and min_batch_line <= 5.0
    )

    headline = {
        "pass_gate": pass_gate,
        "surfaces": len(parity_rows),
        "rows": len(rows),
        "exact_id_parity": all(row["exact_id_parity"] for row in parity_rows),
        "packet_payload_bytes": sorted({row["payload_bytes"] for row in packet_rows}),
        "matched_packet_min_delta_vs_target": min_packet_delta,
        "max_source_destroying_control_delta_vs_target": max_control_delta,
        "query_aware_text_raw_ratio_min": min_query_raw_ratio,
        "query_aware_text_line_ratio_min": min_query_line_ratio,
        "full_log_raw_ratio_min": min_full_log_raw_ratio,
        "full_log_line_ratio_min": min_full_log_line_ratio,
        "full_log_qjl_1bit_kv_delta_bytes_per_packet_byte_min": min_qjl_full_log_ratio,
        "packet_batch64_line_bytes_per_request_min": min_batch_line,
        "packet_batch64_dma_bytes_per_request_min": min_batch_dma,
        "source_text_exposed_rows": sum(bool(row["source_text_exposed"]) for row in rows),
        "source_kv_exposed_rows": sum(bool(row["source_kv_exposed"]) for row in rows),
        "private_text_relay_rows": len(text_rows),
        "positive_packet_rows": len(positive_rows),
    }

    payload = {
        "gate": "source_private_mac_unified_memory_transport_profile",
        "pass_gate": pass_gate,
        "headline": headline,
        "parity": parity_rows,
        "host_profile": _host_profile(),
        "model_config": kv_baseline["model_config"],
        "kv_bytes_per_prompt_token": kv_baseline["bytes_per_token"],
        "rows": rows,
        "sources": {
            "endpoint_rows": [str(path) for path in endpoint_rows],
            "endpoint_row_sha256": {str(path): _sha256_file(path) for path in endpoint_rows},
            "endpoint_summaries": [str(path) for path in endpoint_summaries],
            "endpoint_summary_sha256": {str(path): _sha256_file(path) for path in endpoint_summaries},
            "kv_baseline_table": str(kv_baseline_table),
            "kv_baseline_table_sha256": _sha256_file(kv_baseline_table),
            "packet_isa_frontier": str(packet_isa_frontier),
            "packet_isa_frontier_sha256": _sha256_file(packet_isa_frontier),
        },
        "interpretation": (
            "Mac unified-memory transport profile over existing CPU endpoint rows. It separates raw packet bytes, "
            "64B line rounding, 128B DMA-burst rounding, batch-64 packet packing, prompt/KV byte floors, TTFT "
            "proxy telemetry, and private source-state exposure."
        ),
        "non_claims": [
            "This is not a native MPS/GPU kernel benchmark and not a production serving benchmark.",
            "Unified-memory line/DMA values are deterministic accounting proxies, not measured hardware counters.",
            "KV rows are prompt-token byte floors under published quantization-style bit widths, not a KVComm implementation.",
            "Short query-aware text can tie a single packet at one 64B line while still exposing private text.",
        ],
    }

    json_path = output_dir / "mac_unified_memory_transport_profile.json"
    csv_path = output_dir / "mac_unified_memory_transport_profile.csv"
    md_path = output_dir / "mac_unified_memory_transport_profile.md"
    manifest_path = output_dir / "manifest.json"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})
    _write_markdown(md_path, payload)

    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "headline": headline,
        "artifacts": [
            json_path.name,
            csv_path.name,
            md_path.name,
            manifest_path.name,
            "manifest.md",
        ],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Mac Unified-Memory Transport Profile Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- surfaces: `{headline['surfaces']}`",
                f"- packet payload bytes: `{headline['packet_payload_bytes']}`",
                f"- query-aware text raw ratio min: `{headline['query_aware_text_raw_ratio_min']:.2f}x`",
                f"- query-aware text line ratio min: `{headline['query_aware_text_line_ratio_min']:.2f}x`",
                f"- full-log raw ratio min: `{headline['full_log_raw_ratio_min']:.2f}x`",
                f"- packet batch-64 line bytes/request min: `{headline['packet_batch64_line_bytes_per_request_min']:.2f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Mac Unified-Memory Transport Profile",
        "",
        "This artifact profiles the existing `n=160` core and held-out Mac endpoint rows as",
        "source-target transport objects: raw bytes, cache-line rounding, DMA-burst rounding,",
        "batch-packed packet records, prompt/KV byte floors, and private-state exposure.",
        "",
        "## Headline",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- exact ID parity: `{headline['exact_id_parity']}`",
        f"- packet payload bytes: `{headline['packet_payload_bytes']}`",
        f"- matched packet min delta vs target: `{headline['matched_packet_min_delta_vs_target']:.3f}`",
        f"- max source-destroying control delta vs target: `{headline['max_source_destroying_control_delta_vs_target']:.3f}`",
        f"- query-aware text raw ratio: `{headline['query_aware_text_raw_ratio_min']:.2f}x`",
        f"- query-aware text 64B-line ratio: `{headline['query_aware_text_line_ratio_min']:.2f}x`",
        f"- full-log raw ratio: `{headline['full_log_raw_ratio_min']:.2f}x`",
        f"- full-log 64B-line ratio: `{headline['full_log_line_ratio_min']:.2f}x`",
        f"- full-log QJL 1-bit prompt-KV delta / packet byte: "
        f"`{headline['full_log_qjl_1bit_kv_delta_bytes_per_packet_byte_min']:.1f}x`",
        f"- packet batch-64 packed line bytes/request: "
        f"`{headline['packet_batch64_line_bytes_per_request_min']:.2f}`",
        f"- packet batch-64 packed DMA bytes/request: "
        f"`{headline['packet_batch64_dma_bytes_per_request_min']:.2f}`",
        "",
        "## Host Profile",
        "",
    ]
    host = payload["host_profile"]
    for key in ("mac_model", "cpu_brand", "machine", "memory_gib", "platform"):
        lines.append(f"- {key}: `{host.get(key)}`")
    lines.extend(
        [
            f"- execution note: {host['execution_note']}",
            "",
        ]
    )
    lines.extend(
        [
        "## Rows",
        "",
        "| Surface | Condition | Acc | Payload B | 64B line B | 128B DMA B | Prompt delta | QJL KV delta B | TTFT p50 | Exposure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        exposure = []
        if row["source_private"]:
            exposure.append("packet")
        if row["source_text_exposed"]:
            exposure.append("text")
        if row["source_kv_exposed"]:
            exposure.append("KV")
        if not exposure:
            exposure.append("none")
        lines.append(
            f"| {row['surface']} | `{row['condition']}` | {row['accuracy']:.3f} | "
            f"{row['payload_bytes']:.1f} | {row['single_request_line_bytes_64b']:.1f} | "
            f"{row['single_request_dma_bytes_128b']:.1f} | {row['prompt_token_delta_vs_packet']:.2f} | "
            f"{row['qjl_1bit_kv_delta_bytes_vs_packet']:.1f} | {row['p50_ttft_ms']:.1f} | "
            f"{', '.join(exposure)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "The important boundary is not just raw byte count. A 2-byte packet still rounds",
            "to one transfer quantum for a single request. The win is that the packet avoids",
            "private source text and source KV/cache movement, and it becomes materially tiny",
            "when packed across a batch. Short query-aware text can tie the packet at one 64B",
            "line, but it is a private-text relay and uses 7x raw semantic payload.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-row", action="append", type=pathlib.Path)
    parser.add_argument("--endpoint-summary", action="append", type=pathlib.Path)
    parser.add_argument("--kv-baseline-table", type=pathlib.Path, default=DEFAULT_KV_BASELINE_TABLE)
    parser.add_argument("--packet-isa-frontier", type=pathlib.Path, default=DEFAULT_PACKET_ISA_FRONTIER)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_mac_unified_memory_transport_profile_20260430"),
    )
    parser.add_argument("--amortized-batch-size", type=int, default=64)
    args = parser.parse_args()
    payload = build_mac_unified_memory_transport_profile(
        endpoint_rows=args.endpoint_row or list(DEFAULT_ENDPOINT_ROWS),
        endpoint_summaries=args.endpoint_summary or list(DEFAULT_ENDPOINT_SUMMARIES),
        kv_baseline_table=args.kv_baseline_table,
        packet_isa_frontier=args.packet_isa_frontier,
        output_dir=args.output_dir,
        amortized_batch_size=args.amortized_batch_size,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(
        json.dumps(
            {"output_dir": str(output_dir), "pass_gate": payload["pass_gate"], "headline": payload["headline"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
