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
DEFAULT_QWEN3_CONFIG = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/config.json"
)

BITS_PER_ELEMENT = {
    "fp16_bf16": 16.0,
    "int8": 8.0,
    "int4": 4.0,
    "turboquant_3p5bit_proxy": 3.5,
    "turboquant_2p5bit_proxy": 2.5,
    "kivi_2bit_proxy": 2.0,
    "qjl_1bit_sign_proxy": 1.0,
}

CONDITION_ORDER = (
    "matched_packet",
    "matched_byte_text_2",
    "query_aware_diag_span",
    "structured_free_text_diag",
    "structured_json_diag",
    "full_hidden_log",
)

BASELINE_COMPARISON_ROWS = (
    {
        "method": "LatentWire 2-byte source-private packet",
        "source_private": True,
        "decoder_side_information": True,
        "source_destroying_controls": True,
        "systems_axis": "extreme-rate private evidence communication",
        "paper_use": "headline method row",
    },
    {
        "method": "TurboQuant-style KV/vector quantization",
        "source_private": False,
        "decoder_side_information": False,
        "source_destroying_controls": False,
        "systems_axis": "same-model vector/KV compression",
        "paper_use": "byte-floor baseline and caveat",
    },
    {
        "method": "QJL-style 1-bit sign sketch",
        "source_private": False,
        "decoder_side_information": False,
        "source_destroying_controls": False,
        "systems_axis": "inner-product-preserving KV sketch",
        "paper_use": "same-byte sketch ablation if sparse receiver becomes live",
    },
    {
        "method": "KIVI/KVQuant-style low-bit KV cache",
        "source_private": False,
        "decoder_side_information": False,
        "source_destroying_controls": False,
        "systems_axis": "same-model long-context KV memory reduction",
        "paper_use": "cache payload accounting baseline",
    },
    {
        "method": "SnapKV/CacheGen-style pruning or cache streaming",
        "source_private": False,
        "decoder_side_information": False,
        "source_destroying_controls": False,
        "systems_axis": "selected/cache-streamed model-visible context",
        "paper_use": "higher-byte systems comparator",
    },
    {
        "method": "vLLM/PagedAttention/DistServe-style serving systems",
        "source_private": False,
        "decoder_side_information": False,
        "source_destroying_controls": False,
        "systems_axis": "throughput, TTFT, TPOT, memory scheduling",
        "paper_use": "metric convention and future GPU-serving comparator",
    },
)

CSV_COLUMNS = (
    "surface",
    "condition",
    "accuracy",
    "payload_bytes",
    "payload_bytes_vs_packet",
    "prompt_token_delta_vs_packet",
    "prompt_token_delta_vs_target",
    "kv_payload_bytes_fp16_bf16",
    "kv_payload_bytes_int8",
    "kv_payload_bytes_int4",
    "kv_payload_bytes_turboquant_3p5bit_proxy",
    "kv_payload_bytes_turboquant_2p5bit_proxy",
    "kv_payload_bytes_kivi_2bit_proxy",
    "kv_payload_bytes_qjl_1bit_sign_proxy",
    "qjl_1bit_bytes_vs_packet",
    "kivi_2bit_bytes_vs_packet",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p50_e2e_ms",
    "p95_e2e_ms",
    "mean_prompt_tokens",
    "mean_generated_tokens",
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _kv_bytes_per_token(config: dict[str, Any], bits_per_element: float) -> float:
    kv_heads = config.get("num_key_value_heads") or config["num_attention_heads"]
    head_dim = config.get("head_dim") or config["hidden_size"] / config["num_attention_heads"]
    elements = config["num_hidden_layers"] * 2 * kv_heads * head_dim
    return elements * bits_per_element / 8.0


def _surface_name(path: pathlib.Path, summary: dict[str, Any]) -> str:
    name = path.parent.name
    if name.startswith("core_"):
        return f"core n{summary['n']} {summary.get('prompt_style', 'canonical')}"
    if name.startswith("holdout_"):
        return f"holdout n{summary['n']} {summary.get('prompt_style', 'canonical')}"
    return name


def _condition_row(
    *,
    summary_path: pathlib.Path,
    summary: dict[str, Any],
    condition: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = summary["metrics"][condition]
    packet = summary["metrics"]["matched_packet"]
    target = summary["metrics"]["target_only"]
    prompt_delta_packet = max(0.0, metrics["mean_prompt_tokens"] - packet["mean_prompt_tokens"])
    prompt_delta_target = max(0.0, metrics["mean_prompt_tokens"] - target["mean_prompt_tokens"])
    payload_bytes = metrics["mean_payload_bytes"]
    packet_bytes = max(packet["mean_payload_bytes"], 1e-9)
    row: dict[str, Any] = {
        "surface": _surface_name(summary_path, summary),
        "condition": condition,
        "accuracy": metrics["accuracy"],
        "payload_bytes": payload_bytes,
        "payload_bytes_vs_packet": payload_bytes / packet_bytes,
        "prompt_token_delta_vs_packet": prompt_delta_packet,
        "prompt_token_delta_vs_target": prompt_delta_target,
        "p50_ttft_ms": metrics["p50_ttft_ms"],
        "p95_ttft_ms": metrics["p95_ttft_ms"],
        "p50_e2e_ms": metrics["p50_e2e_ms"],
        "p95_e2e_ms": metrics["p95_e2e_ms"],
        "mean_prompt_tokens": metrics["mean_prompt_tokens"],
        "mean_generated_tokens": metrics["mean_generated_tokens"],
    }
    for label, bits in BITS_PER_ELEMENT.items():
        bytes_for_delta = prompt_delta_packet * _kv_bytes_per_token(config, bits)
        row[f"kv_payload_bytes_{label}"] = bytes_for_delta
    row["qjl_1bit_bytes_vs_packet"] = row["kv_payload_bytes_qjl_1bit_sign_proxy"] / packet_bytes
    row["kivi_2bit_bytes_vs_packet"] = row["kv_payload_bytes_kivi_2bit_proxy"] / packet_bytes
    return row


def build_kv_cache_baseline_table(
    *,
    endpoint_summaries: list[pathlib.Path],
    qwen3_config: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(qwen3_config)
    rows: list[dict[str, Any]] = []
    for summary_path in endpoint_summaries:
        summary = _read_json(summary_path)
        for condition in CONDITION_ORDER:
            rows.append(_condition_row(summary_path=summary_path, summary=summary, condition=condition, config=config))

    packet_rows = [row for row in rows if row["condition"] == "matched_packet"]
    non_packet_rows = [row for row in rows if row["condition"] != "matched_packet"]
    min_qjl_ratio = min(row["qjl_1bit_bytes_vs_packet"] for row in non_packet_rows if row["prompt_token_delta_vs_packet"] > 0)
    min_kivi_ratio = min(row["kivi_2bit_bytes_vs_packet"] for row in non_packet_rows if row["prompt_token_delta_vs_packet"] > 0)
    payload = {
        "gate": "source_private_kv_cache_baseline_table",
        "source_endpoint_summaries": [str(path) for path in endpoint_summaries],
        "model_config": {
            "path": str(qwen3_config),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "hidden_size": config["hidden_size"],
        },
        "bytes_per_token": {
            label: _kv_bytes_per_token(config, bits) for label, bits in BITS_PER_ELEMENT.items()
        },
        "rows": rows,
        "headline": {
            "surfaces": len(packet_rows),
            "packet_payload_bytes": sorted({row["payload_bytes"] for row in packet_rows}),
            "min_non_packet_qjl_1bit_bytes_vs_packet": min_qjl_ratio,
            "min_non_packet_kivi_2bit_bytes_vs_packet": min_kivi_ratio,
        },
        "baseline_comparison_rows": BASELINE_COMPARISON_ROWS,
        "interpretation": (
            "Derived byte accounting only: this is not a kernel implementation of TurboQuant, QJL, KIVI, or KVQuant. "
            "It estimates the minimum KV-cache payload needed to relay the extra private payload tokens in the "
            "existing endpoint summaries under several bits-per-element assumptions."
        ),
    }

    json_path = output_dir / "kv_cache_baseline_table.json"
    csv_path = output_dir / "kv_cache_baseline_table.csv"
    md_path = output_dir / "kv_cache_baseline_table.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": payload["headline"],
        "model_config": payload["model_config"],
        "qwen3_config_sha256": _sha256_file(qwen3_config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private KV/Cache Baseline Table",
        "",
        "This is derived accounting over existing endpoint summaries. It is a systems",
        "baseline table, not a KV quantization kernel benchmark.",
        "",
        "## Model Geometry",
        "",
    ]
    cfg = payload["model_config"]
    for key in ("model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size"):
        lines.append(f"- {key}: `{cfg.get(key)}`")
    lines.extend(["", "## Bytes Per Prompt Token", "", "| Scheme | Bytes/token |", "|---|---:|"])
    for label, value in payload["bytes_per_token"].items():
        lines.append(f"| `{label}` | {value:.1f} |")
    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"- packet payload bytes: `{payload['headline']['packet_payload_bytes']}`",
            f"- minimum non-packet QJL-style 1-bit cache bytes / packet bytes: `{payload['headline']['min_non_packet_qjl_1bit_bytes_vs_packet']:.1f}x`",
            f"- minimum non-packet KIVI-style 2-bit cache bytes / packet bytes: `{payload['headline']['min_non_packet_kivi_2bit_bytes_vs_packet']:.1f}x`",
            "",
            "## Rows",
            "",
            "| Surface | Condition | Acc | Payload bytes | Prompt delta vs packet | QJL 1-bit bytes | KIVI 2-bit bytes | p50 TTFT ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['surface']} | `{row['condition']}` | {row['accuracy']:.3f} | "
            f"{row['payload_bytes']:.1f} | {row['prompt_token_delta_vs_packet']:.2f} | "
            f"{row['kv_payload_bytes_qjl_1bit_sign_proxy']:.1f} | "
            f"{row['kv_payload_bytes_kivi_2bit_proxy']:.1f} | {row['p50_ttft_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Comparison Axes",
            "",
            "| Method family | Source-private? | Decoder side info? | Source-destroying controls? | Systems axis | Paper use |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["baseline_comparison_rows"]:
        lines.append(
            f"| {row['method']} | `{row['source_private']}` | `{row['decoder_side_information']}` | "
            f"`{row['source_destroying_controls']}` | {row['systems_axis']} | {row['paper_use']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Reviewer caveat: cache quantization methods such as TurboQuant, QJL, KIVI,",
            "KVQuant, SnapKV, and CacheGen attack model-visible KV/context movement.",
            "LatentWire's claim should remain source-private residual communication,",
            "not generic KV-cache compression superiority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-summary", action="append", type=pathlib.Path)
    parser.add_argument("--qwen3-config", type=pathlib.Path, default=DEFAULT_QWEN3_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_kv_cache_baseline_table_20260429"))
    args = parser.parse_args()
    summaries = args.endpoint_summary or [pathlib.Path(path) for path in DEFAULT_ENDPOINT_SUMMARIES]
    summaries = [path if path.is_absolute() else ROOT / path for path in summaries]
    config = args.qwen3_config if args.qwen3_config.is_absolute() else ROOT / args.qwen3_config
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_kv_cache_baseline_table(endpoint_summaries=summaries, qwen3_config=config, output_dir=output)
    print(json.dumps({"output_dir": str(output), "rows": len(payload["rows"]), "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
