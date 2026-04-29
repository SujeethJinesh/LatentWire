from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _budget_summary(path: pathlib.Path, budget: int = 2) -> dict[str, Any]:
    sweep = _read_json(path)
    for row in sweep["budget_summaries"]:
        if row["budget_bytes"] == budget:
            return row
    raise ValueError(f"budget {budget} not found in {path}")


def _metric(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary["metrics"][name]


def deterministic_system_rows(surface: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    packet = _metric(summary, "matched_repair_packet")
    full_log = _metric(summary, "full_hidden_log")
    full_diag = _metric(summary, "full_diag_text")
    rows = [
        ("target-only", "no source", "target_only"),
        ("2-byte diagnostic packet", "method", "matched_repair_packet"),
        ("2-byte hidden-log truncation", "matched-byte text", "structured_text_matched"),
        ("2-byte JSON relay", "matched-byte text", "structured_json_matched"),
        ("2-byte free-text relay", "matched-byte text", "structured_free_text_matched"),
        ("full hidden-log relay", "oracle text relay", "full_hidden_log"),
        ("full diagnostic text", "oracle diagnostic text", "full_diag_text"),
    ]
    out: list[dict[str, Any]] = []
    for label, kind, name in rows:
        m = _metric(summary, name)
        bytes_mean = m["mean_payload_bytes"]
        out.append(
            {
                "surface": surface,
                "interface": label,
                "kind": kind,
                "accuracy": m["accuracy"],
                "mean_bytes": bytes_mean,
                "mean_tokens": m["mean_payload_tokens"],
                "p50_latency_ms": m["p50_latency_ms"],
                "bytes_vs_packet": (bytes_mean / packet["mean_payload_bytes"]) if packet["mean_payload_bytes"] else None,
                "compression_vs_full_log": (full_log["mean_payload_bytes"] / bytes_mean) if bytes_mean else None,
                "compression_vs_full_diag": (full_diag["mean_payload_bytes"] / bytes_mean) if bytes_mean else None,
            }
        )
    return out


def model_packet_rows(medium_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in medium_summary["rows"]:
        if row["prompt_mode"] != "trace_no_hint":
            continue
        p50 = row["p50_source_latency_ms"]
        mean_tokens = row["mean_packet_tokens"]
        rows.append(
            {
                "surface": "500-example model packet",
                "model": row["model"],
                "run_id": row["run_id"],
                "accuracy": row["matched_accuracy"],
                "target_accuracy": row["target_only_accuracy"],
                "best_control_accuracy": row["best_source_destroying_control_accuracy"],
                "valid_rate": row["packet_valid_rate"],
                "mean_bytes": row["mean_packet_bytes"],
                "mean_tokens": mean_tokens,
                "p50_latency_ms": p50,
                "p95_latency_ms": row["p95_source_latency_ms"],
                "packets_per_second_p50": 1000.0 / p50 if p50 else None,
                "generated_tokens_per_second_p50": mean_tokens / (p50 / 1000.0) if p50 else None,
            }
        )
    return rows


def target_decoder_rows(paths: list[tuple[str, pathlib.Path]]) -> list[dict[str, Any]]:
    rows = []
    for label, path in paths:
        if not path.exists():
            continue
        summary = _read_json(path)
        matched = summary["metrics"]["matched_packet"]
        rows.append(
            {
                "surface": label,
                "accuracy": summary["matched_accuracy"],
                "target_accuracy": summary["target_only_accuracy"],
                "best_control_accuracy": summary["best_control_accuracy"],
                "valid_rate": matched["valid_prediction_rate"],
                "mean_bytes": matched["mean_payload_bytes"],
                "mean_tokens": matched["mean_payload_tokens"],
                "p50_latency_ms": matched["p50_latency_ms"],
                "generated_tokens": matched["mean_generated_tokens"],
            }
        )
    return rows


def aggregate_systems_summary(
    *,
    deterministic: list[tuple[str, dict[str, Any]]],
    medium_summary: dict[str, Any],
    target_paths: list[tuple[str, pathlib.Path]],
) -> dict[str, Any]:
    deterministic_rows = [row for surface, summary in deterministic for row in deterministic_system_rows(surface, summary)]
    model_rows = model_packet_rows(medium_summary)
    decoder_rows = target_decoder_rows(target_paths)
    packet_rows = [row for row in deterministic_rows if row["interface"] == "2-byte diagnostic packet"]
    full_log_rows = [row for row in deterministic_rows if row["interface"] == "full hidden-log relay"]
    full_diag_rows = [row for row in deterministic_rows if row["interface"] == "full diagnostic text"]
    text_control_rows = [row for row in deterministic_rows if row["kind"] == "matched-byte text"]
    return {
        "gate": "source_private_systems_summary_20260428",
        "status": "systems evidence from existing artifacts",
        "claim": (
            "At the far-left rate point, 2-byte source-private packets recover hidden diagnostic evidence while "
            "matched-byte text relays stay at target floor, using about 183-187x fewer bytes than full hidden-log relay."
        ),
        "deterministic_rows": deterministic_rows,
        "model_packet_rows": model_rows,
        "target_decoder_rows": decoder_rows,
        "headline": {
            "packet_accuracy_min": min(row["accuracy"] for row in packet_rows),
            "matched_byte_text_accuracy_max": max(row["accuracy"] for row in text_control_rows),
            "full_log_accuracy_min": min(row["accuracy"] for row in full_log_rows),
            "packet_bytes": statistics.fmean(row["mean_bytes"] for row in packet_rows),
            "full_log_bytes_min": min(row["mean_bytes"] for row in full_log_rows),
            "full_log_bytes_max": max(row["mean_bytes"] for row in full_log_rows),
            "full_diag_bytes": statistics.fmean(row["mean_bytes"] for row in full_diag_rows),
            "compression_vs_full_log_min": min(row["compression_vs_full_log"] for row in packet_rows if row["compression_vs_full_log"]),
            "compression_vs_full_log_max": max(row["compression_vs_full_log"] for row in packet_rows if row["compression_vs_full_log"]),
            "compression_vs_full_diag": statistics.fmean(row["compression_vs_full_diag"] for row in packet_rows if row["compression_vs_full_diag"]),
        },
        "caveat": (
            "Local p50 latency is single-request wall-clock timing, not server throughput or TTFT. "
            "Future endpoint runs should add TTFT and streaming decode timing."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Systems Summary",
        "",
        f"- gate: `{payload['gate']}`",
        f"- status: {payload['status']}",
        f"- claim: {payload['claim']}",
        "",
        "## Headline",
        "",
        f"- packet accuracy minimum: `{h['packet_accuracy_min']:.3f}`",
        f"- matched-byte text accuracy maximum: `{h['matched_byte_text_accuracy_max']:.3f}`",
        f"- full hidden-log relay accuracy minimum: `{h['full_log_accuracy_min']:.3f}`",
        f"- packet bytes: `{h['packet_bytes']:.2f}`",
        f"- full hidden-log bytes: `{h['full_log_bytes_min']:.2f}-{h['full_log_bytes_max']:.2f}`",
        f"- compression vs full hidden-log relay: `{h['compression_vs_full_log_min']:.1f}x-{h['compression_vs_full_log_max']:.1f}x`",
        f"- compression vs full diagnostic text: `{h['compression_vs_full_diag']:.1f}x`",
        "",
        "## Deterministic Rate Rows",
        "",
        "| Surface | Interface | Kind | Accuracy | Bytes | Tokens | p50 latency ms | Compression vs full log |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["deterministic_rows"]:
        compression = row["compression_vs_full_log"]
        lines.append(
            f"| {row['surface']} | {row['interface']} | {row['kind']} | {row['accuracy']:.3f} | "
            f"{row['mean_bytes']:.2f} | {row['mean_tokens']:.2f} | {row['p50_latency_ms']:.3f} | "
            f"{compression:.1f}x |" if compression else
            f"| {row['surface']} | {row['interface']} | {row['kind']} | {row['accuracy']:.3f} | "
            f"{row['mean_bytes']:.2f} | {row['mean_tokens']:.2f} | {row['p50_latency_ms']:.3f} | - |"
        )
    lines.extend(
        [
            "",
            "## Model Packet Rows",
            "",
            "| Model | Run | Accuracy | Target | Best control | Valid | Bytes | Tokens | p50 ms | p95 ms | Packets/s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["model_packet_rows"]:
        lines.append(
            f"| {row['model']} | {row['run_id']} | {row['accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['valid_rate']:.3f} | {row['mean_bytes']:.2f} | "
            f"{row['mean_tokens']:.2f} | {row['p50_latency_ms']:.1f} | {row['p95_latency_ms']:.1f} | "
            f"{row['packets_per_second_p50']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Target Decoder Rows",
            "",
            "| Surface | Accuracy | Target | Best control | Valid | Packet bytes | p50 ms | Generated tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["target_decoder_rows"]:
        lines.append(
            f"| {row['surface']} | {row['accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['valid_rate']:.3f} | {row['mean_bytes']:.2f} | "
            f"{row['p50_latency_ms']:.1f} | {row['generated_tokens']:.2f} |"
        )
    lines.extend(["", "## Caveat", "", payload["caveat"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_systems_summary_20260428"))
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    deterministic_paths = [
        ("core seed29", ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/sweep_summary.json"),
        ("holdout seed30", ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/sweep_summary.json"),
    ]
    target_paths = [
        ("core seed29 qwen3 n16 mps", ROOT / "results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n16/summary.json"),
        ("holdout seed30 qwen3 n32 mps", ROOT / "results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n32/summary.json"),
        ("core seed29 qwen3 n64 cpu", ROOT / "results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n64_cpu/summary.json"),
        ("holdout seed30 qwen3 n64 cpu", ROOT / "results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n64_cpu/summary.json"),
    ]
    payload = aggregate_systems_summary(
        deterministic=[(surface, _budget_summary(path, budget=2)) for surface, path in deterministic_paths],
        medium_summary=_read_json(ROOT / "results/source_private_hidden_repair_packet_medium_llm_20260429/medium_summary.json"),
        target_paths=target_paths,
    )
    (out / "systems_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out / "systems_summary.md", payload)
    manifest = {
        "command": " ".join(["scripts/build_source_private_systems_summary.py", "--output-dir", str(args.output_dir)]),
        "artifacts": ["systems_summary.json", "systems_summary.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(out / name)
            for name in ["systems_summary.json", "systems_summary.md"]
        },
        "summary": payload["headline"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Systems Summary Manifest",
                "",
                f"- gate: `{payload['gate']}`",
                f"- packet bytes: `{payload['headline']['packet_bytes']:.2f}`",
                f"- full-log compression: `{payload['headline']['compression_vs_full_log_min']:.1f}x-{payload['headline']['compression_vs_full_log_max']:.1f}x`",
                f"- pass: `{payload['headline']['packet_accuracy_min'] > payload['headline']['matched_byte_text_accuracy_max']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
