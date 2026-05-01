from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/"
    "arc_challenge_fixed_packet_gate.json"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/"
    "arc_challenge_fixed_packet_gate.json"
)
DEFAULT_SEED_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/"
    "arc_challenge_seed_stability.json"
)
DEFAULT_SEED_TEST = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/"
    "arc_challenge_seed_stability.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_systems_trace_20260501")

CSV_COLUMNS = (
    "row_id",
    "split",
    "method",
    "measurement_status",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "payload_bytes",
    "record_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "batch64_cacheline_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "eval_rows",
    "candidate_pairs",
    "candidate_feature_cache_mib_float64",
    "candidate_feature_cache_mib_float32_floor",
    "projection_matrix_kib_float64",
    "accuracy",
    "target_accuracy",
    "same_byte_text_accuracy",
    "best_control_accuracy",
    "ci95_low_vs_target",
    "source_scoring_total_s",
    "source_scoring_ms_per_question",
    "receiver_decode_p50_us",
    "receiver_decode_p95_us",
    "packet_encode_decode_all_conditions_s",
    "peak_rss_mib",
    "native_kernel_status",
    "claim_scope",
    "next_gate",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mib(byte_count: float) -> float:
    return float(byte_count) / (1024.0 * 1024.0)


def _kib(byte_count: float) -> float:
    return float(byte_count) / 1024.0


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _condition(payload: dict[str, Any], condition: str) -> dict[str, Any]:
    return payload["condition_metrics"][condition]


def _systems_row(split: str, artifact: dict[str, Any]) -> dict[str, Any]:
    systems = artifact["systems_trace"]
    phases = systems["phase_timings_s"]
    matched = _condition(artifact, "matched_source_private_packet")
    text = _condition(artifact, "same_byte_structured_text")
    source_total = float(phases["source_scoring"])
    eval_rows = int(artifact["eval_rows"])
    return {
        "row_id": f"arc_shared_basis_{split}",
        "split": split,
        "method": "ARC shared-basis fixed 12B source-private packet",
        "measurement_status": "mac_local_phase_trace",
        "source_private": bool(systems["source_private"]),
        "source_text_exposed": bool(systems["source_text_exposed"]),
        "source_kv_exposed": bool(systems["source_kv_exposed"]),
        "payload_bytes": float(systems["raw_payload_bytes_per_request"]),
        "record_bytes": float(systems["record_bytes_with_header_crc"]),
        "single_request_cacheline_bytes": float(systems["single_request_cacheline_bytes"]),
        "single_request_dma_bytes": float(systems["single_request_dma_bytes"]),
        "batch64_cacheline_bytes_per_request": float(systems["batch64_cacheline_bytes_per_request"]),
        "batch64_dma_bytes_per_request": float(systems["batch64_dma_bytes_per_request"]),
        "eval_rows": eval_rows,
        "candidate_pairs": int(systems["eval_candidate_pairs"]),
        "candidate_feature_cache_mib_float64": _mib(systems["feature_cache_bytes_eval_float64"]),
        "candidate_feature_cache_mib_float32_floor": _mib(systems["feature_cache_bytes_eval_float32_floor"]),
        "projection_matrix_kib_float64": _kib(systems["projection_matrix_bytes_float64"]),
        "accuracy": float(artifact["headline"]["matched_accuracy"]),
        "target_accuracy": float(artifact["headline"]["target_accuracy"]),
        "same_byte_text_accuracy": float(artifact["headline"]["same_byte_structured_text_accuracy"]),
        "best_control_accuracy": float(artifact["headline"]["best_destructive_control_accuracy"]),
        "ci95_low_vs_target": float(artifact["headline"]["paired_ci95_vs_target"]["ci95_low"]),
        "source_scoring_total_s": source_total,
        "source_scoring_ms_per_question": source_total * 1000.0 / max(1, eval_rows),
        "receiver_decode_p50_us": float(matched["p50_latency_ms"]) * 1000.0,
        "receiver_decode_p95_us": float(matched["p95_latency_ms"]) * 1000.0,
        "packet_encode_decode_all_conditions_s": float(phases["packet_encode_decode_all_conditions"]),
        "peak_rss_mib": float(systems["peak_rss_mib"]),
        "native_kernel_status": "mac_python_trace_only",
        "claim_scope": (
            "Mac-local accuracy, bytes, process RSS, and Python/NumPy/PyTorch phase timings. "
            "This does not measure native GPU serving throughput or HBM traffic."
        ),
        "next_gate": "Run native vLLM/NVIDIA TTFT, TPOT, goodput, GPU memory, HBM, C2C/KVComm/TurboQuant/QJL rows.",
    }


def _native_pending_rows() -> list[dict[str, Any]]:
    base = {column: None for column in CSV_COLUMNS}
    rows = []
    for row_id, method, source_url, source_kv in (
        (
            "pending_c2c_native",
            "C2C cache-to-cache native baseline",
            "https://openreview.net/forum?id=LeatkxrBCi",
            True,
        ),
        (
            "pending_kvcomm_native",
            "KVComm/KVCOMM selected-KV native baseline",
            "https://arxiv.org/abs/2510.03346",
            True,
        ),
        (
            "pending_turboquant_native",
            "TurboQuant low-bit KV native baseline",
            "https://arxiv.org/abs/2504.19874",
            True,
        ),
        (
            "pending_qjl_native",
            "QJL sign-sketch source-state baseline",
            "https://arxiv.org/abs/2406.03482",
            True,
        ),
        (
            "pending_vllm_serving",
            "vLLM/PagedAttention serving substrate",
            "https://docs.vllm.ai/en/stable/design/metrics/",
            False,
        ),
    ):
        next_gate = (
            f"Run {method} with TTFT/TPOT/goodput/KV-cache metrics and source exposure annotated. "
            f"Primary source: {source_url}"
        )
        rows.append(
            {
                **base,
                "row_id": row_id,
                "split": "pending_native",
                "method": method,
                "measurement_status": "pending_native_required",
                "source_private": False,
                "source_text_exposed": False,
                "source_kv_exposed": source_kv,
                "native_kernel_status": "pending_native_required",
                "claim_scope": "Related-work/native baseline to run; not a defeated comparator in this Mac artifact.",
                "next_gate": next_gate,
            }
        )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC-Challenge Shared-Basis Systems Trace",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- COLM systems ready: `{h['colm_systems_trace_ready']}`",
        f"- ICLR native systems complete: `{h['iclr_native_systems_complete']}`",
        f"- test matched/target/text: `{h['test_matched_accuracy']:.3f}` / "
        f"`{h['test_target_accuracy']:.3f}` / `{h['test_same_byte_text_accuracy']:.3f}`",
        f"- test CI95 lower bound vs target: `{h['test_ci95_low_vs_target']:.3f}`",
        f"- test source scoring: `{h['test_source_scoring_ms_per_question']:.1f}` ms/question",
        f"- test receiver sparse decode p50/p95: `{h['test_receiver_decode_p50_us']:.1f}` / "
        f"`{h['test_receiver_decode_p95_us']:.1f}` us",
        f"- record bytes/cacheline/DMA: `{h['record_bytes']}B` / "
        f"`{h['single_request_cacheline_bytes']}B` / `{h['single_request_dma_bytes']}B`",
        f"- batch-64 line/DMA bytes per request: `{h['batch64_cacheline_bytes_per_request']:.1f}` / "
        f"`{h['batch64_dma_bytes_per_request']:.1f}`",
        f"- test candidate feature cache: `{h['test_candidate_feature_cache_mib_float64']:.2f}` MiB float64 "
        f"(`{h['test_candidate_feature_cache_mib_float32_floor']:.2f}` MiB fp32 floor)",
        f"- test peak process RSS: `{h['test_peak_rss_mib']:.1f}` MiB",
        "",
        "## Claim Boundary",
        "",
        "This is a Mac-local systems trace for the ARC shared-basis packet endpoint. It supports a "
        "byte-boundary and phase-timing claim: the method sends a 12B source-private payload, framed "
        "as a 15B record, and decodes against receiver-local public candidate features in tens of "
        "microseconds per question in the Python path. It does not establish native GPU serving, HBM, "
        "or C2C/KVComm/TurboQuant wins.",
        "",
        "## Pending Native Rows",
        "",
        "- vLLM TTFT/TPOT/goodput/KV-cache metrics;",
        "- GPU peak memory and HBM read/write bytes;",
        "- C2C and KVComm/KVCOMM source-KV exposure baselines;",
        "- TurboQuant and QJL low-bit source-state baselines.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arc_challenge_systems_trace(
    *,
    validation_artifact: pathlib.Path = DEFAULT_VALIDATION,
    test_artifact: pathlib.Path = DEFAULT_TEST,
    seed_validation_artifact: pathlib.Path = DEFAULT_SEED_VALIDATION,
    seed_test_artifact: pathlib.Path = DEFAULT_SEED_TEST,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _read_json(validation_artifact)
    test = _read_json(test_artifact)
    seed_validation = _read_json(seed_validation_artifact)
    seed_test = _read_json(seed_test_artifact)
    rows = [_systems_row("validation", validation), _systems_row("test", test), *_native_pending_rows()]
    test_row = rows[1]
    headline = {
        "validation_pass_gate": bool(validation["pass_gate"]),
        "test_pass_gate": bool(test["pass_gate"]),
        "validation_seed_pass_count": int(seed_validation["aggregate"]["pass_count"]),
        "test_seed_pass_count": int(seed_test["aggregate"]["pass_count"]),
        "test_matched_accuracy": float(test["headline"]["matched_accuracy"]),
        "test_target_accuracy": float(test["headline"]["target_accuracy"]),
        "test_same_byte_text_accuracy": float(test["headline"]["same_byte_structured_text_accuracy"]),
        "test_ci95_low_vs_target": float(test["headline"]["paired_ci95_vs_target"]["ci95_low"]),
        "test_source_scoring_ms_per_question": float(test_row["source_scoring_ms_per_question"]),
        "test_receiver_decode_p50_us": float(test_row["receiver_decode_p50_us"]),
        "test_receiver_decode_p95_us": float(test_row["receiver_decode_p95_us"]),
        "payload_bytes": float(test_row["payload_bytes"]),
        "record_bytes": float(test_row["record_bytes"]),
        "single_request_cacheline_bytes": float(test_row["single_request_cacheline_bytes"]),
        "single_request_dma_bytes": float(test_row["single_request_dma_bytes"]),
        "batch64_cacheline_bytes_per_request": float(test_row["batch64_cacheline_bytes_per_request"]),
        "batch64_dma_bytes_per_request": float(test_row["batch64_dma_bytes_per_request"]),
        "test_candidate_pairs": int(test_row["candidate_pairs"]),
        "test_candidate_feature_cache_mib_float64": float(test_row["candidate_feature_cache_mib_float64"]),
        "test_candidate_feature_cache_mib_float32_floor": float(test_row["candidate_feature_cache_mib_float32_floor"]),
        "test_peak_rss_mib": float(test_row["peak_rss_mib"]),
        "mac_measured_rows": 2,
        "pending_native_rows": 5,
        "colm_systems_trace_ready": True,
        "iclr_native_systems_complete": False,
    }
    pass_gate = bool(
        headline["validation_pass_gate"]
        and headline["test_pass_gate"]
        and headline["validation_seed_pass_count"] >= 5
        and headline["test_seed_pass_count"] >= 5
    )
    payload = {
        "artifact": "source_private_arc_challenge_systems_trace",
        "validation_artifact": _rel(validation_artifact),
        "test_artifact": _rel(test_artifact),
        "seed_validation_artifact": _rel(seed_validation_artifact),
        "seed_test_artifact": _rel(seed_test_artifact),
        "headline": headline,
        "rows": rows,
        "pass_gate": pass_gate,
        "claim_boundary": (
            "Mac-local phase/RSS/byte trace for the source-computable ARC endpoint. "
            "Native NVIDIA/vLLM serving and source-KV baselines remain pending."
        ),
    }
    json_path = output_dir / "arc_challenge_systems_trace.json"
    csv_path = output_dir / "arc_challenge_systems_trace.csv"
    md_path = output_dir / "arc_challenge_systems_trace.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, "manifest.json", "manifest.md"],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "pass_gate": pass_gate,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# ARC-Challenge Shared-Basis Systems Trace Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- Mac measured rows: `{headline['mac_measured_rows']}`",
                f"- pending native rows: `{headline['pending_native_rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-artifact", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-artifact", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--seed-validation-artifact", type=pathlib.Path, default=DEFAULT_SEED_VALIDATION)
    parser.add_argument("--seed-test-artifact", type=pathlib.Path, default=DEFAULT_SEED_TEST)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_arc_challenge_systems_trace(
        validation_artifact=args.validation_artifact,
        test_artifact=args.test_artifact,
        seed_validation_artifact=args.seed_validation_artifact,
        seed_test_artifact=args.seed_test_artifact,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
