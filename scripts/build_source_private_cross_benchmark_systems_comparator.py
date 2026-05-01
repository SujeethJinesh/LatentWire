from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_CONFIG = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775/config.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_cross_benchmark_systems_comparator_20260501")

DEFAULT_BENCHMARKS = (
    {
        "row_id": "arc_challenge_test_12b",
        "dataset": "ARC-Challenge",
        "split": "test",
        "paper_role": "headline_public_benchmark",
        "seed_artifact": pathlib.Path(
            "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/"
            "arc_challenge_seed_stability.json"
        ),
        "phase_artifact": pathlib.Path(
            "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/"
            "arc_challenge_fixed_packet_gate.json"
        ),
        "label_copy_artifact": None,
    },
    {
        "row_id": "openbookqa_test_3b",
        "dataset": "OpenBookQA",
        "split": "test",
        "paper_role": "headline_second_public_benchmark",
        "seed_artifact": pathlib.Path(
            "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/"
            "arc_challenge_seed_stability.json"
        ),
        "phase_artifact": None,
        "label_copy_artifact": None,
    },
    {
        "row_id": "hellaswag_validation1024_2b",
        "dataset": "HellaSwag",
        "split": "validation_first1024",
        "paper_role": "diagnostic_not_headline_label_copy_threat",
        "seed_artifact": pathlib.Path(
            "results/source_private_hellaswag_seed_stability_20260501_qwen05_hashed_validation1024_2b_5seed/"
            "arc_challenge_seed_stability.json"
        ),
        "phase_artifact": pathlib.Path(
            "results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/"
            "arc_challenge_fixed_packet_gate.json"
        ),
        "label_copy_artifact": pathlib.Path(
            "results/source_private_hellaswag_control_suite_20260501/hellaswag_control_suite.json"
        ),
    },
)

CSV_COLUMNS = (
    "row_id",
    "dataset",
    "split",
    "paper_role",
    "artifact_path",
    "artifact_sha256",
    "phase_artifact_path",
    "phase_artifact_sha256",
    "pass_gate",
    "headline_eligible",
    "label_copy_threat",
    "eval_rows",
    "seed_count",
    "pass_count",
    "payload_bytes",
    "framed_record_bytes",
    "batch64_cacheline_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "matched_accuracy_mean",
    "matched_accuracy_min",
    "target_accuracy",
    "same_byte_text_accuracy",
    "best_destructive_accuracy",
    "matched_minus_target_min",
    "matched_minus_same_byte_text_min",
    "matched_minus_best_destructive_min",
    "paired_ci95_low_vs_target_min",
    "source_label_copy_accuracy",
    "matched_minus_source_label_copy",
    "source_text_exposed",
    "source_kv_exposed",
    "source_config_model_type",
    "kv_elements_per_source_token",
    "one_token_fp16_kv_bytes",
    "one_token_kvcomm30_fp16_bytes",
    "one_token_turboquant_3p5bit_bytes",
    "one_token_qjl_1bit_bytes",
    "one_token_qjl_30pct_bytes",
    "fp16_one_token_ratio_vs_framed",
    "kvcomm30_fp16_ratio_vs_framed",
    "turboquant_3p5bit_ratio_vs_framed",
    "qjl_1bit_ratio_vs_framed",
    "qjl_30pct_ratio_vs_framed",
    "phase_trace_available",
    "source_scoring_ms_per_question",
    "receiver_decode_p50_us",
    "receiver_decode_p95_us",
    "peak_rss_mib",
    "claim_allowed",
    "claim_forbidden",
    "next_native_gate",
)

EXTERNAL_BASELINES = (
    {
        "method": "C2C cache-to-cache communication",
        "source": "https://arxiv.org/abs/2510.03215",
        "communicated_object": "projected/fused source KV cache",
        "fair_axis": "accuracy, latency, bytes, and source-KV exposure on the same task",
        "local_status": "not_run_native",
        "claim_boundary": "closest semantic-cache baseline; byte floors are not a native C2C result",
    },
    {
        "method": "KVComm selective KV sharing",
        "source": "https://arxiv.org/abs/2510.03346",
        "communicated_object": "selected source KV layers/pairs",
        "fair_axis": "fraction of KV layers, accuracy, latency, and source-KV exposure",
        "local_status": "not_run_native",
        "claim_boundary": "30% layer fraction is used only as an assumption row unless rerun locally",
    },
    {
        "method": "KVCOMM cross-context KV-cache communication",
        "source": "https://arxiv.org/abs/2510.12872",
        "communicated_object": "aligned/reused KV caches for multi-agent prefill",
        "fair_axis": "prefill reuse, anchor alignment, and source-KV exposure",
        "local_status": "not_run_native",
        "claim_boundary": "systems neighbor, not same threat model as source-private public packets",
    },
    {
        "method": "QJL 1-bit KV sketch",
        "source": "https://arxiv.org/abs/2406.03482",
        "communicated_object": "1-bit JL/sign sketch of KV/cache state",
        "fair_axis": "state-sketch bytes and accuracy under source-state exposure",
        "local_status": "byte_floor_only",
        "claim_boundary": "mathematical byte-floor comparator, not a defeated native baseline",
    },
    {
        "method": "TurboQuant online vector quantization",
        "source": "https://arxiv.org/abs/2504.19874",
        "communicated_object": "low-bit vector/KV state",
        "fair_axis": "bits per vector/cache element, latency, and quality",
        "local_status": "byte_floor_only",
        "claim_boundary": "quantization inspiration and byte-floor proxy only",
    },
    {
        "method": "vLLM/PagedAttention",
        "source": "https://arxiv.org/abs/2309.06180",
        "communicated_object": "paged KV-cache serving substrate",
        "fair_axis": "TTFT, TPOT, goodput, peak GPU memory, and HBM traffic",
        "local_status": "pending_nvidia",
        "claim_boundary": "native serving target; Mac rows cannot close this gate",
    },
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _ceil_quantum(value: float, quantum: int) -> float:
    n = int(value)
    return float(max(quantum, ((n + quantum - 1) // quantum) * quantum))


def _kv_elements_per_token(config: dict[str, Any]) -> float:
    kv_heads = config.get("num_key_value_heads") or config["num_attention_heads"]
    head_dim = config.get("head_dim") or config["hidden_size"] / config["num_attention_heads"]
    return float(config["num_hidden_layers"]) * 2.0 * float(kv_heads) * float(head_dim)


def _state_bytes(elements: float, bits_per_element: float, layer_fraction: float = 1.0) -> float:
    return elements * bits_per_element * layer_fraction / 8.0


def _phase_metrics(phase_artifact: pathlib.Path | None) -> dict[str, Any]:
    if phase_artifact is None:
        return {
            "phase_trace_available": False,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "batch64_cacheline_bytes_per_request": None,
            "batch64_dma_bytes_per_request": None,
            "source_scoring_ms_per_question": None,
            "receiver_decode_p50_us": None,
            "receiver_decode_p95_us": None,
            "peak_rss_mib": None,
        }
    artifact = _read_json(phase_artifact)
    trace = artifact["systems_trace"]
    eval_rows = max(1, int(artifact["eval_rows"]))
    matched = artifact["condition_metrics"]["matched_source_private_packet"]
    return {
        "phase_trace_available": True,
        "source_text_exposed": bool(trace["source_text_exposed"]),
        "source_kv_exposed": bool(trace["source_kv_exposed"]),
        "batch64_cacheline_bytes_per_request": trace["batch64_cacheline_bytes_per_request"],
        "batch64_dma_bytes_per_request": trace["batch64_dma_bytes_per_request"],
        "source_scoring_ms_per_question": trace["phase_timings_s"]["source_scoring"] * 1000.0 / eval_rows,
        "receiver_decode_p50_us": matched["p50_latency_ms"] * 1000.0,
        "receiver_decode_p95_us": matched["p95_latency_ms"] * 1000.0,
        "peak_rss_mib": trace["peak_rss_mib"],
    }


def _benchmark_row(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seed_artifact = pathlib.Path(spec["seed_artifact"])
    phase_artifact = spec.get("phase_artifact")
    if phase_artifact is not None:
        phase_artifact = pathlib.Path(phase_artifact)
    label_copy_artifact = spec.get("label_copy_artifact")
    if label_copy_artifact is not None:
        label_copy_artifact = pathlib.Path(label_copy_artifact)

    seed = _read_json(seed_artifact)
    aggregate = seed["aggregate"]
    payload_bytes = float(seed["budget_bytes"])
    framed_record_bytes = payload_bytes + 3.0
    label_copy_threat = False
    source_label_copy_accuracy = None
    matched_minus_source_label_copy = None
    if label_copy_artifact is not None:
        label_copy = _read_json(label_copy_artifact)
        label_headline = label_copy["headline"]
        label_copy_threat = bool(label_headline["label_copy_threat_present"])
        source_label_copy_accuracy = label_headline["source_label_text_copy_accuracy"]
        matched_minus_source_label_copy = label_headline["matched_minus_source_label_text_copy"]

    elements = _kv_elements_per_token(config)
    fp16_bytes = _state_bytes(elements, 16.0)
    kvcomm30_fp16_bytes = _state_bytes(elements, 16.0, layer_fraction=0.30)
    turboquant_bytes = _state_bytes(elements, 3.5)
    qjl_bytes = _state_bytes(elements, 1.0)
    qjl_30pct_bytes = _state_bytes(elements, 1.0, layer_fraction=0.30)
    phase = _phase_metrics(phase_artifact)

    headline_eligible = (
        bool(seed["pass_gate"])
        and aggregate["all_seeds_pass"] is True
        and aggregate["paired_ci95_low_vs_target_min"] > 0.0
        and aggregate["matched_minus_same_byte_text_min"] > 0.0
        and not label_copy_threat
    )
    source_text_exposed = bool(phase["source_text_exposed"])
    source_kv_exposed = bool(phase["source_kv_exposed"])

    return {
        "row_id": spec["row_id"],
        "dataset": spec["dataset"],
        "split": spec["split"],
        "paper_role": spec["paper_role"],
        "artifact_path": _rel(seed_artifact),
        "artifact_sha256": _sha256_file(seed_artifact),
        "phase_artifact_path": _rel(phase_artifact),
        "phase_artifact_sha256": _sha256_file(phase_artifact),
        "pass_gate": bool(seed["pass_gate"]),
        "headline_eligible": headline_eligible,
        "label_copy_threat": label_copy_threat,
        "eval_rows": int(seed["eval_rows"]),
        "seed_count": int(aggregate["seed_count"]),
        "pass_count": int(aggregate["pass_count"]),
        "payload_bytes": payload_bytes,
        "framed_record_bytes": framed_record_bytes,
        "batch64_cacheline_bytes_per_request": phase["batch64_cacheline_bytes_per_request"],
        "batch64_dma_bytes_per_request": phase["batch64_dma_bytes_per_request"],
        "matched_accuracy_mean": aggregate["matched_accuracy_mean"],
        "matched_accuracy_min": aggregate["matched_accuracy_min"],
        "target_accuracy": aggregate["target_accuracy"],
        "same_byte_text_accuracy": aggregate["same_byte_structured_text_accuracy"],
        "best_destructive_accuracy": aggregate["candidate_derangement_accuracy_max"],
        "matched_minus_target_min": aggregate["matched_minus_target_min"],
        "matched_minus_same_byte_text_min": aggregate["matched_minus_same_byte_text_min"],
        "matched_minus_best_destructive_min": aggregate["matched_minus_best_destructive_min"],
        "paired_ci95_low_vs_target_min": aggregate["paired_ci95_low_vs_target_min"],
        "source_label_copy_accuracy": source_label_copy_accuracy,
        "matched_minus_source_label_copy": matched_minus_source_label_copy,
        "source_text_exposed": source_text_exposed,
        "source_kv_exposed": source_kv_exposed,
        "source_config_model_type": config.get("model_type"),
        "kv_elements_per_source_token": elements,
        "one_token_fp16_kv_bytes": fp16_bytes,
        "one_token_kvcomm30_fp16_bytes": kvcomm30_fp16_bytes,
        "one_token_turboquant_3p5bit_bytes": turboquant_bytes,
        "one_token_qjl_1bit_bytes": qjl_bytes,
        "one_token_qjl_30pct_bytes": qjl_30pct_bytes,
        "fp16_one_token_ratio_vs_framed": fp16_bytes / framed_record_bytes,
        "kvcomm30_fp16_ratio_vs_framed": kvcomm30_fp16_bytes / framed_record_bytes,
        "turboquant_3p5bit_ratio_vs_framed": turboquant_bytes / framed_record_bytes,
        "qjl_1bit_ratio_vs_framed": qjl_bytes / framed_record_bytes,
        "qjl_30pct_ratio_vs_framed": qjl_30pct_bytes / framed_record_bytes,
        "phase_trace_available": phase["phase_trace_available"],
        "source_scoring_ms_per_question": phase["source_scoring_ms_per_question"],
        "receiver_decode_p50_us": phase["receiver_decode_p50_us"],
        "receiver_decode_p95_us": phase["receiver_decode_p95_us"],
        "peak_rss_mib": phase["peak_rss_mib"],
        "claim_allowed": (
            "Source-private packet byte/accounting comparison with explicit public side information and "
            "local benchmark accuracy."
        ),
        "claim_forbidden": (
            "No native C2C/KVComm/TurboQuant/QJL throughput or quality win is claimed; this row is a "
            "state-exposure byte-floor comparator until those baselines are run natively."
        ),
        "next_native_gate": (
            "Run the same public benchmark rows with vLLM/SGLang TTFT, TPOT, goodput, GPU memory, HBM, "
            "and native C2C/KVComm/QJL/TurboQuant baselines."
        ),
    }


def _checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headline_rows = [row for row in rows if row["headline_eligible"]]
    hellaswag_rows = [row for row in rows if row["dataset"] == "HellaSwag"]
    return [
        {
            "check": "two_public_headline_benchmarks_eligible",
            "pass": len(headline_rows) >= 2,
            "value": len(headline_rows),
        },
        {
            "check": "all_packet_rows_source_private_boundary",
            "pass": all(not row["source_text_exposed"] and not row["source_kv_exposed"] for row in rows),
            "value": "no source text or source KV exposed",
        },
        {
            "check": "one_token_qjl_floor_at_least_50x_framed_packet",
            "pass": min(row["qjl_1bit_ratio_vs_framed"] for row in rows) >= 50.0,
            "value": min(row["qjl_1bit_ratio_vs_framed"] for row in rows),
        },
        {
            "check": "kvcomm30_qjl_floor_at_least_15x_framed_packet",
            "pass": min(row["qjl_30pct_ratio_vs_framed"] for row in rows) >= 15.0,
            "value": min(row["qjl_30pct_ratio_vs_framed"] for row in rows),
        },
        {
            "check": "hellaswag_label_copy_threat_marked_not_headline",
            "pass": bool(hellaswag_rows)
            and all(row["label_copy_threat"] and not row["headline_eligible"] for row in hellaswag_rows),
            "value": [(row["row_id"], row["label_copy_threat"], row["headline_eligible"]) for row in hellaswag_rows],
        },
        {
            "check": "native_baseline_non_claims_explicit",
            "pass": all("No native" in row["claim_forbidden"] for row in rows),
            "value": "native C2C/KVComm/QJL/TurboQuant wins forbidden",
        },
    ]


def build_comparator(
    *,
    output_dir: pathlib.Path,
    source_config: pathlib.Path,
    benchmarks: tuple[dict[str, Any], ...] = DEFAULT_BENCHMARKS,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(source_config)
    rows = [_benchmark_row(spec, config) for spec in benchmarks]
    checks = _checks(rows)
    headline = {
        "pass_gate": all(check["pass"] for check in checks),
        "headline_eligible_benchmarks": sum(1 for row in rows if row["headline_eligible"]),
        "diagnostic_benchmarks": sum(1 for row in rows if not row["headline_eligible"]),
        "min_qjl_1bit_ratio_vs_framed": min(row["qjl_1bit_ratio_vs_framed"] for row in rows),
        "min_qjl_30pct_ratio_vs_framed": min(row["qjl_30pct_ratio_vs_framed"] for row in rows),
        "min_kvcomm30_fp16_ratio_vs_framed": min(row["kvcomm30_fp16_ratio_vs_framed"] for row in rows),
        "min_turboquant_3p5bit_ratio_vs_framed": min(row["turboquant_3p5bit_ratio_vs_framed"] for row in rows),
        "min_framed_packet_bytes": min(row["framed_record_bytes"] for row in rows),
        "max_framed_packet_bytes": max(row["framed_record_bytes"] for row in rows),
        "native_systems_complete": False,
        "claim_scope": (
            "Cross-benchmark byte/state-exposure comparator for source-private packets. KV/cache rows are "
            "one-source-token byte floors from the local Qwen2.5-0.5B config, not native C2C/KVComm/QJL/"
            "TurboQuant quality or throughput measurements."
        ),
    }
    payload = {
        "gate": "source_private_cross_benchmark_systems_comparator",
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "checks": checks,
        "source_model_config": {
            "path": _rel(source_config),
            "sha256": _sha256_file(source_config),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "hidden_size": config["hidden_size"],
            "kv_elements_per_source_token": _kv_elements_per_token(config),
        },
        "rows": rows,
        "external_baselines": EXTERNAL_BASELINES,
        "assumption_notes": [
            "KV byte floors are intentionally conservative: they count one source token of K+V state.",
            "KVComm30 rows apply a 30% layer fraction to that one-token state floor.",
            "QJL and TurboQuant rows are bits-per-element proxies, not local kernel implementations.",
            "HellaSwag remains diagnostic because source-label-copy beats the packet.",
        ],
    }
    json_path = output_dir / "cross_benchmark_systems_comparator.json"
    csv_path = output_dir / "cross_benchmark_systems_comparator.csv"
    md_path = output_dir / "cross_benchmark_systems_comparator.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": headline,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Cross-Benchmark Systems Comparator",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- headline-eligible benchmarks: `{h['headline_eligible_benchmarks']}`",
        f"- diagnostic benchmarks: `{h['diagnostic_benchmarks']}`",
        f"- framed packet bytes: `{h['min_framed_packet_bytes']:.0f}-{h['max_framed_packet_bytes']:.0f}B`",
        f"- min QJL 1-bit one-token KV floor vs framed packet: `{h['min_qjl_1bit_ratio_vs_framed']:.1f}x`",
        f"- min QJL 30%-layer one-token KV floor vs framed packet: `{h['min_qjl_30pct_ratio_vs_framed']:.1f}x`",
        f"- min KVComm30 fp16 one-token KV floor vs framed packet: `{h['min_kvcomm30_fp16_ratio_vs_framed']:.1f}x`",
        f"- min TurboQuant 3.5-bit one-token KV floor vs framed packet: "
        f"`{h['min_turboquant_3p5bit_ratio_vs_framed']:.1f}x`",
        f"- native systems complete: `{h['native_systems_complete']}`",
        "",
        "## Benchmark Rows",
        "",
        "| Dataset | Role | Seeds | Packet | Accuracy | Target | Text | QJL 1-bit floor | HellaSwag label-copy threat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['paper_role']} | "
            f"{row['pass_count']}/{row['seed_count']} | "
            f"{row['payload_bytes']:.0f}B raw / {row['framed_record_bytes']:.0f}B framed | "
            f"{row['matched_accuracy_mean']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['same_byte_text_accuracy']:.3f} | "
            f"{row['qjl_1bit_ratio_vs_framed']:.1f}x | "
            f"`{row['label_copy_threat']}` |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Pass | Value |",
            "|---|---:|---|",
        ]
    )
    for check in payload["checks"]:
        lines.append(f"| `{check['check']}` | `{check['pass']}` | `{check['value']}` |")
    lines.extend(
        [
            "",
            "## External Baselines",
            "",
            "| Method | Source | Communicated object | Local status | Claim boundary |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["external_baselines"]:
        lines.append(
            f"| {row['method']} | {row['source']} | {row['communicated_object']} | "
            f"{row['local_status']} | {row['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["headline"]["claim_scope"],
            "",
            "This artifact strengthens the systems story by making the source-state exposure cost explicit. "
            "It does not close the NVIDIA/vLLM native systems blocker.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-config", type=pathlib.Path, default=DEFAULT_SOURCE_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_comparator(output_dir=args.output_dir, source_config=args.source_config)


if __name__ == "__main__":
    main()
