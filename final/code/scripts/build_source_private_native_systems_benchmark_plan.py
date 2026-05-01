from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = pathlib.Path("results/source_private_native_systems_benchmark_plan_20260501")

REQUIRED_METRICS = (
    {
        "metric": "benchmark",
        "unit": "string",
        "required": True,
        "why": "native rows must map back to a frozen benchmark surface",
    },
    {
        "metric": "split",
        "unit": "string",
        "required": True,
        "why": "methods must run on the same split before paired claims are valid",
    },
    {
        "metric": "model_pair",
        "unit": "string",
        "required": True,
        "why": "source and target model identities define the communication setting",
    },
    {
        "metric": "method",
        "unit": "string",
        "required": True,
        "why": "each row must identify the protocol or baseline under test",
    },
    {
        "metric": "implementation",
        "unit": "string",
        "required": True,
        "why": "native, faithful proxy, and byte-floor rows must remain separable",
    },
    {
        "metric": "commit_hash",
        "unit": "string",
        "required": True,
        "why": "native baselines need exact reproducibility provenance",
    },
    {
        "metric": "gpu_name",
        "unit": "string",
        "required": True,
        "why": "GPU SKU materially changes serving and memory traffic results",
    },
    {
        "metric": "cuda_version",
        "unit": "string",
        "required": True,
        "why": "CUDA version affects kernels, profilers, and serving engines",
    },
    {
        "metric": "driver_version",
        "unit": "string",
        "required": True,
        "why": "driver version is part of the native systems configuration",
    },
    {
        "metric": "serving_engine",
        "unit": "string",
        "required": True,
        "why": "vLLM and SGLang rows must be distinguishable",
    },
    {
        "metric": "batch_size",
        "unit": "count",
        "required": True,
        "why": "batch shape changes throughput and cache pressure",
    },
    {
        "metric": "concurrency",
        "unit": "count",
        "required": True,
        "why": "online goodput depends on load concurrency",
    },
    {
        "metric": "context_len",
        "unit": "tokens",
        "required": True,
        "why": "KV/cache traffic scales with context length",
    },
    {
        "metric": "max_new_tokens",
        "unit": "tokens",
        "required": True,
        "why": "decode work must be fixed across native rows",
    },
    {
        "metric": "precision",
        "unit": "string",
        "required": True,
        "why": "precision affects memory, bandwidth, and accuracy",
    },
    {
        "metric": "num_examples",
        "unit": "count",
        "required": True,
        "why": "paired uncertainty requires the shared example count",
    },
    {
        "metric": "accuracy",
        "unit": "fraction",
        "required": True,
        "why": "quality must be matched before any systems win is meaningful",
    },
    {
        "metric": "paired_delta_vs_target",
        "unit": "fraction",
        "required": True,
        "why": "same-example accuracy deltas avoid independent-sample noise",
    },
    {
        "metric": "paired_ci95_low_vs_target",
        "unit": "fraction",
        "required": True,
        "why": "reviewers need uncertainty on quality deltas",
    },
    {
        "metric": "ttft_ms_p50",
        "unit": "ms",
        "required": True,
        "why": "time to first token captures prefill and packet/KV setup cost",
    },
    {
        "metric": "ttft_ms_p95",
        "unit": "ms",
        "required": True,
        "why": "tail latency is a serving-quality constraint",
    },
    {
        "metric": "tpot_ms_p50",
        "unit": "ms/token",
        "required": True,
        "why": "time per output token captures decode path efficiency",
    },
    {
        "metric": "tpot_ms_p95",
        "unit": "ms/token",
        "required": True,
        "why": "tail decode latency catches scheduler/cache path regressions",
    },
    {
        "metric": "itl_ms_p50",
        "unit": "ms/token",
        "required": True,
        "why": "inter-token latency is a standard online serving metric",
    },
    {
        "metric": "goodput_requests_per_s",
        "unit": "requests/s",
        "required": True,
        "why": "systems claims need throughput at an accepted latency SLO",
    },
    {
        "metric": "generated_tokens_per_s",
        "unit": "tokens/s",
        "required": True,
        "why": "token throughput makes vLLM/SGLang comparisons interpretable",
    },
    {
        "metric": "prefill_ms_p50",
        "unit": "ms",
        "required": True,
        "why": "packet and KV baselines differ most in prefill/cache setup",
    },
    {
        "metric": "decode_ms_p50",
        "unit": "ms",
        "required": True,
        "why": "decode latency separates one-time packet setup from token generation",
    },
    {
        "metric": "peak_gpu_memory_gb",
        "unit": "GB",
        "required": True,
        "why": "KV/cache baselines may trade bytes for GPU memory",
    },
    {
        "metric": "hbm_read_bytes_per_request",
        "unit": "bytes/request",
        "required": True,
        "why": "hardware-facing systems wins should show memory-traffic movement",
    },
    {
        "metric": "hbm_write_bytes_per_request",
        "unit": "bytes/request",
        "required": True,
        "why": "cache-sharing and quantized-KV methods write different state volumes",
    },
    {
        "metric": "pcie_or_nvlink_rx_bytes_per_request",
        "unit": "bytes/request",
        "required": True,
        "why": "multi-device or host-device communication must be accounted",
    },
    {
        "metric": "pcie_or_nvlink_tx_bytes_per_request",
        "unit": "bytes/request",
        "required": True,
        "why": "transferred source state is central to the claim boundary",
    },
    {
        "metric": "payload_bytes_per_request",
        "unit": "bytes/request",
        "required": True,
        "why": "packet/KV/text baselines must report actual communicated bytes",
    },
    {
        "metric": "framed_bytes_per_request",
        "unit": "bytes/request",
        "required": True,
        "why": "wire-format overhead must be separated from raw payload bytes",
    },
    {
        "metric": "transferred_source_state_bytes",
        "unit": "bytes/request",
        "required": True,
        "why": "cache/KV/vector baselines move source state that packets avoid exposing",
    },
    {
        "metric": "source_text_exposed",
        "unit": "bool",
        "required": True,
        "why": "source-private claims fail if source text is exposed",
    },
    {
        "metric": "source_kv_exposed",
        "unit": "bool",
        "required": True,
        "why": "C2C/KVComm/QJL/TurboQuant expose different source-state objects",
    },
    {
        "metric": "source_hidden_or_score_vector_exposed",
        "unit": "bool",
        "required": True,
        "why": "raw hidden/score vectors are not equivalent to fixed-byte packets",
    },
    {
        "metric": "hardware",
        "unit": "string",
        "required": True,
        "why": "GPU SKU and interconnect define systems comparability",
    },
    {
        "metric": "software_commit",
        "unit": "string",
        "required": True,
        "why": "serving/runtime versions must be reproducible",
    },
    {
        "metric": "batch_size_or_concurrency",
        "unit": "count",
        "required": True,
        "why": "serving throughput depends on load shape",
    },
    {
        "metric": "input_output_token_counts",
        "unit": "tokens",
        "required": True,
        "why": "TTFT/TPOT/goodput are meaningless without token lengths",
    },
    {
        "metric": "wall_time_s",
        "unit": "seconds",
        "required": True,
        "why": "total runtime catches benchmark harness and setup regressions",
    },
)

BENCHMARK_ROWS = (
    {
        "benchmark": "ARC-Challenge",
        "split": "test",
        "paper_role": "headline",
        "local_artifact": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/arc_challenge_seed_stability.json",
        "native_priority": 1,
        "promotion_condition": "preserve positive packet delta over target, same-byte text, and destructive controls",
    },
    {
        "benchmark": "OpenBookQA",
        "split": "test",
        "paper_role": "headline_second_benchmark",
        "local_artifact": "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/arc_challenge_seed_stability.json",
        "native_priority": 2,
        "promotion_condition": "preserve seed-stable 3B packet lift over target and same-byte text",
    },
    {
        "benchmark": "HellaSwag",
        "split": "validation_first1024_diagnostic",
        "paper_role": "diagnostic_until_label_copy_gate_passes",
        "local_artifact": "results/source_private_hellaswag_repair_systems_acceptance_card_20260501/hellaswag_repair_systems_acceptance_card.json",
        "native_priority": 99,
        "promotion_condition": "run native systems only after a repair beats source-label and trained-label copy by 0.02",
    },
)

BASELINE_ROWS = (
    {
        "row_id": "latentwire_packet_cached_source",
        "family": "this_work",
        "method": "LatentWire packet receiver, cached source packet",
        "source_url": "local",
        "serving_substrate": "vLLM and SGLang",
        "communicated_object": "fixed-byte source-private packet",
        "source_private": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "required_for_native_gate": True,
        "claim_role": "headline systems row if quality is preserved",
    },
    {
        "row_id": "latentwire_packet_end_to_end_source_scoring",
        "family": "this_work",
        "method": "LatentWire end-to-end source scoring plus packet receiver",
        "source_url": "local",
        "serving_substrate": "vLLM and SGLang",
        "communicated_object": "source scoring plus fixed-byte packet",
        "source_private": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "required_for_native_gate": True,
        "claim_role": "end-to-end cost row; may weaken speed claims but preserves privacy/bandwidth claims",
    },
    {
        "row_id": "target_only_vllm",
        "family": "serving_baseline",
        "method": "Target-only vLLM/PagedAttention",
        "source_url": "https://arxiv.org/abs/2309.06180",
        "serving_substrate": "vLLM",
        "communicated_object": "none",
        "source_private": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "required_for_native_gate": True,
        "claim_role": "serving baseline for accuracy and latency",
    },
    {
        "row_id": "target_only_sglang",
        "family": "serving_baseline",
        "method": "Target-only SGLang/RadixAttention",
        "source_url": "https://arxiv.org/abs/2312.07104",
        "serving_substrate": "SGLang",
        "communicated_object": "none",
        "source_private": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "required_for_native_gate": True,
        "claim_role": "second serving substrate / scheduler sensitivity check",
    },
    {
        "row_id": "same_byte_visible_text",
        "family": "text_control",
        "method": "Same-byte visible text packet",
        "source_url": "local",
        "serving_substrate": "vLLM and SGLang",
        "communicated_object": "visible text control with matched byte budget",
        "source_private": False,
        "source_text_exposed": True,
        "source_kv_exposed": False,
        "required_for_native_gate": True,
        "claim_role": "privacy and text-relay control",
    },
    {
        "row_id": "source_label_copy_control",
        "family": "label_copy_control",
        "method": "Source-label-copy / trained-label-copy control",
        "source_url": "https://arxiv.org/abs/2309.03882",
        "serving_substrate": "offline control plus serving row when applicable",
        "communicated_object": "source selected option id",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "required_for_native_gate": True,
        "claim_role": "fatal shortcut control for HellaSwag and MCQ surfaces",
    },
    {
        "row_id": "c2c_cache_to_cache",
        "family": "cache_communication",
        "method": "C2C cache-to-cache communication",
        "source_url": "https://arxiv.org/abs/2510.03215",
        "serving_substrate": "native cache-fusion implementation",
        "communicated_object": "projected/fused source KV cache",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "required_for_native_gate": True,
        "claim_role": "closest high-rate internal-state baseline",
    },
    {
        "row_id": "kvcomm_selective_kv",
        "family": "kv_communication",
        "method": "KVComm selective KV sharing",
        "source_url": "https://arxiv.org/abs/2510.03346",
        "serving_substrate": "native KV-sharing implementation or faithful proxy",
        "communicated_object": "selected source KV layers/pairs",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "required_for_native_gate": True,
        "claim_role": "selective KV communication baseline",
    },
    {
        "row_id": "kvcomm_online_cross_context",
        "family": "kv_communication",
        "method": "KVCOMM online cross-context KV-cache communication",
        "source_url": "https://arxiv.org/abs/2510.12872",
        "serving_substrate": "native or faithful online KV-cache communication proxy",
        "communicated_object": "aligned/reused source KV caches",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "required_for_native_gate": True,
        "claim_role": "multi-agent KV reuse systems neighbor",
    },
    {
        "row_id": "qjl_1bit_source_state",
        "family": "quantized_projection",
        "method": "QJL 1-bit source-state sketch",
        "source_url": "https://arxiv.org/abs/2406.03482",
        "serving_substrate": "native or faithful sign-sketch proxy",
        "communicated_object": "1-bit JL/sign sketch of source KV/vector state",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "required_for_native_gate": True,
        "claim_role": "low-bit vector-state sketch baseline",
    },
    {
        "row_id": "turboquant_lowbit_source_state",
        "family": "quantized_kv",
        "method": "TurboQuant-style low-bit source-state quantization",
        "source_url": "https://arxiv.org/abs/2504.19874",
        "serving_substrate": "native low-bit vector/KV kernel or faithful proxy",
        "communicated_object": "low-bit quantized source KV/vector state",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "required_for_native_gate": True,
        "claim_role": "quantized vector-state baseline",
    },
)

PROFILER_REQUIREMENTS = (
    "vLLM or SGLang JSON serving benchmark output with TTFT/TPOT/ITL/goodput",
    "Nsight Systems trace for request timeline, CPU/GPU overlap, and PCIe/NVLink transfer accounting",
    "Nsight Compute or CUPTI counters for HBM read/write bytes on representative kernels",
    "nvidia-smi or NVML peak memory and power log sampled during each run",
    "exact hardware SKU, driver, CUDA, runtime commit, model revision, precision, batch/concurrency, and token lengths",
)

NON_CLAIMS = (
    "Do not claim native throughput, TTFT, TPOT, HBM, or peak-memory wins until all required native rows are measured.",
    "Do not claim LatentWire beats C2C, KVComm, KVCOMM, QJL, TurboQuant, vLLM, or SGLang from byte-floor accounting alone.",
    "Do not run or require SSH in the artifact; remote execution must be done manually by the user from the runbook.",
    "Do not promote HellaSwag native rows until the HellaSwag method gate beats source-label and trained-label copy controls.",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path) -> str:
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


def _checks() -> list[dict[str, Any]]:
    metric_names = {row["metric"] for row in REQUIRED_METRICS}
    baseline_ids = {row["row_id"] for row in BASELINE_ROWS}
    benchmark_roles = {row["paper_role"] for row in BENCHMARK_ROWS}
    source_urls = [row["source_url"] for row in BASELINE_ROWS if row["source_url"] != "local"]
    checks = [
        (
            "all_required_quality_latency_memory_traffic_metrics_listed",
            {
                "accuracy",
                "benchmark",
                "commit_hash",
                "cuda_version",
                "driver_version",
                "framed_bytes_per_request",
                "gpu_name",
                "paired_ci95_low_vs_target",
                "split",
                "transferred_source_state_bytes",
                "ttft_ms_p50",
                "tpot_ms_p50",
                "goodput_requests_per_s",
                "peak_gpu_memory_gb",
                "hbm_read_bytes_per_request",
                "hbm_write_bytes_per_request",
                "pcie_or_nvlink_rx_bytes_per_request",
                "payload_bytes_per_request",
                "wall_time_s",
            }
            <= metric_names,
        ),
        (
            "headline_benchmarks_are_arc_and_openbookqa",
            {"headline", "headline_second_benchmark"} <= benchmark_roles,
        ),
        (
            "hellaswag_marked_diagnostic_until_label_copy_gate",
            any(row["benchmark"] == "HellaSwag" and "diagnostic" in row["paper_role"] for row in BENCHMARK_ROWS),
        ),
        (
            "packet_rows_include_cached_and_end_to_end_modes",
            {"latentwire_packet_cached_source", "latentwire_packet_end_to_end_source_scoring"} <= baseline_ids,
        ),
        ("serving_substrates_include_vllm_and_sglang", {"target_only_vllm", "target_only_sglang"} <= baseline_ids),
        (
            "competitor_rows_include_cache_kv_quantized_baselines",
            {"c2c_cache_to_cache", "kvcomm_selective_kv", "kvcomm_online_cross_context", "qjl_1bit_source_state", "turboquant_lowbit_source_state"}
            <= baseline_ids,
        ),
        (
            "source_exposure_flags_required_for_every_baseline",
            all(
                {"source_private", "source_text_exposed", "source_kv_exposed"} <= set(row)
                for row in BASELINE_ROWS
            ),
        ),
        (
            "external_baselines_have_primary_sources",
            all(url.startswith("https://") for url in source_urls),
        ),
        (
            "profiler_requirements_include_serving_json_nsight_and_nvml",
            any("JSON" in item for item in PROFILER_REQUIREMENTS)
            and any("Nsight Systems" in item for item in PROFILER_REQUIREMENTS)
            and any("Nsight Compute" in item or "CUPTI" in item for item in PROFILER_REQUIREMENTS)
            and any("NVML" in item for item in PROFILER_REQUIREMENTS),
        ),
        (
            "no_ssh_policy_recorded",
            any("SSH" in item for item in NON_CLAIMS),
        ),
        (
            "native_systems_complete_false_until_measurements_ingested",
            True,
        ),
        (
            "native_win_non_claims_recorded",
            any("Do not claim native throughput" in item for item in NON_CLAIMS)
            and any("byte-floor accounting" in item for item in NON_CLAIMS),
        ),
    ]
    return [{"check": name, "pass": bool(value)} for name, value in checks]


def _write_csv(path: pathlib.Path, rows: tuple[dict[str, Any], ...], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fieldnames})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Native Systems Benchmark Plan",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- native systems complete: `{h['native_systems_complete']}`",
        f"- required metrics: `{h['required_metric_count']}`",
        f"- required baselines: `{h['required_baseline_count']}`",
        f"- headline benchmarks: `{', '.join(h['headline_benchmarks'])}`",
        f"- diagnostic benchmarks: `{', '.join(h['diagnostic_benchmarks'])}`",
        "",
        "## Baseline Rows",
        "",
        "| Row | Method | Substrate | Source private | Source KV exposed | Role |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in payload["baseline_rows"]:
        lines.append(
            f"| `{row['row_id']}` | {row['method']} | {row['serving_substrate']} | "
            f"`{row['source_private']}` | `{row['source_kv_exposed']}` | {row['claim_role']} |"
        )
    lines.extend(
        [
            "",
            "## Required Metrics",
            "",
            "| Metric | Unit | Required | Why |",
            "|---|---|---:|---|",
        ]
    )
    for row in payload["required_metrics"]:
        lines.append(f"| `{row['metric']}` | {row['unit']} | `{row['required']}` | {row['why']} |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for check in payload["checks"]:
        lines.append(f"| `{check['check']}` | `{check['pass']}` |")
    lines.extend(
        [
            "",
            "## Runbook",
            "",
            "1. Freeze ARC-Challenge test and OpenBookQA test as headline native rows; keep HellaSwag diagnostic until the label-copy gate passes.",
            "2. Run target-only vLLM and target-only SGLang first to establish serving baselines.",
            "3. Run LatentWire in cached-source-packet mode and end-to-end source-scoring mode.",
            "4. Run same-byte visible text and source-label/trained-label controls on the same example IDs.",
            "5. Run C2C, KVComm/KVCOMM, QJL, and TurboQuant rows only as native or faithful source-state baselines with exposure flags set.",
            "6. Collect serving JSON plus Nsight/NVML traces for every row; do not use SSH inside this artifact.",
            "7. Mark `native_systems_complete=true` only after every required baseline has accuracy, latency, memory, traffic, payload bytes, and exposure fields.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    for item in payload["non_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_native_systems_plan(*, output_dir: pathlib.Path = DEFAULT_OUTPUT, run_date: str = "2026-05-01") -> dict[str, Any]:
    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checks = _checks()
    headline_benchmarks = [row["benchmark"] for row in BENCHMARK_ROWS if row["paper_role"].startswith("headline")]
    diagnostic_benchmarks = [row["benchmark"] for row in BENCHMARK_ROWS if "diagnostic" in row["paper_role"]]
    required_baselines = [row for row in BASELINE_ROWS if row["required_for_native_gate"]]
    headline = {
        "pass_gate": all(check["pass"] for check in checks),
        "native_systems_complete": False,
        "required_metric_count": len([row for row in REQUIRED_METRICS if row["required"]]),
        "required_baseline_count": len(required_baselines),
        "headline_benchmarks": headline_benchmarks,
        "diagnostic_benchmarks": diagnostic_benchmarks,
        "serving_substrates": ["vLLM", "SGLang"],
        "profiler_stack": ["serving JSON", "Nsight Systems", "Nsight Compute or CUPTI", "NVML"],
        "acceptance_rule": (
            "Native systems complete only when all required rows report quality, TTFT, TPOT, ITL, goodput, "
            "tokens/s, peak GPU memory, HBM read/write, PCIe/NVLink transfer, payload bytes, and source-exposure flags."
        ),
    }
    payload = {
        "gate": "source_private_native_systems_benchmark_plan",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "required_metrics": list(REQUIRED_METRICS),
        "benchmark_rows": list(BENCHMARK_ROWS),
        "baseline_rows": list(BASELINE_ROWS),
        "profiler_requirements": list(PROFILER_REQUIREMENTS),
        "non_claims": list(NON_CLAIMS),
        "checks": checks,
        "interpretation": (
            "This is a systems acceptance plan, not native evidence. It converts the NVIDIA blocker into "
            "a concrete table schema and runbook so future vLLM/SGLang/C2C/KVComm/QJL/TurboQuant measurements "
            "can be added without changing the claim boundary."
        ),
    }
    json_path = output / "native_systems_benchmark_plan.json"
    md_path = output / "native_systems_benchmark_plan.md"
    baseline_csv = output / "native_systems_baseline_rows.csv"
    metric_csv = output / "native_systems_metric_schema.csv"
    manifest_path = output / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(
        baseline_csv,
        BASELINE_ROWS,
        (
            "row_id",
            "family",
            "method",
            "source_url",
            "serving_substrate",
            "communicated_object",
            "source_private",
            "source_text_exposed",
            "source_kv_exposed",
            "required_for_native_gate",
            "claim_role",
        ),
    )
    _write_csv(metric_csv, REQUIRED_METRICS, ("metric", "unit", "required", "why"))
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [json_path.name, md_path.name, baseline_csv.name, metric_csv.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            md_path.name: _sha256_file(md_path),
            baseline_csv.name: _sha256_file(baseline_csv),
            metric_csv.name: _sha256_file(metric_csv),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the native NVIDIA systems benchmark plan for LatentWire.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()
    payload = build_native_systems_plan(output_dir=args.output_dir, run_date=args.run_date)
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
