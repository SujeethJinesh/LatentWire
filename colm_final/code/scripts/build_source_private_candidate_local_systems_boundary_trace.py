from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_SYSTEMS_WATERFALL = pathlib.Path(
    "results/source_private_candidate_local_residual_systems_waterfall_20260430/"
    "candidate_local_residual_systems_waterfall.json"
)
DEFAULT_MEMORY_LEDGER = pathlib.Path("results/source_private_memory_traffic_ledger_20260430/memory_traffic_ledger.json")
DEFAULT_SYSTEMS_RATE_FRONTIER = pathlib.Path(
    "results/source_private_systems_rate_assumption_frontier_20260430/systems_rate_assumption_frontier.json"
)
DEFAULT_COMPETITOR_TABLE = pathlib.Path(
    "results/source_private_candidate_local_competitor_basis_table_20260430/"
    "candidate_local_competitor_basis_table.json"
)
DEFAULT_KV_CACHE_TABLE = pathlib.Path(
    "results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_systems_boundary_trace_20260430")

CSV_COLUMNS = (
    "row_class",
    "method",
    "measurement_status",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "payload_bytes",
    "record_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "single_request_page_bytes",
    "batch64_line_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "kv_layer_fraction",
    "kv_bits_k",
    "kv_bits_v",
    "accuracy_min",
    "accuracy_max",
    "target_accuracy",
    "best_control_accuracy_max",
    "pass_rows",
    "rows",
    "resident_sparse_decode_p50_us",
    "measured_on",
    "native_kernel_status",
    "nvidia_vllm_required",
    "claim_scope",
    "primary_source",
    "next_action",
    "overclaim_guard",
)

PRIMARY_URLS = {
    "C2C cache-to-cache communication": "https://arxiv.org/abs/2510.03215",
    "KVComm / KVCOMM selective KV communication": "https://arxiv.org/abs/2510.03346; https://arxiv.org/abs/2510.12872",
    "Q-KVComm adaptive compressed KV communication": "https://arxiv.org/abs/2512.17914",
    "TurboQuant": "https://arxiv.org/abs/2504.19874",
    "CacheGen": "https://arxiv.org/abs/2310.07240",
    "KIVI/KVQuant-style 2-bit source KV byte floor": "https://arxiv.org/abs/2402.02750; https://arxiv.org/abs/2401.18079",
    "QJL-style 1-bit source KV byte floor": "https://arxiv.org/abs/2406.03482",
    "vLLM / PagedAttention serving substrate": "https://arxiv.org/abs/2309.06180",
}


KV_PROXY_METHODS = (
    {
        "method": "C2C fp16 source-KV lower-bound proxy",
        "payload_key": "kv_payload_bytes_fp16_bf16",
        "kv_layer_fraction": 1.0,
        "kv_bits_k": 16.0,
        "kv_bits_v": 16.0,
        "primary_source": "https://arxiv.org/abs/2510.03215",
        "claim_scope": "optimistic C2C byte floor for sending the extra private-token source KV, not a cache-fuser implementation",
        "next_action": "Run native C2C cache-fuser accuracy/latency with explicit source-KV byte accounting on NVIDIA.",
    },
    {
        "method": "KVComm 30%-layer fp16 source-KV lower-bound proxy",
        "payload_key": "kv_payload_bytes_fp16_bf16",
        "scale": 0.30,
        "kv_layer_fraction": 0.30,
        "kv_bits_k": 16.0,
        "kv_bits_v": 16.0,
        "primary_source": "https://arxiv.org/abs/2510.03346; https://arxiv.org/abs/2510.12872",
        "claim_scope": "optimistic KVComm byte floor using the reported selective-layer regime, not a native KV-sharing run",
        "next_action": "Run native/proxy KVComm with layer-selection metadata and source-KV exposure flagged.",
    },
    {
        "method": "Q-KVComm 6x compressed source-KV lower-bound proxy",
        "payload_key": "kv_payload_bytes_fp16_bf16",
        "scale": 1.0 / 6.0,
        "kv_layer_fraction": None,
        "kv_bits_k": None,
        "kv_bits_v": None,
        "primary_source": "https://arxiv.org/abs/2512.17914",
        "claim_scope": "optimistic Q-KVComm byte floor using the best reported compression-ratio endpoint, not a native adaptive compressor",
        "next_action": "Run native/proxy Q-KVComm with adaptive bit allocation and source-KV exposure flagged.",
    },
    {
        "method": "TurboQuant 3.5-bit source-KV lower-bound proxy",
        "payload_key": "kv_payload_bytes_turboquant_3p5bit_proxy",
        "kv_layer_fraction": 1.0,
        "kv_bits_k": 3.5,
        "kv_bits_v": 3.5,
        "primary_source": "https://arxiv.org/abs/2504.19874",
        "claim_scope": "TurboQuant quality-neutral byte-floor proxy for source KV, not a native low-bit kernel run",
        "next_action": "Run native TurboQuant/KV-cache kernel on NVIDIA or supported serving stack.",
    },
    {
        "method": "TurboQuant 2.5-bit aggressive source-KV lower-bound proxy",
        "payload_key": "kv_payload_bytes_turboquant_2p5bit_proxy",
        "kv_layer_fraction": 1.0,
        "kv_bits_k": 2.5,
        "kv_bits_v": 2.5,
        "primary_source": "https://arxiv.org/abs/2504.19874",
        "claim_scope": "TurboQuant aggressive byte-floor proxy for source KV with possible quality degradation, not a native kernel run",
        "next_action": "Run native TurboQuant/KV-cache kernel and report quality loss before using this as a systems row.",
    },
    {
        "method": "CacheGen 4.3x compressed source-KV lower-bound proxy",
        "payload_key": "kv_payload_bytes_fp16_bf16",
        "scale": 1.0 / 4.3,
        "kv_layer_fraction": None,
        "kv_bits_k": None,
        "kv_bits_v": None,
        "primary_source": "https://arxiv.org/abs/2310.07240",
        "claim_scope": "optimistic CacheGen compression byte-floor proxy, not a KV streaming implementation",
        "next_action": "Keep as serving/KV-streaming byte-floor contrast unless implementing CacheGen.",
    },
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
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


def _row(
    *,
    row_class: str,
    method: str,
    measurement_status: str,
    source_private: bool,
    source_text_exposed: bool,
    source_kv_exposed: bool,
    payload_bytes: float | None,
    record_bytes: float | None,
    single_request_cacheline_bytes: float | None,
    single_request_dma_bytes: float | None,
    single_request_page_bytes: float | None,
    batch64_line_bytes_per_request: float | None,
    batch64_dma_bytes_per_request: float | None,
    kv_layer_fraction: float | None,
    kv_bits_k: float | None,
    kv_bits_v: float | None,
    accuracy_min: float | None,
    accuracy_max: float | None,
    target_accuracy: float | None,
    best_control_accuracy_max: float | None,
    pass_rows: int | None,
    rows: int | None,
    resident_sparse_decode_p50_us: float | None,
    measured_on: str,
    native_kernel_status: str,
    nvidia_vllm_required: bool,
    claim_scope: str,
    primary_source: str,
    next_action: str,
    overclaim_guard: str,
) -> dict[str, Any]:
    return {
        "row_class": row_class,
        "method": method,
        "measurement_status": measurement_status,
        "source_private": source_private,
        "source_text_exposed": source_text_exposed,
        "source_kv_exposed": source_kv_exposed,
        "payload_bytes": payload_bytes,
        "record_bytes": record_bytes,
        "single_request_cacheline_bytes": single_request_cacheline_bytes,
        "single_request_dma_bytes": single_request_dma_bytes,
        "single_request_page_bytes": single_request_page_bytes,
        "batch64_line_bytes_per_request": batch64_line_bytes_per_request,
        "batch64_dma_bytes_per_request": batch64_dma_bytes_per_request,
        "kv_layer_fraction": kv_layer_fraction,
        "kv_bits_k": kv_bits_k,
        "kv_bits_v": kv_bits_v,
        "accuracy_min": accuracy_min,
        "accuracy_max": accuracy_max,
        "target_accuracy": target_accuracy,
        "best_control_accuracy_max": best_control_accuracy_max,
        "pass_rows": pass_rows,
        "rows": rows,
        "resident_sparse_decode_p50_us": resident_sparse_decode_p50_us,
        "measured_on": measured_on,
        "native_kernel_status": native_kernel_status,
        "nvidia_vllm_required": nvidia_vllm_required,
        "claim_scope": claim_scope,
        "primary_source": primary_source,
        "next_action": next_action,
        "overclaim_guard": overclaim_guard,
    }


def _condition_rows(systems: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    return [row for row in systems["rows"] if row["condition"] == condition]


def _aggregate_condition(
    systems: dict[str, Any],
    *,
    condition: str,
    method: str,
    source_private: bool,
    source_text_exposed: bool,
    source_kv_exposed: bool,
    claim_scope: str,
    next_action: str,
    overclaim_guard: str,
) -> dict[str, Any]:
    rows = _condition_rows(systems, condition)
    if not rows:
        return _row(
            row_class="missing_same_slice_row",
            method=method,
            measurement_status="missing",
            source_private=source_private,
            source_text_exposed=source_text_exposed,
            source_kv_exposed=source_kv_exposed,
            payload_bytes=None,
            record_bytes=None,
            single_request_cacheline_bytes=None,
            single_request_dma_bytes=None,
            single_request_page_bytes=None,
            batch64_line_bytes_per_request=None,
            batch64_dma_bytes_per_request=None,
            kv_layer_fraction=None,
            kv_bits_k=None,
            kv_bits_v=None,
            accuracy_min=None,
            accuracy_max=None,
            target_accuracy=None,
            best_control_accuracy_max=None,
            pass_rows=0,
            rows=0,
            resident_sparse_decode_p50_us=None,
            measured_on="not_measured",
            native_kernel_status="missing",
            nvidia_vllm_required=False,
            claim_scope=claim_scope,
            primary_source="local systems waterfall",
            next_action=next_action,
            overclaim_guard=overclaim_guard,
        )
    pass_values = [row["pass_gate"] for row in rows if row["pass_gate"] is not None]
    headline = systems.get("headline", {})
    payload_bytes = float(headline.get("budget_bytes", max(float(row["payload_bytes"]) for row in rows)))
    record_bytes = float(headline.get("packet_record_bytes", max(float(row["record_bytes"]) for row in rows)))
    cacheline_bytes = float(
        headline.get("packet_single_request_cacheline_bytes", max(float(row["single_request_cacheline_bytes"]) for row in rows))
    )
    dma_bytes = float(
        headline.get("packet_single_request_dma_bytes", max(float(row["single_request_dma_bytes"]) for row in rows))
    )
    batch_line_bytes = float(
        headline.get("packet_batch_line_bytes_per_request", max(float(row["batch_line_bytes_per_request"]) for row in rows))
    )
    batch_dma_bytes = float(
        headline.get("packet_batch_dma_bytes_per_request", max(float(row["batch_dma_bytes_per_request"]) for row in rows))
    )
    return _row(
        row_class="same_slice_measured",
        method=method,
        measurement_status="measured_same_slice_mac",
        source_private=source_private,
        source_text_exposed=source_text_exposed,
        source_kv_exposed=source_kv_exposed,
        payload_bytes=payload_bytes,
        record_bytes=record_bytes,
        single_request_cacheline_bytes=cacheline_bytes,
        single_request_dma_bytes=dma_bytes,
        single_request_page_bytes=4096.0,
        batch64_line_bytes_per_request=batch_line_bytes,
        batch64_dma_bytes_per_request=batch_dma_bytes,
        kv_layer_fraction=None,
        kv_bits_k=None,
        kv_bits_v=None,
        accuracy_min=min(float(row["accuracy"]) for row in rows),
        accuracy_max=max(float(row["accuracy"]) for row in rows),
        target_accuracy=max(float(row["target_accuracy"]) for row in rows),
        best_control_accuracy_max=max(float(row["best_control_accuracy"]) for row in rows),
        pass_rows=sum(bool(value) for value in pass_values) if pass_values else None,
        rows=len(rows),
        resident_sparse_decode_p50_us=max(
            (float(row["resident_sparse_decode_p50_us"]) for row in rows if row["resident_sparse_decode_p50_us"] is not None),
            default=None,
        ),
        measured_on="Mac local artifact",
        native_kernel_status="resident sparse decode microbench" if condition == "learned_synonym_dictionary_packet" else "Python summary row",
        nvidia_vllm_required=False,
        claim_scope=claim_scope,
        primary_source="results/source_private_candidate_local_residual_systems_waterfall_20260430/",
        next_action=next_action,
        overclaim_guard=overclaim_guard,
    )


def _memory_method_row(
    memory: dict[str, Any],
    *,
    method: str,
    row_class: str,
    measurement_status: str,
    claim_scope: str,
    next_action: str,
    overclaim_guard: str,
) -> dict[str, Any]:
    candidates = [row for row in memory["rows"] if row["method"] == method]
    if not candidates:
        raise ValueError(f"missing memory traffic row for {method!r}")
    row = candidates[0]
    return _row(
        row_class=row_class,
        method=method,
        measurement_status=measurement_status,
        source_private=bool(row["source_private"]),
        source_text_exposed=bool(row["source_text_exposed"]),
        source_kv_exposed=bool(row["source_kv_exposed"]),
        payload_bytes=float(row["raw_payload_bytes"]),
        record_bytes=float(row["raw_payload_bytes"]),
        single_request_cacheline_bytes=float(row["single_request_cacheline_bytes"]),
        single_request_dma_bytes=float(row["single_request_dma_bytes"]),
        single_request_page_bytes=4096.0,
        batch64_line_bytes_per_request=row["batch64_packet_line_bytes_per_request"],
        batch64_dma_bytes_per_request=row["batch64_packet_dma_bytes_per_request"],
        kv_layer_fraction=None,
        kv_bits_k=None,
        kv_bits_v=None,
        accuracy_min=row["accuracy"],
        accuracy_max=row["accuracy"],
        target_accuracy=row["target_accuracy"],
        best_control_accuracy_max=None,
        pass_rows=None,
        rows=1,
        resident_sparse_decode_p50_us=None,
        measured_on="Mac accounting artifact",
        native_kernel_status="deterministic byte accounting",
        nvidia_vllm_required=False,
        claim_scope=claim_scope,
        primary_source=PRIMARY_URLS.get(method, "results/source_private_memory_traffic_ledger_20260430/"),
        next_action=next_action,
        overclaim_guard=overclaim_guard,
    )


def _external_reference_row(method: str, *, primary_source: str | None = None) -> dict[str, Any]:
    source = primary_source or PRIMARY_URLS[method]
    kv_layer_fraction = None
    kv_bits_k = None
    kv_bits_v = None
    source_kv_exposed = True
    if method.startswith("C2C"):
        claim = "cache-to-cache source KV projection/fusion competitor; not source-private packet equivalent"
        next_action = "Run only with explicit source-KV exposure and byte accounting on NVIDIA/vLLM-capable stack."
        kernel = "native cache fuser required"
    elif method.startswith("KVComm"):
        claim = "multi-agent KV reuse/communication competitor; different access model from endpoint packet"
        next_action = "Run or proxy only after source/target KV tensors and offsets are observable."
        kernel = "native KV reuse implementation required"
        kv_layer_fraction = 0.30
    elif method.startswith("Q-KVComm"):
        claim = "adaptive compressed KV communication competitor; different access model from endpoint packet"
        next_action = "Run or proxy only after source/target KV tensors and layer-wise compression metadata are observable."
        kernel = "native quantized KV communication implementation required"
    elif method.startswith("TurboQuant"):
        claim = "online vector/KV quantization byte-floor competitor; native task is KV compression"
        next_action = "Report as byte floor now; run native KV quantization kernels on NVIDIA later."
        kernel = "native low-bit KV kernel required"
        kv_bits_k = 3.5
        kv_bits_v = 3.5
    elif method.startswith("CacheGen"):
        claim = "KV-cache compression/streaming serving baseline; exposes/fetches KV tensors"
        next_action = "Keep as serving byte-floor citation unless implementing KV-cache streaming."
        kernel = "native KV streaming/compression implementation required"
    else:
        claim = "serving substrate for later GPU trace"
        next_action = "Use for TTFT/TPOT/goodput/HBM trace when NVIDIA is available."
        kernel = "serving runtime required"
        source_kv_exposed = False
    return _row(
        row_class="external_native_required",
        method=method,
        measurement_status="pending_native_systems_row",
        source_private=False,
        source_text_exposed=False,
        source_kv_exposed=source_kv_exposed,
        payload_bytes=None,
        record_bytes=None,
        single_request_cacheline_bytes=None,
        single_request_dma_bytes=None,
        single_request_page_bytes=None,
        batch64_line_bytes_per_request=None,
        batch64_dma_bytes_per_request=None,
        kv_layer_fraction=kv_layer_fraction,
        kv_bits_k=kv_bits_k,
        kv_bits_v=kv_bits_v,
        accuracy_min=None,
        accuracy_max=None,
        target_accuracy=None,
        best_control_accuracy_max=None,
        pass_rows=None,
        rows=None,
        resident_sparse_decode_p50_us=None,
        measured_on="not run locally",
        native_kernel_status=kernel,
        nvidia_vllm_required=True,
        claim_scope=claim,
        primary_source=source,
        next_action=next_action,
        overclaim_guard="Do not claim a win over this method until the native row is run or explicitly scoped as a byte-floor contrast.",
    )


def _ceil_quantum(value: float, quantum: int) -> float:
    return float(math.ceil(value / quantum) * quantum)


def _min_positive_payload(kv_table: dict[str, Any], key: str) -> tuple[float, float]:
    candidates = [
        (float(row[key]), float(row["prompt_token_delta_vs_packet"]))
        for row in kv_table["rows"]
        if row.get("condition") != "matched_packet"
        and row.get("prompt_token_delta_vs_packet", 0) > 0
        and row.get(key) not in {None, ""}
    ]
    if not candidates:
        raise ValueError(f"no positive KV proxy payload rows for {key}")
    return min(candidates, key=lambda item: item[0])


def _kv_proxy_rows(kv_table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in KV_PROXY_METHODS:
        raw_payload, prompt_delta = _min_positive_payload(kv_table, str(spec["payload_key"]))
        payload_bytes = raw_payload * float(spec.get("scale", 1.0))
        rows.append(
            _row(
                row_class="kv_native_proxy_byte_floor",
                method=str(spec["method"]),
                measurement_status="mac_proxy_byte_floor_only",
                source_private=False,
                source_text_exposed=False,
                source_kv_exposed=True,
                payload_bytes=payload_bytes,
                record_bytes=payload_bytes,
                single_request_cacheline_bytes=_ceil_quantum(payload_bytes, 64),
                single_request_dma_bytes=_ceil_quantum(payload_bytes, 128),
                single_request_page_bytes=_ceil_quantum(payload_bytes, 4096),
                batch64_line_bytes_per_request=payload_bytes,
                batch64_dma_bytes_per_request=payload_bytes,
                kv_layer_fraction=spec.get("kv_layer_fraction"),  # type: ignore[arg-type]
                kv_bits_k=spec.get("kv_bits_k"),  # type: ignore[arg-type]
                kv_bits_v=spec.get("kv_bits_v"),  # type: ignore[arg-type]
                accuracy_min=None,
                accuracy_max=None,
                target_accuracy=None,
                best_control_accuracy_max=None,
                pass_rows=None,
                rows=1,
                resident_sparse_decode_p50_us=None,
                measured_on="Mac deterministic KV byte-floor proxy from endpoint summaries and model config",
                native_kernel_status="proxy only; native KV/cache kernel not run",
                nvidia_vllm_required=True,
                claim_scope=str(spec["claim_scope"]),
                primary_source=str(spec["primary_source"]),
                next_action=str(spec["next_action"]),
                overclaim_guard=(
                    f"Byte floor assumes {prompt_delta:.3f} extra private prompt tokens from the endpoint proxy; "
                    "do not claim native accuracy, latency, or quality equivalence."
                ),
            )
        )
    return rows


def _headline(rows: list[dict[str, Any]], systems: dict[str, Any], competitor: dict[str, Any]) -> dict[str, Any]:
    live = next(row for row in rows if row["method"] == "candidate-local residual chart with row/payload normalization")
    pending = [row for row in rows if row["measurement_status"] == "pending_native_systems_row"]
    kv_rows = [row for row in rows if row["row_class"] == "kv_byte_floor_accounting"]
    proxy_rows = [row for row in rows if row["row_class"] == "kv_native_proxy_byte_floor"]
    proxy_payloads = [float(row["payload_bytes"]) for row in proxy_rows if row["payload_bytes"] is not None]
    by_method = {row["method"]: row for row in proxy_rows}
    return {
        "pass_gate": bool(live["rows"]) and live["pass_rows"] == live["rows"],
        "live_pass_rows": live["pass_rows"],
        "live_rows": live["rows"],
        "live_accuracy_min": live["accuracy_min"],
        "live_accuracy_max": live["accuracy_max"],
        "live_packet_payload_bytes": live["payload_bytes"],
        "live_packet_record_bytes": live["record_bytes"],
        "live_batch64_line_bytes_per_request": live["batch64_line_bytes_per_request"],
        "live_resident_sparse_decode_p50_us": live["resident_sparse_decode_p50_us"],
        "source_text_exposed": live["source_text_exposed"],
        "source_kv_exposed": live["source_kv_exposed"],
        "measured_boundary_rows": sum(row["measurement_status"] != "pending_native_systems_row" for row in rows),
        "pending_native_systems_rows": len(pending),
        "pending_native_methods": [row["method"] for row in pending],
        "kv_byte_floor_rows": len(kv_rows),
        "kv_native_proxy_byte_floor_rows": len(proxy_rows),
        "min_kv_native_proxy_payload_bytes": min(proxy_payloads) if proxy_payloads else None,
        "min_kv_native_proxy_record_ratio_vs_live": (
            min(proxy_payloads) / float(live["record_bytes"])
            if proxy_payloads and live["record_bytes"]
            else None
        ),
        "c2c_fp16_proxy_payload_bytes": (
            by_method["C2C fp16 source-KV lower-bound proxy"]["payload_bytes"]
            if "C2C fp16 source-KV lower-bound proxy" in by_method
            else None
        ),
        "kvcomm_30pct_proxy_payload_bytes": (
            by_method["KVComm 30%-layer fp16 source-KV lower-bound proxy"]["payload_bytes"]
            if "KVComm 30%-layer fp16 source-KV lower-bound proxy" in by_method
            else None
        ),
        "turboquant_3p5_proxy_payload_bytes": (
            by_method["TurboQuant 3.5-bit source-KV lower-bound proxy"]["payload_bytes"]
            if "TurboQuant 3.5-bit source-KV lower-bound proxy" in by_method
            else None
        ),
        "iclr_systems_complete": False,
        "colm_systems_table_ready": True,
        "method_competitor_measured_rows": competitor["headline"].get("measured_table_rows", 0),
        "method_competitor_pending_rows": competitor["headline"].get("pending_required_rows", 0),
        "systems_waterfall_pass_gate": systems["pass_gate"],
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Candidate-Local Systems Boundary Trace",
        "",
        "This artifact separates the live packet method from text relays, KV byte floors, and native",
        "cache-communication systems that still require NVIDIA/vLLM-style access.",
        "",
        "## Headline",
        "",
        f"- pass gate: `{h['pass_gate']}`",
        f"- live packet rows: `{h['live_pass_rows']}/{h['live_rows']}`",
        f"- live accuracy range: `{h['live_accuracy_min']:.3f}-{h['live_accuracy_max']:.3f}`",
        f"- live packet payload/record bytes: `{h['live_packet_payload_bytes']:.1f}/{h['live_packet_record_bytes']:.1f}`",
        f"- live batch64 line bytes/request: `{h['live_batch64_line_bytes_per_request']:.2f}`",
        f"- live resident sparse decode p50: `{h['live_resident_sparse_decode_p50_us']:.3f} us`",
        f"- pending native systems rows: `{h['pending_native_systems_rows']}`",
        f"- KV native proxy byte-floor rows: `{h['kv_native_proxy_byte_floor_rows']}`",
        (
            "- min KV native proxy payload/live record ratio: "
            f"`{h['min_kv_native_proxy_record_ratio_vs_live']:.1f}x`"
            if h["min_kv_native_proxy_record_ratio_vs_live"] is not None
            else "- min KV native proxy payload/live record ratio: ``"
        ),
        f"- ICLR systems complete: `{h['iclr_systems_complete']}`",
        f"- COLM systems table ready: `{h['colm_systems_table_ready']}`",
        "",
        "## Rows",
        "",
        "| Class | Method | Status | Payload B | Record B | Batch64 line B/req | Accuracy | Exposure | Native status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        accuracy = ""
        if row["accuracy_min"] is not None:
            accuracy = f"{row['accuracy_min']:.3f}-{row['accuracy_max']:.3f}"
        exposure = []
        if row["source_private"]:
            exposure.append("source-private")
        if row["source_text_exposed"]:
            exposure.append("text")
        if row["source_kv_exposed"]:
            exposure.append("KV")
        payload_bytes = "" if row["payload_bytes"] is None else f"{row['payload_bytes']:.1f}"
        record_bytes = "" if row["record_bytes"] is None else f"{row['record_bytes']:.1f}"
        batch = "" if row["batch64_line_bytes_per_request"] is None else f"{row['batch64_line_bytes_per_request']:.2f}"
        lines.append(
            f"| `{row['row_class']}` | {row['method']} | `{row['measurement_status']}` | "
            f"{payload_bytes} | {record_bytes} | {batch} | {accuracy} | "
            f"{', '.join(exposure) or 'none'} | {row['native_kernel_status']} |"
        )
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
    lines.extend(f"- {item}" for item in payload["non_claims"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_candidate_local_systems_boundary_trace(
    *,
    systems_waterfall: pathlib.Path,
    memory_ledger: pathlib.Path,
    systems_rate_frontier: pathlib.Path,
    competitor_table: pathlib.Path,
    kv_cache_table: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    systems = _read_json(systems_waterfall)
    memory = _read_json(memory_ledger)
    systems_rate = _read_json(systems_rate_frontier)
    competitor = _read_json(competitor_table)
    kv_table = _read_json(kv_cache_table)
    rows = [
        _aggregate_condition(
            systems,
            condition="learned_synonym_dictionary_packet",
            method="candidate-local residual chart with row/payload normalization",
            source_private=True,
            source_text_exposed=False,
            source_kv_exposed=False,
            claim_scope="main positive source-private packet systems row",
            next_action="Run the same boundary row under vLLM/NVIDIA when available.",
            overclaim_guard="This is a Mac resident sparse decode trace, not production GPU serving throughput.",
        ),
        _aggregate_condition(
            systems,
            condition="structured_text_matched",
            method="matched-byte structured text/log prefix",
            source_private=False,
            source_text_exposed=True,
            source_kv_exposed=False,
            claim_scope="same-byte visible text control",
            next_action="Keep as a negative text row at 8B.",
            overclaim_guard="Do not call text generally weak; this only covers matched 8B text prefixes.",
        ),
        _aggregate_condition(
            systems,
            condition="random_same_byte",
            method="random same-byte source-private packet",
            source_private=True,
            source_text_exposed=False,
            source_kv_exposed=False,
            claim_scope="byte-budget control",
            next_action="Keep as source-destroying byte control.",
            overclaim_guard="This is not a method row.",
        ),
        _memory_method_row(
            memory,
            method="query-aware diagnostic text",
            row_class="text_relay_accounting",
            measurement_status="mac_accounting_proxy",
            claim_scope="higher-rate text comparator",
            next_action="Only promote if source text exposure is acceptable in the task threat model.",
            overclaim_guard="Text can tie one cache-line quantum for a single request but exposes private text.",
        ),
        _memory_method_row(
            memory,
            method="full hidden-log relay",
            row_class="text_relay_accounting",
            measurement_status="mac_accounting_proxy",
            claim_scope="visible private-text upper relay",
            next_action="Use as a privacy/rate contrast, not a source-private method.",
            overclaim_guard="Full logs are a different exposure model and should not be treated as a fair privacy-equivalent baseline.",
        ),
        _memory_method_row(
            memory,
            method="QJL-style 1-bit source KV byte floor",
            row_class="kv_byte_floor_accounting",
            measurement_status="byte_floor_accounting_only",
            claim_scope="KV/cache byte-floor contrast",
            next_action="Run native KV/sign-sketch compression only on NVIDIA/vLLM-capable stack.",
            overclaim_guard="This row is accounting only; no native KV compression quality or throughput claim.",
        ),
        _memory_method_row(
            memory,
            method="KIVI/KVQuant-style 2-bit source KV byte floor",
            row_class="kv_byte_floor_accounting",
            measurement_status="byte_floor_accounting_only",
            claim_scope="KV/cache byte-floor contrast",
            next_action="Run native low-bit KV cache kernels only on NVIDIA/vLLM-capable stack.",
            overclaim_guard="This row is accounting only; no native KIVI/KVQuant throughput claim.",
        ),
        _external_reference_row("C2C cache-to-cache communication"),
        _external_reference_row("KVComm / KVCOMM selective KV communication"),
        _external_reference_row("Q-KVComm adaptive compressed KV communication"),
        _external_reference_row("TurboQuant"),
        _external_reference_row("CacheGen"),
        _external_reference_row("vLLM / PagedAttention serving substrate"),
    ]
    rows.extend(_kv_proxy_rows(kv_table))
    headline = _headline(rows, systems, competitor)
    payload = {
        "gate": "source_private_candidate_local_systems_boundary_trace",
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "rows": rows,
        "systems_rate_related_work": systems_rate.get("related_work", []),
        "interpretation": (
            "The systems contribution is a boundary-interface claim: the live method sends an 8B "
            "source-private payload as an 11B record with no source text or source KV exposure, and "
            "the current Mac artifact validates resident sparse decoding over receiver-local public "
            "candidate residuals. The added KV proxy rows are deterministic byte floors for C2C, "
            "KVComm/Q-KVComm, TurboQuant, and CacheGen under the local Qwen3 endpoint summaries. "
            "They sharpen the systems boundary, but the native systems rows remain not defeated."
        ),
        "non_claims": [
            "No production NVIDIA/HBM/vLLM throughput claim from this artifact.",
            "No claim of beating C2C/KVComm/TurboQuant/CacheGen on their native KV/cache tasks.",
            "KV proxy rows are byte floors only; they do not measure native compression quality or latency.",
            "No claim that private text relays are impossible; they are different exposure and rate points.",
            "No claim that the current receiver is protocol-free latent transfer.",
        ],
        "sources": {
            "systems_waterfall": str(systems_waterfall),
            "systems_waterfall_sha256": _sha256_file(systems_waterfall),
            "memory_ledger": str(memory_ledger),
            "memory_ledger_sha256": _sha256_file(memory_ledger),
            "systems_rate_frontier": str(systems_rate_frontier),
            "systems_rate_frontier_sha256": _sha256_file(systems_rate_frontier),
            "competitor_table": str(competitor_table),
            "competitor_table_sha256": _sha256_file(competitor_table),
            "kv_cache_table": str(kv_cache_table),
            "kv_cache_table_sha256": _sha256_file(kv_cache_table),
        },
    }
    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "candidate_local_systems_boundary_trace.json"
    csv_path = output / "candidate_local_systems_boundary_trace.csv"
    md_path = output / "candidate_local_systems_boundary_trace.md"
    manifest_path = output / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name, "manifest.md"],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "manifest.md").write_text(
        "\n".join(
            [
                "# Candidate-Local Systems Boundary Trace Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- live packet rows: `{headline['live_pass_rows']}/{headline['live_rows']}`",
                f"- pending native systems rows: `{headline['pending_native_systems_rows']}`",
                f"- ICLR systems complete: `{headline['iclr_systems_complete']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-waterfall", type=pathlib.Path, default=DEFAULT_SYSTEMS_WATERFALL)
    parser.add_argument("--memory-ledger", type=pathlib.Path, default=DEFAULT_MEMORY_LEDGER)
    parser.add_argument("--systems-rate-frontier", type=pathlib.Path, default=DEFAULT_SYSTEMS_RATE_FRONTIER)
    parser.add_argument("--competitor-table", type=pathlib.Path, default=DEFAULT_COMPETITOR_TABLE)
    parser.add_argument("--kv-cache-table", type=pathlib.Path, default=DEFAULT_KV_CACHE_TABLE)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_candidate_local_systems_boundary_trace(
        systems_waterfall=args.systems_waterfall,
        memory_ledger=args.memory_ledger,
        systems_rate_frontier=args.systems_rate_frontier,
        competitor_table=args.competitor_table,
        kv_cache_table=args.kv_cache_table,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(_resolve(args.output_dir)),
                "pass_gate": payload["pass_gate"],
                "headline": payload["headline"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
