from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_COMPARATOR = pathlib.Path(
    "results/source_private_cross_benchmark_systems_comparator_20260502/"
    "cross_benchmark_systems_comparator.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_byte_amplification_ablation_20260502")

CSV_COLUMNS = (
    "benchmark_row_id",
    "dataset",
    "split",
    "interface_id",
    "interface_group",
    "communicated_object",
    "accuracy_mean",
    "target_accuracy",
    "best_destructive_accuracy",
    "exact_prediction_equivalence_to_packet",
    "equivalence_status",
    "payload_bytes",
    "framed_or_state_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "batch64_line_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "ratio_vs_framed_packet",
    "ratio_vs_cacheline_packet",
    "source_scoring_ms_per_question",
    "packet_build_ms_per_question",
    "receiver_decode_p50_us",
    "receiver_decode_p95_us",
    "source_private",
    "source_text_exposed",
    "source_score_vector_exposed",
    "source_hidden_vector_exposed",
    "source_kv_exposed",
    "native_measured",
    "measurement_status",
    "claim_allowed",
    "overclaim_guard",
    "source_url",
    "notes",
)

PRIMARY_SOURCES = {
    "c2c": "https://arxiv.org/abs/2510.03215",
    "kvcomm": "https://arxiv.org/abs/2510.03346",
    "interlat": "https://arxiv.org/abs/2511.09149",
    "qjl": "https://arxiv.org/abs/2406.03482",
    "turboquant": "https://arxiv.org/abs/2504.19874",
    "kivi": "https://arxiv.org/abs/2402.02750",
    "kvquant": "https://arxiv.org/abs/2401.18079",
    "vllm": "https://arxiv.org/abs/2309.06180",
    "wyner_ziv": "https://ieeexplore.ieee.org/document/1055039",
}


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


def _round_up(value: float, quantum: int) -> float:
    if value <= 0:
        return 0.0
    return float(quantum * math.ceil(value / quantum))


def _packed_bytes_per_request(record_bytes: float, *, quantum: int, batch_size: int = 64) -> float:
    if record_bytes <= 0:
        return 0.0
    total = _round_up(record_bytes * batch_size, quantum)
    return total / batch_size


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_bytes(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}B"
    return f"{value:.1f}B"


def _kv_bytes(source_config: dict[str, Any], *, bits_per_element: float, layer_fraction: float = 1.0) -> float:
    return (
        float(source_config["kv_elements_per_source_token"])
        * float(bits_per_element)
        * float(layer_fraction)
        / 8.0
    )


def _source_private(*, text: bool, scores: bool, hidden: bool, kv: bool) -> bool:
    return not text and not scores and not hidden and not kv


def _record(
    *,
    benchmark: dict[str, Any],
    source_config: dict[str, Any],
    interface_id: str,
    interface_group: str,
    communicated_object: str,
    payload_bytes: float,
    framed_or_state_bytes: float,
    exact_prediction_equivalence_to_packet: float,
    equivalence_status: str,
    source_text_exposed: bool = False,
    source_score_vector_exposed: bool = False,
    source_hidden_vector_exposed: bool = False,
    source_kv_exposed: bool = False,
    native_measured: bool = False,
    measurement_status: str,
    source_scoring_ms_per_question: float | None = None,
    packet_build_ms_per_question: float | None = None,
    receiver_decode_p50_us: float | None = None,
    receiver_decode_p95_us: float | None = None,
    claim_allowed: str,
    overclaim_guard: str,
    source_url: str,
    notes: str,
) -> dict[str, Any]:
    framed_packet = float(benchmark["framed_record_bytes"])
    cacheline_packet = _round_up(framed_packet, 64)
    private = _source_private(
        text=source_text_exposed,
        scores=source_score_vector_exposed,
        hidden=source_hidden_vector_exposed,
        kv=source_kv_exposed,
    )
    return {
        "benchmark_row_id": benchmark["row_id"],
        "dataset": benchmark["dataset"],
        "split": benchmark["split"],
        "interface_id": interface_id,
        "interface_group": interface_group,
        "communicated_object": communicated_object,
        "accuracy_mean": float(benchmark["matched_accuracy_mean"]),
        "target_accuracy": (
            None if benchmark.get("target_accuracy") is None else float(benchmark["target_accuracy"])
        ),
        "best_destructive_accuracy": (
            None
            if benchmark.get("best_destructive_accuracy") is None
            else float(benchmark["best_destructive_accuracy"])
        ),
        "exact_prediction_equivalence_to_packet": float(exact_prediction_equivalence_to_packet),
        "equivalence_status": equivalence_status,
        "payload_bytes": float(payload_bytes),
        "framed_or_state_bytes": float(framed_or_state_bytes),
        "single_request_cacheline_bytes": _round_up(float(framed_or_state_bytes), 64),
        "single_request_dma_bytes": _round_up(float(framed_or_state_bytes), 128),
        "batch64_line_bytes_per_request": _packed_bytes_per_request(
            float(framed_or_state_bytes), quantum=64
        ),
        "batch64_dma_bytes_per_request": _packed_bytes_per_request(
            float(framed_or_state_bytes), quantum=128
        ),
        "ratio_vs_framed_packet": float(framed_or_state_bytes) / framed_packet,
        "ratio_vs_cacheline_packet": float(framed_or_state_bytes) / cacheline_packet,
        "source_scoring_ms_per_question": source_scoring_ms_per_question,
        "packet_build_ms_per_question": packet_build_ms_per_question,
        "receiver_decode_p50_us": receiver_decode_p50_us,
        "receiver_decode_p95_us": receiver_decode_p95_us,
        "source_private": private,
        "source_text_exposed": bool(source_text_exposed),
        "source_score_vector_exposed": bool(source_score_vector_exposed),
        "source_hidden_vector_exposed": bool(source_hidden_vector_exposed),
        "source_kv_exposed": bool(source_kv_exposed),
        "native_measured": bool(native_measured),
        "measurement_status": measurement_status,
        "claim_allowed": claim_allowed,
        "overclaim_guard": overclaim_guard,
        "source_url": source_url,
        "notes": notes,
        "source_model_config": {
            "hidden_size": source_config["hidden_size"],
            "kv_elements_per_source_token": source_config["kv_elements_per_source_token"],
            "model_type": source_config.get("model_type"),
        },
    }


def _interface_rows_for_benchmark(
    benchmark: dict[str, Any],
    *,
    source_config: dict[str, Any],
    choice_count: int,
) -> list[dict[str, Any]]:
    framed = float(benchmark["framed_record_bytes"])
    payload = float(benchmark["payload_bytes"])
    score_bytes = float(choice_count) * 2.0
    hidden_bytes = float(source_config["hidden_size"]) * 2.0
    qjl_bytes = _kv_bytes(source_config, bits_per_element=1.0)
    kivi_bytes = _kv_bytes(source_config, bits_per_element=2.0)
    kvquant_bytes = _kv_bytes(source_config, bits_per_element=3.0)
    turbo_bytes = _kv_bytes(source_config, bits_per_element=3.5)
    kvcomm30_bytes = _kv_bytes(source_config, bits_per_element=16.0, layer_fraction=0.30)
    fp16_kv_bytes = _kv_bytes(source_config, bits_per_element=16.0)
    packet_claim = (
        "Cached packet row; source-private byte/exposure claim only, not native GPU throughput."
    )
    fixed_equivalence = "measured_same_cached_packet_predictions"
    floor_equivalence = (
        "counterfactual_same_predictions_for_byte_accounting_only_not_a_native_quality_result"
    )
    source_scoring_ms = benchmark.get("source_scoring_ms_per_question")
    receiver_p50_us = benchmark.get("receiver_decode_p50_us")
    receiver_p95_us = benchmark.get("receiver_decode_p95_us")
    e2e_status = (
        "mac_local_end_to_end_source_scoring_disclosed"
        if source_scoring_ms is not None
        else "mac_local_end_to_end_source_scoring_missing_phase_trace"
    )
    e2e_notes = (
        f"source_scoring_ms_per_question={source_scoring_ms}; "
        f"receiver_decode_p50_us={receiver_p50_us}; "
        f"receiver_decode_p95_us={receiver_p95_us}"
    )
    return [
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="latentwire_packet_cached_source",
            interface_group="source_private_packet",
            communicated_object="cached-source framed source-private task packet",
            payload_bytes=payload,
            framed_or_state_bytes=framed,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=fixed_equivalence,
            native_measured=False,
            measurement_status="cached_source_communication_object",
            claim_allowed=packet_claim,
            overclaim_guard=(
                "This row counts only the communicated packet object; source scoring and serving "
                "latency are intentionally excluded."
            ),
            source_url=benchmark["artifact_path"],
            notes="Communication-object row only; use paired end-to-end row for source-side work.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="latentwire_packet_end_to_end_source_scoring",
            interface_group="source_private_packet",
            communicated_object="same packet with source scoring disclosed separately",
            payload_bytes=payload,
            framed_or_state_bytes=framed,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=fixed_equivalence,
            native_measured=False,
            measurement_status=e2e_status,
            source_scoring_ms_per_question=(
                None if source_scoring_ms is None else float(source_scoring_ms)
            ),
            packet_build_ms_per_question=None,
            receiver_decode_p50_us=None if receiver_p50_us is None else float(receiver_p50_us),
            receiver_decode_p95_us=None if receiver_p95_us is None else float(receiver_p95_us),
            claim_allowed=(
                "Honest end-to-end disclosure row: packet bytes are unchanged, but source scoring "
                "is not hidden from the systems accounting."
            ),
            overclaim_guard=(
                "Do not merge this row with the cached-source packet row when making latency, "
                "TTFT, TPOT, HBM, or goodput claims."
            ),
            source_url=benchmark["artifact_path"],
            notes=e2e_notes,
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="latentwire_cacheline_padded_packet",
            interface_group="source_private_packet",
            communicated_object="same packet padded to one 64B cache line",
            payload_bytes=payload,
            framed_or_state_bytes=64.0,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=fixed_equivalence,
            native_measured=False,
            measurement_status="cacheline_amplification_counterfactual",
            claim_allowed="Single-request cache-line movement stress case; packet content unchanged.",
            overclaim_guard="Use only as packet amplification accounting, not measured serving latency.",
            source_url=benchmark["artifact_path"],
            notes="Worst-case one packet per 64B line; batch packing can recover framed-byte amortization.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="source_score_vector_fp16_floor",
            interface_group="source_state_floor",
            communicated_object=f"{choice_count}-choice fp16 source score vector",
            payload_bytes=score_bytes,
            framed_or_state_bytes=score_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_score_vector_exposed=True,
            measurement_status="score_vector_floor_not_source_private",
            claim_allowed="Reviewer-facing exposure boundary: small bytes but raw source scores cross.",
            overclaim_guard="This is a source-score relay floor, not a source-private packet.",
            source_url=benchmark["artifact_path"],
            notes="Included because byte count alone is insufficient; exposure/threat model matters.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="source_logit_vector_fp16_floor",
            interface_group="source_state_floor",
            communicated_object=f"{choice_count}-choice fp16 source logit vector",
            payload_bytes=score_bytes,
            framed_or_state_bytes=score_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_score_vector_exposed=True,
            measurement_status="logit_vector_floor_not_source_private",
            claim_allowed="Reviewer-facing exposure boundary: logits are byte-small but leak source state.",
            overclaim_guard="This is a source-logit relay floor, not a source-private packet.",
            source_url=benchmark["artifact_path"],
            notes="Separate alias from score vector so reviewers cannot collapse source-state relays into packet rows.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="source_hidden_vector_fp16_floor",
            interface_group="source_state_floor",
            communicated_object="one fp16 source hidden vector",
            payload_bytes=hidden_bytes,
            framed_or_state_bytes=hidden_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_hidden_vector_exposed=True,
            measurement_status="hidden_vector_floor_not_native",
            claim_allowed="Source-state exposure byte floor only.",
            overclaim_guard="This exposes a hidden vector and is not source-private.",
            source_url="local Qwen2.5-0.5B config",
            notes=f"hidden_size={source_config['hidden_size']}",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="qjl_1bit_kv_floor",
            interface_group="kv_cache_floor",
            communicated_object="one-token K+V state at 1 bit/element",
            payload_bytes=qjl_bytes,
            framed_or_state_bytes=qjl_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_kv_exposed=True,
            measurement_status="optimistic_qjl_byte_floor_not_native",
            claim_allowed="Optimistic KV-state byte floor comparator only.",
            overclaim_guard="QJL is a KV-cache sketch comparator; no native quality or throughput win is claimed.",
            source_url=PRIMARY_SOURCES["qjl"],
            notes="Uses local source config K+V elements per token times 1 bit.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="kivi_2bit_kv_floor",
            interface_group="kv_cache_floor",
            communicated_object="one-token K+V state at 2 bits/element",
            payload_bytes=kivi_bytes,
            framed_or_state_bytes=kivi_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_kv_exposed=True,
            measurement_status="kivi_byte_floor_not_native",
            claim_allowed="KV-cache quantization byte-floor comparator only.",
            overclaim_guard="KIVI compresses KV state; it is not a source-private task packet.",
            source_url=PRIMARY_SOURCES["kivi"],
            notes="Uses local source config K+V elements per token times 2 bits.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="kvquant_3bit_kv_floor",
            interface_group="kv_cache_floor",
            communicated_object="one-token K+V state at 3 bits/element",
            payload_bytes=kvquant_bytes,
            framed_or_state_bytes=kvquant_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_kv_exposed=True,
            measurement_status="kvquant_proxy_floor_not_native",
            claim_allowed="Sub-4-bit KV-cache comparator only.",
            overclaim_guard="KVQuant is a source-state codec comparator; native quality/throughput is pending.",
            source_url=PRIMARY_SOURCES["kvquant"],
            notes="Uses a 3-bit proxy floor for a sub-4-bit KV quantizer.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="turboquant_3p5bit_kv_floor",
            interface_group="kv_cache_floor",
            communicated_object="one-token K+V state at 3.5 bits/element",
            payload_bytes=turbo_bytes,
            framed_or_state_bytes=turbo_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_kv_exposed=True,
            measurement_status="turboquant_byte_floor_not_native",
            claim_allowed="Vector/KV quantization byte-floor comparator only.",
            overclaim_guard="TurboQuant is not a native LatentWire baseline until run in the same serving stack.",
            source_url=PRIMARY_SOURCES["turboquant"],
            notes="Uses local source config K+V elements per token times 3.5 bits.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="kvcomm30_fp16_floor",
            interface_group="kv_communication_floor",
            communicated_object="30% selected source KV layers at fp16",
            payload_bytes=kvcomm30_bytes,
            framed_or_state_bytes=kvcomm30_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_kv_exposed=True,
            measurement_status="kvcomm30_floor_not_native",
            claim_allowed="Selective-KV communication byte-floor comparator only.",
            overclaim_guard="KVComm shares KV pairs; this is not source-private and not a native run.",
            source_url=PRIMARY_SOURCES["kvcomm"],
            notes="Uses 30% layer fraction as an optimistic accounting row.",
        ),
        _record(
            benchmark=benchmark,
            source_config=source_config,
            interface_id="c2c_fp16_kv_floor",
            interface_group="kv_communication_floor",
            communicated_object="one-token source K+V cache at fp16",
            payload_bytes=fp16_kv_bytes,
            framed_or_state_bytes=fp16_kv_bytes,
            exact_prediction_equivalence_to_packet=1.0,
            equivalence_status=floor_equivalence,
            source_kv_exposed=True,
            measurement_status="c2c_floor_not_native",
            claim_allowed="Closest cache-transfer byte-floor comparator only.",
            overclaim_guard="C2C fuses source KV cache; do not claim superiority without native C2C.",
            source_url=PRIMARY_SOURCES["c2c"],
            notes="Uses local source config K+V elements per token times fp16.",
        ),
    ]


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Byte-Amplification Ablation",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- benchmark rows: `{h['benchmark_row_count']}`",
        f"- interface rows: `{h['interface_row_count']}`",
        f"- packet framed-byte range: `{h['min_packet_framed_bytes']:.0f}-"
        f"{h['max_packet_framed_bytes']:.0f}B`",
        f"- max single-request cache-line amplification: "
        f"`{h['max_single_request_cacheline_amplification']:.1f}x`",
        f"- minimum KV floor vs 64B padded packet: "
        f"`{h['min_kv_floor_ratio_vs_cacheline_packet']:.1f}x`",
        f"- score-vector floor vs minimum packet: `{h['score_floor_ratio_vs_min_packet']:.1f}x`",
        "",
        "## Table",
        "",
        "| Dataset | Split | Interface | Object | Accuracy | Bytes | Cacheline | Batch64 | Private | Exposure | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    selected_ids = {
        "latentwire_packet_cached_source",
        "latentwire_packet_end_to_end_source_scoring",
        "latentwire_cacheline_padded_packet",
        "source_score_vector_fp16_floor",
        "source_logit_vector_fp16_floor",
        "qjl_1bit_kv_floor",
        "turboquant_3p5bit_kv_floor",
        "c2c_fp16_kv_floor",
    }
    for row in payload["rows"]:
        if row["interface_id"] not in selected_ids:
            continue
        exposures = []
        if row["source_score_vector_exposed"]:
            exposures.append("score")
        if row["source_hidden_vector_exposed"]:
            exposures.append("hidden")
        if row["source_kv_exposed"]:
            exposures.append("KV")
        if row["source_text_exposed"]:
            exposures.append("text")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["split"],
                    row["interface_id"],
                    row["communicated_object"],
                    f"{row['accuracy_mean']:.3f}",
                    _fmt_bytes(row["framed_or_state_bytes"]),
                    _fmt_bytes(row["single_request_cacheline_bytes"]),
                    _fmt_bytes(row["batch64_line_bytes_per_request"]),
                    "`true`" if row["source_private"] else "`false`",
                    ", ".join(exposures) if exposures else "none",
                    row["measurement_status"],
                ]
            )
            + " |"
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_manifest(output_dir: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    files = [
        "byte_amplification_ablation.json",
        "byte_amplification_ablation.csv",
        "byte_amplification_ablation.md",
    ]
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": payload["headline"],
        "files": [
            {"path": _rel(output_dir / filename), "sha256": _sha256_file(output_dir / filename)}
            for filename in files
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_byte_amplification_ablation(
    *,
    comparator_path: pathlib.Path = DEFAULT_COMPARATOR,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    choice_count: int = 4,
) -> dict[str, Any]:
    comparator = _read_json(comparator_path)
    source_config = comparator["source_model_config"]
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for benchmark in comparator["rows"]:
        rows.extend(
            _interface_rows_for_benchmark(
                benchmark,
                source_config=source_config,
                choice_count=choice_count,
            )
        )
    packet_rows = [row for row in rows if row["interface_id"] == "latentwire_packet_cached_source"]
    e2e_packet_rows = [
        row for row in rows if row["interface_id"] == "latentwire_packet_end_to_end_source_scoring"
    ]
    padded_rows = [row for row in rows if row["interface_id"] == "latentwire_cacheline_padded_packet"]
    kv_floor_rows = [row for row in rows if row["interface_group"] in {"kv_cache_floor", "kv_communication_floor"}]
    score_rows = [row for row in rows if row["interface_id"] == "source_score_vector_fp16_floor"]
    logit_rows = [row for row in rows if row["interface_id"] == "source_logit_vector_fp16_floor"]
    min_packet = min(row["framed_or_state_bytes"] for row in packet_rows)
    max_packet = max(row["framed_or_state_bytes"] for row in packet_rows)
    min_kv_floor = min(row["framed_or_state_bytes"] for row in kv_floor_rows)
    checks = [
        {
            "check": "packet_and_padded_packet_rows_remain_prediction_equivalent",
            "pass": all(
                row["exact_prediction_equivalence_to_packet"] == 1.0
                for row in packet_rows + e2e_packet_rows + padded_rows
            ),
            "value": len(packet_rows + e2e_packet_rows + padded_rows),
        },
        {
            "check": "packet_and_padded_packet_rows_remain_source_private",
            "pass": all(row["source_private"] for row in packet_rows + e2e_packet_rows + padded_rows),
            "value": len(packet_rows + e2e_packet_rows + padded_rows),
        },
        {
            "check": "cached_source_and_end_to_end_packet_rows_are_split",
            "pass": len(packet_rows) == len(e2e_packet_rows) == len(comparator["rows"]),
            "value": {
                "cached_source_packet_rows": len(packet_rows),
                "end_to_end_packet_rows": len(e2e_packet_rows),
            },
        },
        {
            "check": "score_floor_is_marked_non_private",
            "pass": all(
                row["source_score_vector_exposed"] and not row["source_private"] for row in score_rows
            ),
            "value": len(score_rows),
        },
        {
            "check": "logit_floor_is_marked_non_private",
            "pass": all(
                row["source_score_vector_exposed"] and not row["source_private"] for row in logit_rows
            ),
            "value": len(logit_rows),
        },
        {
            "check": "kv_floors_exceed_64b_padded_packet_by_10x",
            "pass": min_kv_floor / 64.0 >= 10.0,
            "value": min_kv_floor / 64.0,
        },
        {
            "check": "kv_and_hidden_rows_are_non_native_non_claims",
            "pass": all(
                (not row["native_measured"])
                and ("native" in row["overclaim_guard"].lower() or "source-private" in row["overclaim_guard"].lower())
                for row in rows
                if row["source_kv_exposed"] or row["source_hidden_vector_exposed"]
            ),
            "value": sum(1 for row in rows if row["source_kv_exposed"] or row["source_hidden_vector_exposed"]),
        },
    ]
    headline = {
        "pass_gate": all(check["pass"] for check in checks),
        "benchmark_row_count": len(comparator["rows"]),
        "interface_row_count": len(rows),
        "min_packet_framed_bytes": min_packet,
        "max_packet_framed_bytes": max_packet,
        "max_single_request_cacheline_amplification": max(
            row["single_request_cacheline_bytes"] / row["framed_or_state_bytes"] for row in packet_rows
        ),
        "max_single_request_dma_amplification": max(
            row["single_request_dma_bytes"] / row["framed_or_state_bytes"] for row in packet_rows
        ),
        "min_kv_floor_bytes": min_kv_floor,
        "min_kv_floor_ratio_vs_max_packet": min_kv_floor / max_packet,
        "min_kv_floor_ratio_vs_cacheline_packet": min_kv_floor / 64.0,
        "score_floor_bytes": min(row["framed_or_state_bytes"] for row in score_rows),
        "score_floor_ratio_vs_min_packet": min(row["framed_or_state_bytes"] for row in score_rows) / min_packet,
        "native_nvidia_complete": False,
    }
    payload = {
        "gate": "source_private_byte_amplification_ablation",
        "pass_gate": headline["pass_gate"],
        "source_comparator": {
            "path": _rel(comparator_path),
            "sha256": _sha256_file(comparator_path),
        },
        "choice_count": int(choice_count),
        "source_model_config": source_config,
        "headline": headline,
        "checks": checks,
        "primary_sources": PRIMARY_SOURCES,
        "rows": rows,
        "interpretation": (
            "This ablation holds cached packet predictions fixed and varies the communicated object. "
            "It shows how much byte movement is added by single-request cache-line padding, and how far "
            "even optimistic one-token KV/source-state floors remain from the packet regime. The fp16 "
            "score-vector row is intentionally included as a reviewer stress test: it is byte-small, "
            "but it exposes raw source scores and is not source-private."
        ),
        "non_claims": [
            "This is not a native NVIDIA serving benchmark.",
            "QJL, TurboQuant, KIVI, KVQuant, KVComm, and C2C rows are byte floors or counterfactual same-prediction accounting rows, not defeated native baselines.",
            "Exact prediction equivalence for source-state rows is a held-fixed accounting assumption, not a measured quality result.",
            "Small source-score vectors are not source-private and should not be merged with LatentWire packet rows.",
        ],
    }
    (output_dir / "byte_amplification_ablation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "byte_amplification_ablation.csv", rows)
    _write_markdown(output_dir / "byte_amplification_ablation.md", payload)
    _make_manifest(output_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator", type=pathlib.Path, default=DEFAULT_COMPARATOR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--choice-count", type=int, default=4)
    args = parser.parse_args()
    payload = build_byte_amplification_ablation(
        comparator_path=args.comparator,
        output_dir=args.output_dir,
        choice_count=int(args.choice_count),
    )
    print(
        json.dumps(
            {
                "output_dir": str(_resolve(args.output_dir)),
                "pass_gate": payload["pass_gate"],
                "min_kv_floor_ratio_vs_cacheline_packet": payload["headline"][
                    "min_kv_floor_ratio_vs_cacheline_packet"
                ],
                "score_floor_ratio_vs_min_packet": payload["headline"][
                    "score_floor_ratio_vs_min_packet"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
