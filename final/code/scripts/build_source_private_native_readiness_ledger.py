from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_FRONTIER = pathlib.Path(
    "results/source_private_train_donor_antishuffle_locked_rate_frontier_20260501/"
    "train_donor_locked_rate_frontier.json"
)
DEFAULT_PACKET_RING = pathlib.Path(
    "results/source_private_mac_packet_ring_transport_microbench_20260501/"
    "packet_ring_transport_microbench.json"
)
DEFAULT_SERVING_SLO = pathlib.Path(
    "results/source_private_serving_slo_envelope_20260501/serving_slo_envelope.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_native_readiness_ledger_20260501")

CSV_COLUMNS = (
    "row_id",
    "method_family",
    "method",
    "primary_source_url",
    "native_problem",
    "communicated_object",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "receiver_public_side_info",
    "same_model_required",
    "cross_model_supported",
    "evidence_class",
    "measured_on",
    "artifact_path",
    "artifact_sha256",
    "payload_bytes",
    "record_bytes",
    "single_request_cacheline_bytes",
    "batch64_line_bytes_per_request",
    "kv_layer_fraction",
    "kv_bits_k",
    "kv_bits_v",
    "accuracy_surface",
    "accuracy",
    "target_accuracy",
    "best_control_accuracy",
    "latency_surface",
    "mac_packet_ring_p50_ns",
    "mac_sparse_decode_p50_us",
    "ttft_ms",
    "tpot_ms",
    "throughput_qps",
    "peak_gpu_mem_gb",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "native_kernel_status",
    "native_blocker",
    "fair_comparison_axis",
    "claim_allowed",
    "claim_forbidden",
    "next_native_gate",
)

EXTERNAL_ROWS = (
    {
        "row_id": "external_c2c",
        "method_family": "cache_reuse",
        "method": "C2C cache-to-cache communication",
        "primary_source_url": "https://arxiv.org/abs/2510.03215",
        "native_problem": "cross-model KV/cache communication",
        "communicated_object": "source KV/cache state",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "receiver_public_side_info": "native cache-fuser implementation",
        "same_model_required": False,
        "cross_model_supported": True,
        "evidence_class": "external_reference_only",
        "measured_on": "not measured locally",
        "kv_layer_fraction": 1.0,
        "kv_bits_k": 16.0,
        "kv_bits_v": 16.0,
        "native_kernel_status": "pending_native_required",
        "native_blocker": "requires NVIDIA/vLLM-compatible native cache-fuser run",
        "fair_comparison_axis": "accuracy/latency/bytes with source-KV exposure flagged",
        "claim_allowed": "related-work native baseline to run",
        "claim_forbidden": "Do not claim LatentWire beats C2C native throughput or quality yet.",
        "next_native_gate": "Run C2C or closest available KV cache-fusion baseline with identical source-private task rows.",
    },
    {
        "row_id": "external_kvcomm",
        "method_family": "kv_communication",
        "method": "KVComm/KVCOMM selective KV communication",
        "primary_source_url": "https://openreview.net/forum?id=F7rUng23nw; https://arxiv.org/abs/2510.03346",
        "native_problem": "multi-agent KV-cache communication",
        "communicated_object": "selected source KV layers",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "receiver_public_side_info": "native KV layer selector",
        "same_model_required": False,
        "cross_model_supported": True,
        "evidence_class": "external_reference_only",
        "measured_on": "not measured locally",
        "kv_layer_fraction": 0.30,
        "kv_bits_k": 16.0,
        "kv_bits_v": 16.0,
        "native_kernel_status": "pending_native_required",
        "native_blocker": "requires native KV selection/compression run",
        "fair_comparison_axis": "source exposure, bytes, accuracy, TTFT/TPOT",
        "claim_allowed": "related-work systems comparator",
        "claim_forbidden": "Do not treat byte-floor accounting as native KVComm evidence.",
        "next_native_gate": "Implement or run a KVComm proxy with layer-selection metadata and source-KV exposure recorded.",
    },
    {
        "row_id": "external_turboquant",
        "method_family": "quantized_kv",
        "method": "TurboQuant-style low-bit KV cache",
        "primary_source_url": "https://arxiv.org/abs/2504.19874",
        "native_problem": "low-bit KV cache compression",
        "communicated_object": "quantized source KV/cache state",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "receiver_public_side_info": "native low-bit KV kernels",
        "same_model_required": True,
        "cross_model_supported": False,
        "evidence_class": "external_reference_only",
        "measured_on": "not measured locally",
        "kv_layer_fraction": 1.0,
        "kv_bits_k": 3.5,
        "kv_bits_v": 3.5,
        "native_kernel_status": "pending_native_required",
        "native_blocker": "requires NVIDIA-supported low-bit KV kernel path",
        "fair_comparison_axis": "KV bytes/latency under same accuracy target",
        "claim_allowed": "quantization substrate inspiration and byte-floor comparator",
        "claim_forbidden": "Do not claim native TurboQuant speedup or quality without running it.",
        "next_native_gate": "Run low-bit KV cache baseline when NVIDIA hardware is available.",
    },
    {
        "row_id": "external_qjl",
        "method_family": "quantized_projection",
        "method": "QJL-style quantized Johnson-Lindenstrauss sketch",
        "primary_source_url": "https://arxiv.org/abs/2406.03482",
        "native_problem": "compressed embedding/KV sketching",
        "communicated_object": "1-bit projected source state",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "receiver_public_side_info": "shared random projection/sign sketch",
        "same_model_required": False,
        "cross_model_supported": True,
        "evidence_class": "external_reference_only",
        "measured_on": "not measured locally",
        "kv_layer_fraction": 1.0,
        "kv_bits_k": 1.0,
        "kv_bits_v": 1.0,
        "native_kernel_status": "pending_native_required",
        "native_blocker": "requires native sketch quality/latency baseline",
        "fair_comparison_axis": "source-state sketch bytes versus packet bytes at matched accuracy",
        "claim_allowed": "mathematical systems inspiration and byte-floor comparator",
        "claim_forbidden": "Do not claim QJL is beaten before a native or faithful proxy run.",
        "next_native_gate": "Run sign-sketch baseline as a strict source-state exposure comparator.",
    },
    {
        "row_id": "external_vllm",
        "method_family": "serving_substrate",
        "method": "vLLM/PagedAttention serving substrate",
        "primary_source_url": "https://arxiv.org/abs/2309.06180",
        "native_problem": "production LLM serving throughput",
        "communicated_object": "paged KV/cache blocks",
        "source_private": False,
        "source_text_exposed": False,
        "source_kv_exposed": True,
        "receiver_public_side_info": "vLLM scheduler and paged cache",
        "same_model_required": True,
        "cross_model_supported": False,
        "evidence_class": "external_reference_only",
        "measured_on": "not measured locally",
        "native_kernel_status": "pending_native_required",
        "native_blocker": "Mac-local artifacts do not measure HBM, TPOT, or vLLM goodput",
        "fair_comparison_axis": "TTFT/TPOT/goodput/peak memory with source exposure annotated",
        "claim_allowed": "native serving target for the next experiment phase",
        "claim_forbidden": "Do not claim production serving win from Mac proxy measurements.",
        "next_native_gate": "Run vLLM endpoint with packet receiver and text/KV baselines on NVIDIA.",
    },
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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _base_row(**updates: Any) -> dict[str, Any]:
    row = {column: None for column in CSV_COLUMNS}
    row.update(updates)
    return row


def _frontier_row(frontier_path: pathlib.Path, frontier: dict[str, Any]) -> dict[str, Any]:
    policy = frontier.get("policies", {}).get("per_seed", {})
    rows = [
        row
        for row in frontier.get("selected_eval_rows", [])
        if row.get("phase") == "eval" and row.get("policy") == "per_seed"
    ]
    if not rows:
        rows = [row for row in frontier.get("selected_eval_rows", []) if row.get("phase") == "eval"]
    budgets = sorted({int(row["budget_bytes"]) for row in rows if row.get("budget_bytes") is not None})
    budget_label = f"{budgets[0]}-{budgets[-1]}" if len(budgets) > 1 else (str(budgets[0]) if budgets else "")
    return _base_row(
        row_id="latentwire_train_donor_frontier",
        method_family="source_private_packet",
        method=f"LatentWire train-donor anti-shuffle packet ({budget_label}B frontier)",
        primary_source_url="local artifact",
        native_problem="source-private cross-model communication",
        communicated_object="short answer-conditioned latent packet",
        source_private=True,
        source_text_exposed=False,
        source_kv_exposed=False,
        receiver_public_side_info="candidate set, train-only calibration, public receiver prompt contract",
        same_model_required=False,
        cross_model_supported=True,
        evidence_class="measured_mac_same_slice",
        measured_on="Apple Silicon / local Python synthetic cross-family gate",
        artifact_path=_rel(frontier_path),
        artifact_sha256=_sha256_file(frontier_path),
        payload_bytes=float(min(budgets)) if budgets else None,
        record_bytes=float(max(budgets)) if budgets else None,
        single_request_cacheline_bytes=64.0 if budgets else None,
        batch64_line_bytes_per_request=float(max(budgets)) if budgets else None,
        accuracy_surface="selected n512 cross-family rows, seeds 47/53/59",
        accuracy=policy.get("min_selected_candidate_accuracy"),
        target_accuracy=0.25,
        best_control_accuracy=policy.get("max_selected_best_control_accuracy"),
        latency_surface="accuracy artifact only; native serving unmeasured",
        native_kernel_status="mac_accuracy_only",
        native_blocker="needs native receiver implementation with GPU/vLLM endpoint measurements",
        fair_comparison_axis="source exposure + payload bytes + accuracy + TTFT/TPOT/goodput",
        claim_allowed="source-private cross-family packet accuracy and byte-boundary evidence",
        claim_forbidden="native GPU/vLLM throughput win, HBM traffic win, or claim of beating C2C/KVComm/TurboQuant native systems",
        next_native_gate="Run the same selected packet rows in a native serving harness with text, KV, C2C/KVComm, and low-bit KV baselines.",
    )


def _packet_ring_row(packet_path: pathlib.Path, packet: dict[str, Any]) -> dict[str, Any]:
    headline = packet.get("headline", {})
    return _base_row(
        row_id="latentwire_mac_packet_ring",
        method_family="transport_microbench",
        method="LatentWire packet-ring packed-record microbench",
        primary_source_url="local artifact",
        native_problem="host-side packet transport accounting",
        communicated_object="packed packet records",
        source_private=True,
        source_text_exposed=False,
        source_kv_exposed=False,
        receiver_public_side_info="contiguous packet ring",
        same_model_required=False,
        cross_model_supported=True,
        evidence_class="measured_mac_packet_ring",
        measured_on="Apple Silicon local C microbench",
        artifact_path=_rel(packet_path),
        artifact_sha256=_sha256_file(packet_path),
        payload_bytes=2.0,
        record_bytes=headline.get("packet_batch64_record_bytes"),
        single_request_cacheline_bytes=64.0,
        batch64_line_bytes_per_request=headline.get("packet_batch64_line_bytes_per_request"),
        latency_surface="batch-64 packet-ring p50",
        mac_packet_ring_p50_ns=headline.get("packet_batch64_p50_ns_per_request"),
        native_kernel_status="mac_transport_proxy",
        native_blocker="does not measure GPU HBM, TPOT, or vLLM scheduler effects",
        fair_comparison_axis="record bytes and host transport overhead",
        claim_allowed="Mac-local packed-record transport sanity check",
        claim_forbidden="production accelerator throughput or end-to-end serving win",
        next_native_gate="Port packet receiver to native endpoint and record GPU counters.",
    )


def _serving_row(serving_path: pathlib.Path, serving: dict[str, Any]) -> dict[str, Any]:
    endpoint_rows = [row for row in serving.get("rows", []) if row.get("row_class") == "endpoint_packet"]
    accuracy = min((float(row["accuracy"]) for row in endpoint_rows if row.get("accuracy") is not None), default=None)
    target = min((float(row["accuracy"] - row["delta_vs_target"]) for row in endpoint_rows if row.get("delta_vs_target") is not None), default=None)
    ttft = min((float(row["ttft_p50_ms"]) for row in endpoint_rows if row.get("ttft_p50_ms") is not None), default=None)
    return _base_row(
        row_id="latentwire_mac_serving_proxy",
        method_family="serving_proxy",
        method="LatentWire Mac endpoint packet proxy",
        primary_source_url="local artifact",
        native_problem="endpoint TTFT proxy",
        communicated_object="diagnostic atom packet",
        source_private=True,
        source_text_exposed=False,
        source_kv_exposed=False,
        receiver_public_side_info="candidate set + receiver prompt contract",
        same_model_required=False,
        cross_model_supported=True,
        evidence_class="measured_mac_same_slice",
        measured_on="Apple Silicon local endpoint proxy",
        artifact_path=_rel(serving_path),
        artifact_sha256=_sha256_file(serving_path),
        payload_bytes=serving.get("headline", {}).get("packet_min_raw_bytes"),
        record_bytes=serving.get("headline", {}).get("packet_min_batch64_line_bytes"),
        single_request_cacheline_bytes=64.0,
        batch64_line_bytes_per_request=serving.get("headline", {}).get("packet_min_batch64_line_bytes"),
        accuracy_surface="legacy endpoint packet n160 label-strict rows",
        accuracy=accuracy,
        target_accuracy=target,
        latency_surface="Mac endpoint TTFT proxy",
        ttft_ms=ttft,
        native_kernel_status="mac_endpoint_proxy",
        native_blocker=serving.get("headline", {}).get("native_serving_gap"),
        fair_comparison_axis="TTFT only; TPOT/goodput pending",
        claim_allowed="Mac TTFT proxy and source-exposure accounting",
        claim_forbidden="native production serving throughput or GPU memory claim",
        next_native_gate="Run packet/text/KV baselines in vLLM/SGLang on NVIDIA with TTFT, TPOT, throughput, and peak memory.",
    )


def _external_row(spec: dict[str, Any]) -> dict[str, Any]:
    return _base_row(
        artifact_path="external literature",
        artifact_sha256="",
        claim_allowed=spec["claim_allowed"],
        claim_forbidden=spec["claim_forbidden"],
        communicated_object=spec["communicated_object"],
        cross_model_supported=spec["cross_model_supported"],
        evidence_class=spec["evidence_class"],
        fair_comparison_axis=spec["fair_comparison_axis"],
        kv_bits_k=spec.get("kv_bits_k"),
        kv_bits_v=spec.get("kv_bits_v"),
        kv_layer_fraction=spec.get("kv_layer_fraction"),
        measured_on=spec["measured_on"],
        method=spec["method"],
        method_family=spec["method_family"],
        native_blocker=spec["native_blocker"],
        native_kernel_status=spec["native_kernel_status"],
        native_problem=spec["native_problem"],
        next_native_gate=spec["next_native_gate"],
        primary_source_url=spec["primary_source_url"],
        receiver_public_side_info=spec["receiver_public_side_info"],
        row_id=spec["row_id"],
        same_model_required=spec["same_model_required"],
        source_kv_exposed=spec["source_kv_exposed"],
        source_private=spec["source_private"],
        source_text_exposed=spec["source_text_exposed"],
    )


def _headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    local_rows = [row for row in rows if str(row["evidence_class"]).startswith("measured_mac")]
    native_pending = [row for row in rows if row["native_kernel_status"] == "pending_native_required"]
    forbidden = sorted({row["claim_forbidden"] for row in rows if row.get("claim_forbidden")})
    return {
        "pass_gate": False,
        "local_measured_rows": len(local_rows),
        "pending_native_rows": len(native_pending),
        "source_private_local_rows": sum(1 for row in local_rows if row["source_private"]),
        "native_ready": False,
        "claim_allowed": "Mac-local source-private packet accuracy/byte-boundary/transport proxy evidence.",
        "claim_forbidden_count": len(forbidden),
        "native_blocker": "NVIDIA/vLLM/SGLang run is still required for systems claims.",
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    headline = payload["headline"]
    lines = [
        "# Source-Private Native Readiness Ledger",
        "",
        f"- native ready: `{headline['native_ready']}`",
        f"- local measured rows: `{headline['local_measured_rows']}`",
        f"- pending native rows: `{headline['pending_native_rows']}`",
        f"- blocker: {headline['native_blocker']}",
        "",
        "## Readiness Rows",
        "",
        "| Row | Evidence | Source private | Source KV exposed | Accuracy | Native status | Claim allowed |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["evidence_class"]),
                    _fmt(row["source_private"]),
                    _fmt(row["source_kv_exposed"]),
                    _fmt(row["accuracy"]),
                    str(row["native_kernel_status"]),
                    str(row["claim_allowed"]),
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
    for non_claim in payload["non_claims"]:
        lines.append(f"- {non_claim}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_native_readiness_ledger(
    *,
    frontier_path: pathlib.Path = DEFAULT_FRONTIER,
    packet_ring_path: pathlib.Path = DEFAULT_PACKET_RING,
    serving_slo_path: pathlib.Path = DEFAULT_SERVING_SLO,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    frontier = _read_json(frontier_path)
    packet_ring = _read_json(packet_ring_path)
    serving_slo = _read_json(serving_slo_path)
    rows = [
        _frontier_row(frontier_path, frontier),
        _packet_ring_row(packet_ring_path, packet_ring),
        _serving_row(serving_slo_path, serving_slo),
        *[_external_row(spec) for spec in EXTERNAL_ROWS],
    ]
    payload = {
        "gate": "source_private_native_readiness_ledger",
        "pass_gate": False,
        "headline": _headline(rows),
        "rows": rows,
        "interpretation": (
            "This ledger makes the systems boundary explicit. The paper may claim Mac-local "
            "source-private packet accuracy, byte accounting, and packed-record transport proxy "
            "evidence. It may not claim native NVIDIA/vLLM throughput, HBM traffic, or wins over "
            "C2C, KVComm, TurboQuant, QJL, or vLLM until those rows are measured in their native "
            "serving setting."
        ),
        "non_claims": [
            "No native GPU throughput claim.",
            "No HBM read/write or peak-memory claim.",
            "No claim that byte-floor KV/cache accounting beats native C2C/KVComm/TurboQuant.",
            "No claim that visible text relay is privacy-equivalent to source-private packets.",
        ],
    }

    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "native_readiness_ledger.json"
    csv_path = output / "native_readiness_ledger.csv"
    md_path = output / "native_readiness_ledger.md"
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
        "headline": payload["headline"],
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
                "# Source-Private Native Readiness Ledger Manifest",
                "",
                f"- native ready: `{payload['headline']['native_ready']}`",
                f"- local measured rows: `{payload['headline']['local_measured_rows']}`",
                f"- pending native rows: `{payload['headline']['pending_native_rows']}`",
                f"- blocker: {payload['headline']['native_blocker']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", type=pathlib.Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--packet-ring", type=pathlib.Path, default=DEFAULT_PACKET_RING)
    parser.add_argument("--serving-slo", type=pathlib.Path, default=DEFAULT_SERVING_SLO)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_native_readiness_ledger(
        frontier_path=args.frontier,
        packet_ring_path=args.packet_ring,
        serving_slo_path=args.serving_slo,
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
