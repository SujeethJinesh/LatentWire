from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import statistics
from collections import defaultdict
from typing import Any


SCRIPT_PATH = pathlib.Path(__file__).resolve()
if SCRIPT_PATH.parents[1].name == "code" and SCRIPT_PATH.parents[2].name == "final":
    ROOT = SCRIPT_PATH.parents[2]
else:
    ROOT = SCRIPT_PATH.parents[1]

DEFAULT_PQ_GATE = pathlib.Path(
    "results/source_private_product_codebook_packet_gate_n500_20260430/product_codebook_packet_gate.json"
)
DEFAULT_GEOMETRY_STRESS = pathlib.Path(
    "results/source_private_product_codebook_geometry_knockout_stress_n500_20260430/"
    "product_codebook_geometry_knockout_stress.json"
)
DEFAULT_DECODE_FRONTIER = pathlib.Path(
    "results/source_private_product_codebook_decode_frontier_n500_20260430/product_codebook_decode_frontier.json"
)
DEFAULT_VERIFIER_TRACE = pathlib.Path(
    "results/source_private_verifier_consumption_trace_20260430/"
    "qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu/verifier_consumption_trace.json"
)
DEFAULT_PACKET_TRACE_CARD = pathlib.Path(
    "results/source_private_packet_trace_card_v2_20260430/packet_trace_card_v2.json"
)
DEFAULT_RATE_FRONTIER = pathlib.Path(
    "results/source_private_systems_rate_assumption_frontier_20260430/systems_rate_assumption_frontier.json"
)

CSV_COLUMNS = (
    "row_group",
    "method",
    "surface",
    "artifact",
    "n",
    "remaps",
    "budget_bytes",
    "record_bytes",
    "batch64_line_bytes_per_request",
    "communicated_object",
    "receiver_compute",
    "source_private",
    "source_text_exposed",
    "source_kv_exposed",
    "decoder_side_info",
    "accuracy_min",
    "accuracy_max",
    "target_accuracy",
    "best_control_accuracy_max",
    "delta_vs_target_min",
    "delta_vs_best_control_min",
    "controls_ok_all",
    "pass_rule_result",
    "cached_decode_p50_ms_max",
    "cached_decode_p95_ms_max",
    "request_public_decode_p50_ms_max",
    "source_encode_p50_ms_max",
    "verifier_p50_ms_max",
    "unique_payloads_min",
    "unique_payloads_max",
    "collision_subset_accuracy_min",
    "public_mean_lift_removed_min",
    "top_worst_lift_removed_min",
    "text_byte_ratio_vs_packet_min",
    "kv_byte_ratio_vs_packet_min",
    "paper_use",
    "caveat",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else statistics.fmean(present)


def _min(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else min(present)


def _max(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else max(present)


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {column: kwargs.get(column) for column in CSV_COLUMNS}
    if isinstance(row["artifact"], pathlib.Path):
        row["artifact"] = _rel(row["artifact"])
    accuracy_min = row["accuracy_min"]
    target = row["target_accuracy"]
    best_control = row["best_control_accuracy_max"]
    row["delta_vs_target_min"] = (
        None if accuracy_min is None or target is None else float(accuracy_min) - float(target)
    )
    row["delta_vs_best_control_min"] = (
        None if accuracy_min is None or best_control is None else float(accuracy_min) - float(best_control)
    )
    return row


def _group_by_variant(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    return dict(grouped)


def _decode_aggregate(decode_payload: dict[str, Any]) -> dict[str, float | None]:
    rows = decode_payload["rows"]
    return {
        "cached_decode_p50_ms_max": _max([row["cached_receiver_p50_ms"] for row in rows]),
        "cached_decode_p95_ms_max": _max([row["cached_receiver_p95_ms"] for row in rows]),
        "request_public_decode_p50_ms_max": _max([row["request_public_table_decode_p50_ms"] for row in rows]),
        "source_encode_p50_ms_max": _max([row["source_encode_p50_ms"] for row in rows]),
    }


def _geometry_rows(
    geometry_path: pathlib.Path,
    geometry_payload: dict[str, Any],
    decode_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    decode_stats = _decode_aggregate(decode_payload)
    output: list[dict[str, Any]] = []
    display = {
        "canonical": (
            "canonical 4-byte product-codebook packet",
            "table_distance_decode",
            "baseline compression-native PQ packet",
            "near-unique payloads; geometry mitigation needed for lookup-risk objection",
        ),
        "utility_opq_procrustes": (
            "utility-OPQ product-codebook packet",
            "dense_opq_rotation_plus_table_decode",
            "public-mean-sensitive geometry mitigation",
            "does not reduce payload uniqueness; dense rotation is less hardware friendly",
        ),
        "protected_hadamard": (
            "protected Hadamard product-codebook packet",
            "sign_permutation_hadamard_plus_table_decode",
            "hardware-friendly geometry mitigation",
            "cached decode for this rotated variant is not separately timed in this artifact",
        ),
        "utility_protected_hadamard": (
            "utility-protected Hadamard product-codebook packet",
            "utility_grouping_sign_permutation_hadamard_plus_table_decode",
            "strongest lookup-risk mitigation row",
            "cached decode for this rotated variant is not separately timed in this artifact",
        ),
    }
    for variant, rows in sorted(_group_by_variant(geometry_payload["rows"]).items()):
        if variant not in display:
            continue
        method, compute, paper_use, caveat = display[variant]
        remaps = sorted({row["remap_slot_seed"] for row in rows})
        source_acc = [row["source_accuracy"] for row in rows]
        best_control = [row["best_control_accuracy"] for row in rows]
        target = [row["target_accuracy"] for row in rows]
        unique_payloads = [row["payload_entropy"]["unique_payloads"] for row in rows]
        collision_acc = [row["payload_reuse"]["collision_accuracy"] for row in rows]
        public_mean = [row["top_mean_lift_removed_fraction"] for row in rows]
        top_worst = [row["top_worst_lift_removed_fraction"] for row in rows]
        controls_ok = all(row["source_packet_pass"] for row in rows)
        adversarial_ok = all(row["adversarial_knockout_pass"] for row in rows)
        if variant == "canonical":
            pass_result = "pass_source_controls_lookup_risk"
        else:
            mitigation_ok = all(row["mitigation_pass"] for row in rows)
            pass_result = "pass_mitigation" if mitigation_ok else "source_pass_no_mitigation"
        output.append(
            _row(
                row_group="pq_geometry_method",
                method=method,
                surface="n500 remaps 101/103/107 slot candidate view",
                artifact=geometry_path,
                n=500,
                remaps=remaps,
                budget_bytes=4.0,
                record_bytes=5.0,
                batch64_line_bytes_per_request=None,
                communicated_object="source-private residual codeword packet",
                receiver_compute=compute,
                source_private=True,
                source_text_exposed=False,
                source_kv_exposed=False,
                decoder_side_info=True,
                accuracy_min=min(source_acc),
                accuracy_max=max(source_acc),
                target_accuracy=max(target),
                best_control_accuracy_max=max(best_control),
                controls_ok_all=controls_ok and adversarial_ok,
                pass_rule_result=pass_result,
                cached_decode_p50_ms_max=decode_stats["cached_decode_p50_ms_max"] if variant == "canonical" else None,
                cached_decode_p95_ms_max=decode_stats["cached_decode_p95_ms_max"] if variant == "canonical" else None,
                request_public_decode_p50_ms_max=(
                    decode_stats["request_public_decode_p50_ms_max"] if variant == "canonical" else None
                ),
                source_encode_p50_ms_max=decode_stats["source_encode_p50_ms_max"] if variant == "canonical" else None,
                verifier_p50_ms_max=None,
                unique_payloads_min=min(unique_payloads),
                unique_payloads_max=max(unique_payloads),
                collision_subset_accuracy_min=_min(collision_acc),
                public_mean_lift_removed_min=min(public_mean),
                top_worst_lift_removed_min=min(top_worst),
                text_byte_ratio_vs_packet_min=None,
                kv_byte_ratio_vs_packet_min=None,
                paper_use=paper_use,
                caveat=caveat,
            )
        )
    return output


def _scalar_rows(pq_path: pathlib.Path, pq_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = pq_payload["rows"]
    return [
        _row(
            row_group="source_coding_baseline",
            method="scalar Wyner-Ziv residual packet",
            surface="n500 remaps 101/103/107 slot candidate view",
            artifact=pq_path,
            n=500,
            remaps=sorted({row["remap_slot_seed"] for row in rows}),
            budget_bytes=4.0,
            record_bytes=5.0,
            batch64_line_bytes_per_request=None,
            communicated_object="source-private scalar residual packet",
            receiver_compute="scalar_projection_decode",
            source_private=True,
            source_text_exposed=False,
            source_kv_exposed=False,
            decoder_side_info=True,
            accuracy_min=min(row["scalar_wyner_ziv_accuracy"] for row in rows),
            accuracy_max=max(row["scalar_wyner_ziv_accuracy"] for row in rows),
            target_accuracy=max(row["target_accuracy"] for row in rows),
            best_control_accuracy_max=max(row["best_product_codebook_control_accuracy"] for row in rows),
            controls_ok_all=True,
            pass_rule_result="source_coding_comparator",
            cached_decode_p50_ms_max=None,
            cached_decode_p95_ms_max=None,
            request_public_decode_p50_ms_max=None,
            source_encode_p50_ms_max=None,
            verifier_p50_ms_max=None,
            unique_payloads_min=None,
            unique_payloads_max=None,
            collision_subset_accuracy_min=None,
            public_mean_lift_removed_min=None,
            top_worst_lift_removed_min=None,
            text_byte_ratio_vs_packet_min=None,
            kv_byte_ratio_vs_packet_min=None,
            paper_use="direct learned residual-code comparator",
            caveat="slightly lower accuracy than PQ on at least one remap; useful as scalar syndrome baseline",
        )
    ]


def _verifier_rows(verifier_path: pathlib.Path, verifier_payload: dict[str, Any]) -> list[dict[str, Any]]:
    matched = [row for row in verifier_payload["rows"] if row["condition"] == "matched_packet"]
    controls = [row for row in verifier_payload["rows"] if row["role"] == "source_destroying_control"]
    return [
        _row(
            row_group="model_mediated_receiver",
            method="frozen Qwen3 binary-verifier packet",
            surface="seed31 core+holdout n160 combined controls",
            artifact=verifier_path,
            n=min(row["examples"] for row in matched),
            remaps=None,
            budget_bytes=max(row["mean_payload_bytes"] for row in matched),
            record_bytes=max(row["mean_packet_record_bytes"] for row in matched),
            batch64_line_bytes_per_request=max(row["batch64_line_bytes_per_request"] for row in matched),
            communicated_object="source-private diagnostic packet",
            receiver_compute="four frozen binary logprob forwards per example",
            source_private=True,
            source_text_exposed=False,
            source_kv_exposed=False,
            decoder_side_info=True,
            accuracy_min=min(row["accuracy"] for row in matched),
            accuracy_max=max(row["accuracy"] for row in matched),
            target_accuracy=verifier_payload["headline"]["max_target_only_accuracy"],
            best_control_accuracy_max=max(row["accuracy"] for row in controls),
            controls_ok_all=verifier_payload["pass_gate"],
            pass_rule_result="pass_model_mediated_controls",
            cached_decode_p50_ms_max=None,
            cached_decode_p95_ms_max=None,
            request_public_decode_p50_ms_max=None,
            source_encode_p50_ms_max=None,
            verifier_p50_ms_max=max(row["p50_latency_ms"] for row in matched),
            unique_payloads_min=None,
            unique_payloads_max=None,
            collision_subset_accuracy_min=None,
            public_mean_lift_removed_min=None,
            top_worst_lift_removed_min=None,
            text_byte_ratio_vs_packet_min=None,
            kv_byte_ratio_vs_packet_min=None,
            paper_use="model-mediated consumption evidence",
            caveat="Mac CPU timing is not production serving telemetry",
        )
    ]


def _rate_rows(rate_path: pathlib.Path, rate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_row in rate_payload["rows"]:
        method = source_row["method"]
        row_class = source_row["row_class"]
        if row_class not in {"rate_frontier_text", "endpoint_text_relay", "kv_byte_floor"}:
            continue
        if method == "LatentWire endpoint packet":
            continue
        pass_rule = "accounting_contrast" if row_class == "kv_byte_floor" else source_row["claim_allowed"]
        output.append(
            _row(
                row_group=row_class,
                method=method,
                surface=source_row["surface"],
                artifact=rate_path,
                n=None,
                remaps=None,
                budget_bytes=source_row["private_payload_bytes"],
                record_bytes=None,
                batch64_line_bytes_per_request=None,
                communicated_object=source_row["communicated_object"],
                receiver_compute=source_row["latency_metric_scope"],
                source_private=source_row["source_private"],
                source_text_exposed=source_row["source_text_exposed"],
                source_kv_exposed=source_row["source_kv_exposed"],
                decoder_side_info=True,
                accuracy_min=source_row["accuracy"],
                accuracy_max=source_row["accuracy"],
                target_accuracy=source_row["target_accuracy"],
                best_control_accuracy_max=source_row["best_control_accuracy"],
                controls_ok_all=None,
                pass_rule_result=pass_rule,
                cached_decode_p50_ms_max=None,
                cached_decode_p95_ms_max=None,
                request_public_decode_p50_ms_max=None,
                source_encode_p50_ms_max=None,
                verifier_p50_ms_max=source_row["ttft_ms"],
                unique_payloads_min=None,
                unique_payloads_max=None,
                collision_subset_accuracy_min=None,
                public_mean_lift_removed_min=None,
                top_worst_lift_removed_min=None,
                text_byte_ratio_vs_packet_min=source_row["byte_ratio_vs_packet"],
                kv_byte_ratio_vs_packet_min=source_row["kv_byte_floor"],
                paper_use=source_row["paper_use"],
                caveat=source_row["caveat"],
            )
        )
    return output


def _external_reference_rows(rate_path: pathlib.Path, rate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_row in rate_payload["rows"]:
        if source_row["row_class"] != "external_reference":
            continue
        output.append(
            _row(
                row_group="external_reference",
                method=source_row["method"],
                surface=source_row["surface"],
                artifact=rate_path,
                n=None,
                remaps=None,
                budget_bytes=None,
                record_bytes=None,
                batch64_line_bytes_per_request=None,
                communicated_object=source_row["communicated_object"],
                receiver_compute=source_row["native_problem"],
                source_private=source_row["source_private"],
                source_text_exposed=source_row["source_text_exposed"],
                source_kv_exposed=source_row["source_kv_exposed"],
                decoder_side_info=True,
                accuracy_min=None,
                accuracy_max=None,
                target_accuracy=None,
                best_control_accuracy_max=None,
                controls_ok_all=None,
                pass_rule_result="reference_only",
                cached_decode_p50_ms_max=None,
                cached_decode_p95_ms_max=None,
                request_public_decode_p50_ms_max=None,
                source_encode_p50_ms_max=None,
                verifier_p50_ms_max=None,
                unique_payloads_min=None,
                unique_payloads_max=None,
                collision_subset_accuracy_min=None,
                public_mean_lift_removed_min=None,
                top_worst_lift_removed_min=None,
                text_byte_ratio_vs_packet_min=None,
                kv_byte_ratio_vs_packet_min=None,
                paper_use=source_row["paper_use"],
                caveat=source_row["caveat"],
            )
        )
    return output


def build_pq_systems_comparison_table(
    *,
    pq_gate: pathlib.Path,
    geometry_stress: pathlib.Path,
    decode_frontier: pathlib.Path,
    verifier_trace: pathlib.Path,
    packet_trace_card: pathlib.Path,
    rate_frontier: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pq_gate = _resolve(pq_gate)
    geometry_stress = _resolve(geometry_stress)
    decode_frontier = _resolve(decode_frontier)
    verifier_trace = _resolve(verifier_trace)
    packet_trace_card = _resolve(packet_trace_card)
    rate_frontier = _resolve(rate_frontier)

    pq_payload = _read_json(pq_gate)
    geometry_payload = _read_json(geometry_stress)
    decode_payload = _read_json(decode_frontier)
    verifier_payload = _read_json(verifier_trace)
    packet_payload = _read_json(packet_trace_card)
    rate_payload = _read_json(rate_frontier)

    rows = [
        *_geometry_rows(geometry_stress, geometry_payload, decode_payload),
        *_scalar_rows(pq_gate, pq_payload),
        *_verifier_rows(verifier_trace, verifier_payload),
        *_rate_rows(rate_frontier, rate_payload),
        *_external_reference_rows(rate_frontier, rate_payload),
    ]
    geometry_methods = [row for row in rows if row["row_group"] == "pq_geometry_method"]
    mitigation_methods = [row for row in geometry_methods if row["pass_rule_result"] == "pass_mitigation"]
    verifier_rows = [row for row in rows if row["row_group"] == "model_mediated_receiver"]
    text_rows = [row for row in rows if row["row_group"] in {"rate_frontier_text", "endpoint_text_relay"}]
    kv_rows = [row for row in rows if row["row_group"] == "kv_byte_floor"]
    same_byte = [row for row in text_rows if row["method"] == "same-byte structured text"]
    headline = {
        "pass_gate": (
            geometry_payload["pass_gate"]
            and decode_payload["pass_gate"]
            and verifier_payload["pass_gate"]
            and packet_payload["pass_gate"]
            and bool(mitigation_methods)
        ),
        "pq_geometry_rows": len(geometry_methods),
        "pq_mitigation_rows": len(mitigation_methods),
        "pq_min_delta_vs_best_control": min(
            row["delta_vs_best_control_min"] for row in geometry_methods if row["delta_vs_best_control_min"] is not None
        ),
        "pq_max_cached_decode_p50_ms": max(
            row["cached_decode_p50_ms_max"] or 0.0 for row in geometry_methods
        ),
        "protected_hadamard_unique_payloads_min": min(
            row["unique_payloads_min"]
            for row in geometry_methods
            if "Hadamard" in row["method"] and row["unique_payloads_min"] is not None
        ),
        "verifier_min_accuracy": min(row["accuracy_min"] for row in verifier_rows),
        "verifier_max_p50_latency_ms": max(row["verifier_p50_ms_max"] for row in verifier_rows),
        "same_byte_text_accuracy_max": max(row["accuracy_max"] for row in same_byte),
        "query_aware_text_raw_ratio": packet_payload["headline"]["query_aware_text_raw_ratio"],
        "full_log_raw_ratio_min": packet_payload["headline"]["full_log_raw_ratio_min"],
        "kv_raw_ratio_min": packet_payload["headline"]["kv_raw_ratio_min"],
        "claim_boundary": (
            "This table supports a source-private boundary-traffic and residual-code systems claim. "
            "It does not claim production GPU serving speedup, native KV-cache compression superiority, "
            "or protocol-free latent reasoning."
        ),
    }
    payload = {
        "gate": "source_private_pq_systems_comparison_table",
        "pass_gate": headline["pass_gate"],
        "inputs": {
            "pq_gate": _rel(pq_gate),
            "geometry_stress": _rel(geometry_stress),
            "decode_frontier": _rel(decode_frontier),
            "verifier_trace": _rel(verifier_trace),
            "packet_trace_card": _rel(packet_trace_card),
            "rate_frontier": _rel(rate_frontier),
        },
        "input_sha256": {
            "pq_gate": _sha256_file(pq_gate),
            "geometry_stress": _sha256_file(geometry_stress),
            "decode_frontier": _sha256_file(decode_frontier),
            "verifier_trace": _sha256_file(verifier_trace),
            "packet_trace_card": _sha256_file(packet_trace_card),
            "rate_frontier": _sha256_file(rate_frontier),
        },
        "headline": headline,
        "rows": rows,
        "sources": {
            "packet_trace_card": packet_payload.get("sources", {}),
            "rate_frontier": rate_payload.get("sources", {}),
            "rate_related_work": rate_payload.get("related_work", {}),
        },
        "non_claims": [
            "No native GPU/vLLM TTFT, TPOT, or goodput claim is made.",
            "KV/cache rows are byte-floor or related-work comparators, not implemented kernel baselines.",
            "PQ/OPQ/Hadamard rows are source-private residual-code methods on a controlled task, not broad latent reasoning.",
        ],
    }

    json_path = output_dir / "source_private_pq_systems_comparison_table.json"
    csv_path = output_dir / "source_private_pq_systems_comparison_table.csv"
    md_path = output_dir / "source_private_pq_systems_comparison_table.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
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
        "inputs": payload["inputs"],
        "input_sha256": payload["input_sha256"],
        "headline": headline,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Source-Private PQ Systems Comparison Table",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- PQ geometry rows: `{headline['pq_geometry_rows']}`",
        f"- PQ mitigation rows: `{headline['pq_mitigation_rows']}`",
        f"- PQ min delta vs best source-destroying control: `{headline['pq_min_delta_vs_best_control']:.3f}`",
        f"- canonical PQ max cached decode p50: `{headline['pq_max_cached_decode_p50_ms']:.4f} ms`",
        f"- protected Hadamard min unique payloads: `{headline['protected_hadamard_unique_payloads_min']}`",
        f"- frozen verifier min accuracy: `{headline['verifier_min_accuracy']:.3f}`",
        f"- frozen verifier max Mac CPU p50: `{headline['verifier_max_p50_latency_ms']:.1f} ms`",
        f"- same-byte text max accuracy: `{headline['same_byte_text_accuracy_max']:.3f}`",
        f"- query-aware text raw-byte ratio: `{headline['query_aware_text_raw_ratio']:.1f}x`",
        f"- full-log raw-byte ratio min: `{headline['full_log_raw_ratio_min']:.2f}x`",
        f"- KV raw-byte ratio min: `{headline['kv_raw_ratio_min']:.1f}x`",
        "",
        "## Compact Rows",
        "",
        "| Group | Method | Bytes | Accuracy | Target | Best control | Pass | Exposes text | Exposes KV | Paper use |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['row_group']} | {row['method']} | {_fmt(row['budget_bytes'])} | "
            f"{_fmt(row['accuracy_min'])}-{_fmt(row['accuracy_max'])} | {_fmt(row['target_accuracy'])} | "
            f"{_fmt(row['best_control_accuracy_max'])} | {row['pass_rule_result']} | "
            f"{_fmt(row['source_text_exposed'])} | {_fmt(row['source_kv_exposed'])} | {row['paper_use']} |"
        )
    lines.extend(
        [
            "",
            f"Claim boundary: {headline['claim_boundary']}",
            "",
            "## Non-Claims",
            "",
        ]
    )
    for claim in payload["non_claims"]:
        lines.append(f"- {claim}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq-gate", type=pathlib.Path, default=DEFAULT_PQ_GATE)
    parser.add_argument("--geometry-stress", type=pathlib.Path, default=DEFAULT_GEOMETRY_STRESS)
    parser.add_argument("--decode-frontier", type=pathlib.Path, default=DEFAULT_DECODE_FRONTIER)
    parser.add_argument("--verifier-trace", type=pathlib.Path, default=DEFAULT_VERIFIER_TRACE)
    parser.add_argument("--packet-trace-card", type=pathlib.Path, default=DEFAULT_PACKET_TRACE_CARD)
    parser.add_argument("--rate-frontier", type=pathlib.Path, default=DEFAULT_RATE_FRONTIER)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_pq_systems_comparison_table_20260430"),
    )
    args = parser.parse_args()
    payload = build_pq_systems_comparison_table(
        pq_gate=args.pq_gate,
        geometry_stress=args.geometry_stress,
        decode_frontier=args.decode_frontier,
        verifier_trace=args.verifier_trace,
        packet_trace_card=args.packet_trace_card,
        rate_frontier=args.rate_frontier,
        output_dir=_resolve(args.output_dir),
    )
    print(
        json.dumps(
            {
                "output_dir": str(_resolve(args.output_dir)),
                "headline": payload["headline"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
