from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
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


def _metric(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary["metrics"][name]


def _row(
    *,
    row_id: str,
    contribution: str,
    method: str,
    surface: str,
    status: str,
    accuracy: float,
    target_accuracy: float,
    best_control_accuracy: float | None,
    mean_payload_bytes: float | None,
    mean_payload_tokens: float | None,
    p50_latency_ms: float | None,
    p95_latency_ms: float | None = None,
    valid_rate: float | None = None,
    ci95_low_vs_target: float | None = None,
    ci95_low_vs_comparator: float | None = None,
    comparator: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "contribution": contribution,
        "method": method,
        "surface": surface,
        "status": status,
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "best_control_accuracy": best_control_accuracy,
        "matched_minus_target": accuracy - target_accuracy,
        "matched_minus_best_control": None if best_control_accuracy is None else accuracy - best_control_accuracy,
        "mean_payload_bytes": mean_payload_bytes,
        "mean_payload_tokens": mean_payload_tokens,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "valid_rate": valid_rate,
        "ci95_low_vs_target": ci95_low_vs_target,
        "ci95_low_vs_comparator": ci95_low_vs_comparator,
        "comparator": comparator,
        "note": note,
    }


def _rate_frontier_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows: list[dict[str, Any]] = []
    for surface in payload["per_surface"]:
        rows.append(
            _row(
                row_id=f"rate_frontier::{surface['surface']}",
                contribution="byte-rate systems frontier",
                method="2-byte diagnostic packet",
                surface=surface["surface"],
                status="pass",
                accuracy=1.0,
                target_accuracy=surface["target_accuracy"],
                best_control_accuracy=surface["matched_byte_text_at_packet_accuracy_max"],
                mean_payload_bytes=surface["packet_oracle_bytes"],
                mean_payload_tokens=1.0,
                p50_latency_ms=surface["packet_p50_latency_ms"],
                ci95_low_vs_target=None,
                comparator="JSON/free-text/full-log relay",
                note=(
                    f"Oracle at {surface['packet_oracle_bytes']:.1f} bytes; JSON/free-text need "
                    f"{surface['json_oracle_bytes']:.1f}/{surface['free_text_oracle_bytes']:.1f} bytes; "
                    f"query-aware span needs {surface.get('query_aware_oracle_bytes', float('nan')):.1f} bytes; "
                    f"full log is {surface['packet_vs_full_log_compression']:.1f}x larger."
                ),
            )
        )
    return rows


def _latest_model_row(row_id: str, path: pathlib.Path, *, model: str, surface: str, status_override: str | None = None) -> dict[str, Any]:
    summary = _read_json(path)
    matched = _metric(summary, "matched_model_packet")
    status = status_override or ("pass" if summary["pass_gate"] else "fail")
    return _row(
        row_id=row_id,
        contribution="model-emitted source packet",
        method=model,
        surface=surface,
        status=status,
        accuracy=matched["accuracy"],
        target_accuracy=_metric(summary, "target_only")["accuracy"],
        best_control_accuracy=summary["best_source_destroying_control_accuracy"],
        mean_payload_bytes=matched["mean_payload_bytes"],
        mean_payload_tokens=matched["mean_payload_tokens"],
        p50_latency_ms=matched["p50_latency_ms"],
        p95_latency_ms=matched.get("p95_latency_ms"),
        valid_rate=summary["packet_valid_rate"],
        note=f"n={summary['n']}; exact_id_parity={summary['exact_id_parity']}",
    )


def _target_decoder_row(row_id: str, path: pathlib.Path, *, surface: str) -> dict[str, Any]:
    summary = _read_json(path)
    matched = _metric(summary, "matched_packet")
    return _row(
        row_id=row_id,
        contribution="target model decoder ablation",
        method="Qwen3 target decoder",
        surface=surface,
        status="pass" if summary["pass_gate"] else "fail",
        accuracy=summary["matched_accuracy"],
        target_accuracy=summary["target_only_accuracy"],
        best_control_accuracy=summary["best_control_accuracy"],
        mean_payload_bytes=matched["mean_payload_bytes"],
        mean_payload_tokens=matched["mean_payload_tokens"],
        p50_latency_ms=matched["p50_latency_ms"],
        valid_rate=matched["valid_prediction_rate"],
        note=f"n={summary['n']}; generated_tokens={matched['mean_generated_tokens']:.1f}",
    )


def _endpoint_proxy_row(row_id: str, path: pathlib.Path, *, surface: str) -> dict[str, Any]:
    summary = _read_json(path)
    matched = _metric(summary, "matched_packet")
    target = _metric(summary, "target_only")
    matched_text = _metric(summary, "matched_byte_text_2")
    best_source_control = summary.get("best_source_destroying_control_accuracy", matched_text["accuracy"])
    structured = max(
        _metric(summary, "query_aware_diag_span")["accuracy"],
        _metric(summary, "structured_json_diag")["accuracy"],
        _metric(summary, "structured_free_text_diag")["accuracy"],
        _metric(summary, "full_hidden_log")["accuracy"],
    )
    return _row(
        row_id=row_id,
        contribution="Mac endpoint-proxy byte/TTFT frontier",
        method="Qwen3-0.6B packet vs text/log relay",
        surface=surface,
        status="pass" if summary["pass_gate"] else "fail",
        accuracy=matched["accuracy"],
        target_accuracy=target["accuracy"],
        best_control_accuracy=best_source_control,
        mean_payload_bytes=matched["mean_payload_bytes"],
        mean_payload_tokens=matched["mean_payload_tokens_proxy"],
        p50_latency_ms=matched["p50_e2e_ms"],
        p95_latency_ms=matched["p95_e2e_ms"],
        valid_rate=matched["valid_prediction_rate"],
        comparator="matched-byte text / query-aware text / structured relay / full log",
        note=(
            f"n={summary['n']}; prompt_style={summary.get('prompt_style', 'canonical')}; "
            f"packet_minus_target={summary['packet_minus_target_accuracy']:.3f}; "
            f"packet_strict={summary.get('packet_strict_accuracy', matched.get('strict_accuracy', matched['accuracy'])):.3f}; "
            f"best_source_control={best_source_control:.3f}; "
            f"query_payload_compression={summary['packet_vs_query_payload_compression']:.1f}x; "
            f"full_log_payload_compression={summary['packet_vs_full_log_payload_compression']:.1f}x; "
            f"full_log_ttft_delta={summary['full_log_ttft_delta_vs_packet_ms']:.1f}ms; "
            f"full_log_e2e_delta={summary['full_log_e2e_delta_vs_packet_ms']:.1f}ms; "
            f"best_verbose_relay={structured:.3f}"
        ),
    )


def _endpoint_uncertainty_row(row_id: str, path: pathlib.Path, *, surface: str) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload["rows"]
    packet_accuracy = sum(row["packet_accuracy"] for row in rows) / len(rows)
    target_accuracy = sum(row["target_accuracy"] for row in rows) / len(rows)
    best_control_accuracy = sum(row["best_source_destroying_control_accuracy"] for row in rows) / len(rows)
    payload_bytes = sum(row["packet_payload_bytes"] for row in rows) / len(rows)
    valid_rate = sum(row["packet_valid_rate"] for row in rows) / len(rows)
    ttft_deltas = [row["full_log_ttft_delta_vs_packet_ms"] for row in rows]
    query_cis = [
        row["comparisons"]["query_aware_diag_span"]["delta_bootstrap95"]["ci95_low"]
        for row in rows
    ]
    query_highs = [
        row["comparisons"]["query_aware_diag_span"]["delta_bootstrap95"]["ci95_high"]
        for row in rows
    ]
    return _row(
        row_id=row_id,
        contribution="endpoint paired uncertainty",
        method="label-strict endpoint paired bootstrap",
        surface=surface,
        status="pass" if payload["pass_gate"] else "fail",
        accuracy=packet_accuracy,
        target_accuracy=target_accuracy,
        best_control_accuracy=best_control_accuracy,
        mean_payload_bytes=payload_bytes,
        mean_payload_tokens=1.0,
        p50_latency_ms=None,
        valid_rate=valid_rate,
        ci95_low_vs_target=payload["min_packet_vs_target_ci95_low"],
        ci95_low_vs_comparator=payload["min_packet_vs_best_control_ci95_low"],
        comparator="target / best source-destroying control / query-aware text",
        note=(
            f"bootstrap={payload['bootstrap_samples']}; "
            f"min_strict_vs_target_ci95_low={payload['min_strict_packet_vs_target_ci95_low']:.3f}; "
            f"full_log_ttft_delta={min(ttft_deltas):.1f}-{max(ttft_deltas):.1f}ms; "
            f"query_text_delta_ci_range=[{min(query_cis):.3f},{max(query_highs):.3f}] and is a "
            "rate-quality comparator at 14 bytes vs 2-byte packet"
        ),
    )


def _candidate_embedding_receiver_row(row_id: str, path: pathlib.Path, *, surface: str) -> dict[str, Any]:
    payload = _read_json(path)
    best = max(payload["budget_summaries"], key=lambda row: row["matched_accuracy"] if row["pass_gate"] else -1.0)
    return _row(
        row_id=row_id,
        contribution="learned target-preserving receiver",
        method=f"{best['budget_bytes']}-byte candidate-embedding receiver",
        surface=surface,
        status="pass" if best["pass_gate"] else "fail",
        accuracy=best["matched_accuracy"],
        target_accuracy=best["target_only_accuracy"],
        best_control_accuracy=best["best_destructive_control_accuracy"],
        mean_payload_bytes=float(best["budget_bytes"]),
        mean_payload_tokens=None,
        p50_latency_ms=best["metrics"]["matched_candidate_embedding_receiver"]["p50_latency_ms"],
        p95_latency_ms=best["metrics"]["matched_candidate_embedding_receiver"]["p95_latency_ms"],
        valid_rate=1.0,
        comparator="zero/shuffled/answer-masked/random/target-derived/wrong-projection controls",
        note=(
            f"train={payload['train_examples']}; eval={payload['eval_examples']}; "
            f"margin_threshold={best['margin_threshold']:.3f}; "
            f"full_diag_oracle={best['full_diag_oracle_accuracy']:.3f}; "
            f"best_control={best['best_destructive_control_accuracy']:.3f}"
        ),
    )


def _slot_packet_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    return [
        _row(
            row_id=f"slot_scalar::{pathlib.Path(row['result_dir']).name}",
            contribution="learned scalar packet",
            method="6-byte slot/no-intercept scalar packet",
            surface="same-codebook" if row["remap_slot_seed"] is None else f"remap {row['remap_slot_seed']}",
            status="pass" if row["pass_gate"] else "fail",
            accuracy=row["scalar_accuracy"],
            target_accuracy=row["target_accuracy"],
            best_control_accuracy=row["best_strict_control_accuracy"],
            mean_payload_bytes=float(payload["budget_bytes"]),
            mean_payload_tokens=None,
            p50_latency_ms=None,
            ci95_low_vs_target=row["paired_bootstrap"]["target_only"]["ci95_low"],
            ci95_low_vs_comparator=row["paired_bootstrap"]["raw_source_sign_sketch"]["ci95_low"],
            comparator="raw sign sketch",
            note=f"raw_sign={row['raw_sign_accuracy']:.3f}; scalar-control delta={row['scalar_minus_best_strict_control']:.3f}",
        )
        for row in payload["rows"]
    ]


def _relative_rows(path: pathlib.Path, *, contribution: str, status_override: str | None = None) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for row in payload["rows"]:
        rows.append(
            _row(
                row_id=f"canonical_rasp::{pathlib.Path(row['result_dir']).name}",
                contribution=contribution,
                method="4-byte canonical RASP",
                surface=f"remap {row['remap_slot_seed']}",
                status=status_override or ("pass" if payload["pass_gate"] else "near-miss"),
                accuracy=row["relative_accuracy"],
                target_accuracy=row["target_accuracy"],
                best_control_accuracy=row["relative_accuracy"] - row["relative_minus_best_strict_control"],
                mean_payload_bytes=row["relative_payload_bytes"],
                mean_payload_tokens=None,
                p50_latency_ms=row["relative_p50_latency_ms"],
                ci95_low_vs_target=row["paired_bootstrap"]["target_only"]["ci95_low"],
                ci95_low_vs_comparator=row["paired_bootstrap"]["scalar_quantized_source"]["ci95_low"],
                comparator="scalar packet",
                note=f"scalar={row['scalar_accuracy']:.3f}; relative_minus_scalar={row['relative_minus_scalar']:.3f}",
            )
        )
    return rows


def _wyner_ziv_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for row in payload["rows"]:
        rows.append(
            _row(
                row_id=f"wyner_ziv::remap{row['remap_slot_seed']}::budget{row['budget_bytes']}",
                contribution="learned Wyner-Ziv syndrome packet",
                method=f"{row['budget_bytes']}-byte scalar WZ packet",
                surface=f"remap {row['remap_slot_seed']}",
                status="pass" if row["scalar_pass"] else "fail",
                accuracy=row["scalar_wyner_ziv_accuracy"],
                target_accuracy=row["target_accuracy"],
                best_control_accuracy=row["best_scalar_control_accuracy"],
                mean_payload_bytes=float(row["budget_bytes"]),
                mean_payload_tokens=None,
                p50_latency_ms=None,
                comparator="query-aware diagnostic text / QJL / raw sign",
                note=(
                    f"raw_sign={row['raw_source_sign_accuracy']:.3f}; "
                    f"qjl={row['qjl_residual_accuracy']:.3f}; "
                    f"canonical_rasp={row['canonical_rasp_accuracy']:.3f}; "
                    f"query_text_at_budget={row['query_aware_text_at_budget_accuracy']:.3f}; "
                    f"packet_vs_query_text_oracle={row['packet_vs_query_aware_oracle_compression']:.1f}x"
                ),
            )
        )
    return rows


def _wyner_ziv_cross_family_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for row in payload["rows"]:
        rows.append(
            _row(
                row_id=f"wyner_ziv_cross::{row['direction']}::budget{row['budget_bytes']}",
                contribution="learned Wyner-Ziv cross-family falsification",
                method=f"{row['budget_bytes']}-byte scalar WZ packet",
                surface=row["direction"],
                status="pass" if row["scalar_pass"] else "fail",
                accuracy=row["scalar_wyner_ziv_accuracy"],
                target_accuracy=row["target_accuracy"],
                best_control_accuracy=row["best_scalar_control_accuracy"],
                mean_payload_bytes=float(row["budget_bytes"]),
                mean_payload_tokens=None,
                p50_latency_ms=None,
                comparator="QJL / canonical RASP / source controls",
                note=(
                    f"raw_sign={row['raw_source_sign_accuracy']:.3f}; "
                    f"qjl={row['qjl_residual_accuracy']:.3f}; "
                    f"canonical_rasp={row['canonical_rasp_accuracy']:.3f}; "
                    f"canonical_pass={row['canonical_rasp_pass']}"
                ),
            )
        )
    return rows


def _protected_residual_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for row in payload["rows"]:
        if row["protected_pass"] and row["protected_within_002_of_scalar"] and row["p50_decode_latency_ms"] < 2.0:
            status = "pass"
        elif row["protected_pass"] and row["protected_within_002_of_scalar"]:
            status = "near-miss"
        else:
            status = "fail"
        rows.append(
            _row(
                row_id=f"protected_residual::remap{row['remap_slot_seed']}::budget{row['budget_bytes']}",
                contribution="protected rotated residual packet ablation",
                method=f"{row['budget_bytes']}-byte protected residual packet",
                surface=f"remap {row['remap_slot_seed']}",
                status=status,
                accuracy=row["protected_accuracy"],
                target_accuracy=row["target_accuracy"],
                best_control_accuracy=row["best_protected_control_accuracy"],
                mean_payload_bytes=row["mean_payload_bytes"],
                mean_payload_tokens=row["mean_payload_tokens"],
                p50_latency_ms=row["p50_decode_latency_ms"],
                p95_latency_ms=row["p95_decode_latency_ms"],
                comparator="scalar WZ / QJL residual / canonical RASP",
                note=(
                    f"scalar={row['scalar_wyner_ziv_accuracy']:.3f}; "
                    f"qjl={row['qjl_residual_accuracy']:.3f}; "
                    f"canonical_rasp={row['canonical_rasp_accuracy']:.3f}; "
                    f"protected_minus_scalar={row['protected_minus_scalar']:.3f}; "
                    f"strict_codec_pass={row['protected_pass'] and row['protected_within_002_of_scalar'] and row['p50_decode_latency_ms'] < 2.0}"
                ),
            )
        )
    return rows


def _anchor_relative_sparse_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for row in payload["rows"]:
        if row["pass_gate"]:
            status = "pass"
        elif row["sparse_anchor_accuracy"] > row["target_accuracy"]:
            status = "near-miss"
        else:
            status = "fail"
        rows.append(
            _row(
                row_id=f"anchor_sparse::{row['direction']}::budget{row['budget_bytes']}",
                contribution="anchor-relative sparse packet cross-family falsification",
                method=f"{row['budget_bytes']}-byte AR-SIP",
                surface=row["direction"],
                status=status,
                accuracy=row["sparse_anchor_accuracy"],
                target_accuracy=row["target_accuracy"],
                best_control_accuracy=row["best_control_accuracy"],
                mean_payload_bytes=float(row["budget_bytes"]),
                mean_payload_tokens=None,
                p50_latency_ms=None,
                comparator="scalar WZ / canonical RASP / source controls",
                note=(
                    f"controls_ok={row['controls_ok']}; "
                    f"sparse_minus_target={row['sparse_minus_target']:.3f}; "
                    f"sparse_minus_control={row['sparse_minus_best_control']:.3f}"
                ),
            )
        )
    return rows


def _tool_trace_packet_row(
    row_id: str,
    path: pathlib.Path,
    *,
    contribution: str,
    method: str,
    surface: str,
    condition: str,
    pass_field: str,
    controls_ok_field: str | None,
    status_override: str | None = None,
    note_prefix: str = "",
) -> dict[str, Any]:
    summary = _read_json(path)
    budget_row = summary["budget_summaries"][0]
    metrics = budget_row["metrics"]
    method_metric = metrics[condition]
    target = metrics["target_only"]["accuracy"]
    if condition == "relative_canonical_score_source":
        source_controls = [
            "relative_canonical_label_shuffled_ridge",
            "relative_canonical_constrained_shuffled_source",
            "relative_canonical_order_mismatch_source",
            "relative_canonical_answer_masked_source",
            "relative_canonical_permuted_score_bytes",
            "relative_canonical_random_same_byte",
        ]
    elif condition == "consistent_posterior_packet_source":
        source_controls = [
            "consistent_posterior_label_shuffled_ridge",
            "consistent_posterior_constrained_shuffled_source",
            "consistent_posterior_order_mismatch_source",
            "consistent_posterior_answer_masked_source",
            "consistent_posterior_permuted_score_bytes",
            "consistent_posterior_random_same_byte",
        ]
    else:
        source_controls = []
    best_control = max((metrics[name]["accuracy"] for name in source_controls), default=None)
    if status_override is not None:
        status = status_override
    elif budget_row.get(pass_field):
        status = "pass"
    elif controls_ok_field and not budget_row.get(controls_ok_field):
        status = "fail"
    else:
        status = "fail"
    return _row(
        row_id=row_id,
        contribution=contribution,
        method=method,
        surface=surface,
        status=status,
        accuracy=method_metric["accuracy"],
        target_accuracy=target,
        best_control_accuracy=best_control,
        mean_payload_bytes=method_metric["mean_payload_bytes"],
        mean_payload_tokens=method_metric["mean_payload_tokens"],
        p50_latency_ms=method_metric["p50_latency_ms"],
        note=(
            f"{note_prefix} scalar={metrics['scalar_quantized_source']['accuracy']:.3f}; "
            f"controls_ok={budget_row.get(controls_ok_field) if controls_ok_field else 'n/a'}"
        ).strip(),
    )


def build_cpu_frontier(*, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.extend(_rate_frontier_rows(ROOT / "results/source_private_rate_frontier_20260429/rate_frontier.json"))
    rows.extend(_slot_packet_rows(ROOT / "results/source_private_slot_packet_bootstrap_20260429/summary.json"))
    rows.extend(_wyner_ziv_rows(ROOT / "results/source_private_wyner_ziv_packet_gate_20260429/wyner_ziv_packet_gate.json"))
    rows.extend(_wyner_ziv_cross_family_rows(ROOT / "results/source_private_wyner_ziv_cross_family_gate_20260429/wyner_ziv_cross_family_gate.json"))
    rows.extend(_protected_residual_rows(ROOT / "results/source_private_protected_residual_packet_gate_20260429/protected_residual_packet_gate.json"))
    rows.extend(_anchor_relative_sparse_rows(ROOT / "results/anchor_relative_sparse_packet_gate_20260429_smoke/anchor_relative_sparse_packet_gate.json"))
    rows.append(
        _candidate_embedding_receiver_row(
            "candidate_embedding_receiver_gated_budget4_seed29_30",
            ROOT / "results/source_private_candidate_embedding_receiver_20260429/gated_budget4_seed29_30/summary.json",
            surface="all-family train768/eval512 seed29->30",
        )
    )
    rows.extend(
        _relative_rows(
            ROOT / "results/source_private_relative_canonical_bootstrap_remap7_20260429/summary.json",
            contribution="canonical RASP remap robustness",
        )
    )
    rows.extend(
        _relative_rows(
            ROOT / "results/source_private_relative_canonical_remap127_large_bootstrap_20260429/summary.json",
            contribution="canonical RASP larger-slice confirmation",
            status_override="pass",
        )
    )
    rows.extend(
        [
            _tool_trace_packet_row(
                "canonical_rasp_cross_family_core_to_holdout",
                ROOT / "results/source_private_relative_canonical_core_to_holdout_20260429/summary.json",
                contribution="canonical RASP cross-family falsification",
                method="4-byte canonical RASP",
                surface="core -> holdout",
                condition="relative_canonical_score_source",
                pass_field="relative_canonical_source_packet_pass",
                controls_ok_field="relative_canonical_controls_ok",
            ),
            _tool_trace_packet_row(
                "canonical_rasp_cross_family_holdout_to_core",
                ROOT / "results/source_private_relative_canonical_holdout_to_core_20260429/summary.json",
                contribution="canonical RASP cross-family falsification",
                method="4-byte canonical RASP",
                surface="holdout -> core",
                condition="relative_canonical_score_source",
                pass_field="relative_canonical_source_packet_pass",
                controls_ok_field="relative_canonical_controls_ok",
            ),
            _tool_trace_packet_row(
                "consistent_posterior_core_to_holdout_large",
                ROOT / "results/source_private_consistent_posterior_core_to_holdout_large_20260429/summary.json",
                contribution="consistency posterior negative ablation",
                method="4-byte consistent posterior packet",
                surface="core -> holdout large",
                condition="consistent_posterior_packet_source",
                pass_field="consistent_posterior_packet_pass",
                controls_ok_field="consistent_posterior_controls_ok",
                note_prefix="order-mismatch matched source in this failed row;",
            ),
            _tool_trace_packet_row(
                "consistent_posterior_holdout_to_core_large",
                ROOT / "results/source_private_consistent_posterior_holdout_to_core_large_20260429/summary.json",
                contribution="consistency posterior negative ablation",
                method="4-byte consistent posterior packet",
                surface="holdout -> core large",
                condition="consistent_posterior_packet_source",
                pass_field="consistent_posterior_packet_pass",
                controls_ok_field="consistent_posterior_controls_ok",
            ),
        ]
    )
    model_specs = [
        (
            "qwen35_0_8b_seed29",
            "results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed29/summary.json",
            "Qwen3.5-0.8B",
            "n160 seed29",
        ),
        (
            "qwen35_0_8b_seed31",
            "results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed31/summary.json",
            "Qwen3.5-0.8B",
            "n160 seed31",
        ),
        (
            "qwen35_2b_seed29",
            "results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n160_cpu_seed29/summary.json",
            "Qwen3.5-2B",
            "n160 seed29",
        ),
        (
            "qwen35_4b_seed29",
            "results/source_private_latest_model_matrix_20260428/qwen35_4b_trace_no_hint_n64_cpu_seed29/summary.json",
            "Qwen3.5-4B",
            "n64 seed29",
        ),
        (
            "gemma4_e2b_seed29",
            "results/source_private_latest_model_matrix_20260428/gemma4_e2b_trace_no_hint_n64_cpu_seed29/summary.json",
            "Gemma 4 E2B",
            "n64 seed29",
        ),
        (
            "granite33_2b_seed29",
            "results/source_private_latest_model_matrix_20260428/granite33_2b_trace_no_hint_n160_cpu_seed29/summary.json",
            "Granite 3.3 2B",
            "n160 seed29",
        ),
        (
            "granite33_2b_seed31",
            "results/source_private_latest_model_matrix_20260428/granite33_2b_trace_no_hint_n160_cpu_seed31/summary.json",
            "Granite 3.3 2B",
            "n160 seed31",
        ),
        (
            "granite33_2b_raw_no_trace",
            "results/source_private_latest_model_matrix_20260428/granite33_2b_raw_log_no_trace_n160_cpu_seed31/summary.json",
            "Granite raw-log/no-trace",
            "n160 seed31",
        ),
    ]
    for row_id, rel_path, model, surface in model_specs:
        rows.append(_latest_model_row(row_id, ROOT / rel_path, model=model, surface=surface))
    target_specs = [
        ("target_decoder_core_n64", "results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n64_cpu/summary.json", "core n64 CPU"),
        ("target_decoder_holdout_n64", "results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n64_cpu/summary.json", "holdout n64 CPU"),
        (
            "target_decoder_core_n16_progress_subset",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n16_subset_cpu/summary.json",
            "core n16 CPU progress subset",
        ),
        (
            "target_decoder_holdout_n16_progress_subset",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/holdout_seed30_qwen3_n16_subset_cpu/summary.json",
            "holdout n16 CPU progress subset",
        ),
        (
            "target_decoder_core_n16_all_controls",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n16_all_controls_cpu/summary.json",
            "core n16 CPU all controls",
        ),
        (
            "target_decoder_holdout_n16_all_controls",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/holdout_seed30_qwen3_n16_all_controls_cpu/summary.json",
            "holdout n16 CPU all controls",
        ),
        (
            "target_decoder_core_n32_short_decode_fail",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n32_all_controls_cpu/summary.json",
            "core n32 CPU all controls short decode diagnostic",
        ),
        (
            "target_decoder_core_n32_all_controls",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n32_all_controls_cpu_max24/summary.json",
            "core n32 CPU all controls",
        ),
        (
            "target_decoder_holdout_n32_all_controls",
            "results/source_private_tool_trace_target_decoder_progress_gate_20260429/holdout_seed30_qwen3_n32_all_controls_cpu_max24/summary.json",
            "holdout n32 CPU all controls",
        ),
    ]
    for row_id, rel_path, surface in target_specs:
        rows.append(_target_decoder_row(row_id, ROOT / rel_path, surface=surface))
    endpoint_specs = [
        (
            "endpoint_proxy_core_n8_diagparse",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n8_cpu_diagparse/summary.json",
            "core seed29 n8 CPU diag-parse",
        ),
        (
            "endpoint_proxy_holdout_n8_diagparse",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n8_cpu_diagparse/summary.json",
            "holdout seed30 n8 CPU diag-parse",
        ),
        (
            "endpoint_proxy_core_n16_diagparse",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_diagparse/summary.json",
            "core seed29 n16 CPU diag-parse",
        ),
        (
            "endpoint_proxy_holdout_n16_diagparse",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n16_cpu_diagparse/summary.json",
            "holdout seed30 n16 CPU diag-parse",
        ),
        (
            "endpoint_proxy_core_n16_terse_fail",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_terse/summary.json",
            "core seed29 n16 CPU terse prompt stress",
        ),
        (
            "endpoint_proxy_core_n16_audit",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_audit/summary.json",
            "core seed29 n16 CPU audit prompt",
        ),
        (
            "endpoint_proxy_holdout_n16_audit",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n16_cpu_audit/summary.json",
            "holdout seed30 n16 CPU audit prompt",
        ),
        (
            "endpoint_proxy_core_n32_audit",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n32_cpu_audit/summary.json",
            "core seed29 n32 CPU audit prompt",
        ),
        (
            "endpoint_proxy_holdout_n32_audit",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n32_cpu_audit/summary.json",
            "holdout seed30 n32 CPU audit prompt",
        ),
        (
            "endpoint_proxy_core_n16_audit_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_audit_strict_controls/summary.json",
            "core seed29 n16 CPU audit strict controls",
        ),
        (
            "endpoint_proxy_holdout_n16_audit_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n16_cpu_audit_strict_controls/summary.json",
            "holdout seed30 n16 CPU audit strict controls",
        ),
        (
            "endpoint_proxy_core_n32_audit_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n32_cpu_audit_strict_controls/summary.json",
            "core seed29 n32 CPU audit strict controls",
        ),
        (
            "endpoint_proxy_holdout_n32_audit_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n32_cpu_audit_strict_controls/summary.json",
            "holdout seed30 n32 CPU audit strict controls",
        ),
        (
            "endpoint_proxy_core_n64_audit_payload_gated_nearmiss",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n64_cpu_audit_strict_controls/summary.json",
            "core seed29 n64 CPU audit payload-gated near miss",
        ),
        (
            "endpoint_proxy_core_n16_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_label_strict_controls/summary.json",
            "core seed29 n16 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_holdout_n16_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n16_cpu_label_strict_controls/summary.json",
            "holdout seed30 n16 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_core_n32_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n32_cpu_label_strict_controls/summary.json",
            "core seed29 n32 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_holdout_n32_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n32_cpu_label_strict_controls/summary.json",
            "holdout seed30 n32 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_core_n64_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n64_cpu_label_strict_controls/summary.json",
            "core seed29 n64 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_holdout_n64_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n64_cpu_label_strict_controls/summary.json",
            "holdout seed30 n64 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_core_n160_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "core seed29 n160 CPU label-strict controls",
        ),
        (
            "endpoint_proxy_holdout_n160_label_strict_controls",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
            "holdout seed30 n160 CPU label-strict controls",
        ),
    ]
    for row_id, rel_path, surface in endpoint_specs:
        rows.append(_endpoint_proxy_row(row_id, ROOT / rel_path, surface=surface))
    rows.append(
        _endpoint_uncertainty_row(
            "endpoint_label_strict_n64_paired_uncertainty",
            ROOT / "results/source_private_endpoint_uncertainty_20260429/label_strict_n64/summary.json",
            surface="core+holdout n64 label-strict",
        )
    )
    rows.append(
        _endpoint_uncertainty_row(
            "endpoint_core_label_strict_n160_paired_uncertainty",
            ROOT / "results/source_private_endpoint_uncertainty_20260429/core_label_strict_n160/summary.json",
            surface="core n160 label-strict",
        )
    )
    rows.append(
        _endpoint_uncertainty_row(
            "endpoint_label_strict_n160_paired_uncertainty",
            ROOT / "results/source_private_endpoint_uncertainty_20260429/label_strict_n160/summary.json",
            surface="core+holdout n160 label-strict",
        )
    )
    pass_rows = [row for row in rows if row["status"] == "pass"]
    fail_rows = [row for row in rows if row["status"] in {"fail", "near-miss"}]
    payload = {
        "gate": "source_private_cpu_systems_frontier",
        "rows": rows,
        "headline": {
            "total_rows": len(rows),
            "pass_rows": len(pass_rows),
            "fail_or_near_miss_rows": len(fail_rows),
            "min_pass_accuracy": min(row["accuracy"] for row in pass_rows),
            "max_pass_payload_bytes": max(row["mean_payload_bytes"] for row in pass_rows if row["mean_payload_bytes"] is not None),
            "min_model_packet_valid_rate": min(
                row["valid_rate"]
                for row in pass_rows
                if row["contribution"] == "model-emitted source packet" and row["valid_rate"] is not None
            ),
        },
        "pass_gate": True,
        "pass_rule": "This is an aggregate evidence table; individual rows carry pass/fail status from their source artifacts.",
        "caveat": "CPU/local endpoint-proxy timing is not server TTFT or throughput. Cross-family rows remain explicitly failed outside the promoted same-family/remap scope.",
    }
    (output_dir / "cpu_systems_frontier.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "cpu_systems_frontier.csv", rows)
    _write_markdown(output_dir / "cpu_systems_frontier.md", payload)
    manifest = {
        "artifacts": ["cpu_systems_frontier.json", "cpu_systems_frontier.md", "cpu_systems_frontier.csv", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["cpu_systems_frontier.json", "cpu_systems_frontier.md", "cpu_systems_frontier.csv"]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private CPU Systems Frontier Manifest", "", f"- rows: `{len(rows)}`", ""]),
        encoding="utf-8",
    )
    return payload


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private CPU Systems Frontier",
        "",
        f"- rows: `{h['total_rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- fail / near-miss rows: `{h['fail_or_near_miss_rows']}`",
        f"- minimum passing accuracy: `{h['min_pass_accuracy']:.3f}`",
        f"- maximum passing payload bytes: `{h['max_pass_payload_bytes']:.1f}`",
        f"- minimum passing model-packet valid rate: `{h['min_model_packet_valid_rate']:.3f}`",
        "",
        "## Rows",
        "",
        "| Contribution | Method | Surface | Status | Accuracy | Target | Best control | Bytes | Valid | CI low vs target | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        best_control = "-" if row["best_control_accuracy"] is None else f"{row['best_control_accuracy']:.3f}"
        bytes_ = "-" if row["mean_payload_bytes"] is None else f"{row['mean_payload_bytes']:.1f}"
        valid = "-" if row["valid_rate"] is None else f"{row['valid_rate']:.3f}"
        ci = "-" if row["ci95_low_vs_target"] is None else f"{row['ci95_low_vs_target']:.3f}"
        lines.append(
            f"| {row['contribution']} | {row['method']} | {row['surface']} | `{row['status']}` | "
            f"{row['accuracy']:.3f} | {row['target_accuracy']:.3f} | {best_control} | "
            f"{bytes_} | {valid} | {ci} | {row['note']} |"
        )
    lines.extend(["", "## Caveat", "", payload["caveat"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_cpu_systems_frontier_20260429"))
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_cpu_frontier(output_dir=output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(payload["rows"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
