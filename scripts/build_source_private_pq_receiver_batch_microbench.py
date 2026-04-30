from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_source_private_product_codebook_geometry_gate import (  # noqa: E402
    _decode_geometry_packet,
    _dimension_utilities,
    _fit_geometry_codebook,
    _geometry_packet,
    _groups_for_variant,
)
from scripts.build_source_private_product_codebook_geometry_knockout_stress import (  # noqa: E402
    _decode_table,
    _geometry_distance_tables,
)
from scripts.run_source_private_hidden_repair_packet_smoke import make_benchmark  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _fit_ridge_encoder_for_view,
    _prior_prediction,
    _remap_candidate_slots,
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(values)
    return {
        "p50_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]),
        "p99_ms": float(ordered[min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))]),
        "mean_ms": float(statistics.fmean(ordered)),
    }


def _cacheline_bytes(default: int = 128) -> int:
    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], text=True).strip()
        value = int(output)
        return value if value > 0 else default
    except (OSError, subprocess.CalledProcessError, ValueError):
        return default


def _amortized_burst_bytes(*, payload_bytes: int, batch_size: int, burst_bytes: int) -> float:
    total = payload_bytes * batch_size
    return float(math.ceil(total / burst_bytes) * burst_bytes / batch_size)


def _sha256_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _distance_table_bytes(distance_tables: tuple[np.ndarray, ...]) -> int:
    return int(sum(table.nbytes for table in distance_tables))


def _decode_batch_choices(
    *,
    batch_tables: list[tuple[np.ndarray, ...]],
    batch_payloads: list[bytes],
    batch_prior_indices: list[int],
) -> list[int]:
    if not batch_payloads:
        return []
    subspaces = len(batch_tables[0])
    batch = len(batch_payloads)
    scores = np.zeros((batch, batch_tables[0][0].shape[0]), dtype=np.float32)
    codes = np.stack(
        [np.frombuffer(payload[:subspaces], dtype=np.uint8).astype(np.int64) for payload in batch_payloads],
        axis=0,
    )
    for subspace_index in range(subspaces):
        tables = np.stack([tables_for_example[subspace_index] for tables_for_example in batch_tables], axis=0)
        code_indices = codes[:, subspace_index] % tables.shape[2]
        scores += tables[np.arange(batch), :, code_indices]
    choices: list[int] = []
    for row_index, row_scores in enumerate(scores):
        min_score = float(np.min(row_scores))
        tied = np.flatnonzero(np.isclose(row_scores, min_score, rtol=1e-6, atol=1e-8))
        prior_index = int(batch_prior_indices[row_index])
        choices.append(prior_index if prior_index in tied else int(tied[0]))
    return choices


def _prior_index(example: Any) -> int:
    prior = _prior_prediction(example)
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == prior)


def _label_for_index(example: Any, index: int) -> str:
    return example.candidates[int(index)].label


def _batch_latency_rows(
    *,
    eval_rows: list[Any],
    payloads: list[bytes],
    distance_tables: list[tuple[np.ndarray, ...]],
    canonical_predictions: list[str],
    batch_sizes: list[int],
    repeats: int,
    payload_bytes: int,
    packet_record_bytes: int,
    burst_bytes: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    prior_indices = [_prior_index(example) for example in eval_rows]
    exact_id_sha256 = _sha256_strings([example.example_id for example in eval_rows])
    for batch_size in batch_sizes:
        per_request_latencies: list[float] = []
        batch_mismatches: list[str] = []
        first_pass_predictions: list[str] = []
        for repeat_index in range(repeats):
            for start_index in range(0, len(eval_rows), batch_size):
                end_index = min(len(eval_rows), start_index + batch_size)
                batch_tables = distance_tables[start_index:end_index]
                batch_payloads = payloads[start_index:end_index]
                batch_priors = prior_indices[start_index:end_index]
                start = time.perf_counter()
                choices = _decode_batch_choices(
                    batch_tables=batch_tables,
                    batch_payloads=batch_payloads,
                    batch_prior_indices=batch_priors,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                per_request_latencies.append(elapsed_ms / max(1, len(batch_payloads)))
                if repeat_index == 0:
                    labels = [
                        _label_for_index(example, choice)
                        for example, choice in zip(eval_rows[start_index:end_index], choices, strict=True)
                    ]
                    first_pass_predictions.extend(labels)
        for example, canonical_prediction, batch_prediction in zip(
            eval_rows, canonical_predictions, first_pass_predictions, strict=True
        ):
            if canonical_prediction != batch_prediction:
                batch_mismatches.append(example.example_id)
        summary = _latency_summary(per_request_latencies)
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "per_request_p50_ms": summary["p50_ms"],
            "per_request_p95_ms": summary["p95_ms"],
            "per_request_p99_ms": summary["p99_ms"],
            "per_request_mean_ms": summary["mean_ms"],
            "prediction_mismatch_count": len(batch_mismatches),
            "prediction_mismatch_examples": batch_mismatches[:10],
            "prediction_sha256": _sha256_strings(first_pass_predictions),
            "exact_id_sha256": exact_id_sha256,
            "amortized_raw_payload_observed_cacheline_bytes_per_request": _amortized_burst_bytes(
                payload_bytes=payload_bytes,
                batch_size=batch_size,
                burst_bytes=burst_bytes,
            ),
            "amortized_raw_payload_64b_line_bytes_per_request": _amortized_burst_bytes(
                payload_bytes=payload_bytes,
                batch_size=batch_size,
                burst_bytes=64,
            ),
            "amortized_raw_payload_128b_burst_bytes_per_request": _amortized_burst_bytes(
                payload_bytes=payload_bytes,
                batch_size=batch_size,
                burst_bytes=128,
            ),
            "amortized_packet_record_observed_cacheline_bytes_per_request": _amortized_burst_bytes(
                payload_bytes=packet_record_bytes,
                batch_size=batch_size,
                burst_bytes=burst_bytes,
            ),
            "amortized_packet_record_64b_line_bytes_per_request": _amortized_burst_bytes(
                payload_bytes=packet_record_bytes,
                batch_size=batch_size,
                burst_bytes=64,
            ),
            "amortized_packet_record_128b_burst_bytes_per_request": _amortized_burst_bytes(
                payload_bytes=packet_record_bytes,
                batch_size=batch_size,
                burst_bytes=128,
            ),
        }
    return results


def build_pq_receiver_batch_microbench(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    remap_seeds: list[int],
    budget_bytes: int,
    variants: list[str],
    feature_dim: int,
    candidate_view: str,
    ridge: float,
    opq_iterations: int,
    table_repeats: int,
    batch_repeats: int,
    batch_sizes: list[int],
    packet_record_overhead_bytes: int = 3,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cacheline = _cacheline_bytes()
    packet_record_bytes = budget_bytes + packet_record_overhead_bytes
    rows: list[dict[str, Any]] = []
    for remap_seed in remap_seeds:
        train_rows = make_benchmark(examples=train_examples, candidates=4, seed=train_seed, family_set="all")
        eval_rows = make_benchmark(examples=eval_examples, candidates=4, seed=eval_seed, family_set="all")
        train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_seed)
        eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_seed)
        encoder = _fit_ridge_encoder_for_view(
            train_rows,
            feature_dim=feature_dim,
            ridge=ridge,
            candidate_view=candidate_view,
            fit_intercept=False,
        )
        utilities = _dimension_utilities(
            train_rows,
            encoder=encoder,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
        )
        candidate_matrices = [
            _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
            for example in eval_rows
        ]
        for variant in variants:
            groups = _groups_for_variant(
                variant=variant,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                utilities=utilities,
                seed=train_seed * 11003 + eval_seed * 97 + budget_bytes + remap_seed,
            )
            fit_start = time.perf_counter()
            codebook = _fit_geometry_codebook(
                train_rows,
                encoder=encoder,
                feature_dim=feature_dim,
                groups=groups,
                variant=variant,
                utilities=utilities,
                seed=train_seed * 9001 + eval_seed * 17 + budget_bytes,
                opq_iterations=opq_iterations if variant in {"opq_procrustes", "utility_opq_procrustes"} else 0,
            )
            fit_ms = (time.perf_counter() - fit_start) * 1000.0

            payloads: list[bytes] = []
            packet_encode_latencies: list[float] = []
            for example in eval_rows:
                start = time.perf_counter()
                payload = _geometry_packet(
                    example,
                    encoder=encoder,
                    codebook=codebook,
                    feature_dim=feature_dim,
                    mode="matched",
                )
                packet_encode_latencies.append((time.perf_counter() - start) * 1000.0)
                payloads.append(payload)

            table_build_latencies: list[float] = []
            distance_tables: list[tuple[np.ndarray, ...]] = []
            for candidate_matrix in candidate_matrices:
                start = time.perf_counter()
                tables = _geometry_distance_tables(candidate_matrix, codebook=codebook)
                table_build_latencies.append((time.perf_counter() - start) * 1000.0)
                distance_tables.append(tables)

            canonical_predictions: list[str] = []
            table_predictions: list[str] = []
            canonical_latencies: list[float] = []
            resident_table_latencies: list[float] = []
            for _ in range(table_repeats):
                for example, payload, tables in zip(eval_rows, payloads, distance_tables, strict=True):
                    start = time.perf_counter()
                    canonical_prediction, _ = _decode_geometry_packet(
                        example,
                        payload,
                        codebook=codebook,
                        feature_dim=feature_dim,
                        candidate_view=candidate_view,
                    )
                    canonical_latencies.append((time.perf_counter() - start) * 1000.0)
                    start = time.perf_counter()
                    table_prediction = _decode_table(example, payload, distance_tables=tables)
                    resident_table_latencies.append((time.perf_counter() - start) * 1000.0)
                    if len(canonical_predictions) < len(eval_rows):
                        canonical_predictions.append(canonical_prediction)
                        table_predictions.append(table_prediction)

            table_mismatches = [
                example.example_id
                for example, canonical_prediction, table_prediction in zip(
                    eval_rows, canonical_predictions, table_predictions, strict=True
                )
                if canonical_prediction != table_prediction
            ]
            correct = sum(
                1
                for example, prediction in zip(eval_rows, canonical_predictions, strict=True)
                if prediction == example.answer_label
            )
            batch_results = _batch_latency_rows(
                eval_rows=eval_rows,
                payloads=payloads,
                distance_tables=distance_tables,
                canonical_predictions=canonical_predictions,
                batch_sizes=batch_sizes,
                repeats=batch_repeats,
                payload_bytes=budget_bytes,
                packet_record_bytes=packet_record_bytes,
                burst_bytes=cacheline,
            )
            max_batch_mismatches = max(
                (entry["prediction_mismatch_count"] for entry in batch_results.values()),
                default=0,
            )
            batch64 = batch_results.get("64") or batch_results[str(max(batch_sizes))]
            batch_prediction_hashes = {entry["prediction_sha256"] for entry in batch_results.values()}
            table_summary = _latency_summary(resident_table_latencies)
            canonical_summary = _latency_summary(canonical_latencies)
            max_batch_p95 = max((entry["per_request_p95_ms"] for entry in batch_results.values()), default=0.0)
            row_pass = (
                not table_mismatches
                and max_batch_mismatches == 0
                and len(batch_prediction_hashes) == 1
                and table_summary["p50_ms"] < 0.25
                and max_batch_p95 < 0.25
                and batch_results[str(max(batch_sizes))][
                    "amortized_packet_record_128b_burst_bytes_per_request"
                ]
                <= packet_record_bytes
            )
            rows.append(
                {
                    "remap_slot_seed": remap_seed,
                    "variant": variant,
                    "budget_bytes": budget_bytes,
                    "n": eval_examples,
                    "accuracy": correct / eval_examples,
                    "codebook_fit_ms": fit_ms,
                    "packet_encode_p50_ms": _latency_summary(packet_encode_latencies)["p50_ms"],
                    "packet_encode_p95_ms": _latency_summary(packet_encode_latencies)["p95_ms"],
                    "public_distance_table_build_p50_ms": _latency_summary(table_build_latencies)["p50_ms"],
                    "public_distance_table_build_p95_ms": _latency_summary(table_build_latencies)["p95_ms"],
                    "canonical_vector_decode_p50_ms": canonical_summary["p50_ms"],
                    "canonical_vector_decode_p95_ms": canonical_summary["p95_ms"],
                    "resident_table_decode_p50_ms": table_summary["p50_ms"],
                    "resident_table_decode_p95_ms": table_summary["p95_ms"],
                    "resident_distance_table_bytes_per_example": _distance_table_bytes(distance_tables[0]),
                    "payload_bytes_per_request": budget_bytes,
                    "packet_record_bytes_per_request": packet_record_bytes,
                    "packet_record_overhead_bytes": packet_record_overhead_bytes,
                    "cacheline_bytes": cacheline,
                    "rotation_matrix_bytes": 0 if codebook.rotation is None else int(codebook.rotation.nbytes),
                    "runtime_rotation_in_resident_lookup": False,
                    "exact_id_sha256": _sha256_strings([example.example_id for example in eval_rows]),
                    "reference_prediction_sha256": _sha256_strings(canonical_predictions),
                    "batch_prediction_sha256_values": sorted(batch_prediction_hashes),
                    "batch_size_invariant": len(batch_prediction_hashes) == 1,
                    "table_prediction_mismatch_count": len(table_mismatches),
                    "table_prediction_mismatch_examples": table_mismatches[:10],
                    "max_batch_prediction_mismatch_count": max_batch_mismatches,
                    "max_batch_per_request_p95_ms": max_batch_p95,
                    "batch_results": batch_results,
                    "batch64_speedup_vs_scalar_table_p50": table_summary["p50_ms"]
                    / max(batch64["per_request_p50_ms"], 1e-12),
                    "batch256_amortized_128b_raw_payload_bytes_per_request": batch_results[str(max(batch_sizes))][
                        "amortized_raw_payload_128b_burst_bytes_per_request"
                    ],
                    "batch256_amortized_128b_packet_record_bytes_per_request": batch_results[str(max(batch_sizes))][
                        "amortized_packet_record_128b_burst_bytes_per_request"
                    ],
                    "batch_microbench_pass": row_pass,
                }
            )

    pass_rows = [row for row in rows if row["batch_microbench_pass"]]
    required = {(remap_seed, variant) for remap_seed in remap_seeds for variant in variants}
    passed = {(row["remap_slot_seed"], row["variant"]) for row in pass_rows}
    headline = {
        "rows": len(rows),
        "pass_rows": len(pass_rows),
        "remap_seeds": remap_seeds,
        "variants": variants,
        "batch_sizes": batch_sizes,
        "max_table_prediction_mismatch_count": max((row["table_prediction_mismatch_count"] for row in rows), default=0),
        "max_batch_prediction_mismatch_count": max((row["max_batch_prediction_mismatch_count"] for row in rows), default=0),
        "max_resident_table_decode_p50_ms": max((row["resident_table_decode_p50_ms"] for row in rows), default=None),
        "max_batch64_per_request_p50_ms": max(
            (row["batch_results"].get("64", row["batch_results"][str(max(batch_sizes))])["per_request_p50_ms"] for row in rows),
            default=None,
        ),
        "min_batch64_speedup_vs_scalar_table_p50": min(
            (row["batch64_speedup_vs_scalar_table_p50"] for row in rows),
            default=None,
        ),
        "batch256_amortized_128b_raw_payload_bytes_per_request": min(
            (row["batch256_amortized_128b_raw_payload_bytes_per_request"] for row in rows),
            default=None,
        ),
        "batch256_amortized_128b_packet_record_bytes_per_request": min(
            (row["batch256_amortized_128b_packet_record_bytes_per_request"] for row in rows),
            default=None,
        ),
        "packet_record_bytes_per_request": packet_record_bytes,
        "payload_bytes_per_request": budget_bytes,
        "max_public_distance_table_build_p50_ms": max(
            (row["public_distance_table_build_p50_ms"] for row in rows),
            default=None,
        ),
    }
    payload = {
        "gate": "source_private_pq_receiver_batch_microbench",
        "rows": rows,
        "headline": headline,
        "pass_gate": passed == required,
        "pass_rule": (
            "Every remap/variant row must exactly match the canonical geometry decoder under resident table lookup and "
            "all batch kernels; predictions must be invariant across batch sizes; resident table p50 and every batch "
            "p95 must stay below 0.25 ms/request; and the largest batch must amortize 128B packet-record traffic down "
            "to the packet record byte count per request. Codebook fit and public table build costs are reported "
            "separately and are not claimed as per-token model speedups."
        ),
        "interpretation": (
            "This is a receiver-kernel systems gate for geometry-mitigated source-private PQ packets. It measures the "
            "target-side operation after public candidate state has been cached: summing PQ distance-table entries for "
            "a few source-private byte indices. It supports a boundary-traffic and batching claim, not an end-to-end "
            "vLLM/GPU serving claim."
        ),
        "inputs": {
            "train_examples": train_examples,
            "eval_examples": eval_examples,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "budget_bytes": budget_bytes,
            "feature_dim": feature_dim,
            "candidate_view": candidate_view,
            "ridge": ridge,
            "opq_iterations": opq_iterations,
            "table_repeats": table_repeats,
            "batch_repeats": batch_repeats,
            "packet_record_overhead_bytes": packet_record_overhead_bytes,
        },
    }
    json_path = output_dir / "pq_receiver_batch_microbench.json"
    md_path = output_dir / "pq_receiver_batch_microbench.md"
    csv_path = output_dir / "pq_receiver_batch_microbench.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows, batch_sizes)
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [
            "pq_receiver_batch_microbench.json",
            "pq_receiver_batch_microbench.md",
            "pq_receiver_batch_microbench.csv",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "pq_receiver_batch_microbench.json": _sha256_file(json_path),
            "pq_receiver_batch_microbench.md": _sha256_file(md_path),
            "pq_receiver_batch_microbench.csv": _sha256_file(csv_path),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private PQ Receiver Batch Microbench Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], batch_sizes: list[int]) -> None:
    columns = [
        "remap_slot_seed",
        "variant",
        "n",
        "accuracy",
        "resident_table_decode_p50_ms",
        "public_distance_table_build_p50_ms",
        "table_prediction_mismatch_count",
        "max_batch_prediction_mismatch_count",
        "batch64_speedup_vs_scalar_table_p50",
        "batch_microbench_pass",
    ]
    for batch_size in batch_sizes:
        columns.append(f"batch{batch_size}_per_request_p50_ms")
        columns.append(f"batch{batch_size}_raw_payload_128b_bytes")
        columns.append(f"batch{batch_size}_packet_record_128b_bytes")
        columns.append(f"batch{batch_size}_packet_record_64b_bytes")
    lines = [",".join(columns)]
    for row in rows:
        values: list[Any] = [row.get(column) for column in columns[:10]]
        for batch_size in batch_sizes:
            batch = row["batch_results"][str(batch_size)]
            values.append(batch["per_request_p50_ms"])
            values.append(batch["amortized_raw_payload_128b_burst_bytes_per_request"])
            values.append(batch["amortized_packet_record_128b_burst_bytes_per_request"])
            values.append(batch["amortized_packet_record_64b_line_bytes_per_request"])
        lines.append(",".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private PQ Receiver Batch Microbench",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- max resident table p50 ms: `{_fmt(h['max_resident_table_decode_p50_ms'])}`",
        f"- max batch-64 per-request p50 ms: `{_fmt(h['max_batch64_per_request_p50_ms'])}`",
        f"- min batch-64 speedup vs scalar table p50: `{_fmt(h['min_batch64_speedup_vs_scalar_table_p50'])}x`",
        f"- raw payload bytes/request: `{h['payload_bytes_per_request']}`",
        f"- packet record bytes/request: `{h['packet_record_bytes_per_request']}`",
        f"- batch-256 amortized 128B raw payload bytes/request: `{_fmt(h['batch256_amortized_128b_raw_payload_bytes_per_request'], 2)}`",
        f"- batch-256 amortized 128B packet record bytes/request: `{_fmt(h['batch256_amortized_128b_packet_record_bytes_per_request'], 2)}`",
        f"- max table mismatches: `{h['max_table_prediction_mismatch_count']}`",
        f"- max batch mismatches: `{h['max_batch_prediction_mismatch_count']}`",
        "",
        "## Rows",
        "",
        "| Remap | Variant | Acc | Table p50 ms | Batch64 p50 ms | Batch64 speedup | Batch256 record 128B bytes/req | Table bytes | Rotation bytes | Mismatch | Invariant | Pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        batch64 = row["batch_results"].get("64", row["batch_results"][str(max(int(key) for key in row["batch_results"]))])
        lines.append(
            f"| {row['remap_slot_seed']} | {row['variant']} | {row['accuracy']:.3f} | "
            f"{row['resident_table_decode_p50_ms']:.5f} | {batch64['per_request_p50_ms']:.5f} | "
            f"{row['batch64_speedup_vs_scalar_table_p50']:.2f}x | "
            f"{row['batch256_amortized_128b_packet_record_bytes_per_request']:.2f} | "
            f"{row['resident_distance_table_bytes_per_example']} | {row['rotation_matrix_bytes']} | "
            f"{row['table_prediction_mismatch_count']}/{row['max_batch_prediction_mismatch_count']} | "
            f"`{row['batch_size_invariant']}` | "
            f"`{row['batch_microbench_pass']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], "", "## Pass Rule", "", payload["pass_rule"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_pq_receiver_batch_microbench_20260430"))
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "canonical",
            "utility_balanced",
            "opq_procrustes",
            "utility_opq_procrustes",
            "protected_hadamard",
            "utility_protected_hadamard",
        ],
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--candidate-view", default="slot")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--opq-iterations", type=int, default=4)
    parser.add_argument("--table-repeats", type=int, default=1)
    parser.add_argument("--batch-repeats", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--packet-record-overhead-bytes", type=int, default=3)
    args = parser.parse_args()
    payload = build_pq_receiver_batch_microbench(
        output_dir=args.output_dir,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        remap_seeds=args.remap_seeds,
        budget_bytes=args.budget_bytes,
        variants=args.variants,
        feature_dim=args.feature_dim,
        candidate_view=args.candidate_view,
        ridge=args.ridge,
        opq_iterations=args.opq_iterations,
        table_repeats=args.table_repeats,
        batch_repeats=args.batch_repeats,
        batch_sizes=args.batch_sizes,
        packet_record_overhead_bytes=args.packet_record_overhead_bytes,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
