from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import make_benchmark  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    ProductCodebook,
    _candidate_matrix_for_view,
    _decode_product_codebook_packet,
    _fit_product_codebook,
    _fit_ridge_encoder_for_view,
    _prior_prediction,
    _project_source,
    _product_codebook_packet,
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
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": float(statistics.median(values)),
        "p95_ms": float(sorted(values)[max(0, int(0.95 * len(values)) - 1)]),
        "mean_ms": float(statistics.fmean(values)),
    }


def _reconstruct_product_codebook_vector(
    payload: bytes,
    *,
    codebook: ProductCodebook,
    feature_dim: int,
) -> np.ndarray:
    reconstructed = np.zeros(feature_dim, dtype=np.float32)
    used_subspaces = min(len(payload), codebook.subspaces)
    raw_codes = np.frombuffer(payload[:used_subspaces], dtype=np.uint8)
    for subspace_index, code in enumerate(raw_codes):
        sub_centroids = codebook.centroids[subspace_index]
        centroid_index = int(code) % sub_centroids.shape[0]
        reconstructed[codebook.slices[subspace_index]] = sub_centroids[centroid_index]
    return reconstructed


def _decode_product_codebook_packet_cached(
    example: Any,
    payload: bytes | None,
    *,
    codebook: ProductCodebook,
    feature_dim: int,
    candidate_matrix: np.ndarray,
) -> str:
    if not payload:
        return _prior_prediction(example)
    reconstructed = _reconstruct_product_codebook_vector(payload, codebook=codebook, feature_dim=feature_dim)
    distances = np.sum((candidate_matrix - reconstructed[None, :]) ** 2, axis=1)
    min_distance = float(np.min(distances))
    tied = np.flatnonzero(np.isclose(distances, min_distance, rtol=1e-6, atol=1e-8))
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[int(idx)] == prior for idx in tied):
        return prior
    return labels[int(tied[0])]


def _encode_product_codebook_from_vector(vector: np.ndarray, *, codebook: ProductCodebook) -> bytes:
    codes = np.zeros(codebook.subspaces, dtype=np.uint8)
    for subspace_index, (sub_centroids, dim_slice) in enumerate(zip(codebook.centroids, codebook.slices, strict=True)):
        part = vector[dim_slice]
        distances = np.sum((sub_centroids - part[None, :]) ** 2, axis=1)
        codes[subspace_index] = int(np.argmin(distances))
    return codes.tobytes()


def _build_product_codebook_distance_tables(candidate_matrix: np.ndarray, *, codebook: ProductCodebook) -> tuple[np.ndarray, ...]:
    tables: list[np.ndarray] = []
    for sub_centroids, dim_slice in zip(codebook.centroids, codebook.slices, strict=True):
        candidate_part = candidate_matrix[:, dim_slice].astype(np.float32)
        table = np.sum((candidate_part[:, None, :] - sub_centroids[None, :, :]) ** 2, axis=2)
        tables.append(table.astype(np.float32))
    return tuple(tables)


def _decode_product_codebook_packet_table(
    example: Any,
    payload: bytes | None,
    *,
    distance_tables: tuple[np.ndarray, ...],
) -> str:
    if not payload:
        return _prior_prediction(example)
    used_subspaces = min(len(payload), len(distance_tables))
    raw_codes = np.frombuffer(payload[:used_subspaces], dtype=np.uint8)
    scores = np.zeros(distance_tables[0].shape[0], dtype=np.float32)
    for subspace_index, code in enumerate(raw_codes):
        table = distance_tables[subspace_index]
        scores += table[:, int(code) % table.shape[1]]
    min_score = float(np.min(scores))
    tied = np.flatnonzero(np.isclose(scores, min_score, rtol=1e-6, atol=1e-8))
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[int(idx)] == prior for idx in tied):
        return prior
    return labels[int(tied[0])]


def _batch_kernel_latency_ms(
    *,
    payloads: list[bytes],
    candidate_matrices: list[np.ndarray],
    codebook: ProductCodebook,
    feature_dim: int,
    repeats: int,
) -> dict[str, float]:
    candidates = np.stack(candidate_matrices, axis=0).astype(np.float32)
    codes = np.stack(
        [
            np.frombuffer(payload[: codebook.subspaces], dtype=np.uint8).astype(np.int64)
            for payload in payloads
        ],
        axis=0,
    )
    per_example_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        reconstructed = np.zeros((len(payloads), feature_dim), dtype=np.float32)
        for subspace_index, sub_centroids in enumerate(codebook.centroids):
            centroid_indices = codes[:, subspace_index] % sub_centroids.shape[0]
            reconstructed[:, codebook.slices[subspace_index]] = sub_centroids[centroid_indices]
        _ = np.argmin(np.sum((candidates - reconstructed[:, None, :]) ** 2, axis=2), axis=1)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_example_ms.append(elapsed_ms / max(1, len(payloads)))
    return _latency_summary(per_example_ms)


def _gate_row_by_key(gate_payload: dict[str, Any] | None) -> dict[tuple[int, int], dict[str, Any]]:
    if not gate_payload:
        return {}
    return {
        (int(row["remap_slot_seed"]), int(row["budget_bytes"])): row
        for row in gate_payload.get("rows", [])
    }


def build_decode_frontier(
    *,
    output_dir: pathlib.Path,
    product_gate_json: pathlib.Path | None,
    remap_seeds: list[int],
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    train_seed: int,
    eval_seed: int,
    candidate_view: str,
    timing_repeats: int,
    batch_repeats: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_payload = None
    if product_gate_json and product_gate_json.exists():
        gate_payload = json.loads(product_gate_json.read_text(encoding="utf-8"))
    prior_rows = _gate_row_by_key(gate_payload)

    rows: list[dict[str, Any]] = []
    for remap_seed in remap_seeds:
        train_rows = make_benchmark(examples=train_examples, candidates=4, seed=train_seed, family_set="all")
        eval_rows = make_benchmark(examples=eval_examples, candidates=4, seed=eval_seed, family_set="all")
        train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_seed)
        eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_seed)
        encoder = _fit_ridge_encoder_for_view(
            train_rows,
            feature_dim=feature_dim,
            ridge=1e-2,
            candidate_view=candidate_view,
            fit_intercept=False,
        )
        candidate_matrices = [
            _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
            for example in eval_rows
        ]

        for budget in budgets:
            codebook = _fit_product_codebook(
                train_rows,
                encoder=encoder,
                feature_dim=feature_dim,
                budget_bytes=budget,
                seed=train_seed * 9001 + eval_seed * 17 + budget,
            )
            payloads: list[bytes] = []
            source_encode_latencies: list[float] = []
            source_projected_vectors: list[np.ndarray] = []
            for example in eval_rows:
                source_projected_vectors.append(
                    _project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched").astype(np.float32)
                )
                start = time.perf_counter()
                payload = _product_codebook_packet(
                    example,
                    encoder=encoder,
                    codebook=codebook,
                    feature_dim=feature_dim,
                    mode="matched",
                )
                source_encode_latencies.append((time.perf_counter() - start) * 1000.0)
                payloads.append(payload)

            source_packet_kernel_latencies: list[float] = []
            for vector in source_projected_vectors:
                start = time.perf_counter()
                _ = _encode_product_codebook_from_vector(vector, codebook=codebook)
                source_packet_kernel_latencies.append((time.perf_counter() - start) * 1000.0)

            table_build_latencies: list[float] = []
            distance_tables: list[tuple[np.ndarray, ...]] = []
            for candidate_matrix in candidate_matrices:
                start = time.perf_counter()
                tables = _build_product_codebook_distance_tables(candidate_matrix, codebook=codebook)
                table_build_latencies.append((time.perf_counter() - start) * 1000.0)
                distance_tables.append(tables)

            cold_latencies: list[float] = []
            cached_latencies: list[float] = []
            resident_table_latencies: list[float] = []
            request_public_table_latencies: list[float] = []
            cached_predictions: list[str] = []
            cold_predictions: list[str] = []
            table_predictions: list[str] = []
            request_predictions: list[str] = []
            for _ in range(timing_repeats):
                for example, payload, candidate_matrix, tables in zip(
                    eval_rows, payloads, candidate_matrices, distance_tables, strict=True
                ):
                    start = time.perf_counter()
                    cold_prediction, _ = _decode_product_codebook_packet(
                        example,
                        payload,
                        codebook=codebook,
                        feature_dim=feature_dim,
                        candidate_view=candidate_view,
                    )
                    cold_latencies.append((time.perf_counter() - start) * 1000.0)

                    start = time.perf_counter()
                    cached_prediction = _decode_product_codebook_packet_cached(
                        example,
                        payload,
                        codebook=codebook,
                        feature_dim=feature_dim,
                        candidate_matrix=candidate_matrix,
                    )
                    cached_latencies.append((time.perf_counter() - start) * 1000.0)

                    start = time.perf_counter()
                    table_prediction = _decode_product_codebook_packet_table(
                        example,
                        payload,
                        distance_tables=tables,
                    )
                    resident_table_latencies.append((time.perf_counter() - start) * 1000.0)

                    start = time.perf_counter()
                    request_candidate_matrix = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
                    request_tables = _build_product_codebook_distance_tables(request_candidate_matrix, codebook=codebook)
                    request_prediction = _decode_product_codebook_packet_table(
                        example,
                        payload,
                        distance_tables=request_tables,
                    )
                    request_public_table_latencies.append((time.perf_counter() - start) * 1000.0)
                    if len(cached_predictions) < len(eval_rows):
                        cached_predictions.append(cached_prediction)
                        cold_predictions.append(cold_prediction)
                        table_predictions.append(table_prediction)
                        request_predictions.append(request_prediction)

            correct = sum(
                1 for prediction, example in zip(cached_predictions, eval_rows, strict=True) if prediction == example.answer_label
            )
            mismatches = [
                example.example_id
                for example, cold_prediction, cached_prediction in zip(
                    eval_rows, cold_predictions, cached_predictions, strict=True
                )
                if cold_prediction != cached_prediction
            ]
            table_mismatches = [
                example.example_id
                for example, cold_prediction, table_prediction, request_prediction in zip(
                    eval_rows, cold_predictions, table_predictions, request_predictions, strict=True
                )
                if cold_prediction != table_prediction or cold_prediction != request_prediction
            ]
            prior = prior_rows.get((remap_seed, budget), {})
            cached_summary = _latency_summary(cached_latencies)
            cold_summary = _latency_summary(cold_latencies)
            source_summary = _latency_summary(source_encode_latencies)
            source_packet_kernel_summary = _latency_summary(source_packet_kernel_latencies)
            table_build_summary = _latency_summary(table_build_latencies)
            resident_table_summary = _latency_summary(resident_table_latencies)
            request_table_summary = _latency_summary(request_public_table_latencies)
            batch_summary = _batch_kernel_latency_ms(
                payloads=payloads,
                candidate_matrices=candidate_matrices,
                codebook=codebook,
                feature_dim=feature_dim,
                repeats=batch_repeats,
            )
            recorded_p50 = prior.get("p50_decode_latency_ms")
            row_pass = (
                bool(prior.get("product_codebook_pass", False))
                and not mismatches
                and not table_mismatches
                and request_table_summary["p50_ms"] < 2.0
                and request_table_summary["p95_ms"] < 5.0
                and resident_table_summary["p50_ms"] < 0.25
            )
            rows.append(
                {
                    "remap_slot_seed": remap_seed,
                    "budget_bytes": budget,
                    "n": eval_examples,
                    "accuracy": correct / eval_examples,
                    "prior_gate_product_codebook_accuracy": prior.get("product_codebook_accuracy"),
                    "prior_gate_functional_pass": prior.get("product_codebook_pass"),
                    "prior_gate_product_codebook_minus_best_control": prior.get("product_codebook_minus_best_control"),
                    "prior_gate_recorded_p50_ms": recorded_p50,
                    "source_encode_p50_ms": source_summary["p50_ms"],
                    "source_encode_p95_ms": source_summary["p95_ms"],
                    "source_packet_kernel_p50_ms": source_packet_kernel_summary["p50_ms"],
                    "source_packet_kernel_p95_ms": source_packet_kernel_summary["p95_ms"],
                    "public_distance_table_build_p50_ms": table_build_summary["p50_ms"],
                    "public_distance_table_build_p95_ms": table_build_summary["p95_ms"],
                    "cold_receiver_p50_ms": cold_summary["p50_ms"],
                    "cold_receiver_p95_ms": cold_summary["p95_ms"],
                    "cached_receiver_p50_ms": cached_summary["p50_ms"],
                    "cached_receiver_p95_ms": cached_summary["p95_ms"],
                    "request_public_table_decode_p50_ms": request_table_summary["p50_ms"],
                    "request_public_table_decode_p95_ms": request_table_summary["p95_ms"],
                    "resident_table_decode_p50_ms": resident_table_summary["p50_ms"],
                    "resident_table_decode_p95_ms": resident_table_summary["p95_ms"],
                    "cached_batch_amortized_p50_ms": batch_summary["p50_ms"],
                    "cached_batch_amortized_p95_ms": batch_summary["p95_ms"],
                    "cached_speedup_vs_prior_recorded": None
                    if not recorded_p50
                    else float(recorded_p50) / max(cached_summary["p50_ms"], 1e-9),
                    "cached_speedup_vs_cold_receiver": cold_summary["p50_ms"] / max(cached_summary["p50_ms"], 1e-9),
                    "prediction_mismatch_count": len(mismatches),
                    "prediction_mismatch_examples": mismatches[:10],
                    "table_prediction_mismatch_count": len(table_mismatches),
                    "table_prediction_mismatch_examples": table_mismatches[:10],
                    "mean_payload_bytes": float(statistics.fmean(len(payload) for payload in payloads)),
                    "cached_latency_pass": cached_summary["p50_ms"] < 2.0,
                    "request_public_latency_pass": request_table_summary["p50_ms"] < 2.0 and request_table_summary["p95_ms"] < 5.0,
                    "resident_kernel_latency_pass": resident_table_summary["p50_ms"] < 0.25,
                    "decode_frontier_pass": row_pass,
                }
            )

    pass_remaps = sorted({row["remap_slot_seed"] for row in rows if row["decode_frontier_pass"]})
    payload = {
        "gate": "source_private_product_codebook_decode_frontier",
        "rows": rows,
        "headline": {
            "rows": len(rows),
            "pass_rows": sum(1 for row in rows if row["decode_frontier_pass"]),
            "remaps_with_pass": pass_remaps,
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "max_cached_receiver_p50_ms": max((row["cached_receiver_p50_ms"] for row in rows), default=None),
            "max_request_public_table_decode_p50_ms": max(
                (row["request_public_table_decode_p50_ms"] for row in rows),
                default=None,
            ),
            "max_resident_table_decode_p50_ms": max((row["resident_table_decode_p50_ms"] for row in rows), default=None),
            "min_cached_speedup_vs_prior_recorded": min(
                (
                    row["cached_speedup_vs_prior_recorded"]
                    for row in rows
                    if row["cached_speedup_vs_prior_recorded"] is not None
                ),
                default=None,
            ),
            "max_prediction_mismatch_count": max((row["prediction_mismatch_count"] for row in rows), default=0),
            "max_table_prediction_mismatch_count": max((row["table_prediction_mismatch_count"] for row in rows), default=0),
        },
        "pass_gate": len(pass_remaps) == len(remap_seeds),
        "pass_rule": (
            "For every remapped codebook there must be at least one functionally passing product-codebook row "
            "whose cached and table-lookup target-side decoders exactly match the canonical decoder, whose request-public "
            "table decode has p50 <2 ms and p95 <5 ms, and whose resident lookup kernel has p50 <0.25 ms. "
            "Cold receiver decode, source packet construction, and public table construction are reported separately."
        ),
        "interpretation": (
            "The product-codebook packet gate failed the strict systems rule because the prior metric timed source packet "
            "construction and repeated target-side candidate feature hashing inside every row. This frontier isolates the "
            "receiver-side lookup that a real target would run after it already has public prompt/candidate state T, and "
            "adds a direct PQ distance-table path so the systems row matches product-quantization practice rather than "
            "full-vector reconstruction. "
            "A pass here does not claim end-to-end model inference speedup; it shows that the learned byte packet can be "
            "decoded as a low-latency table lookup once target side information is cached."
        ),
        "inputs": {
            "product_gate_json": None if product_gate_json is None else str(product_gate_json),
            "train_examples": train_examples,
            "eval_examples": eval_examples,
            "feature_dim": feature_dim,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "candidate_view": candidate_view,
            "timing_repeats": timing_repeats,
            "batch_repeats": batch_repeats,
        },
    }
    json_path = output_dir / "product_codebook_decode_frontier.json"
    md_path = output_dir / "product_codebook_decode_frontier.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": ["product_codebook_decode_frontier.json", "product_codebook_decode_frontier.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "product_codebook_decode_frontier.json": _sha256_file(json_path),
            "product_codebook_decode_frontier.md": _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Product-Codebook Decode Frontier Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Product-Codebook Decode Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- remaps with pass: `{h['remaps_with_pass']}`",
        f"- max cached receiver p50 ms: `{_fmt(h['max_cached_receiver_p50_ms'], 4)}`",
        f"- max request-public table p50 ms: `{_fmt(h['max_request_public_table_decode_p50_ms'], 4)}`",
        f"- max resident table p50 ms: `{_fmt(h['max_resident_table_decode_p50_ms'], 5)}`",
        f"- min cached speedup vs prior recorded: `{_fmt(h['min_cached_speedup_vs_prior_recorded'])}x`",
        f"- max prediction mismatch count: `{h['max_prediction_mismatch_count']}`",
        f"- max table prediction mismatch count: `{h['max_table_prediction_mismatch_count']}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | Functional pass | Prior p50 ms | Source packet kernel p50 | Request table p50 | Resident table p50 | Cached vector p50 | Batch p50 | Speedup vs prior | Mismatches | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"`{row['prior_gate_functional_pass']}` | {_fmt(row['prior_gate_recorded_p50_ms'])} | "
            f"{row['source_packet_kernel_p50_ms']:.4f} | {row['request_public_table_decode_p50_ms']:.4f} | "
            f"{row['resident_table_decode_p50_ms']:.5f} | {row['cached_receiver_p50_ms']:.4f} | "
            f"{row['cached_batch_amortized_p50_ms']:.5f} | {_fmt(row['cached_speedup_vs_prior_recorded'])}x | "
            f"{row['prediction_mismatch_count']}/{row['table_prediction_mismatch_count']} | `{row['decode_frontier_pass']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Pass Rule",
            "",
            payload["pass_rule"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_product_codebook_decode_frontier_20260430"))
    parser.add_argument(
        "--product-gate-json",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_product_codebook_packet_gate_20260430/product_codebook_packet_gate.json"),
    )
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--candidate-view", default="slot")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--batch-repeats", type=int, default=25)
    args = parser.parse_args()

    payload = build_decode_frontier(
        output_dir=args.output_dir,
        product_gate_json=args.product_gate_json,
        remap_seeds=args.remap_seeds,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        feature_dim=args.feature_dim,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        candidate_view=args.candidate_view,
        timing_repeats=args.timing_repeats,
        batch_repeats=args.batch_repeats,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
