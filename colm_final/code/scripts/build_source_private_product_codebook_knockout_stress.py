from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_source_private_product_codebook_decode_frontier import (  # noqa: E402
    _build_product_codebook_distance_tables,
    _decode_product_codebook_packet_table,
    _encode_product_codebook_from_vector,
)
from scripts.run_source_private_hidden_repair_packet_smoke import make_benchmark  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    ProductCodebook,
    _candidate_matrix_for_view,
    _fit_product_codebook,
    _fit_ridge_encoder_for_view,
    _prior_prediction,
    _project_source,
    _product_codebook_packet,
    _remap_candidate_slots,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


CONDITIONS = [
    "target_only",
    "product_codebook_source",
    "top_codeword_removed_mean",
    "top_codeword_removed_worst",
    "random_codeword_removed_mean",
    "random_codeword_removed_random",
    "top_codeword_only",
    "mean_payload",
]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return float(ordered[index])


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "mean": sum(values) / n,
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def _answer_index(example: Any) -> int:
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _decode_scores(
    *,
    payload: bytes,
    distance_tables: tuple[np.ndarray, ...],
) -> np.ndarray:
    used_subspaces = min(len(payload), len(distance_tables))
    raw_codes = np.frombuffer(payload[:used_subspaces], dtype=np.uint8)
    scores = np.zeros(distance_tables[0].shape[0], dtype=np.float32)
    for subspace_index, code in enumerate(raw_codes):
        table = distance_tables[subspace_index]
        scores += table[:, int(code) % table.shape[1]]
    return scores


def _best_wrong_index(scores: np.ndarray, answer_index: int) -> int:
    wrong = [(float(score), idx) for idx, score in enumerate(scores) if idx != answer_index]
    return min(wrong, key=lambda item: (item[0], item[1]))[1]


def _top_margin_subspace(
    *,
    payload: bytes,
    distance_tables: tuple[np.ndarray, ...],
    answer_index: int,
    wrong_index: int,
) -> tuple[int, list[float]]:
    codes = np.frombuffer(payload[: len(distance_tables)], dtype=np.uint8)
    contributions: list[float] = []
    for subspace_index, code in enumerate(codes):
        table = distance_tables[subspace_index]
        code_index = int(code) % table.shape[1]
        contributions.append(float(table[wrong_index, code_index] - table[answer_index, code_index]))
    top_index = max(range(len(contributions)), key=lambda idx: (contributions[idx], -idx))
    return top_index, contributions


def _replace_code(payload: bytes, *, subspace_index: int, replacement: int) -> bytes:
    codes = bytearray(payload)
    codes[subspace_index] = replacement % 256
    return bytes(codes)


def _random_subspace(payload: bytes, *, seed: int) -> int:
    rng = random.Random(seed)
    return rng.randrange(max(1, len(payload)))


def _random_different_code(*, codebook: ProductCodebook, subspace_index: int, old_code: int, seed: int) -> int:
    centroid_count = codebook.centroids[subspace_index].shape[0]
    if centroid_count <= 1:
        return old_code
    rng = random.Random(seed)
    candidate = rng.randrange(centroid_count - 1)
    if candidate >= old_code % centroid_count:
        candidate += 1
    return candidate


def _worst_answer_margin_code(
    *,
    table: np.ndarray,
    answer_index: int,
    wrong_index: int,
    old_code: int,
) -> int:
    margins = table[wrong_index] - table[answer_index]
    order = np.argsort(margins, kind="stable")
    for candidate in order:
        if int(candidate) != old_code % table.shape[1]:
            return int(candidate)
    return int(order[0])


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": float(statistics.median(values)),
        "p95_ms": float(sorted(values)[max(0, int(0.95 * len(values)) - 1)]),
        "mean_ms": float(statistics.fmean(values)),
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "n": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "accuracy": sum(1 for row in rows if row["correct"]) / max(1, len(rows)),
        "mean_payload_bytes": statistics.fmean(float(row["payload_bytes"]) for row in rows),
        "mean_payload_tokens": statistics.fmean(float(row["payload_tokens"]) for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _payload_entropy_summary(rows: list[dict[str, Any]], *, budget: int) -> dict[str, Any]:
    payload_counts: dict[str, int] = {}
    payload_labels: dict[str, set[str]] = {}
    for row in rows:
        payload = str(row["payload_hex"])
        payload_counts[payload] = payload_counts.get(payload, 0) + 1
        payload_labels.setdefault(payload, set()).add(str(row["answer"]))
    n = max(1, len(rows))
    payload_entropy = -sum((count / n) * math.log2(count / n) for count in payload_counts.values())
    codeword_summaries: list[dict[str, Any]] = []
    for subspace_index in range(budget):
        counts: dict[int, int] = {}
        for row in rows:
            code = bytes.fromhex(str(row["payload_hex"]))[subspace_index]
            counts[code] = counts.get(code, 0) + 1
        entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
        top_codes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        codeword_summaries.append(
            {
                "subspace_index": subspace_index,
                "unique_codes": len(counts),
                "max_frequency": max(counts.values(), default=0),
                "entropy_bits": float(entropy),
                "top5_codes": [{"code": int(code), "count": int(count)} for code, count in top_codes],
            }
        )
    return {
        "unique_payloads": len(payload_counts),
        "unique_once_payloads": sum(1 for count in payload_counts.values() if count == 1),
        "collision_payloads": sum(1 for count in payload_counts.values() if count > 1),
        "max_payload_frequency": max(payload_counts.values(), default=0),
        "payload_entropy_bits": float(payload_entropy),
        "single_label_payloads": sum(1 for labels in payload_labels.values() if len(labels) == 1),
        "codeword_summaries": codeword_summaries,
    }


def _paired_values(
    grouped: dict[str, dict[str, dict[str, Any]]],
    *,
    method: str,
    baseline: str,
) -> list[float]:
    values: list[float] = []
    for example_id in sorted(grouped):
        conditions = grouped[example_id]
        if method in conditions and baseline in conditions:
            values.append(float(bool(conditions[method]["correct"])) - float(bool(conditions[baseline]["correct"])))
    return values


def _paired_counts(values: list[float]) -> dict[str, int]:
    return {
        "wins": sum(1 for value in values if value > 0),
        "losses": sum(1 for value in values if value < 0),
        "ties": sum(1 for value in values if value == 0),
    }


def _mean_payload(
    train_rows: list[Any],
    *,
    encoder: np.ndarray,
    codebook: ProductCodebook,
    feature_dim: int,
) -> bytes:
    train_vectors = np.stack(
        [_project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched") for example in train_rows],
        axis=0,
    ).astype(np.float32)
    return _encode_product_codebook_from_vector(train_vectors.mean(axis=0), codebook=codebook)


def _row_payloads(
    *,
    example: Any,
    example_index: int,
    encoder: np.ndarray,
    codebook: ProductCodebook,
    feature_dim: int,
    candidate_view: str,
    public_mean_payload: bytes,
    seed: int,
) -> tuple[dict[str, bytes | None], dict[str, Any]]:
    source_payload = _product_codebook_packet(
        example,
        encoder=encoder,
        codebook=codebook,
        feature_dim=feature_dim,
        mode="matched",
    )
    candidate_matrix = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
    distance_tables = _build_product_codebook_distance_tables(candidate_matrix, codebook=codebook)
    scores = _decode_scores(payload=source_payload, distance_tables=distance_tables)
    answer_index = _answer_index(example)
    wrong_index = _best_wrong_index(scores, answer_index)
    top_index, contributions = _top_margin_subspace(
        payload=source_payload,
        distance_tables=distance_tables,
        answer_index=answer_index,
        wrong_index=wrong_index,
    )
    codes = np.frombuffer(source_payload, dtype=np.uint8)
    mean_codes = np.frombuffer(public_mean_payload, dtype=np.uint8)
    random_index = _random_subspace(source_payload, seed=seed + example_index * 9173)
    top_mean_code = int(mean_codes[top_index] % codebook.centroids[top_index].shape[0])
    random_mean_code = int(mean_codes[random_index] % codebook.centroids[random_index].shape[0])
    top_worst_code = _worst_answer_margin_code(
        table=distance_tables[top_index],
        answer_index=answer_index,
        wrong_index=wrong_index,
        old_code=int(codes[top_index]),
    )
    random_code = _random_different_code(
        codebook=codebook,
        subspace_index=random_index,
        old_code=int(codes[random_index]),
        seed=seed + example_index * 7919 + 13,
    )
    top_only = bytearray(public_mean_payload)
    top_only[top_index] = int(codes[top_index])
    payloads: dict[str, bytes | None] = {
        "target_only": None,
        "product_codebook_source": source_payload,
        "top_codeword_removed_mean": _replace_code(source_payload, subspace_index=top_index, replacement=top_mean_code),
        "top_codeword_removed_worst": _replace_code(source_payload, subspace_index=top_index, replacement=top_worst_code),
        "random_codeword_removed_mean": _replace_code(source_payload, subspace_index=random_index, replacement=random_mean_code),
        "random_codeword_removed_random": _replace_code(source_payload, subspace_index=random_index, replacement=random_code),
        "top_codeword_only": bytes(top_only),
        "mean_payload": public_mean_payload,
    }
    metadata = {
        "answer_index": answer_index,
        "best_wrong_index": wrong_index,
        "matched_margin": float(scores[wrong_index] - scores[answer_index]),
        "top_subspace_index": top_index,
        "random_subspace_index": random_index,
        "top_margin_contribution": float(contributions[top_index]),
        "random_margin_contribution": float(contributions[random_index]),
        "margin_contributions": contributions,
        "source_payload_hex": source_payload.hex(),
        "mean_payload_hex": public_mean_payload.hex(),
        "top_original_code": int(codes[top_index]),
        "top_mean_code": top_mean_code,
        "top_worst_code": top_worst_code,
        "random_original_code": int(codes[random_index]),
        "random_mean_code": random_mean_code,
        "random_replacement_code": random_code,
    }
    return payloads, metadata


def _predict_payload(
    *,
    condition: str,
    example: Any,
    payload: bytes | None,
    distance_tables: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    start = time.perf_counter()
    if payload is None:
        prediction = _prior_prediction(example)
    else:
        prediction = _decode_product_codebook_packet_table(example, payload, distance_tables=distance_tables)
    latency_ms = (time.perf_counter() - start) * 1000.0
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": latency_ms,
        "payload_hex": payload_hex,
    }


def build_knockout_stress(
    *,
    output_dir: pathlib.Path,
    remap_seeds: list[int],
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    train_seed: int,
    eval_seed: int,
    candidate_view: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
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
        remap_dir = output_dir / f"remap_{remap_seed}"
        remap_dir.mkdir(parents=True, exist_ok=True)
        run_dirs.append(str(remap_dir))
        for budget in budgets:
            codebook = _fit_product_codebook(
                train_rows,
                encoder=encoder,
                feature_dim=feature_dim,
                budget_bytes=budget,
                seed=train_seed * 9001 + eval_seed * 17 + budget,
            )
            public_mean_payload = _mean_payload(
                train_rows,
                encoder=encoder,
                codebook=codebook,
                feature_dim=feature_dim,
            )
            by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
            per_example_metadata: list[dict[str, Any]] = []
            table_build_latencies: list[float] = []
            for example_index, example in enumerate(eval_rows):
                candidate_matrix = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
                start = time.perf_counter()
                distance_tables = _build_product_codebook_distance_tables(candidate_matrix, codebook=codebook)
                table_build_latencies.append((time.perf_counter() - start) * 1000.0)
                payloads, metadata = _row_payloads(
                    example=example,
                    example_index=example_index,
                    encoder=encoder,
                    codebook=codebook,
                    feature_dim=feature_dim,
                    candidate_view=candidate_view,
                    public_mean_payload=public_mean_payload,
                    seed=train_seed * 4001 + eval_seed * 101 + remap_seed * 17 + budget,
                )
                per_example_metadata.append({"example_id": example.example_id, **metadata})
                for condition, payload in payloads.items():
                    row = _predict_payload(
                        condition=condition,
                        example=example,
                        payload=payload,
                        distance_tables=distance_tables,
                    )
                    by_condition[condition].append(
                        row
                        | {
                            "example_id": example.example_id,
                            "family_name": example.family_name,
                            "budget_bytes": budget,
                            "remap_slot_seed": remap_seed,
                            "top_subspace_index": metadata["top_subspace_index"],
                            "random_subspace_index": metadata["random_subspace_index"],
                            "matched_margin": metadata["matched_margin"],
                            "top_margin_contribution": metadata["top_margin_contribution"],
                            "random_margin_contribution": metadata["random_margin_contribution"],
                        }
                    )
            prediction_path = remap_dir / f"predictions_budget{budget}.jsonl"
            with prediction_path.open("w", encoding="utf-8") as handle:
                for condition in CONDITIONS:
                    for row in by_condition[condition]:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
            metadata_path = remap_dir / f"metadata_budget{budget}.jsonl"
            with metadata_path.open("w", encoding="utf-8") as handle:
                for row in per_example_metadata:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            metrics = {condition: _condition_summary(rows) for condition, rows in by_condition.items()}
            grouped: dict[str, dict[str, dict[str, Any]]] = {}
            for condition, rows in by_condition.items():
                for row in rows:
                    grouped.setdefault(row["example_id"], {})[condition] = row
            exact_id_parity = all(
                {row["example_id"] for row in by_condition[condition]} == set(grouped)
                for condition in CONDITIONS
            )
            comparisons: dict[str, Any] = {}
            for baseline in [condition for condition in CONDITIONS if condition != "product_codebook_source"]:
                values = _paired_values(grouped, method="product_codebook_source", baseline=baseline)
                comparisons[baseline] = {
                    "accuracy": metrics[baseline]["accuracy"],
                    "paired_delta": _bootstrap_ci(
                        values,
                        samples=bootstrap_samples,
                        seed=train_seed * 5003 + eval_seed * 101 + remap_seed * 37 + budget * 1009 + len(comparisons),
                    ),
                    "paired_counts": _paired_counts(values),
                }
            source_acc = metrics["product_codebook_source"]["accuracy"]
            target_acc = metrics["target_only"]["accuracy"]
            lift = source_acc - target_acc
            payload_entropy = _payload_entropy_summary(by_condition["product_codebook_source"], budget=budget)

            def removed_fraction(condition: str) -> float | None:
                if lift <= 1e-9:
                    return None
                return (source_acc - metrics[condition]["accuracy"]) / lift

            top_worst_removed = removed_fraction("top_codeword_removed_worst")
            top_mean_removed = removed_fraction("top_codeword_removed_mean")
            random_mean_removed = removed_fraction("random_codeword_removed_mean")
            top_only_lift = metrics["top_codeword_only"]["accuracy"] - target_acc
            adversarial_pass = (
                exact_id_parity
                and lift >= 0.15
                and top_worst_removed is not None
                and top_worst_removed >= 0.50
                and comparisons["top_codeword_removed_worst"]["paired_delta"]["ci95_low"] > 0.05
            )
            public_mean_pass = (
                exact_id_parity
                and lift >= 0.15
                and top_mean_removed is not None
                and random_mean_removed is not None
                and top_mean_removed >= 0.25
                and top_mean_removed >= random_mean_removed + 0.05
                and comparisons["top_codeword_removed_mean"]["paired_delta"]["ci95_low"] > 0.05
            )
            row_payload = {
                "remap_slot_seed": remap_seed,
                "budget_bytes": budget,
                "n": eval_examples,
                "exact_id_parity": exact_id_parity,
                "candidate_pool_recall": 1.0,
                "source_accuracy": source_acc,
                "target_accuracy": target_acc,
                "source_minus_target": lift,
                "top_worst_accuracy": metrics["top_codeword_removed_worst"]["accuracy"],
                "top_mean_accuracy": metrics["top_codeword_removed_mean"]["accuracy"],
                "random_mean_accuracy": metrics["random_codeword_removed_mean"]["accuracy"],
                "random_random_accuracy": metrics["random_codeword_removed_random"]["accuracy"],
                "top_only_accuracy": metrics["top_codeword_only"]["accuracy"],
                "mean_payload_accuracy": metrics["mean_payload"]["accuracy"],
                "top_worst_lift_removed_fraction": top_worst_removed,
                "top_mean_lift_removed_fraction": top_mean_removed,
                "random_mean_lift_removed_fraction": random_mean_removed,
                "top_mean_minus_random_mean_removed_fraction": (
                    None
                    if top_mean_removed is None or random_mean_removed is None
                    else top_mean_removed - random_mean_removed
                ),
                "top_only_lift_vs_target": top_only_lift,
                "mean_top_margin_contribution": statistics.fmean(
                    float(row["top_margin_contribution"]) for row in per_example_metadata
                ),
                "mean_random_margin_contribution": statistics.fmean(
                    float(row["random_margin_contribution"]) for row in per_example_metadata
                ),
                "mean_matched_margin": statistics.fmean(float(row["matched_margin"]) for row in per_example_metadata),
                "mean_payload_bytes": metrics["product_codebook_source"]["mean_payload_bytes"],
                "mean_payload_tokens": metrics["product_codebook_source"]["mean_payload_tokens"],
                "payload_entropy": payload_entropy,
                "table_build_latency": _latency_summary(table_build_latencies),
                "metrics": metrics,
                "paired_comparisons": comparisons,
                "adversarial_knockout_pass": adversarial_pass,
                "public_mean_knockout_pass": public_mean_pass,
                "predictions_file": str(prediction_path),
                "metadata_file": str(metadata_path),
            }
            aggregate_rows.append(row_payload)
            summary_path = remap_dir / f"summary_budget{budget}.json"
            summary_path.write_text(json.dumps(row_payload, indent=2, sort_keys=True), encoding="utf-8")
    adversarial_pass_remaps = sorted({row["remap_slot_seed"] for row in aggregate_rows if row["adversarial_knockout_pass"]})
    public_pass_remaps = sorted({row["remap_slot_seed"] for row in aggregate_rows if row["public_mean_knockout_pass"]})
    payload = {
        "gate": "source_private_product_codebook_knockout_stress",
        "rows": aggregate_rows,
        "run_dirs": run_dirs,
        "headline": {
            "rows": len(aggregate_rows),
            "adversarial_pass_rows": sum(1 for row in aggregate_rows if row["adversarial_knockout_pass"]),
            "public_mean_pass_rows": sum(1 for row in aggregate_rows if row["public_mean_knockout_pass"]),
            "adversarial_remaps_with_pass": adversarial_pass_remaps,
            "public_mean_remaps_with_pass": public_pass_remaps,
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "min_top_worst_lift_removed_fraction": min(
                (
                    row["top_worst_lift_removed_fraction"]
                    for row in aggregate_rows
                    if row["top_worst_lift_removed_fraction"] is not None
                ),
                default=None,
            ),
            "min_top_mean_lift_removed_fraction": min(
                (
                    row["top_mean_lift_removed_fraction"]
                    for row in aggregate_rows
                    if row["top_mean_lift_removed_fraction"] is not None
                ),
                default=None,
            ),
            "max_random_mean_lift_removed_fraction": max(
                (
                    row["random_mean_lift_removed_fraction"]
                    for row in aggregate_rows
                    if row["random_mean_lift_removed_fraction"] is not None
                ),
                default=None,
            ),
            "max_source_accuracy": max((row["source_accuracy"] for row in aggregate_rows), default=None),
            "min_source_minus_target": min((row["source_minus_target"] for row in aggregate_rows), default=None),
            "min_unique_payloads": min((row["payload_entropy"]["unique_payloads"] for row in aggregate_rows), default=None),
            "max_payload_frequency": max((row["payload_entropy"]["max_payload_frequency"] for row in aggregate_rows), default=None),
            "min_payload_entropy_bits": min(
                (row["payload_entropy"]["payload_entropy_bits"] for row in aggregate_rows),
                default=None,
            ),
        },
        "adversarial_pass_gate": len(adversarial_pass_remaps) == len(remap_seeds),
        "public_mean_pass_gate": len(public_pass_remaps) == len(remap_seeds),
        "pass_gate": len(adversarial_pass_remaps) == len(remap_seeds),
        "pass_rule": (
            "Adversarial pass: every remapped codebook must have a matched-source lift >=0.15 over target-only, exact ID "
            "parity, and top-margin oracle codeword replacement must remove at least 50% of matched lift with paired "
            "CI95 low >0.05 versus the matched packet. Public-mean pass is stricter: replacing the same top-margin byte "
            "with a train-public mean code must remove at least 25% of lift and at least 5 points more lift fraction than "
            "a random-subspace public-mean replacement."
        ),
        "interpretation": (
            "This is a diagnostic stress test for the n500 product-codebook packet. The source packet is one centroid "
            "index per byte. For each example, the analyzer decomposes the target-side PQ distance margin by byte, selects "
            "the byte that most helps the gold candidate against the nearest wrong candidate, and then corrupts that byte. "
            "The adversarial replacement is an oracle analysis, not a deployable control; the public-mean replacement is "
            "a stronger source-erasure test because it uses only train/public codebook statistics."
        ),
        "inputs": {
            "train_examples": train_examples,
            "eval_examples": eval_examples,
            "feature_dim": feature_dim,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "candidate_view": candidate_view,
            "bootstrap_samples": bootstrap_samples,
        },
    }
    json_path = output_dir / "product_codebook_knockout_stress.json"
    md_path = output_dir / "product_codebook_knockout_stress.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": ["product_codebook_knockout_stress.json", "product_codebook_knockout_stress.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "product_codebook_knockout_stress.json": _sha256_file(json_path),
            "product_codebook_knockout_stress.md": _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
        "adversarial_pass_gate": payload["adversarial_pass_gate"],
        "public_mean_pass_gate": payload["public_mean_pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Product-Codebook Knockout Stress Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- adversarial pass gate: `{payload['adversarial_pass_gate']}`",
                f"- public-mean pass gate: `{payload['public_mean_pass_gate']}`",
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
        "# Source-Private Product-Codebook Knockout Stress",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- adversarial pass gate: `{payload['adversarial_pass_gate']}`",
        f"- public-mean pass gate: `{payload['public_mean_pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- adversarial pass rows: `{h['adversarial_pass_rows']}`",
        f"- public-mean pass rows: `{h['public_mean_pass_rows']}`",
        f"- adversarial remaps with pass: `{h['adversarial_remaps_with_pass']}`",
        f"- public-mean remaps with pass: `{h['public_mean_remaps_with_pass']}`",
        f"- min top-worst lift removed fraction: `{_fmt(h['min_top_worst_lift_removed_fraction'])}`",
        f"- min top-mean lift removed fraction: `{_fmt(h['min_top_mean_lift_removed_fraction'])}`",
        f"- max random-mean lift removed fraction: `{_fmt(h['max_random_mean_lift_removed_fraction'])}`",
        f"- min source-target lift: `{_fmt(h['min_source_minus_target'])}`",
        f"- min unique payloads: `{h['min_unique_payloads']}`",
        f"- max payload frequency: `{h['max_payload_frequency']}`",
        f"- min payload entropy bits: `{_fmt(h['min_payload_entropy_bits'])}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | N | Source | Target | Top worst | Top mean | Random mean | Random random | Top only | Mean payload | Top worst lift removed | Top mean lift removed | Random mean lift removed | Adv pass | Public pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['source_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['top_worst_accuracy']:.3f} | {row['top_mean_accuracy']:.3f} | "
            f"{row['random_mean_accuracy']:.3f} | {row['random_random_accuracy']:.3f} | "
            f"{row['top_only_accuracy']:.3f} | {row['mean_payload_accuracy']:.3f} | "
            f"{_fmt(row['top_worst_lift_removed_fraction'])} | "
            f"{_fmt(row['top_mean_lift_removed_fraction'])} | "
            f"{_fmt(row['random_mean_lift_removed_fraction'])} | "
            f"`{row['adversarial_knockout_pass']}` | `{row['public_mean_knockout_pass']}` |"
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
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_product_codebook_knockout_stress_20260430"))
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4])
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "slot"], default="slot")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_knockout_stress(
        output_dir=out,
        remap_seeds=args.remap_seeds,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        feature_dim=args.feature_dim,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        candidate_view=args.candidate_view,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload["headline"] | {"pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
