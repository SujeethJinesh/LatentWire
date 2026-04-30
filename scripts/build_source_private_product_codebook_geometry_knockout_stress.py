from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts.build_source_private_product_codebook_geometry_gate import (  # noqa: E402
    CONTROL_CONDITIONS,
    GeometryProductCodebook,
    _dimension_utilities,
    _fit_geometry_codebook,
    _geometry_packet,
    _groups_for_variant,
    _payload_for_condition,
)
from scripts.build_source_private_product_codebook_knockout_stress import (  # noqa: E402
    _bootstrap_ci,
    _paired_counts,
    _paired_values,
    _payload_entropy_summary,
    _sha256_file,
)
from scripts.run_source_private_hidden_repair_packet_smoke import make_benchmark  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _fit_ridge_encoder_for_view,
    _prior_prediction,
    _project_source,
    _remap_candidate_slots,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


STRESS_CONDITIONS = [
    "target_only",
    "source",
    *CONTROL_CONDITIONS,
    "top_codeword_removed_mean",
    "top_codeword_removed_worst",
    "random_codeword_removed_mean",
    "random_codeword_removed_random",
    "top_codeword_only",
    "mean_payload",
]


def _answer_index(example: Any) -> int:
    return next(index for index, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _geometry_distance_tables(
    candidate_matrix: np.ndarray,
    *,
    codebook: GeometryProductCodebook,
) -> tuple[np.ndarray, ...]:
    rotated_candidates = candidate_matrix
    if codebook.rotation is not None:
        rotated_candidates = candidate_matrix @ codebook.rotation
    tables: list[np.ndarray] = []
    for sub_centroids, group in zip(codebook.centroids, codebook.groups, strict=True):
        candidate_part = rotated_candidates[:, group].astype(np.float32)
        table = np.sum((candidate_part[:, None, :] - sub_centroids[None, :, :]) ** 2, axis=2)
        tables.append(table.astype(np.float32))
    return tuple(tables)


def _decode_table(example: Any, payload: bytes | None, *, distance_tables: tuple[np.ndarray, ...]) -> str:
    if not payload:
        return _prior_prediction(example)
    raw_codes = np.frombuffer(payload[: len(distance_tables)], dtype=np.uint8)
    scores = np.zeros(distance_tables[0].shape[0], dtype=np.float32)
    for subspace_index, code in enumerate(raw_codes):
        table = distance_tables[subspace_index]
        scores += table[:, int(code) % table.shape[1]]
    min_score = float(np.min(scores))
    tied = np.flatnonzero(np.isclose(scores, min_score, rtol=1e-6, atol=1e-8))
    prior = _prior_prediction(example)
    if any(example.candidates[int(index)].label == prior for index in tied):
        return prior
    return example.candidates[int(tied[0])].label


def _decode_scores(payload: bytes, *, distance_tables: tuple[np.ndarray, ...]) -> np.ndarray:
    raw_codes = np.frombuffer(payload[: len(distance_tables)], dtype=np.uint8)
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
    top_index = max(range(len(contributions)), key=lambda index: (contributions[index], -index))
    return top_index, contributions


def _replace_code(payload: bytes, *, subspace_index: int, replacement: int) -> bytes:
    codes = bytearray(payload)
    codes[subspace_index] = int(replacement) % 256
    return bytes(codes)


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


def _random_different_code(
    *,
    codebook: GeometryProductCodebook,
    subspace_index: int,
    old_code: int,
    seed: int,
) -> int:
    centroid_count = codebook.centroids[subspace_index].shape[0]
    if centroid_count <= 1:
        return old_code
    rng = random.Random(seed)
    candidate = rng.randrange(centroid_count - 1)
    if candidate >= old_code % centroid_count:
        candidate += 1
    return candidate


def _encode_rotated_vector(vector: np.ndarray, *, codebook: GeometryProductCodebook) -> bytes:
    rotated = vector
    if codebook.rotation is not None:
        rotated = vector @ codebook.rotation
    codes = np.zeros(codebook.subspaces, dtype=np.uint8)
    for subspace_index, (sub_centroids, group) in enumerate(zip(codebook.centroids, codebook.groups, strict=True)):
        part = rotated[group]
        distances = np.sum((sub_centroids - part[None, :]) ** 2, axis=1)
        codes[subspace_index] = int(np.argmin(distances))
    return codes.tobytes()


def _mean_payload(
    train_rows: list[Any],
    *,
    encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
) -> bytes:
    vectors = np.stack(
        [_project_source(example, encoder=encoder, feature_dim=feature_dim, mode="matched") for example in train_rows],
        axis=0,
    ).astype(np.float32)
    return _encode_rotated_vector(vectors.mean(axis=0), codebook=codebook)


def _stress_payloads(
    *,
    example: Any,
    example_index: int,
    encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    public_mean_payload: bytes,
    distance_tables: tuple[np.ndarray, ...],
    seed: int,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    source_payload = _geometry_packet(
        example,
        encoder=encoder,
        codebook=codebook,
        feature_dim=feature_dim,
        mode="matched",
    )
    scores = _decode_scores(source_payload, distance_tables=distance_tables)
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
    rng = random.Random(seed + example_index * 9173)
    random_index = rng.randrange(max(1, len(source_payload)))
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
    payloads = {
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


def _predict(
    *,
    condition: str,
    example: Any,
    payload: bytes | None,
    distance_tables: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    start = time.perf_counter()
    prediction = _decode_table(example, payload, distance_tables=distance_tables)
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


def _payload_reuse_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        payload = str(row["payload_hex"])
        counts[payload] = counts.get(payload, 0) + 1
    singleton_rows = [row for row in rows if counts[str(row["payload_hex"])] == 1]
    collision_rows = [row for row in rows if counts[str(row["payload_hex"])] > 1]

    def accuracy(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(1 for row in subset if row["correct"]) / len(subset)

    return {
        "singleton_n": len(singleton_rows),
        "singleton_accuracy": accuracy(singleton_rows),
        "collision_n": len(collision_rows),
        "collision_accuracy": accuracy(collision_rows),
        "max_payload_frequency": max(counts.values(), default=0),
    }


def build_geometry_knockout_stress(
    *,
    output_dir: pathlib.Path,
    remap_seeds: list[int],
    budgets: list[int],
    variants: list[str],
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    train_seed: int,
    eval_seed: int,
    candidate_view: str,
    bootstrap_samples: int,
    opq_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        label_shuffle_encoder = _fit_ridge_encoder_for_view(
            train_rows,
            feature_dim=feature_dim,
            ridge=1e-2,
            candidate_view=candidate_view,
            fit_intercept=False,
            label_shuffle_seed=train_seed * 5003 + eval_seed,
        )
        utilities = _dimension_utilities(train_rows, encoder=encoder, feature_dim=feature_dim, candidate_view=candidate_view)
        for budget in budgets:
            for variant in variants:
                groups = _groups_for_variant(
                    variant=variant,
                    feature_dim=feature_dim,
                    budget_bytes=budget,
                    utilities=utilities,
                    seed=train_seed * 11003 + eval_seed * 97 + budget + remap_seed,
                )
                codebook = _fit_geometry_codebook(
                    train_rows,
                    encoder=encoder,
                    feature_dim=feature_dim,
                    groups=groups,
                    variant=variant,
                    utilities=utilities,
                    seed=train_seed * 9001 + eval_seed * 17 + budget,
                    opq_iterations=opq_iterations if variant in {"opq_procrustes", "utility_opq_procrustes"} else 0,
                )
                public_mean_payload = _mean_payload(
                    train_rows,
                    encoder=encoder,
                    codebook=codebook,
                    feature_dim=feature_dim,
                )
                run_dir = output_dir / f"remap_{remap_seed}" / f"budget_{budget}" / variant
                run_dir.mkdir(parents=True, exist_ok=True)
                rng = random.Random(train_seed * 4001 + eval_seed + remap_seed * 13 + budget)
                by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in STRESS_CONDITIONS}
                metadata_rows: list[dict[str, Any]] = []
                for example_index, example in enumerate(eval_rows):
                    candidate_matrix = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
                    distance_tables = _geometry_distance_tables(candidate_matrix, codebook=codebook)
                    payloads: dict[str, bytes | None] = {
                        condition: _payload_for_condition(
                            condition=condition,
                            example=example,
                            eval_rows=eval_rows,
                            index=example_index,
                            encoder=encoder,
                            label_shuffle_encoder=label_shuffle_encoder,
                            codebook=codebook,
                            feature_dim=feature_dim,
                            rng=rng,
                        )
                        for condition in ["target_only", "source", *CONTROL_CONDITIONS]
                    }
                    stress_payloads, metadata = _stress_payloads(
                        example=example,
                        example_index=example_index,
                        encoder=encoder,
                        codebook=codebook,
                        feature_dim=feature_dim,
                        public_mean_payload=public_mean_payload,
                        distance_tables=distance_tables,
                        seed=train_seed * 4001 + eval_seed * 101 + remap_seed * 17 + budget,
                    )
                    payloads.update(stress_payloads)
                    metadata_rows.append({"example_id": example.example_id, **metadata})
                    for condition, payload in payloads.items():
                        by_condition[condition].append(
                            _predict(
                                condition=condition,
                                example=example,
                                payload=payload,
                                distance_tables=distance_tables,
                            )
                            | {
                                "example_id": example.example_id,
                                "family_name": example.family_name,
                                "budget_bytes": budget,
                                "remap_slot_seed": remap_seed,
                                "variant": variant,
                                "top_subspace_index": metadata["top_subspace_index"],
                                "random_subspace_index": metadata["random_subspace_index"],
                                "matched_margin": metadata["matched_margin"],
                                "top_margin_contribution": metadata["top_margin_contribution"],
                                "random_margin_contribution": metadata["random_margin_contribution"],
                            }
                        )
                prediction_path = run_dir / "predictions.jsonl"
                prediction_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for condition in STRESS_CONDITIONS
                        for row in by_condition[condition]
                    ),
                    encoding="utf-8",
                )
                metadata_path = run_dir / "metadata.jsonl"
                metadata_path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows),
                    encoding="utf-8",
                )
                metrics = {condition: _condition_summary(condition_rows) for condition, condition_rows in by_condition.items()}
                grouped: dict[str, dict[str, dict[str, Any]]] = {}
                for condition_rows in by_condition.values():
                    for row in condition_rows:
                        grouped.setdefault(row["example_id"], {})[row["condition"]] = row
                exact_id_parity = all(
                    {row["example_id"] for row in condition_rows} == set(grouped)
                    for condition_rows in by_condition.values()
                )
                comparisons = {}
                for baseline in [condition for condition in STRESS_CONDITIONS if condition != "source"]:
                    values = _paired_values(grouped, method="source", baseline=baseline)
                    comparisons[baseline] = {
                        "accuracy": metrics[baseline]["accuracy"],
                        "paired_delta": _bootstrap_ci(
                            values,
                            samples=bootstrap_samples,
                            seed=train_seed * 5003 + eval_seed * 101 + remap_seed * 37 + budget * 1009 + len(comparisons),
                        ),
                        "paired_counts": _paired_counts(values),
                    }
                source_acc = metrics["source"]["accuracy"]
                target_acc = metrics["target_only"]["accuracy"]
                lift = source_acc - target_acc

                def removed_fraction(condition: str) -> float | None:
                    if lift <= 1e-9:
                        return None
                    return (source_acc - metrics[condition]["accuracy"]) / lift

                control_accuracies = {condition: metrics[condition]["accuracy"] for condition in CONTROL_CONDITIONS}
                best_control_name = max(control_accuracies, key=control_accuracies.get)
                controls_ok = all(accuracy <= target_acc + 0.05 for accuracy in control_accuracies.values())
                top_worst_removed = removed_fraction("top_codeword_removed_worst")
                top_mean_removed = removed_fraction("top_codeword_removed_mean")
                random_mean_removed = removed_fraction("random_codeword_removed_mean")
                row_payload = {
                    "remap_slot_seed": remap_seed,
                    "budget_bytes": budget,
                    "variant": variant,
                    "n": eval_examples,
                    "exact_id_parity": exact_id_parity,
                    "candidate_pool_recall": 1.0,
                    "source_accuracy": source_acc,
                    "target_accuracy": target_acc,
                    "best_control_condition": best_control_name,
                    "best_control_accuracy": control_accuracies[best_control_name],
                    "source_minus_target": lift,
                    "source_minus_best_control": source_acc - control_accuracies[best_control_name],
                    "controls_ok": controls_ok,
                    "source_packet_pass": exact_id_parity and controls_ok and lift >= 0.15,
                    "top_worst_accuracy": metrics["top_codeword_removed_worst"]["accuracy"],
                    "top_mean_accuracy": metrics["top_codeword_removed_mean"]["accuracy"],
                    "random_mean_accuracy": metrics["random_codeword_removed_mean"]["accuracy"],
                    "random_random_accuracy": metrics["random_codeword_removed_random"]["accuracy"],
                    "top_only_accuracy": metrics["top_codeword_only"]["accuracy"],
                    "mean_payload_accuracy": metrics["mean_payload"]["accuracy"],
                    "top_worst_lift_removed_fraction": top_worst_removed,
                    "top_mean_lift_removed_fraction": top_mean_removed,
                    "random_mean_lift_removed_fraction": random_mean_removed,
                    "public_mean_delta_over_random_mean": (
                        None
                        if top_mean_removed is None or random_mean_removed is None
                        else top_mean_removed - random_mean_removed
                    ),
                    "mean_top_margin_contribution": statistics.fmean(float(row["top_margin_contribution"]) for row in metadata_rows),
                    "mean_random_margin_contribution": statistics.fmean(float(row["random_margin_contribution"]) for row in metadata_rows),
                    "mean_matched_margin": statistics.fmean(float(row["matched_margin"]) for row in metadata_rows),
                    "mean_payload_bytes": metrics["source"]["mean_payload_bytes"],
                    "mean_payload_tokens": metrics["source"]["mean_payload_tokens"],
                    "payload_entropy": _payload_entropy_summary(by_condition["source"], budget=budget),
                    "payload_reuse": _payload_reuse_summary(by_condition["source"]),
                    "paired_comparisons": comparisons,
                    "metrics": metrics,
                    "predictions_file": str(prediction_path),
                    "metadata_file": str(metadata_path),
                    "variant_sha256": hashlib.sha256(
                        b"".join(centroid.tobytes() for centroid in codebook.centroids)
                        + b"".join(group.tobytes() for group in codebook.groups)
                        + (b"" if codebook.rotation is None else codebook.rotation.tobytes())
                    ).hexdigest(),
                }
                row_payload["adversarial_knockout_pass"] = (
                    row_payload["source_packet_pass"]
                    and top_worst_removed is not None
                    and top_worst_removed >= 0.50
                    and comparisons["top_codeword_removed_worst"]["paired_delta"]["ci95_low"] > 0.05
                )
                row_payload["public_mean_knockout_pass"] = (
                    row_payload["source_packet_pass"]
                    and top_mean_removed is not None
                    and random_mean_removed is not None
                    and top_mean_removed >= 0.25
                    and top_mean_removed >= random_mean_removed + 0.05
                    and comparisons["top_codeword_removed_mean"]["paired_delta"]["ci95_low"] > 0.05
                )
                rows.append(row_payload)
                (run_dir / "summary.json").write_text(json.dumps(row_payload, indent=2, sort_keys=True), encoding="utf-8")

    canonical_by_key = {
        (row["remap_slot_seed"], row["budget_bytes"]): row
        for row in rows
        if row["variant"] == "canonical"
    }
    for row in rows:
        canonical = canonical_by_key.get((row["remap_slot_seed"], row["budget_bytes"]))
        if canonical is None:
            continue
        row["source_minus_canonical"] = row["source_accuracy"] - canonical["source_accuracy"]
        row["top_mean_removed_minus_canonical"] = (
            None
            if row["top_mean_lift_removed_fraction"] is None or canonical["top_mean_lift_removed_fraction"] is None
            else row["top_mean_lift_removed_fraction"] - canonical["top_mean_lift_removed_fraction"]
        )
        row["unique_payload_delta_vs_canonical"] = row["payload_entropy"]["unique_payloads"] - canonical["payload_entropy"]["unique_payloads"]
        collision_accuracy = row["payload_reuse"]["collision_accuracy"]
        collision_lift_ok = (
            collision_accuracy is not None
            and row["payload_reuse"]["collision_n"] >= 50
            and collision_accuracy >= row["target_accuracy"] + 0.10
        )
        row["mitigation_pass"] = (
            row["variant"] != "canonical"
            and row["source_packet_pass"]
            and row["source_accuracy"] >= canonical["source_accuracy"] - 0.02
            and (
                (row["top_mean_removed_minus_canonical"] is not None and row["top_mean_removed_minus_canonical"] >= 0.05)
                or (row["unique_payload_delta_vs_canonical"] <= -25 and collision_lift_ok)
            )
        )
    mitigation_rows = [row for row in rows if row.get("mitigation_pass")]
    mitigation_remaps = sorted({row["remap_slot_seed"] for row in mitigation_rows})
    payload = {
        "gate": "source_private_product_codebook_geometry_knockout_stress",
        "rows": rows,
        "headline": {
            "rows": len(rows),
            "source_pass_rows": sum(1 for row in rows if row["source_packet_pass"]),
            "adversarial_pass_rows": sum(1 for row in rows if row["adversarial_knockout_pass"]),
            "public_mean_pass_rows": sum(1 for row in rows if row["public_mean_knockout_pass"]),
            "mitigation_pass_rows": len(mitigation_rows),
            "mitigation_remaps": mitigation_remaps,
            "variants": variants,
            "remap_seeds": remap_seeds,
            "budgets": budgets,
            "max_source_accuracy": max((row["source_accuracy"] for row in rows), default=None),
            "max_noncanonical_source_minus_canonical": max(
                (row.get("source_minus_canonical", -1.0) for row in rows if row["variant"] != "canonical"),
                default=None,
            ),
            "max_noncanonical_top_mean_removed_minus_canonical": max(
                (
                    row["top_mean_removed_minus_canonical"]
                    for row in rows
                    if row["variant"] != "canonical" and row.get("top_mean_removed_minus_canonical") is not None
                ),
                default=None,
            ),
            "min_noncanonical_unique_payload_delta_vs_canonical": min(
                (row["unique_payload_delta_vs_canonical"] for row in rows if row["variant"] != "canonical"),
                default=None,
            ),
        },
        "pass_gate": len(mitigation_remaps) == len(remap_seeds),
        "pass_rule": (
            "A noncanonical geometry variant must pass source controls, keep source accuracy within 0.02 of canonical "
            "PQ at the same remap/budget, and either improve public-mean top-codeword lift removal by >=0.05 or reduce "
            "unique matched payloads by at least 25 at n500 while the reused-payload subset still beats target by >=0.10."
        ),
        "interpretation": (
            "This gate tests whether OPQ/protected geometry mitigates the lookup-like uniqueness observed in the n500 "
            "canonical PQ packet while preserving the source-private control pass and byte-causality diagnostics."
        ),
        "inputs": {
            "train_examples": train_examples,
            "eval_examples": eval_examples,
            "feature_dim": feature_dim,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "candidate_view": candidate_view,
            "bootstrap_samples": bootstrap_samples,
            "opq_iterations": opq_iterations,
        },
    }
    json_path = output_dir / "product_codebook_geometry_knockout_stress.json"
    md_path = output_dir / "product_codebook_geometry_knockout_stress.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [
            "product_codebook_geometry_knockout_stress.json",
            "product_codebook_geometry_knockout_stress.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "product_codebook_geometry_knockout_stress.json": _sha256_file(json_path),
            "product_codebook_geometry_knockout_stress.md": _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Product-Codebook Geometry Knockout Stress Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
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
        "# Source-Private Product-Codebook Geometry Knockout Stress",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows: `{h['rows']}`",
        f"- source pass rows: `{h['source_pass_rows']}`",
        f"- adversarial pass rows: `{h['adversarial_pass_rows']}`",
        f"- public-mean pass rows: `{h['public_mean_pass_rows']}`",
        f"- mitigation pass rows: `{h['mitigation_pass_rows']}`",
        f"- mitigation remaps: `{h['mitigation_remaps']}`",
        f"- max noncanonical source-canonical: `{_fmt(h['max_noncanonical_source_minus_canonical'])}`",
        f"- max noncanonical top-mean-removed-canonical: `{_fmt(h['max_noncanonical_top_mean_removed_minus_canonical'])}`",
        f"- min noncanonical unique-payload delta vs canonical: `{_fmt(h['min_noncanonical_unique_payload_delta_vs_canonical'])}`",
        "",
        "## Rows",
        "",
        "| Remap | Budget | Variant | Source | Target | Best ctrl | Source-ctrl | Source-can | Top worst rem | Top mean rem | Random mean rem | Unique payloads | Collision n | Collision acc | Unique-can | Public pass | Mitigation |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['remap_slot_seed']} | {row['budget_bytes']} | {row['variant']} | "
            f"{row['source_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['source_minus_best_control']:.3f} | "
            f"{_fmt(row.get('source_minus_canonical'))} | {_fmt(row['top_worst_lift_removed_fraction'])} | "
            f"{_fmt(row['top_mean_lift_removed_fraction'])} | {_fmt(row['random_mean_lift_removed_fraction'])} | "
            f"{row['payload_entropy']['unique_payloads']} | {row['payload_reuse']['collision_n']} | "
            f"{_fmt(row['payload_reuse']['collision_accuracy'])} | {_fmt(row.get('unique_payload_delta_vs_canonical'))} | "
            f"`{row['public_mean_knockout_pass']}` | `{row.get('mitigation_pass', False)}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], "", "## Pass Rule", "", payload["pass_rule"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_product_codebook_geometry_knockout_stress_20260430"))
    parser.add_argument("--remap-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--variants",
        choices=[
            "canonical",
            "utility_round_robin",
            "utility_balanced",
            "random_balanced",
            "opq_procrustes",
            "utility_opq_procrustes",
            "protected_hadamard",
            "utility_protected_hadamard",
        ],
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
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "slot"], default="slot")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--opq-iterations", type=int, default=4)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_geometry_knockout_stress(
        output_dir=out,
        remap_seeds=args.remap_seeds,
        budgets=args.budgets,
        variants=args.variants,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        feature_dim=args.feature_dim,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        candidate_view=args.candidate_view,
        bootstrap_samples=args.bootstrap_samples,
        opq_iterations=args.opq_iterations,
    )
    print(json.dumps(payload["headline"] | {"pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
