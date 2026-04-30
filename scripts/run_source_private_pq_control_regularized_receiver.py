from __future__ import annotations

import argparse
import datetime as dt
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
    GeometryProductCodebook,
    _answer_index,
    _dimension_utilities,
    _fit_geometry_codebook,
    _geometry_packet,
    _groups_for_variant,
    _payload_for_condition,
    _permute_payload,
)
from scripts.run_source_private_hidden_repair_packet_smoke import Example, _prior_prediction, make_benchmark  # noqa: E402
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _constrained_nonself_index,
    _fit_ridge_encoder_for_view,
    _project_source,
    _remap_candidate_slots,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


CONTROL_CONDITIONS = [
    "label_shuffled_ridge",
    "constrained_shuffled_source",
    "answer_masked_source",
    "permuted_codes",
    "random_same_byte",
    "deranged_public_table",
]
CONDITIONS = ["target_only", "source", *CONTROL_CONDITIONS]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_vectors(
    example: Example,
    *,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    candidate_view: str,
    permutation: np.ndarray | None,
) -> np.ndarray:
    candidates = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
    if codebook.rotation is not None:
        candidates = candidates @ codebook.rotation
    if permutation is not None:
        candidates = candidates[permutation]
    return candidates.astype(np.float32)


def _codes_for_candidates(candidates: np.ndarray, *, codebook: GeometryProductCodebook) -> np.ndarray:
    rows: list[list[int]] = []
    for candidate in candidates:
        codes: list[int] = []
        for sub_centroids, group in zip(codebook.centroids, codebook.groups, strict=True):
            part = candidate[group]
            distances = np.sum((sub_centroids - part[None, :]) ** 2, axis=1)
            codes.append(int(np.argmin(distances)))
        rows.append(codes)
    return np.asarray(rows, dtype=np.int64)


def _payload_codes(payload: bytes | None, *, codebook: GeometryProductCodebook) -> np.ndarray | None:
    if not payload:
        return None
    raw = np.frombuffer(payload[: codebook.subspaces], dtype=np.uint8).astype(np.int64)
    if raw.shape[0] < codebook.subspaces:
        return None
    return np.asarray(
        [int(code) % codebook.centroids[idx].shape[0] for idx, code in enumerate(raw[: codebook.subspaces])],
        dtype=np.int64,
    )


def _features_and_l2_prediction(
    example: Example,
    payload: bytes | None,
    *,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    candidate_view: str,
    permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    prior = _prior_prediction(example)
    candidates = _candidate_vectors(
        example,
        codebook=codebook,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
        permutation=permutation,
    )
    has_packet = float(payload is not None and len(payload) > 0)
    payload_codes = _payload_codes(payload, codebook=codebook)
    if payload_codes is None:
        distances = np.zeros(len(example.candidates), dtype=np.float32)
        ranks = np.zeros(len(example.candidates), dtype=np.float32)
        equal_frac = np.zeros(len(example.candidates), dtype=np.float32)
        best_distance = 0.0
        gap = 0.0
    else:
        reconstructed = np.zeros(feature_dim, dtype=np.float32)
        for subspace_index, code in enumerate(payload_codes):
            reconstructed[codebook.groups[subspace_index]] = codebook.centroids[subspace_index][int(code)]
        distances = np.sum((candidates - reconstructed[None, :]) ** 2, axis=1).astype(np.float32)
        ranks = np.argsort(np.argsort(distances, kind="stable"), kind="stable").astype(np.float32)
        candidate_codes = _codes_for_candidates(candidates, codebook=codebook)
        equal_frac = np.mean(candidate_codes == payload_codes[None, :], axis=1).astype(np.float32)
        sorted_distances = np.sort(distances)
        best_distance = float(sorted_distances[0])
        gap = float(sorted_distances[1] - sorted_distances[0]) if len(sorted_distances) > 1 else 0.0
    if has_packet:
        min_distance = float(np.min(distances))
        tied = np.flatnonzero(np.isclose(distances, min_distance, rtol=1e-6, atol=1e-8))
        if any(example.candidates[int(idx)].label == prior for idx in tied):
            l2_prediction = prior
        else:
            l2_prediction = example.candidates[int(tied[0])].label
    else:
        l2_prediction = prior
    norm_dist = np.log1p(distances / max(1.0, float(feature_dim)))
    best_norm = float(np.log1p(best_distance / max(1.0, float(feature_dim))))
    gap_norm = float(np.log1p(max(gap, 0.0) / max(1.0, float(feature_dim))))
    spread = float(np.std(norm_dist) + 1e-6)
    rows: list[list[float]] = []
    for idx, candidate in enumerate(example.candidates):
        is_prior = float(candidate.label == prior)
        rank_norm = float(ranks[idx]) / max(1.0, float(len(example.candidates) - 1))
        is_best = float(rank_norm == 0.0 and has_packet)
        rows.append(
            [
                1.0,
                has_packet,
                float(candidate.prior_score),
                is_prior,
                float(norm_dist[idx]),
                -float(norm_dist[idx]),
                rank_norm,
                is_best,
                float(equal_frac[idx]),
                best_norm,
                gap_norm,
                spread,
                is_prior * is_best,
                is_prior * float(equal_frac[idx]),
                is_prior * -float(norm_dist[idx]),
            ]
        )
    return np.asarray(rows, dtype=np.float32), l2_prediction, {
        "l2_min_distance": best_distance,
        "l2_gap": gap,
        "deranged_public_table": permutation is not None,
    }


def _derangement(length: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(32):
        perm = rng.permutation(length)
        if np.all(perm != np.arange(length)):
            return perm.astype(np.int64)
    return np.roll(np.arange(length), 1).astype(np.int64)


def _fit_receiver(
    train_rows: list[Example],
    *,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    candidate_view: str,
    receiver_ridge: float,
    seed: int,
    matched_weight: float,
    control_weight: float,
    target_weight: float,
    deranged_weight: float,
    random_rounds: int,
) -> np.ndarray:
    rng = random.Random(seed)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    weights: list[float] = []

    def add_view(
        example: Example,
        payload: bytes | None,
        *,
        target_label: str,
        weight: float,
        permutation: np.ndarray | None = None,
    ) -> None:
        features, _, _ = _features_and_l2_prediction(
            example,
            payload,
            codebook=codebook,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            permutation=permutation,
        )
        for candidate, feature in zip(example.candidates, features, strict=True):
            x_rows.append(feature)
            y_rows.append(float(candidate.label == target_label))
            weights.append(weight)

    for index, example in enumerate(train_rows):
        prior = _prior_prediction(example)
        matched = _geometry_packet(example, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="matched")
        add_view(example, None, target_label=prior, weight=target_weight)
        add_view(example, matched, target_label=example.answer_label, weight=matched_weight)
        control_payloads = [
            _geometry_packet(example, encoder=label_shuffle_encoder, codebook=codebook, feature_dim=feature_dim, mode="matched"),
            _geometry_packet(
                train_rows[_constrained_nonself_index(index, train_rows)],
                encoder=encoder,
                codebook=codebook,
                feature_dim=feature_dim,
                mode="matched",
            ),
            _geometry_packet(example, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="answer_masked"),
            _permute_payload(matched),
        ]
        for payload in control_payloads:
            add_view(example, payload, target_label=prior, weight=control_weight)
        for _ in range(random_rounds):
            add_view(example, rng.randbytes(codebook.subspaces), target_label=prior, weight=control_weight)
        add_view(
            example,
            matched,
            target_label=prior,
            weight=deranged_weight,
            permutation=_derangement(len(example.candidates), seed=seed * 1009 + index),
        )

    x = np.stack(x_rows, axis=0).astype(np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    sample_weights = np.sqrt(np.asarray(weights, dtype=np.float64))
    xw = x * sample_weights[:, None]
    yw = y * sample_weights
    xtx = xw.T @ xw
    xtx += receiver_ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= receiver_ridge
    return np.linalg.solve(xtx, xw.T @ yw).astype(np.float32)


def _payload_for_eval_condition(
    *,
    condition: str,
    example: Example,
    eval_rows: list[Example],
    index: int,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    rng: random.Random,
) -> bytes | None:
    if condition == "deranged_public_table":
        return _geometry_packet(example, encoder=encoder, codebook=codebook, feature_dim=feature_dim, mode="matched")
    return _payload_for_condition(
        condition=condition,
        example=example,
        eval_rows=eval_rows,
        index=index,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        feature_dim=feature_dim,
        rng=rng,
    )


def _predict_rows(
    eval_rows: list[Example],
    *,
    encoder: np.ndarray,
    label_shuffle_encoder: np.ndarray,
    codebook: GeometryProductCodebook,
    feature_dim: int,
    candidate_view: str,
    receiver_weights: np.ndarray,
    seed: int,
    conditions: list[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(eval_rows):
        for condition in conditions:
            start = time.perf_counter()
            payload = _payload_for_eval_condition(
                condition=condition,
                example=example,
                eval_rows=eval_rows,
                index=index,
                encoder=encoder,
                label_shuffle_encoder=label_shuffle_encoder,
                codebook=codebook,
                feature_dim=feature_dim,
                rng=rng,
            )
            permutation = (
                _derangement(len(example.candidates), seed=seed * 2003 + index)
                if condition == "deranged_public_table"
                else None
            )
            features, l2_prediction, metadata = _features_and_l2_prediction(
                example,
                payload,
                codebook=codebook,
                feature_dim=feature_dim,
                candidate_view=candidate_view,
                permutation=permutation,
            )
            scores = features @ receiver_weights
            max_score = float(np.max(scores))
            tied = np.flatnonzero(np.isclose(scores, max_score, rtol=1e-6, atol=1e-8))
            prior = _prior_prediction(example)
            if any(example.candidates[int(idx)].label == prior for idx in tied):
                learned_prediction = prior
            else:
                learned_prediction = example.candidates[int(tied[0])].label
            latency_ms = (time.perf_counter() - start) * 1000.0
            payload_hex = (payload or b"").hex()
            rows.append(
                {
                    "example_id": example.example_id,
                    "family_name": example.family_name,
                    "condition": condition,
                    "answer_label": example.answer_label,
                    "target_prior_label": prior,
                    "payload_hex": payload_hex,
                    "payload_bytes": len(payload or b""),
                    "payload_tokens": _token_count(payload_hex),
                    "learned_prediction": learned_prediction,
                    "learned_correct": learned_prediction == example.answer_label,
                    "l2_prediction": l2_prediction,
                    "l2_correct": l2_prediction == example.answer_label,
                    "learned_scores": [float(value) for value in scores],
                    "latency_ms": latency_ms,
                    "metadata": metadata,
                }
            )
    return rows


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    correct = [row["example_id"] for row in rows if row[key]]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "correct": len(correct),
        "accuracy": len(correct) / len(rows),
        "correct_ids": correct,
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
    key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    ids = sorted({row["example_id"] for row in rows})
    by = {(row["condition"], row["example_id"]): bool(row[key]) for row in rows}
    diffs = np.asarray(
        [float(by[(condition_a, example_id)]) - float(by[(condition_b, example_id)]) for example_id in ids],
        dtype=np.float32,
    )
    point = float(np.mean(diffs)) if len(diffs) else 0.0
    if len(diffs) <= 1:
        return {"point": point, "ci95_low": point, "ci95_high": point}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float32)
    for idx in range(samples):
        sampled = rng.integers(0, len(diffs), size=len(diffs))
        boot[idx] = float(np.mean(diffs[sampled]))
    return {
        "point": point,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    conditions: list[str],
    bootstrap_samples: int,
    seed: int,
    tolerance_vs_l2: float,
) -> dict[str, Any]:
    example_ids = sorted({row["example_id"] for row in rows})
    condition_ids = {
        condition: [row["example_id"] for row in rows if row["condition"] == condition]
        for condition in conditions
    }
    exact_id_parity = all(
        len(ids) == len(example_ids) and len(set(ids)) == len(example_ids) and set(ids) == set(example_ids)
        for ids in condition_ids.values()
    )
    learned_metrics = {
        condition: _metric([row for row in rows if row["condition"] == condition], "learned_correct")
        for condition in conditions
    }
    l2_metrics = {
        condition: _metric([row for row in rows if row["condition"] == condition], "l2_correct")
        for condition in conditions
    }
    target = learned_metrics["target_only"]["accuracy"]
    source = learned_metrics["source"]["accuracy"]
    l2_source = l2_metrics["source"]["accuracy"]
    best_control_condition = max(CONTROL_CONDITIONS, key=lambda condition: learned_metrics[condition]["accuracy"])
    best_control = learned_metrics[best_control_condition]["accuracy"]
    controls_ok = all(learned_metrics[condition]["accuracy"] <= target + 0.06 for condition in CONTROL_CONDITIONS)
    source_packet_pass = (
        exact_id_parity
        and source >= target + 0.15
        and source >= best_control + 0.15
        and controls_ok
    )
    pass_gate = source_packet_pass and source >= l2_source - tolerance_vs_l2
    return {
        "n": len(example_ids),
        "conditions": conditions,
        "exact_id_parity": exact_id_parity,
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "target_only_accuracy": target,
        "learned_source_accuracy": source,
        "l2_source_accuracy": l2_source,
        "learned_minus_target": source - target,
        "learned_minus_l2": source - l2_source,
        "best_control_condition": best_control_condition,
        "best_control_accuracy": best_control,
        "learned_minus_best_control": source - best_control,
        "controls_ok": controls_ok,
        "source_packet_pass": source_packet_pass,
        "pass_gate": pass_gate,
        "learned_metrics": learned_metrics,
        "l2_metrics": l2_metrics,
        "paired_bootstrap": {
            "learned_source_vs_target": _paired_bootstrap(
                rows,
                condition_a="source",
                condition_b="target_only",
                key="learned_correct",
                samples=bootstrap_samples,
                seed=seed + 17,
            ),
            "learned_source_vs_best_control": _paired_bootstrap(
                rows,
                condition_a="source",
                condition_b=best_control_condition,
                key="learned_correct",
                samples=bootstrap_samples,
                seed=seed + 31,
            ),
        },
        "pass_rule": (
            "Pass requires exact ID parity; learned source >= target+0.15; learned source >= best destructive "
            "control+0.15; all controls including deranged public table <= target+0.06; and learned source "
            f"within {tolerance_vs_l2:.2f} accuracy of deterministic PQ L2 decoding."
        ),
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Source-Private PQ Control-Regularized Receiver",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- examples: `{summary['n']}`",
        f"- remap seed: `{payload['remap_slot_seed']}`",
        f"- variant: `{payload['variant']}`",
        f"- learned source accuracy: `{summary['learned_source_accuracy']:.3f}`",
        f"- deterministic L2 source accuracy: `{summary['l2_source_accuracy']:.3f}`",
        f"- target accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best learned control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
        f"- learned minus best control: `{summary['learned_minus_best_control']:.3f}`",
        "",
        "| Condition | Learned acc | L2 acc | Mean bytes | p50 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        learned = summary["learned_metrics"][condition]
        l2 = summary["l2_metrics"][condition]
        lines.append(
            f"| {condition} | {learned['accuracy']:.3f} | {l2['accuracy']:.3f} | "
            f"{learned['mean_payload_bytes']:.2f} | {learned['p50_latency_ms']:.4f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_start_index: int,
    eval_start_index: int,
    train_family_set: str,
    eval_family_set: str,
    diagnostic_table_mode: str,
    candidates: int,
    feature_dim: int,
    budget_bytes: int,
    variant: str,
    remap_slot_seed: int,
    ridge: float,
    receiver_ridge: float,
    candidate_view: str,
    fit_intercept: bool,
    opq_iterations: int,
    seed: int,
    matched_weight: float,
    control_weight: float,
    target_weight: float,
    deranged_weight: float,
    random_rounds: int,
    bootstrap_samples: int,
    tolerance_vs_l2: float,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(
        examples=train_examples,
        candidates=candidates,
        seed=train_seed,
        family_set=train_family_set,
        start_index=train_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    eval_rows = make_benchmark(
        examples=eval_examples,
        candidates=candidates,
        seed=eval_seed,
        family_set=eval_family_set,
        start_index=eval_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_slot_seed)
    eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_slot_seed)
    encoder = _fit_ridge_encoder_for_view(
        train_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        candidate_view=candidate_view,
        fit_intercept=fit_intercept,
    )
    label_shuffle_encoder = _fit_ridge_encoder_for_view(
        train_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        candidate_view=candidate_view,
        fit_intercept=fit_intercept,
        label_shuffle_seed=train_seed * 5003 + eval_seed,
    )
    utilities = _dimension_utilities(train_rows, encoder=encoder, feature_dim=feature_dim, candidate_view=candidate_view)
    groups = _groups_for_variant(
        variant=variant,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        utilities=utilities,
        seed=train_seed * 11003 + eval_seed * 97 + budget_bytes + remap_slot_seed,
    )
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
    receiver_weights = _fit_receiver(
        train_rows,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
        receiver_ridge=receiver_ridge,
        seed=seed,
        matched_weight=matched_weight,
        control_weight=control_weight,
        target_weight=target_weight,
        deranged_weight=deranged_weight,
        random_rounds=random_rounds,
    )
    eval_conditions = list(conditions or CONDITIONS)
    rows = _predict_rows(
        eval_rows,
        encoder=encoder,
        label_shuffle_encoder=label_shuffle_encoder,
        codebook=codebook,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
        receiver_weights=receiver_weights,
        seed=seed + 104729,
        conditions=eval_conditions,
    )
    summary = _summarize(
        rows,
        conditions=eval_conditions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        tolerance_vs_l2=tolerance_vs_l2,
    )
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "gate": "source_private_pq_control_regularized_receiver",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_start_index": train_start_index,
        "eval_start_index": eval_start_index,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "diagnostic_table_mode": diagnostic_table_mode,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "budget_bytes": budget_bytes,
        "variant": variant,
        "remap_slot_seed": remap_slot_seed,
        "ridge": ridge,
        "receiver_ridge": receiver_ridge,
        "candidate_view": candidate_view,
        "fit_intercept": fit_intercept,
        "opq_iterations": opq_iterations,
        "seed": seed,
        "matched_weight": matched_weight,
        "control_weight": control_weight,
        "target_weight": target_weight,
        "deranged_weight": deranged_weight,
        "random_rounds": random_rounds,
        "tolerance_vs_l2": tolerance_vs_l2,
        "train_eval_id_intersection_count": len({row.example_id for row in train_rows}.intersection(row.example_id for row in eval_rows)),
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "codebook_sha256": hashlib.sha256(
            b"".join(centroid.tobytes() for centroid in codebook.centroids)
            + b"".join(group.tobytes() for group in codebook.groups)
            + (codebook.rotation.tobytes() if codebook.rotation is not None else b"")
        ).hexdigest(),
        "receiver_weights_sha256": hashlib.sha256(receiver_weights.tobytes()).hexdigest(),
        "summary": summary,
        "pass_gate": summary["pass_gate"],
        "prediction_file": "predictions.jsonl",
        "interpretation": (
            "A ridge-trained candidate scorer consumes source-private PQ bytes plus public candidate side "
            "information. Training asks real source packets to select gold while target-only, source-destroying, "
            "random, and deranged-public-table controls fall back to the target prior."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    (output_dir / "receiver_weights.json").write_text(
        json.dumps([float(value) for value in receiver_weights], indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifacts": [
            "run_summary.json",
            "summary.json",
            "summary.md",
            "predictions.jsonl",
            "receiver_weights.json",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["run_summary.json", "summary.json", "summary.md", "predictions.jsonl", "receiver_weights.json"]
        },
        "pass_gate": payload["pass_gate"],
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private PQ Control-Regularized Receiver Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- learned source accuracy: `{summary['learned_source_accuracy']:.3f}`",
                f"- best control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=29)
    parser.add_argument("--train-start-index", type=int, default=10000)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="legacy")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument(
        "--variant",
        choices=[
            "canonical",
            "utility_balanced",
            "opq_procrustes",
            "utility_opq_procrustes",
            "protected_hadamard",
            "utility_protected_hadamard",
        ],
        default="utility_protected_hadamard",
    )
    parser.add_argument("--remap-slot-seed", type=int, default=101)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", default="slot")
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--opq-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--matched-weight", type=float, default=4.0)
    parser.add_argument("--control-weight", type=float, default=2.0)
    parser.add_argument("--target-weight", type=float, default=2.0)
    parser.add_argument("--deranged-weight", type=float, default=0.5)
    parser.add_argument("--random-rounds", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--tolerance-vs-l2", type=float, default=0.08)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="*", default=None)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=output_dir,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        diagnostic_table_mode=args.diagnostic_table_mode,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        budget_bytes=args.budget_bytes,
        variant=args.variant,
        remap_slot_seed=args.remap_slot_seed,
        ridge=args.ridge,
        receiver_ridge=args.receiver_ridge,
        candidate_view=args.candidate_view,
        fit_intercept=args.fit_intercept,
        opq_iterations=args.opq_iterations,
        seed=args.seed,
        matched_weight=args.matched_weight,
        control_weight=args.control_weight,
        target_weight=args.target_weight,
        deranged_weight=args.deranged_weight,
        random_rounds=args.random_rounds,
        bootstrap_samples=args.bootstrap_samples,
        tolerance_vs_l2=args.tolerance_vs_l2,
        conditions=args.conditions,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pass_gate": payload["pass_gate"],
                "learned_source_accuracy": payload["summary"]["learned_source_accuracy"],
                "l2_source_accuracy": payload["summary"]["l2_source_accuracy"],
                "best_control_accuracy": payload["summary"]["best_control_accuracy"],
                "best_control_condition": payload["summary"]["best_control_condition"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
