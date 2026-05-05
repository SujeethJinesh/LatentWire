from __future__ import annotations

"""ARC tokenwise source-evidence repair readout preflight.

This is a discriminative signal probe, not a final packet method.  It asks
whether answer-key-forbidden source token traces contain row-specific evidence
that a tiny held-out receiver can use to repair target errors beyond public
text, source-choice, same-byte, and destructive source controls.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_tokenwise_repair_readout_preflight_20260504_n32_qwen05_to_qwen3"
)
DEFAULT_AUDIT = pathlib.Path(
    "results/source_private_arc_openbookqa_soft_prefix_preflight_20260503_"
    "arc_qwen_token_pool_residual_n32_cpu_label_choice/prediction_audit.jsonl"
)

MATCHED_CONDITION = "matched_tokenwise_repair_readout"
STRICT_CONTROLS = (
    "target_only",
    "public_candidate_readout",
    "zero_source_control",
    "wrong_row_source_control",
    "source_row_shuffle_control",
    "same_source_choice_wrong_row_control",
    "atom_shuffle_control",
    "coefficient_shuffle_control",
    "candidate_roll_control",
    "packet_only_source_index",
    "same_byte_visible_text",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prediction(scores: Sequence[float]) -> int:
    return int(max(range(len(scores)), key=lambda index: (float(scores[index]), -index)))


def _margin(scores: Sequence[float], answer_index: int) -> float:
    gold = float(scores[int(answer_index)])
    distractors = [float(score) for index, score in enumerate(scores) if index != int(answer_index)]
    return gold - max(distractors) if distractors else gold


def _row_offsets(rows: Sequence[arc_gate.ArcRow]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for row in rows:
        end = start + len(row.choices)
        offsets.append((start, end))
        start = end
    return offsets


def _one_hot(index: int, size: int) -> np.ndarray:
    values = np.zeros(int(size), dtype=np.float64)
    values[int(index)] = 1.0
    return values


def _candidate_public_features(rows: Sequence[arc_gate.ArcRow], *, dim: int) -> np.ndarray:
    texts = preflight._source_choice_texts(list(rows))
    return arc_gate._hashed_features(texts, dim=int(dim)).astype(np.float64, copy=False)


def _candidate_labels(rows: Sequence[arc_gate.ArcRow]) -> np.ndarray:
    labels: list[float] = []
    for row in rows:
        for candidate_index in range(len(row.choices)):
            labels.append(1.0 if candidate_index == int(row.answer_index) else 0.0)
    return np.asarray(labels, dtype=np.float64)


def _candidate_design(
    rows: Sequence[arc_gate.ArcRow],
    *,
    source_rows: np.ndarray,
    public_candidates: np.ndarray,
    use_source: bool,
    use_public: bool,
    max_choices: int,
) -> np.ndarray:
    source = np.asarray(source_rows, dtype=np.float64)
    public = np.asarray(public_candidates, dtype=np.float64)
    offsets = _row_offsets(rows)
    if source.shape[0] != len(rows):
        raise ValueError("source_rows must have one row per ARC row")
    if public.shape[0] != offsets[-1][1]:
        raise ValueError("public_candidates must have one row per candidate")
    features: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        start, end = offsets[row_index]
        for candidate_index in range(len(row.choices)):
            parts = [
                source[row_index] if use_source else np.zeros(source.shape[1], dtype=np.float64),
                public[start + candidate_index]
                if use_public
                else np.zeros(public.shape[1], dtype=np.float64),
                _one_hot(candidate_index, max_choices),
            ]
            features.append(np.concatenate(parts))
        if end - start != len(row.choices):
            raise AssertionError("candidate offset accounting failed")
    return np.asarray(features, dtype=np.float64)


def _candidate_indices_for_rows(rows: Sequence[arc_gate.ArcRow], row_indices: Sequence[int]) -> np.ndarray:
    offsets = _row_offsets(rows)
    flat: list[int] = []
    for row_index in row_indices:
        start, end = offsets[int(row_index)]
        flat.extend(range(start, end))
    return np.asarray(flat, dtype=np.int64)


def _standardize_fit_eval(
    train_x: np.ndarray,
    eval_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_x - mean) / std,
        (eval_x - mean) / std,
        {
            "feature_dim": float(train_x.shape[1]),
            "fit_std_min": float(std.min()),
            "fit_std_max": float(std.max()),
        },
    )


def _fit_ridge_scalar(train_x: np.ndarray, train_y: np.ndarray, *, ridge: float) -> np.ndarray:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("ridge expects train_x [n, d] and train_y [n]")
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    gram = x_aug @ x_aug.T
    gram = gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    dual = np.linalg.solve(gram, y)
    return x_aug.T @ dual


def _predict_ridge_scores(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, ridge: float) -> np.ndarray:
    train_z, eval_z, _ = _standardize_fit_eval(train_x, eval_x)
    weights = _fit_ridge_scalar(train_z, train_y, ridge=ridge)
    eval_aug = np.concatenate([np.ones((eval_z.shape[0], 1), dtype=np.float64), eval_z], axis=1)
    return eval_aug @ weights


def _scores_by_eval_row(
    rows: Sequence[arc_gate.ArcRow],
    eval_indices: Sequence[int],
    flat_scores: np.ndarray,
) -> list[list[float]]:
    offsets = _row_offsets(rows)
    scores: list[list[float]] = []
    cursor = 0
    for row_index in eval_indices:
        start, end = offsets[int(row_index)]
        count = end - start
        scores.append([float(value) for value in flat_scores[cursor : cursor + count]])
        cursor += count
    if cursor != len(flat_scores):
        raise ValueError("unused flat scores while reconstructing rows")
    return scores


def _prediction_rows(
    rows: Sequence[arc_gate.ArcRow],
    eval_indices: Sequence[int],
    condition: str,
    scores_by_row: Sequence[Sequence[float]],
    source_predictions: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, scores in zip(eval_indices, scores_by_row, strict=True):
        row = rows[int(row_index)]
        prediction = _prediction(scores)
        records.append(
            {
                "condition": condition,
                "row_id": row.row_id,
                "content_id": row.content_id,
                "source_visible_fields": ["question", "choices", "choice_labels"],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
                "answer_index": int(row.answer_index),
                "answer_label": row.answer_label,
                "prediction_index": int(prediction),
                "prediction_label": row.choice_labels[prediction],
                "correct": bool(prediction == int(row.answer_index)),
                "margin": float(_margin(scores, row.answer_index)),
                "scores": [float(score) for score in scores],
                "source_selected_index": int(source_predictions[int(row_index)]),
                "source_selected_label": row.choice_labels[int(source_predictions[int(row_index)])],
                "source_selected_choice_sha256": hashlib.sha256(
                    row.choices[int(source_predictions[int(row_index)])].encode("utf-8")
                ).hexdigest(),
            }
        )
    return records


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "mean_margin": 0.0, "count": 0.0}
    return {
        "accuracy": float(statistics.fmean(1.0 if row["correct"] else 0.0 for row in rows)),
        "mean_margin": float(statistics.fmean(float(row["margin"]) for row in rows)),
        "count": float(len(rows)),
    }


def _paired_bootstrap(
    matched_rows: Sequence[dict[str, Any]],
    control_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    matched = {str(row["content_id"]): bool(row["correct"]) for row in matched_rows}
    control = {str(row["content_id"]): bool(row["correct"]) for row in control_rows}
    content_ids = sorted(set(matched) & set(control))
    deltas = [(1.0 if matched[key] else 0.0) - (1.0 if control[key] else 0.0) for key in content_ids]
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "paired_count": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "paired_count": float(n),
    }


def _load_eval_audit(path: pathlib.Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_content: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_content.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    return by_content


def _audit_content_ids(path: pathlib.Path, *, condition: str) -> list[str]:
    content_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("condition")) != condition:
                continue
            content_id = str(row["content_id"])
            if content_id not in seen:
                seen.add(content_id)
                content_ids.append(content_id)
    return content_ids


def _audit_condition_rows(
    rows: Sequence[arc_gate.ArcRow],
    eval_indices: Sequence[int],
    source_predictions: Sequence[int],
    audit: dict[str, dict[str, dict[str, Any]]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_index in eval_indices:
        row = rows[int(row_index)]
        audit_row = audit.get(row.content_id, {}).get(condition)
        if audit_row is None:
            raise ValueError(f"audit is missing condition={condition} content_id={row.content_id}")
        scores = [float(value) for value in audit_row["scores"]]
        out.extend(_prediction_rows(rows, [row_index], condition, [scores], source_predictions))
    return out


def _source_index_rows(
    rows: Sequence[arc_gate.ArcRow],
    eval_indices: Sequence[int],
    source_predictions: Sequence[int],
) -> list[dict[str, Any]]:
    scores_by_row: list[list[float]] = []
    for row_index in eval_indices:
        row = rows[int(row_index)]
        scores_by_row.append(preflight._source_index_scores(len(row.choices), int(source_predictions[int(row_index)])))
    return _prediction_rows(rows, eval_indices, "packet_only_source_index", scores_by_row, source_predictions)


def _eval_source_controls(
    source_rows: np.ndarray,
    *,
    eval_indices: Sequence[int],
    source_predictions: Sequence[int],
    seed: int,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    source = np.asarray(source_rows, dtype=np.float64)
    eval_source = source[np.asarray(eval_indices, dtype=np.int64)]
    rng = np.random.default_rng(seed)

    wrong = np.roll(eval_source, shift=1, axis=0)
    shuffled = eval_source.copy()
    rng.shuffle(shuffled, axis=0)
    atom_shuffle = eval_source[:, rng.permutation(eval_source.shape[1])]
    coeff_shuffle = eval_source.copy()
    for row_values in coeff_shuffle:
        rng.shuffle(row_values)

    donors: list[np.ndarray] = []
    donor_hits = 0
    for eval_pos, row_index in enumerate(eval_indices):
        source_choice = int(source_predictions[int(row_index)])
        donor_index = None
        for candidate_index in eval_indices:
            if int(candidate_index) == int(row_index):
                continue
            if int(source_predictions[int(candidate_index)]) == source_choice:
                donor_index = int(candidate_index)
                break
        if donor_index is None:
            donor_index = int(eval_indices[(eval_pos - 1) % len(eval_indices)])
        else:
            donor_hits += 1
        donors.append(source[donor_index])

    return {
        "zero_source_control": (np.zeros_like(eval_source), {"kind": "zero_source"}),
        "wrong_row_source_control": (wrong, {"kind": "rolled_eval_source", "shift": 1}),
        "source_row_shuffle_control": (shuffled, {"kind": "rng_eval_source_shuffle", "seed": int(seed)}),
        "same_source_choice_wrong_row_control": (
            np.asarray(donors, dtype=np.float64),
            {
                "kind": "same_source_choice_wrong_row_where_possible",
                "covered_rows": int(donor_hits),
                "eval_rows": int(len(eval_indices)),
            },
        ),
        "atom_shuffle_control": (atom_shuffle, {"kind": "shared_source_dimension_permutation", "seed": int(seed)}),
        "coefficient_shuffle_control": (coeff_shuffle, {"kind": "rowwise_source_coefficient_shuffle", "seed": int(seed)}),
    }


def evaluate_repair_readout(
    rows: Sequence[arc_gate.ArcRow],
    *,
    source_rows: np.ndarray,
    public_candidates: np.ndarray,
    source_predictions: Sequence[int],
    fit_indices: Sequence[int],
    eval_indices: Sequence[int],
    audit_rows: dict[str, dict[str, dict[str, Any]]],
    ridge: float,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_choices = max(len(row.choices) for row in rows)
    candidate_y = _candidate_labels(rows)
    fit_flat = _candidate_indices_for_rows(rows, fit_indices)
    eval_flat = _candidate_indices_for_rows(rows, eval_indices)

    matched_x = _candidate_design(
        rows,
        source_rows=source_rows,
        public_candidates=public_candidates,
        use_source=True,
        use_public=True,
        max_choices=max_choices,
    )
    public_x = _candidate_design(
        rows,
        source_rows=source_rows,
        public_candidates=public_candidates,
        use_source=False,
        use_public=True,
        max_choices=max_choices,
    )
    source_x = _candidate_design(
        rows,
        source_rows=source_rows,
        public_candidates=public_candidates,
        use_source=True,
        use_public=False,
        max_choices=max_choices,
    )

    condition_rows: dict[str, list[dict[str, Any]]] = {}
    matched_scores = _predict_ridge_scores(
        matched_x[fit_flat],
        candidate_y[fit_flat],
        matched_x[eval_flat],
        ridge=ridge,
    )
    condition_rows[MATCHED_CONDITION] = _prediction_rows(
        rows,
        eval_indices,
        MATCHED_CONDITION,
        _scores_by_eval_row(rows, eval_indices, matched_scores),
        source_predictions,
    )
    public_scores = _predict_ridge_scores(
        public_x[fit_flat],
        candidate_y[fit_flat],
        public_x[eval_flat],
        ridge=ridge,
    )
    condition_rows["public_candidate_readout"] = _prediction_rows(
        rows,
        eval_indices,
        "public_candidate_readout",
        _scores_by_eval_row(rows, eval_indices, public_scores),
        source_predictions,
    )
    source_scores = _predict_ridge_scores(
        source_x[fit_flat],
        candidate_y[fit_flat],
        source_x[eval_flat],
        ridge=ridge,
    )
    condition_rows["source_only_readout"] = _prediction_rows(
        rows,
        eval_indices,
        "source_only_readout",
        _scores_by_eval_row(rows, eval_indices, source_scores),
        source_predictions,
    )

    controls = _eval_source_controls(
        source_rows,
        eval_indices=eval_indices,
        source_predictions=source_predictions,
        seed=seed,
    )
    eval_indices_array = np.asarray(eval_indices, dtype=np.int64)
    train_source = np.asarray(source_rows, dtype=np.float64)
    for condition, (eval_source, _) in controls.items():
        controlled_source = train_source.copy()
        controlled_source[eval_indices_array] = eval_source
        controlled_x = _candidate_design(
            rows,
            source_rows=controlled_source,
            public_candidates=public_candidates,
            use_source=True,
            use_public=True,
            max_choices=max_choices,
        )
        scores = _predict_ridge_scores(
            matched_x[fit_flat],
            candidate_y[fit_flat],
            controlled_x[eval_flat],
            ridge=ridge,
        )
        condition_rows[condition] = _prediction_rows(
            rows,
            eval_indices,
            condition,
            _scores_by_eval_row(rows, eval_indices, scores),
            source_predictions,
        )

    rolled_scores: list[list[float]] = []
    for row in condition_rows[MATCHED_CONDITION]:
        scores = list(row["scores"])
        rolled_scores.append(scores[-1:] + scores[:-1])
    condition_rows["candidate_roll_control"] = _prediction_rows(
        rows,
        eval_indices,
        "candidate_roll_control",
        rolled_scores,
        source_predictions,
    )
    condition_rows["target_only"] = _audit_condition_rows(
        rows, eval_indices, source_predictions, audit_rows, condition="target_only"
    )
    condition_rows["same_byte_visible_text"] = _audit_condition_rows(
        rows, eval_indices, source_predictions, audit_rows, condition="same_byte_visible_text"
    )
    condition_rows["packet_only_source_index"] = _source_index_rows(rows, eval_indices, source_predictions)

    flat_rows: list[dict[str, Any]] = []
    for condition in [MATCHED_CONDITION, *sorted(k for k in condition_rows if k != MATCHED_CONDITION)]:
        flat_rows.extend(condition_rows[condition])

    condition_metrics = {condition: _metrics(rows_) for condition, rows_ in sorted(condition_rows.items())}
    matched = condition_rows[MATCHED_CONDITION]
    paired = {
        condition: _paired_bootstrap(matched, rows_, seed=seed + 31, samples=bootstrap_samples)
        for condition, rows_ in sorted(condition_rows.items())
        if condition != MATCHED_CONDITION
    }
    strict_available = [condition for condition in STRICT_CONTROLS if condition in condition_metrics]
    best_control = max(strict_available, key=lambda name: condition_metrics[name]["accuracy"])
    pass_gate = all(paired[condition]["ci95_low"] > 0.0 for condition in strict_available)

    metadata = {
        "fit_candidate_count": int(len(fit_flat)),
        "eval_candidate_count": int(len(eval_flat)),
        "max_choices": int(max_choices),
        "source_control_metadata": {condition: meta for condition, (_, meta) in controls.items()},
        "strict_controls": strict_available,
        "best_control_by_accuracy": best_control,
        "best_control_accuracy": float(condition_metrics[best_control]["accuracy"]),
        "matched_accuracy": float(condition_metrics[MATCHED_CONDITION]["accuracy"]),
        "matched_minus_best_control_accuracy": float(
            condition_metrics[MATCHED_CONDITION]["accuracy"] - condition_metrics[best_control]["accuracy"]
        ),
        "pass_gate": bool(pass_gate),
        "paired_deltas": paired,
    }
    return {"condition_metrics": condition_metrics, "headline": metadata}, flat_rows


def _load_inputs(args: argparse.Namespace) -> tuple[list[arc_gate.ArcRow], list[int], dict[str, dict[str, dict[str, Any]]]]:
    all_rows = arc_gate._load_rows(_resolve(args.eval_path))
    source_cache = preflight._read_source_cache(_resolve(args.source_cache_path))
    audit_path = _resolve(args.audit_jsonl)
    audit = _load_eval_audit(audit_path)
    eval_content_ids = _audit_content_ids(audit_path, condition="target_only")[: int(args.eval_rows)]
    rows_by_content = {row.content_id: row for row in all_rows}
    eval_rows: list[arc_gate.ArcRow] = []
    for content_id in eval_content_ids:
        row = rows_by_content.get(content_id)
        if row is None:
            raise ValueError(f"audit content_id={content_id} is missing from eval_path")
        if content_id not in source_cache:
            raise ValueError(f"audit content_id={content_id} is missing from source cache")
        if "target_only" not in audit.get(content_id, {}):
            raise ValueError(f"target audit does not align with selected eval row content_id={content_id}")
        eval_rows.append(row)

    eval_set = {row.content_id for row in eval_rows}
    fit_rows: list[arc_gate.ArcRow] = []
    for row in all_rows:
        if row.content_id in eval_set or row.content_id not in source_cache:
            continue
        selected = int(source_cache[row.content_id])
        if 0 <= selected < len(row.choices):
            fit_rows.append(row)
        if len(fit_rows) >= int(args.fit_rows):
            break
    if len(fit_rows) < int(args.fit_rows):
        raise ValueError("not enough source-cache rows for the requested fit_rows")
    if len(eval_rows) < int(args.eval_rows):
        raise ValueError("audit does not contain enough target_only eval rows")

    selected_rows = fit_rows + eval_rows
    source_predictions = [
        int(source_cache[row.content_id])
        for row in selected_rows
    ]
    return selected_rows, source_predictions, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_predictions, audit = _load_inputs(args)
    fit_indices = list(range(int(args.fit_rows)))
    eval_indices = list(range(int(args.fit_rows), int(args.fit_rows) + int(args.eval_rows)))

    source_summary, source_meta = preflight._selected_choice_features(
        rows,
        source_predictions,
        source_feature_mode=str(args.source_feature_mode),
        feature_dim=int(args.source_feature_dim),
        source_model=str(_resolve(args.source_model)),
        source_device=str(args.source_device),
        source_dtype=str(args.source_dtype),
        source_max_length=int(args.source_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        source_token_pool_size=int(args.source_token_pool_size),
        local_files_only=not bool(args.allow_downloads),
        fit_indices=fit_indices,
        sparse_packet_rank=int(args.sparse_packet_rank),
        sparse_packet_top_k=int(args.sparse_packet_top_k),
        sparse_packet_bits=int(args.sparse_packet_bits),
    )
    source_rows = source_summary.detach().cpu().numpy().astype(np.float64)
    source_flat = source_rows.reshape(source_rows.shape[0], -1)
    public_candidates = _candidate_public_features(rows, dim=int(args.candidate_feature_dim))

    evaluation, prediction_rows = evaluate_repair_readout(
        rows,
        source_rows=source_flat,
        public_candidates=public_candidates,
        source_predictions=source_predictions,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        audit_rows=audit,
        ridge=float(args.ridge),
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
    )

    dense_source_bytes = int(source_flat.shape[1]) * 4
    atom_id_bits = int(math.ceil(math.log2(max(source_flat.shape[1], 2))))
    sparse_proxy_bits = int(args.sparse_packet_top_k) * (atom_id_bits + int(args.sparse_packet_bits))
    systems = {
        "diagnostic_not_final_packet": True,
        "dense_source_feature_dim": int(source_flat.shape[1]),
        "dense_source_fp32_bytes_per_row": int(dense_source_bytes),
        "hypothetical_topk_sparse_proxy_bits_per_row": int(sparse_proxy_bits),
        "hypothetical_topk_sparse_proxy_bytes_per_row": float(sparse_proxy_bits / 8.0),
        "sparse_proxy_top_k": int(args.sparse_packet_top_k),
        "sparse_proxy_coeff_bits": int(args.sparse_packet_bits),
        "sparse_proxy_atom_id_bits": int(atom_id_bits),
    }

    result = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "implementation_gate_only": True,
        "benchmark": "ARC-Challenge validation",
        "method": MATCHED_CONDITION,
        "fit_rows": int(args.fit_rows),
        "eval_rows": int(args.eval_rows),
        "row_limit": int(args.row_limit),
        "inputs": {
            "eval_path": _display(args.eval_path),
            "source_cache_path": _display(args.source_cache_path),
            "audit_jsonl": _display(args.audit_jsonl),
        },
        "config": {
            "source_feature_mode": str(args.source_feature_mode),
            "source_feature_dim": int(args.source_feature_dim),
            "source_token_pool_size": int(args.source_token_pool_size),
            "candidate_feature_dim": int(args.candidate_feature_dim),
            "ridge": float(args.ridge),
            "seed": int(args.seed),
            "bootstrap_samples": int(args.bootstrap_samples),
        },
        "feature_metadata": {
            "source": source_meta,
            "source_tensor_shape": [int(value) for value in source_rows.shape],
            "source_flat_shape": [int(value) for value in source_flat.shape],
            "candidate_public_shape": [int(value) for value in public_candidates.shape],
        },
        "systems": systems,
        **evaluation,
        "runtime": {"wall_s": float(time.perf_counter() - start), "peak_rss_mib": float(preflight._peak_rss_mib())},
        "interpretation": (
            "Pass only if matched tokenwise readout beats all strict controls with positive paired CI. "
            "This gate probes source-evidence availability; it is not a low-byte packet claim."
        ),
    }

    result_path = output_dir / "arc_tokenwise_repair_readout_preflight.json"
    prediction_path = output_dir / "prediction_audit.jsonl"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(prediction_path, prediction_rows)
    markdown = _markdown_report(result)
    markdown_path = output_dir / "arc_tokenwise_repair_readout_preflight.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    manifest = {
        "created_utc": result["created_utc"],
        "files": {
            "result": {"path": _display(result_path), "sha256": _sha256_file(result_path)},
            "prediction_audit": {"path": _display(prediction_path), "sha256": _sha256_file(prediction_path)},
            "report": {"path": _display(markdown_path), "sha256": _sha256_file(markdown_path)},
        },
        "inputs": result["inputs"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _markdown_report(result: dict[str, Any]) -> str:
    headline = result["headline"]
    metrics = result["condition_metrics"]
    paired = headline["paired_deltas"]
    lines = [
        "# ARC Tokenwise Repair Readout Preflight",
        "",
        "This is an implementation gate, not a paper-positive result. It tests whether source-token evidence contains held-out repair signal before spending more work on a low-byte packet decoder.",
        "",
        "## Headline",
        "",
        f"- pass_gate: `{headline['pass_gate']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.4f}`",
        f"- best strict control: `{headline['best_control_by_accuracy']}` at `{headline['best_control_accuracy']:.4f}`",
        f"- matched minus best control: `{headline['matched_minus_best_control_accuracy']:.4f}`",
        "",
        "## Condition Metrics",
        "",
        "| condition | accuracy | mean margin | matched delta CI |",
        "|---|---:|---:|---:|",
    ]
    for condition, row in sorted(metrics.items()):
        if condition == MATCHED_CONDITION:
            delta = "-"
        else:
            ci = paired.get(condition, {})
            delta = f"{ci.get('mean', 0.0):+.4f} [{ci.get('ci95_low', 0.0):+.4f}, {ci.get('ci95_high', 0.0):+.4f}]"
        lines.append(
            f"| {condition} | {row['accuracy']:.4f} | {row['mean_margin']:.4f} | {delta} |"
        )
    lines.extend(
        [
            "",
            "## Systems Diagnostic",
            "",
            f"- dense source feature bytes per row: `{result['systems']['dense_source_fp32_bytes_per_row']}`",
            f"- hypothetical top-k sparse proxy bytes per row: `{result['systems']['hypothetical_topk_sparse_proxy_bytes_per_row']:.2f}`",
            "- This run does not claim a final low-byte packet; it only probes whether there is source signal worth compressing.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-path", type=pathlib.Path, default=preflight.DEFAULT_ARC_VALIDATION)
    parser.add_argument("--source-cache-path", type=pathlib.Path, default=preflight.DEFAULT_ARC_SOURCE_CACHE)
    parser.add_argument("--audit-jsonl", type=pathlib.Path, default=DEFAULT_AUDIT)
    parser.add_argument("--row-limit", type=int, default=32)
    parser.add_argument("--fit-rows", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--source-feature-mode", default="hf_choice_token_hidden_pool_residual")
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--source-model", type=pathlib.Path, default=pathlib.Path(preflight.DEFAULT_QWEN_SOURCE))
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--source-dtype", default="float32")
    parser.add_argument("--source-max-length", type=int, default=160)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--source-token-pool-size", type=int, default=16)
    parser.add_argument("--candidate-feature-dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--sparse-packet-rank", type=int, default=64)
    parser.add_argument("--sparse-packet-top-k", type=int, default=16)
    parser.add_argument("--sparse-packet-bits", type=int, default=8)
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
