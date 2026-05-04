from __future__ import annotations

"""Official-train Qwen source-dictionary gate for Qwen-to-Phi transfer.

The protected-rival gate showed that Qwen's hybrid+rival pair has large oracle
headroom, but a tiny validation fit/select receiver cannot choose the rival
safely. This gate asks whether a larger official-train, out-of-bag Qwen source
dictionary can learn that switch using source-side score/packet fields only,
then freeze and emit a byte-scale selected-candidate packet on Qwen-to-Phi.
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402

DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate_20260504_validation1024_2048"
)
DEFAULT_QWEN_TRAIN_CACHE_DIR = official.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_TRAIN_PATH = official.DEFAULT_TRAIN_PATH
DEFAULT_SOURCE_SCORE_CACHE = oracle.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = official.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = official.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = official.DEFAULT_RIDGES
DICTIONARY_RIDGES = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
)
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _answers(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return denoise._paired_ci(
        selected=selected,
        baseline=baseline,
        answers=answers,
        seed=seed,
        samples=samples,
    )


def _official_split_indices(row_count: int, *, dev_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return official._official_split_indices(row_count, dev_fraction=dev_fraction, seed=seed)


def _build_qwen_oob_calibration(
    *,
    train_path: pathlib.Path | str,
    qwen_train_cache_dir: pathlib.Path | str,
    sample_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    component_ridges: tuple[float, ...],
    train_hidden_rows: int,
    dev_fraction: float,
) -> dict[str, Any]:
    all_train_rows = official.arc_gate._load_rows(official._resolve(train_path))
    samples = {
        int(seed): official._load_sample_cache(
            cache_dir=pathlib.Path(qwen_train_cache_dir),
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for seed in sample_seeds
    }
    bank = official._fit_family_model_bank(
        samples=samples,
        split_seeds=split_seeds,
        ridges=component_ridges,
        dev_fraction=dev_fraction,
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "answers": [],
        "scores": [],
        "hybrid": [],
        "mean": [],
        "margin": [],
    }
    duplicate_count = 0
    oob_overlap_drop_count = 0
    seen: set[str] = set()
    for seed in sample_seeds:
        sample = samples[int(seed)]
        included = tuple(item for item in sample_seeds if int(item) != int(seed))
        included_train_ids = {
            row_id
            for included_seed in included
            for row_id in samples[int(included_seed)]["row_ids"]
        }
        predictions = official._packet_predictions_for_sample(
            sample=sample,
            model_bank=bank,
            included_seeds=included,
            aggregation_policy="mean_zscore",
        )
        keep_indices: list[int] = []
        for local_index, row_id in enumerate(sample["row_ids"]):
            if row_id in included_train_ids:
                oob_overlap_drop_count += 1
                continue
            if row_id in seen:
                duplicate_count += 1
                continue
            seen.add(row_id)
            keep_indices.append(local_index)
            rows.append(
                {
                    "row_id": row_id,
                    "answer_index": int(sample["answers"][local_index]),
                    "sample_seed": int(seed),
                    "oob_model_seeds": list(included),
                }
            )
        if not keep_indices:
            continue
        ids = np.asarray(keep_indices, dtype=np.int64)
        arrays["answers"].append(sample["answers"][ids])
        arrays["scores"].append(sample["scores"][ids])
        arrays["hybrid"].append(predictions["hybrid_vote_on_score_agreement_prediction"][ids])
        arrays["mean"].append(predictions["mean_zscore_prediction"][ids])
        arrays["margin"].append(predictions["selected_margin"][ids])
    return {
        "rows": rows,
        "answers": np.concatenate(arrays["answers"], axis=0).astype(np.int64),
        "scores": np.concatenate(arrays["scores"], axis=0).astype(np.float64),
        "hybrid": np.concatenate(arrays["hybrid"], axis=0).astype(np.int64),
        "mean": np.concatenate(arrays["mean"], axis=0).astype(np.int64),
        "margin": np.concatenate(arrays["margin"], axis=0).astype(np.float64),
        "duplicate_row_count": int(duplicate_count),
        "oob_overlap_drop_count": int(oob_overlap_drop_count),
        "sample_cache_rows": [
            {
                "sample_seed": int(seed),
                "row_count": int(len(samples[int(seed)]["row_ids"])),
                "content_digest": samples[int(seed)]["content_digest"],
                "qwen_score_cache": samples[int(seed)]["score_path"],
                "qwen_score_cache_sha256": samples[int(seed)]["score_sha256"],
                "qwen_hidden_cache": samples[int(seed)]["hidden_path"],
                "qwen_hidden_cache_sha256": samples[int(seed)]["hidden_sha256"],
            }
            for seed in sample_seeds
        ],
        "component_rows": [
            {"family": "Qwen2.5", **row}
            for seed in sample_seeds
            for row in bank[int(seed)]["component_rows"]
        ],
    }


def _pair_from_scores(scores: np.ndarray, hybrid: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    rival = np.where(top1 == hybrid, top2, top1)
    return np.stack([hybrid, rival, top1, top2], axis=1).astype(np.int64)


def _feature_matrix(
    *,
    scores: np.ndarray,
    hybrid: np.ndarray,
    mean_prediction: np.ndarray,
    margin: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    hybrid = np.asarray(hybrid, dtype=np.int64)
    mean_prediction = np.asarray(mean_prediction, dtype=np.int64)
    margin = np.asarray(margin, dtype=np.float64)
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    rival = np.where(top1 == hybrid, top2, top1)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    z_scores = centered / np.where(scale > 1e-8, scale, 1.0)
    row_ids = np.arange(scores.shape[0])
    source_margin = scores[row_ids, top1] - scores[row_ids, top2]
    rival_minus_hybrid = scores[row_ids, rival] - scores[row_ids, hybrid]
    features: list[np.ndarray] = [
        np.ones(scores.shape[0], dtype=np.float64),
        z_scores[row_ids, hybrid],
        z_scores[row_ids, rival],
        z_scores[row_ids, top1],
        z_scores[row_ids, top2],
        source_margin,
        rival_minus_hybrid,
        margin,
        (top1 == hybrid).astype(np.float64),
        (top2 == hybrid).astype(np.float64),
        (mean_prediction == hybrid).astype(np.float64),
        (mean_prediction == rival).astype(np.float64),
    ]
    for ids in (hybrid, rival, top1, top2, mean_prediction):
        for candidate in range(4):
            features.append((ids == candidate).astype(np.float64))
    for values in (source_margin, rival_minus_hybrid, margin):
        for threshold in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            features.append((values < threshold).astype(np.float64))
    return np.vstack(features).T.astype(np.float64)


def _fit_dictionary(
    *,
    features: np.ndarray,
    pair: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    ridges: tuple[float, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = pair[:, 0]
    benefit = (pair[:, 1] == answers).astype(np.float64) - (pair[:, 0] == answers).astype(np.float64)
    config_rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, str], dict[str, Any]] | None = None
    for l2 in ridges:
        penalty = float(l2) * np.eye(features.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        lhs = features[fit_indices].T @ features[fit_indices] + penalty
        rhs = features[fit_indices].T @ benefit[fit_indices]
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs
        scores = features[dev_indices] @ weights
        thresholds = sorted(set(float(item) for item in scores))
        noop_threshold = float(np.max(scores) + max(1e-9, abs(float(np.max(scores))) * 1e-6))
        thresholds.append(noop_threshold)
        if len(thresholds) > 80:
            finite = thresholds[:-1]
            thresholds = sorted({finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 61)})
            thresholds.append(noop_threshold)
        for threshold in thresholds:
            predictions = pair[dev_indices, 0].copy()
            switch_mask = scores > float(threshold)
            predictions[switch_mask] = pair[dev_indices, 1][switch_mask]
            paired = _paired_ci(
                selected=predictions,
                baseline=base[dev_indices],
                answers=answers[dev_indices],
                seed=20260504 + int(float(l2) * 1000),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            row = {
                "l2": float(l2),
                "threshold": float(threshold),
                "threshold_is_noop": bool(threshold == noop_threshold),
                "official_dev_accuracy": _accuracy(predictions, answers[dev_indices]),
                "official_dev_delta_vs_hybrid": paired["delta"],
                "official_dev_ci95_low_vs_hybrid": paired["ci95_low"],
                "official_dev_helps_vs_hybrid": paired["helps"],
                "official_dev_harms_vs_hybrid": paired["harms"],
                "official_dev_override_count": int(np.sum(predictions != base[dev_indices])),
            }
            config_rows.append(row)
            key = (
                float(row["official_dev_accuracy"]),
                float(row["official_dev_delta_vs_hybrid"]),
                float(row["official_dev_ci95_low_vs_hybrid"]),
                float(-row["official_dev_override_count"]),
                f"{l2}:{threshold}",
            )
            model = {
                "l2": float(l2),
                "threshold": float(threshold),
                "weights": weights.tolist(),
            }
            if best is None or key > best[0]:
                best = (key, model)
    if best is None:
        raise ValueError("no dictionary configs")
    return best[1], sorted(config_rows, key=lambda item: item["official_dev_accuracy"], reverse=True)


def _predict_dictionary(features: np.ndarray, pair: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    scores = features @ weights
    predictions = pair[:, 0].copy()
    switch_mask = scores > float(model["threshold"])
    predictions[switch_mask] = pair[:, 1][switch_mask]
    return predictions.astype(np.int64)


def _entropy_bits(values: np.ndarray) -> float:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=4).astype(np.float64)
    probabilities = counts[counts > 0.0] / float(np.sum(counts))
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _codes_for_condition(predictions: np.ndarray, *, condition: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    predictions = np.asarray(predictions, dtype=np.int64)
    if condition == "matched":
        return predictions
    if condition == "source_row_shuffle":
        return predictions[rng.permutation(len(predictions))]
    if condition == "random_same_byte":
        return rng.choice(predictions, size=len(predictions), replace=True)
    if condition == "code_value_permutation":
        unique = np.unique(predictions)
        permuted = unique.copy()
        rng.shuffle(permuted)
        mapping = {int(src): int(dst) for src, dst in zip(unique, permuted, strict=True)}
        return np.asarray([mapping[int(item)] for item in predictions], dtype=np.int64)
    if condition == "candidate_roll_code":
        return (predictions + 1) % 4
    raise ValueError(f"unknown condition: {condition}")


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    bootstrap_samples: int,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = _answers(rows)
    vs_hybrid = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=30360504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=30360604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=30360704 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "target_only_accuracy": _accuracy(target_only, answers),
        "delta_vs_fixed_hybrid": vs_hybrid["delta"],
        "ci95_low_vs_fixed_hybrid": vs_hybrid["ci95_low"],
        "ci95_high_vs_fixed_hybrid": vs_hybrid["ci95_high"],
        "helps_vs_fixed_hybrid": vs_hybrid["helps"],
        "harms_vs_fixed_hybrid": vs_hybrid["harms"],
        "delta_vs_candidate_only": vs_candidate["delta"],
        "ci95_low_vs_candidate_only": vs_candidate["ci95_low"],
        "delta_vs_target_only": vs_target["delta"],
        "ci95_low_vs_target_only": vs_target["ci95_low"],
        "override_count_vs_fixed_hybrid": int(np.sum(predictions != fixed_hybrid)),
        "override_rate_vs_fixed_hybrid": float(np.mean(predictions != fixed_hybrid)),
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_private": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "source_score_or_logit_vector_exposed": False,
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _pair_oracle(pair: np.ndarray, answers: np.ndarray) -> np.ndarray:
    predictions = pair[:, 0].copy()
    in_pair = (answers == pair[:, 0]) | (answers == pair[:, 1])
    predictions[in_pair] = answers[in_pair]
    return predictions.astype(np.int64)


def _slice_rows(
    *,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    answers = _answers(rows)
    starts = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    out: list[dict[str, Any]] = []
    for start in sorted(set(starts.tolist())):
        mask = starts == start
        paired = _paired_ci(
            selected=predictions[mask],
            baseline=fixed_hybrid[mask],
            answers=answers[mask],
            seed=40360504 + int(start),
            samples=max(200, min(bootstrap_samples, 1000)),
        )
        out.append(
            {
                "slice_start": int(start),
                "eval_rows": int(np.sum(mask)),
                "method_accuracy": _accuracy(predictions[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]),
                "delta_vs_fixed_hybrid": paired["delta"],
                "ci95_low_vs_fixed_hybrid": paired["ci95_low"],
                "helps_vs_fixed_hybrid": paired["helps"],
                "harms_vs_fixed_hybrid": paired["harms"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Official-Train Source Dictionary Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- official calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- source dictionary accuracy: `{h['source_dictionary_accuracy']:.6f}`",
        f"- source dictionary delta: `{h['source_dictionary_delta_vs_fixed_hybrid']:.6f}`",
        f"- source dictionary CI95 low: `{h['source_dictionary_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- official dev delta: `{h['official_dev_selected_delta_vs_hybrid']:.6f}`",
        f"- hybrid-rival oracle accuracy: `{h['hybrid_rival_oracle_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
    ]
    _resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    train_path: pathlib.Path | str = DEFAULT_TRAIN_PATH,
    qwen_train_cache_dir: pathlib.Path | str = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    component_ridges: tuple[float, ...] = DEFAULT_COMPONENT_RIDGES,
    dictionary_ridges: tuple[float, ...] = DICTIONARY_RIDGES,
    fit_rows_per_slice: int = denoise.FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = denoise.SELECT_ROWS_PER_SLICE,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _build_qwen_oob_calibration(
        train_path=train_path,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        component_ridges=component_ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
    )
    fit_indices, dev_indices = _official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )
    train_pair = _pair_from_scores(calibration["scores"], calibration["hybrid"])
    train_features = _feature_matrix(
        scores=calibration["scores"],
        hybrid=calibration["hybrid"],
        mean_prediction=calibration["mean"],
        margin=calibration["margin"],
    )
    model, config_rows = _fit_dictionary(
        features=train_features,
        pair=train_pair,
        answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=dictionary_ridges,
        bootstrap_samples=bootstrap_samples,
    )
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    eval_scores = np.asarray([row["qwen_source_scores"] for row in eval_rows], dtype=np.float64)
    eval_hybrid = np.asarray([int(row["qwen_hybrid_prediction"]) for row in eval_rows], dtype=np.int64)
    eval_mean = np.asarray([int(row["selected_prediction"]) for row in eval_rows], dtype=np.int64)
    eval_margin = np.asarray([float(row.get("selected_margin", 0.0)) for row in eval_rows], dtype=np.float64)
    eval_pair = _pair_from_scores(eval_scores, eval_hybrid)
    eval_features = _feature_matrix(
        scores=eval_scores,
        hybrid=eval_hybrid,
        mean_prediction=eval_mean,
        margin=eval_margin,
    )
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    selected_predictions = _predict_dictionary(eval_features, eval_pair, model)
    selected_row = next(row for row in config_rows if row["l2"] == model["l2"] and row["threshold"] == model["threshold"])
    label_rng = np.random.default_rng(20260504)
    label_permuted_answers = calibration["answers"][label_rng.permutation(len(calibration["answers"]))]
    label_permutation_model, _ = _fit_dictionary(
        features=train_features,
        pair=train_pair,
        answers=label_permuted_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=dictionary_ridges,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    label_permutation_predictions = _predict_dictionary(eval_features, eval_pair, label_permutation_model)
    method_rows = [
        _method_row(
            name="official_train_source_dictionary_packet",
            rows=eval_rows,
            predictions=selected_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            details={"model": {key: value for key, value in model.items() if key != "weights"}},
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=eval_rows,
            predictions=fixed_hybrid,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="qwen_candidate_only",
            rows=eval_rows,
            predictions=candidate_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="qwen_source_score_top1_baseline",
            rows=eval_rows,
            predictions=eval_pair[:, 2],
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
        ),
        _method_row(
            name="phi_target_only",
            rows=eval_rows,
            predictions=target_only,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
        ),
        _method_row(
            name="hybrid_rival_oracle_diagnostic",
            rows=eval_rows,
            predictions=_pair_oracle(eval_pair[:, :2], answers),
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            details={"oracle": True, "not_promotable": True},
        ),
    ]
    for condition in ("source_row_shuffle", "code_value_permutation", "candidate_roll_code", "random_same_byte"):
        corrupted = _codes_for_condition(
            selected_predictions,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        method_rows.append(
            _method_row(
                name=f"{condition}_source_dictionary_control",
                rows=eval_rows,
                predictions=corrupted,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                details={"condition": condition},
            )
        )
    method_rows.append(
        _method_row(
            name="official_train_label_permutation_dictionary_control",
            rows=eval_rows,
            predictions=label_permutation_predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            details={
                "condition": "official_train_label_permutation",
                "model": {key: value for key, value in label_permutation_model.items() if key != "weights"},
            },
        )
    )
    eval_diag_best: tuple[tuple[float, float, float, str], np.ndarray] | None = None
    weights = np.asarray(model["weights"], dtype=np.float64)
    eval_scores_for_model = eval_features @ weights
    for threshold in sorted(set(float(item) for item in eval_scores_for_model)):
        diag_model = dict(model, threshold=float(threshold))
        diag_predictions = _predict_dictionary(eval_features, eval_pair, diag_model)
        overrides = int(np.sum(diag_predictions != fixed_hybrid))
        if overrides == 0:
            continue
        key = (
            _accuracy(diag_predictions, answers),
            float(np.mean((diag_predictions == answers).astype(float) - (fixed_hybrid == answers).astype(float))),
            float(-overrides),
            str(threshold),
        )
        if eval_diag_best is None or key > eval_diag_best[0]:
            eval_diag_best = (key, diag_predictions)
    if eval_diag_best is not None:
        method_rows.append(
            _method_row(
                name="eval_label_best_threshold_dictionary_diagnostic",
                rows=eval_rows,
                predictions=eval_diag_best[1],
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                details={"not_promotable": True, "eval_label_selected": True},
            )
        )
    method_rows = sorted(method_rows, key=lambda item: item["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "official_train_source_dictionary_packet")
    destructive_rows = [row for row in method_rows if row["method"].endswith("_control")]
    best_destructive = max(destructive_rows, key=lambda item: item["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    pass_gate = bool(
        method_row["delta_vs_fixed_hybrid"] >= 0.005
        and method_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and method_row["ci95_low_vs_candidate_only"] > 0.0
        and method_row["accuracy"] > best_destructive["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and method_row["helps_vs_fixed_hybrid"] > method_row["harms_vs_fixed_hybrid"]
    )
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_qwen_hybrid_accuracy": _accuracy(calibration["hybrid"], calibration["answers"]),
        "official_train_qwen_mean_accuracy": _accuracy(calibration["mean"], calibration["answers"]),
        "official_dev_selected_accuracy": selected_row["official_dev_accuracy"],
        "official_dev_selected_delta_vs_hybrid": selected_row["official_dev_delta_vs_hybrid"],
        "official_dev_selected_ci95_low_vs_hybrid": selected_row["official_dev_ci95_low_vs_hybrid"],
        "selected_l2": float(model["l2"]),
        "selected_threshold": float(model["threshold"]),
        "eval_rows": len(eval_rows),
        "fixed_hybrid_accuracy": next(
            row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement"
        )["accuracy"],
        "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")[
            "accuracy"
        ],
        "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")[
            "accuracy"
        ],
        "source_dictionary_accuracy": method_row["accuracy"],
        "source_dictionary_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "source_dictionary_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "source_dictionary_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "source_dictionary_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "source_dictionary_override_count": method_row["override_count_vs_fixed_hybrid"],
        "hybrid_rival_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "hybrid_rival_oracle_diagnostic"
        )["accuracy"],
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "raw_payload_bytes": 1,
        "framed_record_bytes": 4,
        "native_systems_claim_allowed": False,
    }
    packet_accounting = {
        "raw_payload_bits": 2,
        "raw_payload_bytes": 1,
        "framed_record_bits": 32,
        "framed_record_bytes": 4,
        "header_and_padding_bits": 30,
        "unique_code_count": int(len(set(int(item) for item in selected_predictions.tolist()))),
        "empirical_entropy_bits_per_request": _entropy_bits(selected_predictions),
        "field_bit_allocation": {
            "selected_candidate_id_bits": 2,
            "dictionary_model_id_bits": 0,
            "source_score_vector_bits_transmitted": 0,
            "source_hidden_vector_bits_transmitted": 0,
        },
        "dictionary_static_bytes": int(len(model["weights"]) * 8 + 16),
        "amortized_dictionary_bytes_per_request_on_eval": float((len(model["weights"]) * 8 + 16) / len(eval_rows)),
    }
    comparator_byte_floor_rows = [
        {
            "comparator": "qjl_1bit_source_state_floor",
            "bytes_per_token_floor": 768,
            "threat_model_note": "continuous KV/source-state quantization floor, not a task-packet competitor",
        },
        {
            "comparator": "turboquant_2p5b_source_state_floor",
            "bytes_per_token_floor": 1920,
            "threat_model_note": "low-bit continuous state floor; no native serving comparison claimed here",
        },
        {
            "comparator": "kivi_2b_asymmetric_kv_floor",
            "bytes_per_token_floor": 1536,
            "threat_model_note": "KV-cache compression floor; LatentWire sends a candidate decision packet instead",
        },
        {
            "comparator": "kvquant_3b_source_kv_floor",
            "bytes_per_token_floor": 2304,
            "threat_model_note": "KV-cache compression floor; byte comparison is context only because quality failed",
        },
    ]
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the official-train-selected source dictionary beats fixed Qwen hybrid by at least "
            "0.005 with positive paired CI, beats candidate-only with positive paired CI, is nonnegative on "
            "both cached Phi slices, beats destructive controls, and helps more than it harms."
        ),
        "headline": headline,
        "packet_contract": {
            "receiver_visible_payload": (
                "one byte-scale selected candidate emitted by an official-train Qwen source-side "
                "hybrid-vs-rival dictionary; source score vector and hidden state are not transmitted"
            ),
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "phi_target_scores_used_for_training": False,
        },
        "source_score_metadata": source_score_metadata,
        "slice_metadata": metadata,
        "sample_cache_rows": calibration["sample_cache_rows"],
        "component_rows": calibration["component_rows"],
        "method_rows": method_rows,
        "config_rows": config_rows,
        "slice_rows": slice_rows,
        "packet_accounting": packet_accounting,
        "comparator_byte_floor_rows": comparator_byte_floor_rows,
        "systems_packet_sideband": {
            "raw_payload_bits_per_request": 2,
            "raw_payload_bytes_per_request": 1,
            "framed_record_bytes_per_request": 4,
            "feature_build_and_selection_wall_time_s": float(time.perf_counter() - started),
            "dictionary_static_bytes": packet_accounting["dictionary_static_bytes"],
            "amortized_dictionary_bytes_per_request_on_eval": packet_accounting[
                "amortized_dictionary_bytes_per_request_on_eval"
            ],
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
        },
        "inputs": {
            "train_path": official._display_path(train_path),
            "train_sha256": official._sha256_file(train_path),
            "qwen_train_cache_dir": official._display_path(qwen_train_cache_dir),
            "source_score_cache": denoise._display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate tests whether the official-train source side alone can learn the protected-rival "
            "decision frontier. It uses out-of-bag train rows to avoid scoring a row with a packet model that "
            "trained on that same row. A failure means the larger-data source dictionary does not solve the "
            "Qwen-to-Phi blocker without receiver-side Phi calibration or a richer interface."
        ),
        "lay_explanation": (
            "We trained Qwen on many official training questions to learn when its backup answer should replace "
            "its safe answer. That looked useful on Qwen's training-dev split, but when frozen and tested on the "
            "held-out Qwen-to-Phi slice it made too many bad swaps."
        ),
    }
    _write_json(output_dir / "hellaswag_qwen_to_phi_official_train_source_dictionary_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_official_train_source_dictionary_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_official_train_source_dictionary_gate.json",
                "hellaswag_qwen_to_phi_official_train_source_dictionary_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "slice_rows.csv",
            ],
            "headline": headline,
            "inputs": payload["inputs"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--sample-seeds", type=official._parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=official._parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--component-ridges", type=official._parse_float_tuple, default=DEFAULT_COMPONENT_RIDGES)
    parser.add_argument("--dictionary-ridges", type=official._parse_float_tuple, default=DICTIONARY_RIDGES)
    parser.add_argument("--fit-rows-per-slice", type=int, default=denoise.FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=denoise.SELECT_ROWS_PER_SLICE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        source_score_cache=args.source_score_cache,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        component_ridges=args.component_ridges,
        dictionary_ridges=args.dictionary_ridges,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "official_train_calibration_rows": h["official_train_calibration_rows"],
                "official_dev_selected_delta_vs_hybrid": h["official_dev_selected_delta_vs_hybrid"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "source_dictionary_accuracy": h["source_dictionary_accuracy"],
                "source_dictionary_delta_vs_fixed_hybrid": h["source_dictionary_delta_vs_fixed_hybrid"],
                "hybrid_rival_oracle_accuracy": h["hybrid_rival_oracle_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
