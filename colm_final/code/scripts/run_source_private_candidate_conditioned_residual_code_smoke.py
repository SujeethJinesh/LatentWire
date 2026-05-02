from __future__ import annotations

import argparse
import csv
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

from scripts import run_source_private_learned_synonym_dictionary_packet_gate as syn  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_conditioned_residual_code_smoke_20260501")
BASE_MATCHED_CONDITION = "learned_synonym_dictionary_packet"
MATCHED_CONDITION = "candidate_conditioned_residual_code_packet"
BASE_ORACLE_CONDITION = "oracle_learned_candidate_atoms"
ORACLE_CONDITION = "oracle_candidate_conditioned_residual_code"
STRICT_CONTROLS = tuple(syn.STRICT_SOURCE_DESTROYING_CONTROLS)
EVAL_CONDITIONS = ("target_only", MATCHED_CONDITION, *STRICT_CONTROLS, ORACLE_CONDITION)
TRAIN_BASE_CONDITIONS = (BASE_MATCHED_CONDITION, *STRICT_CONTROLS)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_condition(condition: str) -> str:
    if condition == MATCHED_CONDITION:
        return BASE_MATCHED_CONDITION
    if condition == ORACLE_CONDITION:
        return BASE_ORACLE_CONDITION
    return condition


def _target_index_for_condition(row: dict[str, Any]) -> int:
    return int(row["answer_index"]) if row["condition"] == BASE_MATCHED_CONDITION else int(row["prior_index"])


def _row_scores(row: dict[str, Any]) -> list[float]:
    return [float(value) for value in row.get("metadata", {}).get("scores") or []]


def _candidate_features(row: dict[str, Any], candidate_index: int) -> list[float] | None:
    scores = _row_scores(row)
    if not scores:
        return None
    score = float(scores[candidate_index])
    count = len(scores)
    mean = statistics.fmean(scores)
    variance = statistics.fmean((value - mean) ** 2 for value in scores)
    std = max(variance**0.5, 1e-8)
    prior_index = int(row["prior_index"])
    prior_score = float(scores[prior_index])
    best_score = max(scores)
    sorted_scores = sorted(scores, reverse=True)
    best_margin = best_score - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    row_norms = row.get("metadata", {}).get("candidate_local_row_norms") or [0.0 for _ in scores]
    row_norm = float(row_norms[candidate_index]) if candidate_index < len(row_norms) else 0.0
    payload_l2 = float(row.get("metadata", {}).get("candidate_local_payload_l2") or 0.0)
    is_prior = float(candidate_index == prior_index)
    is_top_raw = float(abs(score - best_score) <= 1e-8)
    rank = sorted(scores, reverse=True).index(score) if score in scores else count - 1
    rank_norm = float(rank) / max(1.0, float(count - 1))
    return [
        1.0,
        score,
        score - mean,
        (score - mean) / std,
        score - prior_score,
        is_prior,
        is_top_raw,
        best_score,
        best_margin,
        payload_l2,
        row_norm,
        score * row_norm,
        is_prior * best_margin,
        is_prior * best_score,
        rank_norm,
    ]


def _feature_matrix(row: dict[str, Any]) -> np.ndarray | None:
    scores = _row_scores(row)
    if not scores:
        return None
    return np.asarray([_candidate_features(row, idx) for idx in range(len(scores))], dtype=np.float64)


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = features.copy()
    standardized[:, 1:] = (standardized[:, 1:] - mean) / std
    return standardized


def _fit_receiver(
    rows: list[dict[str, Any]],
    *,
    ridge: float,
    matched_weight: float,
    control_weight: float,
) -> dict[str, Any]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    sample_weights: list[float] = []
    for row in rows:
        scores = _row_scores(row)
        if not scores:
            continue
        target_index = _target_index_for_condition(row)
        weight = matched_weight if row["condition"] == BASE_MATCHED_CONDITION else control_weight
        for idx in range(len(scores)):
            features = _candidate_features(row, idx)
            if features is None:
                continue
            x_rows.append(features)
            y_rows.append(float(idx == target_index))
            sample_weights.append(weight)
    if not x_rows:
        raise ValueError("no scored training rows available for candidate-conditioned receiver")
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    mean = x[:, 1:].mean(axis=0)
    std = x[:, 1:].std(axis=0)
    std[std < 1e-8] = 1.0
    x_std = _standardize(x, mean, std)
    weights = np.sqrt(np.asarray(sample_weights, dtype=np.float64))
    xw = x_std * weights[:, None]
    yw = y * weights
    xtx = xw.T @ xw
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= ridge
    coeffs = np.linalg.solve(xtx, xw.T @ yw)
    state = {
        "coefficients": [float(value) for value in coeffs],
        "feature_mean": [float(value) for value in mean],
        "feature_std": [float(value) for value in std],
        "ridge": ridge,
        "matched_weight": matched_weight,
        "control_weight": control_weight,
    }
    state["decision_delta_threshold"] = _tune_delta_threshold(rows, state)
    return state


def _calibrated_scores(row: dict[str, Any], state: dict[str, Any]) -> list[float] | None:
    features = _feature_matrix(row)
    if features is None:
        return None
    mean = np.asarray(state["feature_mean"], dtype=np.float64)
    std = np.asarray(state["feature_std"], dtype=np.float64)
    coeffs = np.asarray(state["coefficients"], dtype=np.float64)
    return [float(value) for value in (_standardize(features, mean, std) @ coeffs)]


def _predict_index(row: dict[str, Any], calibrated_scores: list[float] | None, *, threshold: float) -> tuple[int, float]:
    prior_index = int(row["prior_index"])
    if not calibrated_scores:
        return prior_index, 0.0
    best_score = max(calibrated_scores)
    tied = [idx for idx, score in enumerate(calibrated_scores) if abs(score - best_score) <= 1e-8]
    top_index = prior_index if prior_index in tied else tied[0]
    delta = float(calibrated_scores[top_index] - calibrated_scores[prior_index])
    if top_index != prior_index and delta < threshold:
        return prior_index, delta
    return top_index, delta


def _weighted_policy_score(rows: list[dict[str, Any]], state: dict[str, Any], threshold: float) -> tuple[float, dict[str, float]]:
    matched_rows = [row for row in rows if row["condition"] == BASE_MATCHED_CONDITION]
    control_rows = [row for row in rows if row["condition"] in STRICT_CONTROLS]

    def answer_accuracy(group: list[dict[str, Any]]) -> float:
        if not group:
            return 0.0
        return statistics.fmean(
            _predict_index(row, _calibrated_scores(row, state), threshold=threshold)[0] == int(row["answer_index"])
            for row in group
        )

    def prior_accuracy(group: list[dict[str, Any]]) -> float:
        if not group:
            return 0.0
        return statistics.fmean(
            _predict_index(row, _calibrated_scores(row, state), threshold=threshold)[0] == int(row["prior_index"])
            for row in group
        )

    matched = answer_accuracy(matched_rows)
    by_control = {
        condition: answer_accuracy([row for row in rows if row["condition"] == condition])
        for condition in STRICT_CONTROLS
    }
    best_control = max(by_control.values()) if by_control else 0.0
    control_prior = prior_accuracy(control_rows)
    score = matched - 0.75 * best_control + 0.25 * control_prior
    return score, {
        "matched_train_accuracy": matched,
        "best_control_train_accuracy": best_control,
        "control_prior_train_accuracy": control_prior,
    }


def _tune_delta_threshold(rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    deltas: list[float] = [0.0]
    for row in rows:
        scores = _calibrated_scores(row, state)
        if not scores:
            continue
        prior_index = int(row["prior_index"])
        top_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        deltas.append(float(scores[top_index] - scores[prior_index]))
    candidates = sorted(set([min(deltas) - 1e-8, max(deltas) + 1e-8, *deltas]))
    best_score = -1e18
    best_threshold = 0.0
    best_detail: dict[str, float] = {}
    for threshold in candidates:
        score, detail = _weighted_policy_score(rows, state, threshold)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
            best_detail = detail
    return {"threshold": best_threshold, "objective": float(best_score), **best_detail}


def _candidate_label(example: syn.Example, index: int) -> str:
    return example.candidates[index].label


def _convert_base_row(
    *,
    base_row: dict[str, Any],
    display_condition: str,
    example: syn.Example,
    state: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    calibrated = _calibrated_scores(base_row, state)
    threshold = float(state["decision_delta_threshold"]["threshold"])
    prediction_index, delta = _predict_index(base_row, calibrated, threshold=threshold)
    prediction = _candidate_label(example, prediction_index)
    payload_hex = base_row.get("payload_hex", "")
    metadata = dict(base_row.get("metadata", {}))
    if calibrated is not None:
        raw_scores = metadata.get("scores", [])
        metadata.update(
            {
                "decoder": "candidate_conditioned_residual_code",
                "decoder_score_mode": "candidate_conditioned_residual_code",
                "scores": calibrated,
                "raw_candidate_local_scores": raw_scores,
                "candidate_conditioned_decision_delta": delta,
                "candidate_conditioned_delta_threshold": threshold,
                "candidate_conditioned_base_condition": base_row["condition"],
            }
        )
        sorted_scores = sorted(calibrated, reverse=True)
        metadata["best_score"] = float(sorted_scores[0])
        metadata["best_margin"] = float(sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0))
    else:
        metadata.update(
            {
                "decoder": "prior",
                "decoder_score_mode": "candidate_conditioned_residual_code",
                "candidate_conditioned_base_condition": base_row["condition"],
            }
        )
    return {
        "example_id": base_row["example_id"],
        "family_name": base_row["family_name"],
        "budget_bytes": base_row["budget_bytes"],
        "condition": display_condition,
        "answer": example.answer_label,
        "prediction": prediction,
        "correct": prediction == example.answer_label,
        "strict_correct": prediction == example.answer_label,
        "answer_index": int(base_row["answer_index"]),
        "prior_index": int(base_row["prior_index"]),
        "prediction_index": prediction_index,
        "payload_bytes": int(base_row.get("payload_bytes", 0)),
        "payload_tokens": syn._token_count(payload_hex),
        "payload_hex": payload_hex,
        "latency_ms": latency_ms,
        "metadata": metadata,
        "base_prediction": base_row.get("prediction"),
        "base_correct": bool(base_row.get("correct")),
    }


def _make_base_rows(
    *,
    examples: list[syn.Example],
    conditions: tuple[str, ...],
    budget_bytes: int,
    dictionary: Any,
    permuted_teacher_dictionary: Any,
    candidate_atom_view: str,
    decoder_score_mode: str,
    min_decision_score: float,
    seed: int,
) -> list[tuple[dict[str, Any], syn.Example]]:
    rng = random.Random(seed)
    rows: list[tuple[dict[str, Any], syn.Example]] = []
    for index, example in enumerate(examples):
        for condition in conditions:
            row = syn._predict_condition(
                condition=condition,
                example=example,
                eval_examples=examples,
                index=index,
                budget_bytes=budget_bytes,
                dictionary=dictionary,
                permuted_teacher_dictionary=permuted_teacher_dictionary,
                candidate_atom_view=candidate_atom_view,
                decoder_score_mode=decoder_score_mode,
                min_decision_score=min_decision_score,
                rng=rng,
            )
            row |= {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget_bytes}
            rows.append((row, example))
    return rows


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "n": len(rows),
        "accuracy": statistics.fmean(bool(row["correct"]) for row in rows) if rows else 0.0,
        "base_accuracy": statistics.fmean(bool(row.get("base_correct", row["correct"])) for row in rows) if rows else 0.0,
        "prior_prediction_rate": statistics.fmean(row["prediction_index"] == row["prior_index"] for row in rows) if rows else 0.0,
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows) if rows else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = [
        float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"])
        for _, conditions in sorted(by_example.items())
        if condition in conditions and baseline in conditions
    ]
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(deltas),
        "ci95_low": syn._percentile(means, 0.025),
        "ci95_high": syn._percentile(means, 0.975),
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    budget_bytes: int,
    seed: int,
    min_improvement_over_base: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in EVAL_CONDITIONS}
    metrics = {condition: _metric(condition_rows) for condition, condition_rows in by_condition.items()}
    example_ids = sorted({row["example_id"] for row in rows})
    exact_id_parity = all(
        {row["example_id"] for row in by_condition[condition]} == set(example_ids)
        for condition in EVAL_CONDITIONS
    )
    target = metrics["target_only"]["accuracy"]
    matched = metrics[MATCHED_CONDITION]["accuracy"]
    matched_base = metrics[MATCHED_CONDITION]["base_accuracy"]
    best_control_name = max(STRICT_CONTROLS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    controls_ok = all(metrics[condition]["accuracy"] <= target + 0.03 for condition in STRICT_CONTROLS)
    beats_base = matched >= matched_base + min_improvement_over_base
    pass_gate = (
        exact_id_parity
        and matched >= target + 0.15
        and matched >= best_control + 0.10
        and controls_ok
        and beats_base
    )
    return {
        "direction": direction,
        "budget_bytes": budget_bytes,
        "n": len(example_ids),
        "exact_id_parity": exact_id_parity,
        "target_accuracy": target,
        "matched_accuracy": matched,
        "matched_base_accuracy": matched_base,
        "matched_minus_target": matched - target,
        "matched_minus_base": matched - matched_base,
        "best_control_name": best_control_name,
        "best_control_accuracy": best_control,
        "matched_minus_best_control": matched - best_control,
        "oracle_accuracy": metrics[ORACLE_CONDITION]["accuracy"],
        "controls_ok": controls_ok,
        "beats_base": beats_base,
        "pass_gate": pass_gate,
        "metrics": metrics,
        "paired_bootstrap_vs_target": _paired_bootstrap(
            rows,
            condition=MATCHED_CONDITION,
            baseline="target_only",
            samples=bootstrap_samples,
            seed=seed + budget_bytes + 17,
        ),
        "paired_bootstrap_vs_best_control": _paired_bootstrap(
            rows,
            condition=MATCHED_CONDITION,
            baseline=best_control_name,
            samples=bootstrap_samples,
            seed=seed + budget_bytes + 31,
        ),
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Candidate-Conditioned Residual Code Smoke",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- budgets: `{payload['budgets']}`",
        f"- train examples/direction: `{payload['train_examples']}`",
        f"- eval examples/direction: `{payload['eval_examples']}`",
        "",
        "| Direction | Budget | Pass | Matched | Base matched | Target | Best control | Control name |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["direction_summaries"]:
        lines.append(
            "| "
            f"{row['direction']} | {row['budget_bytes']} | `{row['pass_gate']}` | "
            f"{row['matched_accuracy']:.3f} | {row['matched_base_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | {row['best_control_name']} |"
        )
    lines.extend(
        [
            "",
            "Lay explanation: this trains a tiny receiver-side rule that decides whether the packet score surface is strong enough to leave the target prior. The rule is trained to choose the answer for matched packets and to fall back to the prior for destructive controls.",
            "",
            payload["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "direction",
        "budget_bytes",
        "n",
        "pass_gate",
        "target_accuracy",
        "matched_accuracy",
        "matched_base_accuracy",
        "matched_minus_base",
        "best_control_name",
        "best_control_accuracy",
        "matched_minus_best_control",
        "oracle_accuracy",
        "controls_ok",
        "beats_base",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _direction_family_sets(direction: str) -> tuple[str, str]:
    if direction == "core_to_holdout":
        return "core", "holdout"
    if direction == "holdout_to_core":
        return "holdout", "core"
    if direction == "same_family_all":
        return "all", "all"
    raise ValueError(f"unknown direction {direction!r}")


def run_smoke(
    *,
    output_dir: pathlib.Path,
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    calibration_examples: int,
    seed: int,
    feature_dim: int,
    ridge: float,
    top_k: int,
    min_score: float,
    receiver_ridge: float,
    matched_weight: float,
    control_weight: float,
    min_improvement_over_base: float,
    bootstrap_samples: int,
    text_feature_mode: str,
    feature_model: str,
    feature_device: str,
    feature_dtype: str,
    feature_max_length: int,
    local_files_only: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    syn._HF_FEATURE_MODEL = feature_model
    syn._HF_FEATURE_DEVICE = feature_device
    syn._HF_FEATURE_DTYPE = feature_dtype
    syn._HF_FEATURE_MAX_LENGTH = feature_max_length
    syn._HF_FEATURE_LOCAL_FILES_ONLY = local_files_only
    directions = ("core_to_holdout", "holdout_to_core", "same_family_all")
    all_rows: list[dict[str, Any]] = []
    direction_summaries: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}
    for direction_index, direction in enumerate(directions):
        train_family_set, eval_family_set = _direction_family_sets(direction)
        train_seed = seed + 1009 * direction_index
        eval_seed = seed + 1009 * direction_index + 1
        train_examples_rows = syn.make_benchmark(
            examples=train_examples,
            candidates=4,
            seed=train_seed,
            family_set=train_family_set,
        )
        eval_examples_rows = syn.make_benchmark(
            examples=eval_examples,
            candidates=4,
            seed=eval_seed,
            family_set=eval_family_set,
        )
        calibration_rows = syn._calibration_examples(
            mode="all_public_eval_disjoint",
            train_examples=train_examples_rows,
            eval_examples=eval_examples_rows,
            calibration_count=calibration_examples,
            seed=train_seed + 101,
        )
        dictionary_kwargs = dict(
            examples=calibration_rows,
            feature_dim=feature_dim,
            ridge=ridge,
            calibration_atom_view="synonym_stress",
            top_k=top_k,
            min_score=min_score,
            text_feature_mode=text_feature_mode,
            receiver_mode="atom_ridge",
            contrastive_negative_sources=0,
            contrastive_rank=4,
        )
        dictionary = syn._fit_dictionary(
            adapter_target_mode="semantic_anchor_teacher",
            seed=train_seed + 211,
            **dictionary_kwargs,
        )
        permuted_teacher_dictionary = syn._fit_dictionary(
            adapter_target_mode="permuted_semantic_anchor_teacher",
            seed=train_seed + 313,
            **dictionary_kwargs,
        )
        for budget in budgets:
            train_base = [
                row
                for row, _ in _make_base_rows(
                    examples=train_examples_rows,
                    conditions=TRAIN_BASE_CONDITIONS,
                    budget_bytes=budget,
                    dictionary=dictionary,
                    permuted_teacher_dictionary=permuted_teacher_dictionary,
                    candidate_atom_view="heldout_synonym",
                    decoder_score_mode="candidate_local_residual_norm",
                    min_decision_score=0.0,
                    seed=train_seed * 100003 + budget,
                )
            ]
            state = _fit_receiver(
                train_base,
                ridge=receiver_ridge,
                matched_weight=matched_weight,
                control_weight=control_weight,
            )
            eval_rows: list[dict[str, Any]] = []
            for display_condition in EVAL_CONDITIONS:
                base_condition = _base_condition(display_condition)
                for base_row, example in _make_base_rows(
                    examples=eval_examples_rows,
                    conditions=(base_condition,),
                    budget_bytes=budget,
                    dictionary=dictionary,
                    permuted_teacher_dictionary=permuted_teacher_dictionary,
                    candidate_atom_view="heldout_synonym",
                    decoder_score_mode="candidate_local_residual_norm",
                    min_decision_score=0.0,
                    seed=eval_seed * 100003 + budget + len(display_condition),
                ):
                    start = time.perf_counter()
                    converted = _convert_base_row(
                        base_row=base_row,
                        display_condition=display_condition,
                        example=example,
                        state=state,
                        latency_ms=(time.perf_counter() - start) * 1000.0 + float(base_row.get("latency_ms", 0.0)),
                    )
                    converted["direction"] = direction
                    eval_rows.append(converted)
            predictions_name = f"{direction}/predictions_budget{budget}.jsonl"
            (output_dir / direction).mkdir(parents=True, exist_ok=True)
            _write_jsonl(output_dir / predictions_name, eval_rows)
            prediction_files[f"{direction}:{budget}"] = predictions_name
            all_rows.extend(eval_rows)
            direction_summaries.append(
                _summarize_rows(
                    eval_rows,
                    direction=direction,
                    budget_bytes=budget,
                    seed=eval_seed,
                    min_improvement_over_base=min_improvement_over_base,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    pass_gate = all(row["pass_gate"] for row in direction_summaries)
    interpretation = (
        "Promote only if every direction improves over the base residual receiver while preserving strict controls. "
        "If this gate fails, the learned receiver is pruned as an over-conservative or control-leaky calibration layer."
    )
    payload = {
        "gate": "source_private_candidate_conditioned_residual_code_smoke",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "budgets": budgets,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "calibration_examples": calibration_examples,
        "seed": seed,
        "feature_dim": feature_dim,
        "text_feature_mode": text_feature_mode,
        "feature_model": feature_model,
        "receiver_ridge": receiver_ridge,
        "matched_weight": matched_weight,
        "control_weight": control_weight,
        "min_improvement_over_base": min_improvement_over_base,
        "strict_controls": list(STRICT_CONTROLS),
        "matched_condition": MATCHED_CONDITION,
        "prediction_files": prediction_files,
        "direction_summaries": direction_summaries,
        "interpretation": interpretation,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    _write_csv(output_dir / "direction_summary.csv", direction_summaries)
    manifest_artifacts = ["summary.json", "summary.md", "direction_summary.csv", *prediction_files.values()]
    manifest = {
        "artifacts": manifest_artifacts,
        "artifact_sha256": {name: _sha256_file(output_dir / name) for name in manifest_artifacts},
        "pass_gate": pass_gate,
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(part) for part in value.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", type=_parse_budgets, default=[8])
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--calibration-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--receiver-ridge", type=float, default=1.0)
    parser.add_argument("--matched-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument("--min-improvement-over-base", type=float, default=0.03)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--text-feature-mode", default="hf_last_mean")
    parser.add_argument("--feature-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--feature-dtype", default="float32")
    parser.add_argument("--feature-max-length", type=int, default=128)
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_smoke(
        output_dir=output_dir,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        calibration_examples=args.calibration_examples,
        seed=args.seed,
        feature_dim=args.feature_dim,
        ridge=args.ridge,
        top_k=args.top_k,
        min_score=args.min_score,
        receiver_ridge=args.receiver_ridge,
        matched_weight=args.matched_weight,
        control_weight=args.control_weight,
        min_improvement_over_base=args.min_improvement_over_base,
        bootstrap_samples=args.bootstrap_samples,
        text_feature_mode=args.text_feature_mode,
        feature_model=args.feature_model,
        feature_device=args.feature_device,
        feature_dtype=args.feature_dtype,
        feature_max_length=args.feature_max_length,
        local_files_only=not args.allow_downloads,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2))


if __name__ == "__main__":
    main()
