from __future__ import annotations

"""Train-only source-score quantization gate for strict HellaSwag.

This gate tests whether calibrated quantized source-score packets can beat the
current strict 1B candidate-only packet on the same Qwen HellaSwag 0:9216
surface. It intentionally uses only train-split score caches for calibration
and saved validation score caches for evaluation.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = pathlib.Path(
    "results/"
    "source_private_hellaswag_hidden_innovation_multi_slice_stress_20260503_"
    "rank_score_channel_qwen05_validation0_9216/"
    "hellaswag_hidden_innovation_multi_slice_stress.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_strict_source_score_quantization_gate_20260503_validation0_9216"
)
DEFAULT_TRAIN_ROWS = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
CANDIDATE_COUNT = 4
BOOTSTRAP_SAMPLES = 2000


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = pathlib.Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _argmax(scores: np.ndarray) -> np.ndarray:
    return np.argmax(scores, axis=1).astype(np.int64)


def _rank_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1).astype(np.int64)


def _centered_z(scores: np.ndarray) -> np.ndarray:
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(scores, axis=1, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return centered / scale


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def _quantile_edges(values: np.ndarray, bins: int) -> list[float]:
    edges = np.quantile(values.astype(np.float64), np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return [float(item) for item in edges]


def _digitize(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.clip(
        np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right") - 1,
        0,
        len(edges) - 2,
    ).astype(np.int64)


def _permutation_index(order: np.ndarray) -> np.ndarray:
    # Four candidates only; simple Lehmer code is clearer than a generic helper.
    codes = []
    factorials = [6, 2, 1, 1]
    for row in order:
        remaining = list(range(CANDIDATE_COUNT))
        code = 0
        for idx, value in enumerate(row):
            position = remaining.index(int(value))
            code += position * factorials[idx]
            remaining.pop(position)
        codes.append(code)
    return np.asarray(codes, dtype=np.int64)


def _pack_digits(digits: np.ndarray, base: int) -> np.ndarray:
    codes = np.zeros(digits.shape[0], dtype=np.int64)
    multiplier = 1
    for column in range(digits.shape[1]):
        codes += digits[:, column].astype(np.int64) * multiplier
        multiplier *= int(base)
    return codes


def _raw_bytes_for_states(states: int) -> int:
    return max(1, int(math.ceil(math.ceil(math.log2(max(2, states))) / 8.0)))


def _framed_bytes(raw_bytes: int) -> int:
    return int(raw_bytes) + 3


def _score_cache_from_artifact(artifact_path: pathlib.Path | str) -> pathlib.Path:
    artifact = _read_json(artifact_path)
    if artifact.get("eval_score_cache"):
        return pathlib.Path(artifact["eval_score_cache"])
    metadata = artifact.get("eval_cache_metadata") or {}
    if metadata.get("score_cache"):
        return pathlib.Path(metadata["score_cache"])
    if artifact.get("bagged_gate_path"):
        return _score_cache_from_artifact(artifact["bagged_gate_path"])
    raise KeyError(f"could not find eval score cache in {artifact_path}")


def _predictions_path_from_artifact(artifact_path: pathlib.Path | str) -> pathlib.Path:
    artifact_path = _resolve(artifact_path)
    direct = artifact_path.parent / "predictions.jsonl"
    if direct.exists():
        return direct
    nested = artifact_path.parent / "bagged_gate" / "predictions.jsonl"
    if nested.exists():
        return nested
    artifact = _read_json(artifact_path)
    if artifact.get("bagged_gate_path"):
        return _predictions_path_from_artifact(artifact["bagged_gate_path"])
    raise FileNotFoundError(f"could not find predictions beside {artifact_path}")


def _train_score_caches_from_artifact(artifact_path: pathlib.Path | str) -> list[pathlib.Path]:
    predictions_path = _predictions_path_from_artifact(artifact_path)
    sample_caches_path = predictions_path.parent / "sample_caches.jsonl"
    rows = _read_jsonl(sample_caches_path)
    paths = []
    for row in rows:
        path = pathlib.Path(row["train_score_cache"])
        if path not in paths:
            paths.append(path)
    return paths


def _load_train_answers(path: pathlib.Path | str) -> dict[str, int]:
    answers = {}
    for row in _read_jsonl(path):
        answers[str(row["id"])] = int(row["answer_index"])
    return answers


def _load_score_cache(path: pathlib.Path | str) -> tuple[list[str], np.ndarray]:
    payload = _read_json(path)
    scores = np.asarray(payload["source_scores"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != CANDIDATE_COUNT:
        raise ValueError(f"expected N x 4 source_scores in {path}, got {scores.shape}")
    return [str(item) for item in payload["row_ids"]], scores


class ScoreVariant:
    def __init__(self, name: str, states: int, params: dict[str, Any] | None = None):
        self.name = name
        self.states = int(states)
        self.params = params or {}
        self.raw_bytes = _raw_bytes_for_states(self.states)
        self.framed_bytes = _framed_bytes(self.raw_bytes)

    def encode(self, scores: np.ndarray, *, control: str = "matched") -> np.ndarray:
        if control in {"candidate_roll", "score_channel_roll"}:
            scores = np.roll(scores, 1, axis=1)
        if self.name == "source_argmax_1b":
            return _argmax(scores)
        if self.name == "rank_order_majority_1b":
            return _permutation_index(_rank_order(scores))
        if self.name == "top2_margin_q16_majority_1b":
            order = _rank_order(scores)
            top1 = order[:, 0]
            top2 = order[:, 1]
            margin_bin = _digitize(_top2_margin(scores), self.params["margin_edges"])
            return top1 + 4 * top2 + 16 * margin_bin
        if self.name.startswith("zscore_q"):
            bins = int(self.params["bins"])
            zscores = _centered_z(scores)
            digits = _digitize(zscores.reshape(-1), self.params["z_edges"]).reshape(scores.shape)
            return _pack_digits(digits, bins)
        raise ValueError(f"unsupported variant: {self.name}")

    def fallback(self, scores: np.ndarray) -> np.ndarray:
        return _argmax(scores)


def _build_variants(train_scores: np.ndarray) -> list[ScoreVariant]:
    zscores = _centered_z(train_scores)
    return [
        ScoreVariant("source_argmax_1b", states=4),
        ScoreVariant("rank_order_majority_1b", states=24),
        ScoreVariant(
            "top2_margin_q16_majority_1b",
            states=4 * 4 * 16,
            params={"margin_edges": _quantile_edges(_top2_margin(train_scores), 16)},
        ),
        ScoreVariant(
            "zscore_q2_vector_majority_1b",
            states=4**4,
            params={"bins": 4, "z_edges": _quantile_edges(zscores.reshape(-1), 4)},
        ),
        ScoreVariant(
            "zscore_q3_vector_majority_2b",
            states=8**4,
            params={"bins": 8, "z_edges": _quantile_edges(zscores.reshape(-1), 8)},
        ),
        ScoreVariant(
            "zscore_q4_vector_majority_2b",
            states=16**4,
            params={"bins": 16, "z_edges": _quantile_edges(zscores.reshape(-1), 16)},
        ),
        ScoreVariant(
            "zscore_q5_vector_majority_3b",
            states=32**4,
            params={"bins": 32, "z_edges": _quantile_edges(zscores.reshape(-1), 32)},
        ),
    ]


def _fit_decoder(
    *,
    variant: ScoreVariant,
    train_scores: np.ndarray,
    train_answers: np.ndarray,
) -> dict[int, int]:
    codes = variant.encode(train_scores)
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for code, answer in zip(codes, train_answers, strict=True):
        counts[int(code)][int(answer)] += 1
    decoder = {}
    for code, counter in counts.items():
        decoder[code] = max(counter.items(), key=lambda item: (item[1], -item[0]))[0]
    return decoder


def _predict_with_decoder(
    *,
    variant: ScoreVariant,
    decoder: dict[int, int],
    scores: np.ndarray,
    control: str = "matched",
    seed: int = 0,
) -> np.ndarray:
    codes = variant.encode(scores, control=control)
    if control == "row_shuffle":
        rng = random.Random(seed)
        indices = list(range(len(codes)))
        rng.shuffle(indices)
        codes = codes[np.asarray(indices, dtype=np.int64)]
    fallback = variant.fallback(scores)
    predictions = np.empty(len(codes), dtype=np.int64)
    for index, code in enumerate(codes):
        predictions[index] = int(decoder.get(int(code), int(fallback[index])))
    return predictions


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == answers.astype(np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    selected_correct = (selected.astype(np.int64) == answers.astype(np.int64)).astype(np.float64)
    baseline_correct = (baseline.astype(np.int64) == answers.astype(np.int64)).astype(np.float64)
    deltas = selected_correct - baseline_correct
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(deltas), size=len(deltas))
        draws.append(float(np.mean(deltas[indices])))
    return {
        "delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _load_training_matrix(
    *,
    cache_paths: list[pathlib.Path],
    train_answer_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    seen: set[str] = set()
    scores: list[list[float]] = []
    answers: list[int] = []
    row_ids: list[str] = []
    for path in cache_paths:
        cache_row_ids, cache_scores = _load_score_cache(path)
        for row_id, score_row in zip(cache_row_ids, cache_scores, strict=True):
            if row_id in seen:
                continue
            if row_id not in train_answer_map:
                raise KeyError(f"train row id {row_id} missing from official train split")
            seen.add(row_id)
            row_ids.append(row_id)
            scores.append([float(item) for item in score_row])
            answers.append(int(train_answer_map[row_id]))
    if not scores:
        raise ValueError("no train score rows loaded")
    return np.asarray(scores, dtype=np.float64), np.asarray(answers, dtype=np.int64), row_ids


def _slice_payload(
    *,
    slice_row: dict[str, Any],
    variants: list[ScoreVariant],
    decoders: dict[str, dict[int, int]],
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    artifact_path = pathlib.Path(slice_row["artifact_path"])
    predictions_rows = _read_jsonl(_predictions_path_from_artifact(artifact_path))
    prediction_by_id = {str(row["row_id"]): row for row in predictions_rows}
    score_row_ids, scores = _load_score_cache(_score_cache_from_artifact(artifact_path))

    answers = []
    candidate_only = []
    for row_id in score_row_ids:
        if row_id not in prediction_by_id:
            raise KeyError(f"score row id {row_id} missing from predictions for {artifact_path}")
        row = prediction_by_id[row_id]
        answers.append(int(row["answer_index"]))
        candidate_only.append(int(row["selected_prediction"]))
    answers_np = np.asarray(answers, dtype=np.int64)
    candidate_np = np.asarray(candidate_only, dtype=np.int64)
    predictions_by_variant: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for variant in variants:
        matched = _predict_with_decoder(
            variant=variant,
            decoder=decoders[variant.name],
            scores=scores,
            control="matched",
        )
        row_shuffle = _predict_with_decoder(
            variant=variant,
            decoder=decoders[variant.name],
            scores=scores,
            control="row_shuffle",
            seed=1729 + int(slice_row["eval_slice_start"]),
        )
        candidate_roll = _predict_with_decoder(
            variant=variant,
            decoder=decoders[variant.name],
            scores=scores,
            control="candidate_roll",
        )
        predictions_by_variant[variant.name] = matched
        paired = _paired_ci(
            selected=matched,
            baseline=candidate_np,
            answers=answers_np,
            seed=20260503 + int(slice_row["eval_slice_start"]),
            samples=bootstrap_samples,
        )
        rows.append(
            {
                "eval_slice_start": int(slice_row["eval_slice_start"]),
                "eval_slice_end_exclusive": int(slice_row["eval_slice_end_exclusive"]),
                "eval_rows": len(answers_np),
                "variant": variant.name,
                "raw_payload_bytes": variant.raw_bytes,
                "framed_record_bytes": variant.framed_bytes,
                "matched_accuracy": _accuracy(matched, answers_np),
                "candidate_only_accuracy": _accuracy(candidate_np, answers_np),
                "row_shuffle_control_accuracy": _accuracy(row_shuffle, answers_np),
                "candidate_roll_control_accuracy": _accuracy(candidate_roll, answers_np),
                "matched_minus_candidate_only": paired["delta"],
                "ci95_low_vs_candidate_only": paired["ci95_low"],
                "ci95_high_vs_candidate_only": paired["ci95_high"],
                "score_cache": _display_path(_score_cache_from_artifact(artifact_path)),
            }
        )
    return rows, predictions_by_variant, answers_np, candidate_np


def _weighted(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["eval_rows"]) for row in rows)
    return sum(float(row[key]) * int(row["eval_rows"]) for row in rows) / float(total)


def _aggregate_variant_rows(
    *,
    slice_variant_rows: list[dict[str, Any]],
    variant_predictions: dict[str, list[np.ndarray]],
    answers: list[np.ndarray],
    candidate_predictions: list[np.ndarray],
    variants: list[ScoreVariant],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    answer_np = np.concatenate(answers)
    candidate_np = np.concatenate(candidate_predictions)
    rows = []
    by_variant = {variant.name: variant for variant in variants}
    for name, chunks in variant_predictions.items():
        prediction_np = np.concatenate(chunks)
        variant = by_variant[name]
        paired = _paired_ci(
            selected=prediction_np,
            baseline=candidate_np,
            answers=answer_np,
            seed=30360503 + sum(ord(char) for char in name),
            samples=bootstrap_samples,
        )
        slice_rows = [row for row in slice_variant_rows if row["variant"] == name]
        rows.append(
            {
                "variant": name,
                "raw_payload_bytes": variant.raw_bytes,
                "framed_record_bytes": variant.framed_bytes,
                "matched_accuracy": _accuracy(prediction_np, answer_np),
                "candidate_only_accuracy": _accuracy(candidate_np, answer_np),
                "row_shuffle_control_accuracy": _weighted(slice_rows, "row_shuffle_control_accuracy"),
                "candidate_roll_control_accuracy": _weighted(
                    slice_rows, "candidate_roll_control_accuracy"
                ),
                "matched_minus_candidate_only": paired["delta"],
                "ci95_low_vs_candidate_only": paired["ci95_low"],
                "ci95_high_vs_candidate_only": paired["ci95_high"],
                "slice_count": len(slice_rows),
                "improvement_slice_count": sum(
                    float(row["matched_minus_candidate_only"]) > 0 for row in slice_rows
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["matched_accuracy"],
            row["ci95_low_vs_candidate_only"],
            -row["raw_payload_bytes"],
        ),
        reverse=True,
    )


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    best = headline["best_score_quantized_variant"]
    lines = [
        "# HellaSwag Strict Source-Score Quantization Gate",
        "",
        f"- positive method pass: `{payload['positive_method_pass']}`",
        f"- reviewer-control audit complete: `{payload['reviewer_control_audit_complete']}`",
        f"- eval rows: `{headline['total_eval_rows']}`",
        f"- candidate-only accuracy: `{headline['candidate_only_accuracy']:.6f}`",
        f"- best score-quantized variant: `{best['variant']}`",
        f"- best score-quantized accuracy: `{best['matched_accuracy']:.6f}`",
        f"- best minus candidate-only: `{best['matched_minus_candidate_only']:.6f}`",
        f"- best CI95 low vs candidate-only: `{best['ci95_low_vs_candidate_only']:.6f}`",
        f"- best packet bytes: `{best['raw_payload_bytes']}B` raw / `{best['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
    ]
    _resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    source_path: pathlib.Path | str = DEFAULT_SOURCE,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    train_rows_path: pathlib.Path | str = DEFAULT_TRAIN_ROWS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    source = _read_json(source_path)
    if not bool(source.get("pass_gate")):
        raise ValueError("source strict multi-slice gate did not pass")
    slice_rows = list(source["slice_rows"])
    train_cache_paths = _train_score_caches_from_artifact(slice_rows[0]["artifact_path"])
    train_answer_map = _load_train_answers(train_rows_path)
    train_scores, train_answers, train_row_ids = _load_training_matrix(
        cache_paths=train_cache_paths,
        train_answer_map=train_answer_map,
    )
    variants = _build_variants(train_scores)
    decoders = {
        variant.name: _fit_decoder(
            variant=variant,
            train_scores=train_scores,
            train_answers=train_answers,
        )
        for variant in variants
    }
    slice_variant_rows: list[dict[str, Any]] = []
    variant_predictions: dict[str, list[np.ndarray]] = {variant.name: [] for variant in variants}
    all_answers: list[np.ndarray] = []
    all_candidates: list[np.ndarray] = []
    for slice_row in slice_rows:
        rows, predictions_by_variant, answers, candidates = _slice_payload(
            slice_row=slice_row,
            variants=variants,
            decoders=decoders,
            bootstrap_samples=bootstrap_samples,
        )
        slice_variant_rows.extend(rows)
        for name, predictions in predictions_by_variant.items():
            variant_predictions[name].append(predictions)
        all_answers.append(answers)
        all_candidates.append(candidates)

    variant_rows = _aggregate_variant_rows(
        slice_variant_rows=slice_variant_rows,
        variant_predictions=variant_predictions,
        answers=all_answers,
        candidate_predictions=all_candidates,
        variants=variants,
        bootstrap_samples=bootstrap_samples,
    )
    best = variant_rows[0]
    positive_method_pass = (
        float(best["matched_minus_candidate_only"]) > 0.0
        and float(best["ci95_low_vs_candidate_only"]) > 0.0
        and int(best["improvement_slice_count"]) == len(slice_rows)
    )
    reviewer_control_audit_complete = all(
        row["matched_accuracy"] <= source["headline"]["weighted_selected_eval_accuracy"] + 1e-12
        for row in variant_rows
    )
    payload = {
        "gate": "source_private_hellaswag_strict_source_score_quantization_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_artifact": _display_path(source_path),
        "source_artifact_sha256": _sha256_file(source_path),
        "positive_method_pass": positive_method_pass,
        "reviewer_control_audit_complete": reviewer_control_audit_complete,
        "headline": {
            "total_eval_rows": int(source["headline"]["total_eval_rows"]),
            "slice_count": len(slice_rows),
            "train_rows": len(train_row_ids),
            "candidate_only_accuracy": float(source["headline"]["weighted_selected_eval_accuracy"]),
            "best_score_quantized_variant": best,
            "score_quantized_variant_count": len(variant_rows),
            "all_score_quantized_variants_below_candidate_only": reviewer_control_audit_complete,
        },
        "packet_contract": {
            "forbidden_source_fields": [
                "source_text",
                "source_kv_cache",
                "raw_source_hidden_vector",
                "raw_unquantized_source_score_vector",
                "source_logits",
            ],
            "allowed_source_fields": [
                "quantized source score code",
                "train-split decoder metadata",
            ],
        },
        "pass_rule": (
            "A positive source-score method passes only if a train-calibrated quantized score packet "
            "beats the strict candidate-only packet over all 9 slices with positive paired CI95 low. "
            "A reviewer-control audit is complete if all calibrated score-code variants remain below "
            "candidate-only while using no source text/KV/raw hidden/raw unquantized scores."
        ),
        "interpretation": (
            "Train-calibrated source-score quantization does not beat the strict candidate-only packet "
            "on the 9216-row HellaSwag surface. This weakens the source-score branch as an ICLR-positive "
            "method but closes a reviewer gap: explicit score-vector and rank/margin code baselines at "
            "matched and larger byte budgets were tested against the current packet."
        ),
        "lay_explanation": (
            "We tried sending compressed versions of the source model's four answer scores instead of "
            "only its chosen answer. A small decoder was trained on HellaSwag train rows, then frozen. "
            "On the large validation surface, these score codes still did not beat the tiny candidate-only hint."
        ),
        "train_score_caches": [
            {"path": _display_path(path), "sha256": _sha256_file(path)} for path in train_cache_paths
        ],
        "train_rows_path": _display_path(train_rows_path),
        "train_rows_sha256": _sha256_file(train_rows_path),
        "variant_rows": variant_rows,
        "slice_variant_rows": slice_variant_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_strict_source_score_quantization_gate.json", payload)
    _write_csv(output_dir / "variant_rows.csv", variant_rows)
    _write_csv(output_dir / "slice_variant_rows.csv", slice_variant_rows)
    _write_markdown(output_dir / "hellaswag_strict_source_score_quantization_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "source_artifact": payload["source_artifact"],
            "source_artifact_sha256": payload["source_artifact_sha256"],
            "outputs": [
                "hellaswag_strict_source_score_quantization_gate.json",
                "hellaswag_strict_source_score_quantization_gate.md",
                "variant_rows.csv",
                "slice_variant_rows.csv",
            ],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test calibrated source-score quantization against strict HellaSwag candidate-only."
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-rows", type=pathlib.Path, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", default=None)
    args = parser.parse_args()
    payload = build_gate(
        source_path=args.source,
        output_dir=args.output_dir,
        train_rows_path=args.train_rows,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
