from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl"
)
DEFAULT_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json"
)
DEFAULT_TRAIN_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/"
    "source_train_score_cache.json"
)
DEFAULT_TRAIN_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/"
    "source_train_hidden_cache.npz"
)
DEFAULT_EVAL_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/"
    "source_eval_hidden_cache.npz"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_repair_probe_20260501_qwen05_train512_validation1024"
)

STRICT_DELTA = 0.02


def _safe_softmax(scores: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    exps = np.exp(values - np.max(values))
    return exps / np.sum(exps)


def _candidate_score_features(scores: list[float], candidate: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    probs = _safe_softmax(values)
    ranked = top2._ranked_indices(list(values))
    rank_pos = {index: rank for rank, index in enumerate(ranked)}
    entropy = float(-sum(prob * math.log(max(float(prob), 1e-12)) for prob in probs))
    scalar = np.asarray(
        [
            values[candidate],
            values[candidate] - values[ranked[0]],
            values[candidate] - float(np.mean(values)),
            probs[candidate],
            float(rank_pos[candidate]),
            float(candidate),
            values[ranked[0]] - values[ranked[1]],
            entropy,
        ],
        dtype=np.float64,
    )
    rank_one_hot = np.zeros(4, dtype=np.float64)
    rank_one_hot[rank_pos[candidate]] = 1.0
    candidate_one_hot = np.zeros(4, dtype=np.float64)
    candidate_one_hot[candidate] = 1.0
    return np.concatenate([scalar, rank_one_hot, candidate_one_hot])


def _candidate_feature_tensor(
    *,
    scores: list[list[float]],
    hidden: np.ndarray,
    view: str,
) -> np.ndarray:
    layer_hidden = np.asarray(hidden[:, :, 0, :], dtype=np.float64)
    rows: list[list[np.ndarray]] = []
    for row_scores, row_hidden in zip(scores, layer_hidden, strict=True):
        ranked = top2._ranked_indices(row_scores)
        top_hidden = row_hidden[ranked[0]]
        mean_hidden = np.mean(row_hidden, axis=0)
        row_features: list[np.ndarray] = []
        for candidate in range(4):
            parts: list[np.ndarray] = []
            if "score" in view:
                parts.append(_candidate_score_features(row_scores, candidate))
            if "hidden_residual" in view:
                parts.append(row_hidden[candidate] - top_hidden)
            if "hidden_absolute" in view:
                parts.append(row_hidden[candidate])
                parts.append(row_hidden[candidate] - mean_hidden)
            row_features.append(np.concatenate(parts))
        rows.append(row_features)
    return np.asarray(rows, dtype=np.float64)


def _fit_candidate_ridge(
    *,
    features: np.ndarray,
    rows: list[arc_gate.ArcRow],
    score_matrix: list[list[float]],
    fit_indices: list[int],
    ridge: float,
) -> dict[str, Any]:
    x_fit = features[np.asarray(fit_indices, dtype=np.int64)].reshape(-1, features.shape[-1])
    labels: list[float] = []
    weights: list[float] = []
    for index in fit_indices:
        ranked = top2._ranked_indices(score_matrix[index])
        for candidate in range(4):
            labels.append(1.0 if rows[index].answer_index == candidate else -1.0)
            weights.append(1.0 if candidate in ranked[:2] or rows[index].answer_index == candidate else 0.5)
    y = np.asarray(labels, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    mean = np.mean(x_fit, axis=0)
    scale = np.std(x_fit, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_body = (x_fit - mean) / scale
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    weighted_x = x * sample_weights[:, None]
    xtx = x.T @ weighted_x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    xtx[0, 0] -= float(ridge)
    beta = np.linalg.solve(xtx, weighted_x.T @ y)
    return {
        "beta": beta,
        "mean": mean,
        "scale": scale,
        "ridge": float(ridge),
        "feature_dim": int(features.shape[-1]),
        "fit_rows": int(len(fit_indices)),
    }


def _predict_candidate_ridge(features: np.ndarray, model: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    x_body = (features.reshape(-1, features.shape[-1]) - model["mean"]) / model["scale"]
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    candidate_scores = np.asarray(x @ model["beta"], dtype=np.float64).reshape(features.shape[0], 4)
    return [int(value) for value in np.argmax(candidate_scores, axis=1)], candidate_scores


def _fit_and_eval_view(
    *,
    view: str,
    ridge: float,
    train_features: np.ndarray,
    eval_features: np.ndarray,
    fit_indices: list[int],
    dev_indices: list[int],
    train_rows: list[arc_gate.ArcRow],
    train_scores: list[list[float]],
    eval_rows: list[arc_gate.ArcRow],
) -> dict[str, Any]:
    model = _fit_candidate_ridge(
        features=train_features,
        rows=train_rows,
        score_matrix=train_scores,
        fit_indices=fit_indices,
        ridge=ridge,
    )
    train_predictions, _ = _predict_candidate_ridge(train_features, model)
    eval_predictions, eval_candidate_scores = _predict_candidate_ridge(eval_features, model)
    fit_rows = top2._take_rows(train_rows, fit_indices)
    dev_rows = top2._take_rows(train_rows, dev_indices)
    fit_predictions = [train_predictions[index] for index in fit_indices]
    dev_predictions = [train_predictions[index] for index in dev_indices]
    return {
        "view": view,
        "ridge": float(ridge),
        "feature_dim": int(train_features.shape[-1]),
        "fit": top2._evaluate(fit_rows, fit_predictions),
        "internal_dev": top2._evaluate(dev_rows, dev_predictions),
        "train": top2._evaluate(train_rows, train_predictions),
        "eval": top2._evaluate(eval_rows, eval_predictions),
        "eval_predictions": eval_predictions,
        "eval_candidate_scores": eval_candidate_scores,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Hidden-Innovation Repair Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected view: `{h['selected_view']}`",
        f"- selected ridge: `{h['selected_ridge']}`",
        f"- eval accuracy: `{h['selected_eval_accuracy']:.6f}`",
        f"- source-label copy accuracy: `{h['source_label_copy_eval_accuracy']:.6f}`",
        f"- trained-label copy accuracy: `{h['trained_choice_bias_label_copy_eval_accuracy']:.6f}`",
        f"- delta vs best label copy: `{h['selected_minus_best_label_copy']:.6f}`",
        f"- paired CI95 vs best label copy: `[{h['paired_ci95_selected_vs_best_label_copy']['ci95_low']:.6f}, {h['paired_ci95_selected_vs_best_label_copy']['ci95_high']:.6f}]`",
        f"- zero-hidden control accuracy: `{h['zero_hidden_control_accuracy']:.6f}`",
        f"- wrong-example hidden control accuracy: `{h['wrong_example_hidden_control_accuracy']:.6f}`",
        f"- candidate-roll hidden control accuracy: `{h['candidate_roll_hidden_control_accuracy']:.6f}`",
        f"- source top-2 oracle accuracy: `{h['source_top2_oracle_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_probe(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = DEFAULT_TRAIN,
    eval_path: pathlib.Path = DEFAULT_EVAL,
    eval_score_cache: pathlib.Path = DEFAULT_EVAL_SCORE_CACHE,
    train_score_cache: pathlib.Path = DEFAULT_TRAIN_SCORE_CACHE,
    train_hidden_cache: pathlib.Path = DEFAULT_TRAIN_HIDDEN_CACHE,
    eval_hidden_cache: pathlib.Path = DEFAULT_EVAL_HIDDEN_CACHE,
    train_hidden_rows: int = 512,
    selection_seed: int = 1729,
    dev_fraction: float = 0.25,
    ridges: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0),
    bootstrap_samples: int = 500,
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = top2._resolve(train_path)
    eval_path = top2._resolve(eval_path)
    eval_score_cache = top2._resolve(eval_score_cache)
    train_score_cache = top2._resolve(train_score_cache)
    train_hidden_cache = top2._resolve(train_hidden_cache)
    eval_hidden_cache = top2._resolve(eval_hidden_cache)

    all_train_rows = arc_gate._load_rows(train_path)
    train_rows = top2._select_train_rows(all_train_rows, count=train_hidden_rows, seed=selection_seed)
    eval_rows = arc_gate._load_rows(eval_path)
    fit_indices, dev_indices = top2._split_indices(
        len(train_rows), dev_fraction=dev_fraction, seed=selection_seed + 17
    )

    train_scores, _, train_source_model = headroom._load_score_cache(train_score_cache, rows=train_rows)
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
    train_hidden, train_hidden_meta = top2._load_hidden_cache(train_hidden_cache, rows=train_rows)
    eval_hidden, eval_hidden_meta = top2._load_hidden_cache(eval_hidden_cache, rows=eval_rows)

    views = (
        "score_only",
        "hidden_residual_only",
        "score_hidden_residual",
        "score_hidden_absolute",
        "score_hidden_absolute_residual",
    )
    train_features = {
        view: _candidate_feature_tensor(scores=train_scores, hidden=train_hidden, view=view) for view in views
    }
    eval_features = {
        view: _candidate_feature_tensor(scores=eval_scores, hidden=eval_hidden, view=view) for view in views
    }

    candidate_readouts: list[dict[str, Any]] = []
    for view in views:
        for ridge in ridges:
            candidate_readouts.append(
                _fit_and_eval_view(
                    view=view,
                    ridge=ridge,
                    train_features=train_features[view],
                    eval_features=eval_features[view],
                    fit_indices=fit_indices,
                    dev_indices=dev_indices,
                    train_rows=train_rows,
                    train_scores=train_scores,
                    eval_rows=eval_rows,
                )
            )

    selected = max(
        candidate_readouts,
        key=lambda item: (
            item["internal_dev"]["accuracy"],
            item["fit"]["accuracy"],
            item["view"] == "score_hidden_residual",
            -item["feature_dim"],
            -item["ridge"],
        ),
    )
    source_label_eval = top2._source_label_predictions(eval_scores)
    offsets = score_repair._fit_choice_bias_offsets(
        top2._take_rows(train_rows, fit_indices),
        top2._take_scores(train_scores, fit_indices),
    )
    trained_label_eval = [score_repair._predict_calibrated_label(row_scores, offsets) for row_scores in eval_scores]
    source_label_accuracy = top2._accuracy(eval_rows, source_label_eval)
    trained_label_accuracy = top2._accuracy(eval_rows, trained_label_eval)
    best_label_accuracy = max(source_label_accuracy, trained_label_accuracy)
    best_label_predictions = source_label_eval if source_label_accuracy >= trained_label_accuracy else trained_label_eval

    best_score_control = max(
        (item for item in candidate_readouts if item["view"] == "score_only"),
        key=lambda item: (item["internal_dev"]["accuracy"], item["fit"]["accuracy"], item["eval"]["accuracy"]),
    )
    selected_predictions = [int(value) for value in selected["eval_predictions"]]
    selected_view = selected["view"]
    selected_ridge = selected["ridge"]
    refit_model = _fit_candidate_ridge(
        features=train_features[selected_view],
        rows=train_rows,
        score_matrix=train_scores,
        fit_indices=fit_indices,
        ridge=selected_ridge,
    )
    zero_eval_hidden = np.zeros_like(eval_hidden)
    wrong_eval_hidden = np.roll(eval_hidden, 1, axis=0)
    candidate_roll_hidden = np.roll(eval_hidden, 1, axis=1)
    zero_predictions, _ = _predict_candidate_ridge(
        _candidate_feature_tensor(scores=eval_scores, hidden=zero_eval_hidden, view=selected_view),
        refit_model,
    )
    wrong_predictions, _ = _predict_candidate_ridge(
        _candidate_feature_tensor(scores=eval_scores, hidden=wrong_eval_hidden, view=selected_view),
        refit_model,
    )
    candidate_roll_predictions, _ = _predict_candidate_ridge(
        _candidate_feature_tensor(scores=eval_scores, hidden=candidate_roll_hidden, view=selected_view),
        refit_model,
    )

    paired_ci_label = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        best_label_predictions,
        seed=selection_seed + 3001,
        samples=bootstrap_samples,
    )
    paired_ci_score = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        [int(value) for value in best_score_control["eval_predictions"]],
        seed=selection_seed + 3002,
        samples=bootstrap_samples,
    )

    candidate_summary = [
        {
            "view": item["view"],
            "ridge": item["ridge"],
            "feature_dim": item["feature_dim"],
            "fit_accuracy": item["fit"]["accuracy"],
            "internal_dev_accuracy": item["internal_dev"]["accuracy"],
            "eval_accuracy": item["eval"]["accuracy"],
        }
        for item in candidate_readouts
    ]

    predictions = [
        {
            "row_id": row.row_id,
            "answer_index": row.answer_index,
            "selected_prediction": int(prediction),
            "source_label_prediction": int(source_label),
            "trained_label_prediction": int(trained_label),
            "zero_hidden_prediction": int(zero_prediction),
            "wrong_example_hidden_prediction": int(wrong_prediction),
            "candidate_roll_hidden_prediction": int(candidate_roll_prediction),
        }
        for row, prediction, source_label, trained_label, zero_prediction, wrong_prediction, candidate_roll_prediction in zip(
            eval_rows,
            selected_predictions,
            source_label_eval,
            trained_label_eval,
            zero_predictions,
            wrong_predictions,
            candidate_roll_predictions,
            strict=True,
        )
    ]

    selected_eval_accuracy = selected["eval"]["accuracy"]
    zero_accuracy = top2._accuracy(eval_rows, zero_predictions)
    wrong_accuracy = top2._accuracy(eval_rows, wrong_predictions)
    candidate_roll_accuracy = top2._accuracy(eval_rows, candidate_roll_predictions)
    score_control_accuracy = best_score_control["eval"]["accuracy"]
    packet_contract = {
        "packet_name": "hidden_innovation_candidate_selector_packet",
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "fields": [
            "selected candidate id packed into 2 bits",
            "quantized hidden-innovation confidence/debug bin",
            "train-dev-selected model id packed into metadata bits",
        ],
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
        "decoder_rule": "receiver chooses the candidate id computed by the source-side hidden-innovation denoiser",
    }
    headline = {
        "selected_view": selected_view,
        "selected_ridge": selected_ridge,
        "selected_feature_dim": selected["feature_dim"],
        "selected_internal_dev_accuracy": selected["internal_dev"]["accuracy"],
        "selected_fit_accuracy": selected["fit"]["accuracy"],
        "selected_eval_accuracy": selected_eval_accuracy,
        "source_label_copy_eval_accuracy": source_label_accuracy,
        "trained_choice_bias_label_copy_eval_accuracy": trained_label_accuracy,
        "best_label_copy_eval_accuracy": best_label_accuracy,
        "selected_minus_source_label_copy": selected_eval_accuracy - source_label_accuracy,
        "selected_minus_trained_choice_bias_label_copy": selected_eval_accuracy - trained_label_accuracy,
        "selected_minus_best_label_copy": selected_eval_accuracy - best_label_accuracy,
        "score_only_control_accuracy": score_control_accuracy,
        "selected_minus_score_only_control": selected_eval_accuracy - score_control_accuracy,
        "zero_hidden_control_accuracy": zero_accuracy,
        "selected_minus_zero_hidden_control": selected_eval_accuracy - zero_accuracy,
        "wrong_example_hidden_control_accuracy": wrong_accuracy,
        "selected_minus_wrong_example_hidden_control": selected_eval_accuracy - wrong_accuracy,
        "candidate_roll_hidden_control_accuracy": candidate_roll_accuracy,
        "selected_minus_candidate_roll_hidden_control": selected_eval_accuracy - candidate_roll_accuracy,
        "source_top2_oracle_accuracy": top2._topk_oracle(eval_rows, eval_scores, k=2),
        "source_top4_oracle_accuracy": top2._topk_oracle(eval_rows, eval_scores, k=4),
        "paired_ci95_selected_vs_best_label_copy": paired_ci_label,
        "paired_ci95_selected_vs_score_only_control": paired_ci_score,
    }
    pass_gate = bool(
        headline["selected_minus_best_label_copy"] >= STRICT_DELTA
        and headline["paired_ci95_selected_vs_best_label_copy"]["ci95_low"] > 0.0
        and headline["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and wrong_accuracy <= best_label_accuracy
        and candidate_roll_accuracy <= best_label_accuracy
    )

    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_repair_probe",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if a train-only source-side hidden-innovation denoiser beats the best source-label/trained-label "
            "copy control by at least 0.02 with paired CI95 low > 0, beats zero-hidden by at least 0.02, and "
            "wrong-example/candidate-roll hidden controls do not beat label-copy."
        ),
        "train_path": top2._display_path(train_path),
        "train_sha256": top2._sha256_file(train_path),
        "eval_path": top2._display_path(eval_path),
        "eval_sha256": top2._sha256_file(eval_path),
        "eval_score_cache": top2._display_path(eval_score_cache),
        "eval_score_cache_sha256": top2._sha256_file(eval_score_cache),
        "train_score_cache": top2._display_path(train_score_cache),
        "train_score_cache_sha256": top2._sha256_file(train_score_cache),
        "train_hidden_cache": top2._display_path(train_hidden_cache),
        "train_hidden_cache_sha256": top2._sha256_file(train_hidden_cache),
        "eval_hidden_cache": top2._display_path(eval_hidden_cache),
        "eval_hidden_cache_sha256": top2._sha256_file(eval_hidden_cache),
        "all_train_rows": len(all_train_rows),
        "scored_train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "internal_fit_rows": len(fit_indices),
        "internal_dev_rows": len(dev_indices),
        "selection_seed": selection_seed,
        "dev_fraction": dev_fraction,
        "source_model": {
            "score_train": train_source_model,
            "score_eval": eval_source_model,
            "hidden_train": train_hidden_meta,
            "hidden_eval": eval_hidden_meta,
        },
        "packet_contract": packet_contract,
        "headline": headline,
        "candidate_readouts": candidate_summary,
        "control_readouts": {
            "source_label_copy": top2._evaluate(eval_rows, source_label_eval),
            "trained_choice_bias_label_copy": top2._evaluate(eval_rows, trained_label_eval),
            "best_score_only_control": {
                "view": best_score_control["view"],
                "ridge": best_score_control["ridge"],
                **best_score_control["eval"],
            },
            "zero_hidden_control": top2._evaluate(eval_rows, zero_predictions),
            "wrong_example_hidden_control": top2._evaluate(eval_rows, wrong_predictions),
            "candidate_roll_hidden_control": top2._evaluate(eval_rows, candidate_roll_predictions),
        },
        "interpretation": (
            "Unlike the failed top-2 switcher, this branch treats the source hidden state as an innovation signal: "
            "for each candidate, a train-only denoiser scores whether the candidate is the answer using source scores "
            "plus the candidate's hidden residual against the source top choice. The receiver receives only a fixed-byte "
            "candidate/confidence packet, not source text, KV, raw scores, or raw hidden vectors."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }

    (output_dir / "hellaswag_hidden_innovation_repair_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "candidate_readouts.jsonl", candidate_summary)
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_markdown(output_dir / "hellaswag_hidden_innovation_repair_probe.md", payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": payload["headline"],
        "files": [
            {
                "path": top2._display_path(path),
                "sha256": top2._sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                output_dir / "hellaswag_hidden_innovation_repair_probe.json",
                output_dir / "hellaswag_hidden_innovation_repair_probe.md",
                output_dir / "candidate_readouts.jsonl",
                output_dir / "predictions.jsonl",
            )
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--train-score-cache", type=pathlib.Path, default=DEFAULT_TRAIN_SCORE_CACHE)
    parser.add_argument("--train-hidden-cache", type=pathlib.Path, default=DEFAULT_TRAIN_HIDDEN_CACHE)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()

    payload = build_probe(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        train_score_cache=args.train_score_cache,
        train_hidden_cache=args.train_hidden_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        selection_seed=args.selection_seed,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
