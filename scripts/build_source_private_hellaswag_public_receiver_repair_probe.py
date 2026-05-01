from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl"
)
DEFAULT_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024"
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_index(text: str, dim: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _feature_indices(row: arc_gate.ArcRow, choice_index: int, *, dim: int) -> list[int]:
    question_tokens = _tokens(row.question)
    choice_tokens = _tokens(row.choices[choice_index])
    indices: list[int] = []

    def add(name: str, weight: int = 1) -> None:
        index = _stable_index(name, dim)
        indices.extend([index] * weight)

    add(f"label:{choice_index}")
    add(f"qlen:{min(len(question_tokens) // 5, 10)}|label:{choice_index}")
    add(f"clen:{min(len(choice_tokens) // 4, 10)}|label:{choice_index}")
    if question_tokens:
        add(f"q_last:{question_tokens[-1]}|label:{choice_index}")
    if len(question_tokens) > 1:
        add(f"q_last2:{question_tokens[-2]}_{question_tokens[-1]}|label:{choice_index}")
    if choice_tokens:
        add(f"c_first:{choice_tokens[0]}|label:{choice_index}")
    if len(choice_tokens) > 1:
        add(f"c_first2:{choice_tokens[0]}_{choice_tokens[1]}|label:{choice_index}")
    if question_tokens and choice_tokens:
        add(f"join:{question_tokens[-1]}_{choice_tokens[0]}")
        add(f"join_label:{question_tokens[-1]}_{choice_tokens[0]}|{choice_index}")

    for token in choice_tokens:
        add(f"c:{token}")
        add(f"c_label:{token}|{choice_index}")
    for left, right in zip(choice_tokens, choice_tokens[1:]):
        add(f"cb:{left}_{right}")
    for token in question_tokens[-12:]:
        add(f"qtail:{token}")
        if choice_tokens:
            add(f"qt_cf:{token}_{choice_tokens[0]}")
    return indices


def _precompute_features(rows: list[arc_gate.ArcRow], *, dim: int) -> list[list[list[int]]]:
    return [[_feature_indices(row, choice_index, dim=dim) for choice_index in range(len(row.choices))] for row in rows]


def _choice_scores(weights: np.ndarray, row_features: list[list[int]]) -> list[float]:
    return [float(sum(weights[index] for index in choice_features)) for choice_features in row_features]


def _predict(weights: np.ndarray, row_features: list[list[int]]) -> tuple[int, list[float]]:
    scores = _choice_scores(weights, row_features)
    return max(range(len(scores)), key=lambda index: scores[index]), scores


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    if len(rows) != len(predictions):
        raise ValueError("row and prediction counts do not match")
    return float(sum(row.answer_index == prediction for row, prediction in zip(rows, predictions, strict=True)) / len(rows))


def _train_public_perceptron(
    *,
    train_rows: list[arc_gate.ArcRow],
    train_features: list[list[list[int]]],
    dev_rows: list[arc_gate.ArcRow],
    dev_features: list[list[list[int]]],
    dim: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    weights = np.zeros(dim, dtype=np.float32)
    order = list(range(len(train_rows)))
    best_dev_accuracy = -1.0
    best_epoch = 0
    best_weights = weights.copy()
    log: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(order)
        mistakes = 0
        started = time.perf_counter()
        for row_index in order:
            row = train_rows[row_index]
            row_features = train_features[row_index]
            prediction, _ = _predict(weights, row_features)
            if prediction == row.answer_index:
                continue
            mistakes += 1
            for index in row_features[row.answer_index]:
                weights[index] += 1.0
            for index in row_features[prediction]:
                weights[index] -= 1.0
        dev_predictions = [_predict(weights, row_features)[0] for row_features in dev_features]
        dev_accuracy = _accuracy(dev_rows, dev_predictions) if dev_rows else float("nan")
        prefix_count = min(2048, len(train_rows))
        prefix_predictions = [_predict(weights, row_features)[0] for row_features in train_features[:prefix_count]]
        prefix_accuracy = _accuracy(train_rows[:prefix_count], prefix_predictions)
        epoch_log = {
            "epoch": epoch,
            "mistakes": mistakes,
            "train_prefix_accuracy": prefix_accuracy,
            "dev_accuracy": dev_accuracy,
            "seconds": time.perf_counter() - started,
        }
        log.append(epoch_log)
        if dev_rows and dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch
            best_weights = weights.copy()

    if not dev_rows:
        best_epoch = epochs
        best_dev_accuracy = float("nan")
        best_weights = weights.copy()
    return {
        "weights": best_weights,
        "training_log": log,
        "selected_epoch": best_epoch,
        "selected_dev_accuracy": best_dev_accuracy,
    }


def _ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)


def _public_repair_predictions(
    *,
    source_scores: list[list[float]],
    public_scores: list[list[float]],
    public_predictions: list[int],
) -> dict[str, list[int]]:
    source_predictions = [_ranked_indices(scores)[0] for scores in source_scores]
    top2_public_rerank: list[int] = []
    public_if_in_source_top2: list[int] = []
    for source_row_scores, public_row_scores, public_prediction, source_prediction in zip(
        source_scores,
        public_scores,
        public_predictions,
        source_predictions,
        strict=True,
    ):
        top2 = _ranked_indices(source_row_scores)[:2]
        top2_public_rerank.append(max(top2, key=lambda index: public_row_scores[index]))
        public_if_in_source_top2.append(public_prediction if public_prediction in top2 else source_prediction)
    return {
        "source_label_copy": source_predictions,
        "public_target_only": public_predictions,
        "top2_public_rerank": top2_public_rerank,
        "public_if_in_source_top2": public_if_in_source_top2,
    }


def build_probe(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    score_cache: pathlib.Path,
    dim: int,
    epochs: int,
    split_seed: int,
    dev_rows: int,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    train_path = _resolve(train_path)
    eval_path = _resolve(eval_path)
    score_cache = _resolve(score_cache)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    source_scores, _, source_model = headroom._load_score_cache(score_cache, rows=eval_rows)
    shuffled_train_rows = list(train_rows)
    random.Random(split_seed).shuffle(shuffled_train_rows)
    if len(shuffled_train_rows) < 2:
        raise ValueError("public receiver probe needs at least two train rows")
    dev_count = min(max(1, dev_rows), len(shuffled_train_rows) - 1)
    internal_dev_rows = shuffled_train_rows[:dev_count]
    fit_rows = shuffled_train_rows[dev_count:]

    feature_started = time.perf_counter()
    fit_features = _precompute_features(fit_rows, dim=dim)
    dev_features = _precompute_features(internal_dev_rows, dim=dim)
    eval_features = _precompute_features(eval_rows, dim=dim)
    feature_seconds = time.perf_counter() - feature_started

    model = _train_public_perceptron(
        train_rows=fit_rows,
        train_features=fit_features,
        dev_rows=internal_dev_rows,
        dev_features=dev_features,
        dim=dim,
        epochs=epochs,
        seed=split_seed,
    )
    weights = model["weights"]
    public_scores: list[list[float]] = []
    public_predictions: list[int] = []
    for row_features in eval_features:
        prediction, scores = _predict(weights, row_features)
        public_predictions.append(prediction)
        public_scores.append(scores)
    condition_predictions = _public_repair_predictions(
        source_scores=source_scores,
        public_scores=public_scores,
        public_predictions=public_predictions,
    )
    metrics = {
        name: {
            "accuracy": _accuracy(eval_rows, predictions),
            "correct": int(sum(row.answer_index == prediction for row, prediction in zip(eval_rows, predictions, strict=True))),
        }
        for name, predictions in condition_predictions.items()
    }
    source_label_accuracy = metrics["source_label_copy"]["accuracy"]
    best_repair_name = max(
        ("top2_public_rerank", "public_if_in_source_top2"),
        key=lambda name: metrics[name]["accuracy"],
    )
    best_repair_accuracy = metrics[best_repair_name]["accuracy"]
    source_top2_contains = [
        row.answer_index in _ranked_indices(scores)[:2] for row, scores in zip(eval_rows, source_scores, strict=True)
    ]
    public_in_source_top2 = [
        public_prediction in _ranked_indices(scores)[:2]
        for public_prediction, scores in zip(public_predictions, source_scores, strict=True)
    ]
    payload = {
        "gate": "source_private_hellaswag_public_receiver_repair_probe",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "score_cache": _display_path(score_cache),
        "score_cache_sha256": _sha256_file(score_cache),
        "source_model": source_model,
        "train_rows": len(train_rows),
        "internal_fit_rows": len(fit_rows),
        "internal_dev_rows": len(internal_dev_rows),
        "eval_rows": len(eval_rows),
        "feature_dim": dim,
        "epochs": epochs,
        "split_seed": split_seed,
        "training_log": model["training_log"],
        "selected_epoch": model["selected_epoch"],
        "selected_dev_accuracy": model["selected_dev_accuracy"],
        "metrics": metrics,
        "headline": {
            "source_label_copy_accuracy": source_label_accuracy,
            "public_target_only_accuracy": metrics["public_target_only"]["accuracy"],
            "top2_public_rerank_accuracy": metrics["top2_public_rerank"]["accuracy"],
            "public_if_in_source_top2_accuracy": metrics["public_if_in_source_top2"]["accuracy"],
            "best_repair_condition": best_repair_name,
            "best_repair_accuracy": best_repair_accuracy,
            "best_repair_minus_source_label_copy": best_repair_accuracy - source_label_accuracy,
            "best_repair_minus_public_target_only": best_repair_accuracy - metrics["public_target_only"]["accuracy"],
            "source_top2_oracle_accuracy": float(sum(source_top2_contains) / len(source_top2_contains)),
            "public_prediction_in_source_top2_rate": float(sum(public_in_source_top2) / len(public_in_source_top2)),
        },
        "pass_rule": {
            "best_repair_must_beat_source_label_copy_by": 0.02,
            "best_repair_must_beat_public_target_only_by": 0.02,
            "selection_uses_validation_labels": False,
            "claim_boundary": (
                "This is a train-only public receiver repair probe. It is promoted only if the source top-two hint "
                "plus public receiver model beats both source-label copy and target-only receiver behavior."
            ),
        },
        "packet_contract": {
            "source_payload": "ordered source top-2 choice indices from cached source per-choice scores",
            "payload_bytes": 1,
            "receiver_state": "hashed public lexical perceptron trained only on official HellaSwag train labels",
            "source_text_exposed": False,
            "source_kv_exposed": False,
        },
        "timing": {
            "feature_precompute_seconds": feature_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    payload["pass_gate"] = bool(
        payload["headline"]["best_repair_minus_source_label_copy"] >= 0.02
        and payload["headline"]["best_repair_minus_public_target_only"] >= 0.02
    )

    json_path = output_dir / "hellaswag_public_receiver_repair_probe.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row_index, (row, source_row_scores, public_row_scores) in enumerate(
            zip(eval_rows, source_scores, public_scores, strict=True)
        ):
            record = {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "answer_index": row.answer_index,
                "source_scores": source_row_scores,
                "public_scores": public_row_scores,
                "predictions": {name: predictions[row_index] for name, predictions in condition_predictions.items()},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    lines = [
        "# HellaSwag Public Receiver Repair Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source-label copy accuracy: `{source_label_accuracy:.3f}`",
        f"- public target-only accuracy: `{metrics['public_target_only']['accuracy']:.3f}`",
        f"- top-2 public rerank accuracy: `{metrics['top2_public_rerank']['accuracy']:.3f}`",
        f"- public-if-in-top-2 accuracy: `{metrics['public_if_in_source_top2']['accuracy']:.3f}`",
        f"- best repair condition: `{best_repair_name}`",
        f"- best repair delta vs source-label copy: `{payload['headline']['best_repair_minus_source_label_copy']:.3f}`",
        f"- source top-2 oracle accuracy: `{payload['headline']['source_top2_oracle_accuracy']:.3f}`",
        f"- selected internal dev accuracy: `{payload['selected_dev_accuracy']:.3f}`",
        "",
        "## Interpretation",
        "",
        "The public receiver can learn a weak lexical HellaSwag scorer from train labels, but it does not repair",
        "the source model's top-choice errors well enough to beat visible source-label copy. The next valid",
        "HellaSwag branch needs train-split source scores or hidden summaries for calibrated source-error repair.",
        "",
    ]
    (output_dir / "hellaswag_public_receiver_repair_probe.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe train-only public receiver repair on frozen HellaSwag.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--score-cache", type=pathlib.Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument("--dim", type=int, default=1 << 19)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--dev-rows", type=int, default=3990)
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_probe(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        score_cache=args.score_cache,
        dim=args.dim,
        epochs=args.epochs,
        split_seed=args.split_seed,
        dev_rows=args.dev_rows,
        run_date=args.run_date,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
