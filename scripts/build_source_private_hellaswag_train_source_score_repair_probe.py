from __future__ import annotations

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
from collections import Counter, defaultdict
from typing import Any, Callable


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
DEFAULT_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024"
)


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


def _ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)


def _correct_rank(row: arc_gate.ArcRow, scores: list[float]) -> int:
    return _ranked_indices(scores).index(row.answer_index)


def _safe_softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _entropy(scores: list[float]) -> float:
    probs = _safe_softmax(scores)
    return float(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs))


def _margin(scores: list[float], left_rank: int, right_rank: int) -> float:
    ranked = _ranked_indices(scores)
    if right_rank >= len(ranked):
        return 0.0
    return float(scores[ranked[left_rank]] - scores[ranked[right_rank]])


def _quantile_edges(values: list[float], *, bins: int) -> list[float]:
    if bins <= 1 or not values:
        return []
    values = sorted(float(value) for value in values)
    edges: list[float] = []
    for bin_index in range(1, bins):
        offset = int(round(bin_index * (len(values) - 1) / bins))
        edge = values[offset]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def _digitize(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    if len(rows) != len(predictions):
        raise ValueError("row and prediction counts do not match")
    return float(sum(row.answer_index == prediction for row, prediction in zip(rows, predictions, strict=True)) / len(rows))


def _fit_choice_bias_offsets(rows: list[arc_gate.ArcRow], scores: list[list[float]], *, alpha: float = 1.0) -> list[float]:
    choice_count = max(len(row.choices) for row in rows)
    label_counts = [alpha for _ in range(choice_count)]
    prediction_counts = [alpha for _ in range(choice_count)]
    for row, row_scores in zip(rows, scores, strict=True):
        label_counts[row.answer_index] += 1.0
        prediction_counts[_ranked_indices(row_scores)[0]] += 1.0
    label_total = sum(label_counts)
    prediction_total = sum(prediction_counts)
    return [
        math.log(label_counts[index] / label_total) - math.log(prediction_counts[index] / prediction_total)
        for index in range(choice_count)
    ]


def _apply_offsets(scores: list[float], offsets: list[float]) -> list[float]:
    return [score + offsets[index] for index, score in enumerate(scores)]


def _predict_calibrated_label(scores: list[float], offsets: list[float]) -> int:
    calibrated = _apply_offsets(scores, offsets)
    return _ranked_indices(calibrated)[0]


def _global_rank_fallback(rows: list[arc_gate.ArcRow], scores: list[list[float]], *, max_rank: int) -> int:
    counts: Counter[int] = Counter(min(_correct_rank(row, row_scores), max_rank - 1) for row, row_scores in zip(rows, scores, strict=True))
    return counts.most_common(1)[0][0]


FeatureFn = Callable[[list[float]], tuple[Any, ...]]


def _fit_rank_decoder(
    *,
    rows: list[arc_gate.ArcRow],
    scores: list[list[float]],
    feature_fn: FeatureFn,
    max_rank: int,
) -> dict[str, Any]:
    counts_by_code: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    global_rank = _global_rank_fallback(rows, scores, max_rank=max_rank)
    for row, row_scores in zip(rows, scores, strict=True):
        code = feature_fn(row_scores)
        rank = min(_correct_rank(row, row_scores), max_rank - 1)
        counts_by_code[code][rank] += 1
    selected_rank_by_code = {
        json.dumps(code, sort_keys=True): counter.most_common(1)[0][0] for code, counter in counts_by_code.items()
    }
    return {
        "max_rank": max_rank,
        "global_fallback_rank": global_rank,
        "selected_rank_by_code": selected_rank_by_code,
        "seen_codes": len(selected_rank_by_code),
    }


def _predict_rank_decoder(scores: list[list[float]], decoder: dict[str, Any], feature_fn: FeatureFn) -> list[int]:
    predictions: list[int] = []
    by_code = decoder["selected_rank_by_code"]
    fallback_rank = int(decoder["global_fallback_rank"])
    for row_scores in scores:
        ranked = _ranked_indices(row_scores)
        rank = by_code.get(json.dumps(feature_fn(row_scores), sort_keys=True), fallback_rank)
        predictions.append(ranked[min(int(rank), len(ranked) - 1)])
    return predictions


def _make_feature_fns(fit_scores: list[list[float]]) -> dict[str, tuple[FeatureFn, dict[str, Any]]]:
    margin12_edges_2 = _quantile_edges([_margin(scores, 0, 1) for scores in fit_scores], bins=2)
    margin12_edges_4 = _quantile_edges([_margin(scores, 0, 1) for scores in fit_scores], bins=4)
    margin12_edges_8 = _quantile_edges([_margin(scores, 0, 1) for scores in fit_scores], bins=8)
    margin23_edges_3 = _quantile_edges([_margin(scores, 1, 2) for scores in fit_scores], bins=3)
    entropy_edges_3 = _quantile_edges([_entropy(scores) for scores in fit_scores], bins=3)

    def top2_margin_2bin(scores: list[float]) -> tuple[Any, ...]:
        return (_digitize(_margin(scores, 0, 1), margin12_edges_2),)

    def top2_margin_4bin(scores: list[float]) -> tuple[Any, ...]:
        return (_digitize(_margin(scores, 0, 1), margin12_edges_4),)

    def top2_margin_8bin(scores: list[float]) -> tuple[Any, ...]:
        return (_digitize(_margin(scores, 0, 1), margin12_edges_8),)

    def score_shape_3bin(scores: list[float]) -> tuple[Any, ...]:
        return (
            _digitize(_margin(scores, 0, 1), margin12_edges_4),
            _digitize(_margin(scores, 1, 2), margin23_edges_3),
            _digitize(_entropy(scores), entropy_edges_3),
        )

    def score_shape_3bin_with_top1(scores: list[float]) -> tuple[Any, ...]:
        ranked = _ranked_indices(scores)
        return (
            ranked[0],
            _digitize(_margin(scores, 0, 1), margin12_edges_4),
            _digitize(_margin(scores, 1, 2), margin23_edges_3),
            _digitize(_entropy(scores), entropy_edges_3),
        )

    def full_rank_shape_3bin(scores: list[float]) -> tuple[Any, ...]:
        ranked = _ranked_indices(scores)
        return (
            tuple(ranked),
            _digitize(_margin(scores, 0, 1), margin12_edges_4),
            _digitize(_margin(scores, 1, 2), margin23_edges_3),
            _digitize(_entropy(scores), entropy_edges_3),
        )

    metadata = {
        "margin12_edges_2": margin12_edges_2,
        "margin12_edges_4": margin12_edges_4,
        "margin12_edges_8": margin12_edges_8,
        "margin23_edges_3": margin23_edges_3,
        "entropy_edges_3": entropy_edges_3,
    }
    return {
        "top2_margin_2bin": (top2_margin_2bin, metadata),
        "top2_margin_4bin": (top2_margin_4bin, metadata),
        "top2_margin_8bin": (top2_margin_8bin, metadata),
        "score_shape_3bin": (score_shape_3bin, metadata),
        "score_shape_3bin_with_top1": (score_shape_3bin_with_top1, metadata),
        "full_rank_shape_3bin": (full_rank_shape_3bin, metadata),
    }


def _select_train_rows(rows: list[arc_gate.ArcRow], *, count: int, seed: int) -> list[arc_gate.ArcRow]:
    if count >= len(rows):
        return list(rows)
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[: min(count, len(selected))]


def _split_train_rows(
    rows: list[arc_gate.ArcRow],
    scores: list[list[float]],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[list[arc_gate.ArcRow], list[list[float]], list[arc_gate.ArcRow], list[list[float]]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    dev_count = max(1, min(len(indices) - 1, int(round(len(indices) * dev_fraction))))
    dev_indices = set(indices[:dev_count])
    fit_rows: list[arc_gate.ArcRow] = []
    fit_scores: list[list[float]] = []
    dev_rows: list[arc_gate.ArcRow] = []
    dev_scores: list[list[float]] = []
    for index, (row, row_scores) in enumerate(zip(rows, scores, strict=True)):
        if index in dev_indices:
            dev_rows.append(row)
            dev_scores.append(row_scores)
        else:
            fit_rows.append(row)
            fit_scores.append(row_scores)
    return fit_rows, fit_scores, dev_rows, dev_scores


def _evaluate_policy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> dict[str, Any]:
    return {
        "accuracy": _accuracy(rows, predictions),
        "correct": int(sum(row.answer_index == prediction for row, prediction in zip(rows, predictions, strict=True))),
        "rows": len(rows),
    }


def build_probe(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    eval_score_cache: pathlib.Path,
    train_score_cache: pathlib.Path | None,
    train_score_rows: int,
    selection_seed: int,
    dev_fraction: float,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    train_path = _resolve(train_path)
    eval_path = _resolve(eval_path)
    eval_score_cache = _resolve(eval_score_cache)
    train_score_cache = _resolve(train_score_cache) if train_score_cache is not None else output_dir / "source_train_score_cache.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    all_train_rows = arc_gate._load_rows(train_path)
    selected_train_rows = _select_train_rows(all_train_rows, count=train_score_rows, seed=selection_seed)
    eval_rows = arc_gate._load_rows(eval_path)

    train_scores, train_source_predictions, train_source_model, train_score_cache_sha256 = headroom._source_scores(
        rows=selected_train_rows,
        score_cache=train_score_cache,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    eval_scores, eval_source_predictions, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)

    fit_rows, fit_scores, dev_rows, dev_scores = _split_train_rows(
        selected_train_rows,
        train_scores,
        dev_fraction=dev_fraction,
        seed=selection_seed + 17,
    )
    feature_fns = _make_feature_fns(fit_scores)
    policy_payloads: dict[str, dict[str, Any]] = {}

    source_label_train = [_ranked_indices(scores)[0] for scores in train_scores]
    source_label_eval = [_ranked_indices(scores)[0] for scores in eval_scores]
    policy_payloads["source_label_copy"] = {
        "kind": "baseline",
        "payload_bytes": 1,
        "train": _evaluate_policy(selected_train_rows, source_label_train),
        "internal_dev": _evaluate_policy(dev_rows, [_ranked_indices(scores)[0] for scores in dev_scores]),
        "eval": _evaluate_policy(eval_rows, source_label_eval),
    }

    offsets = _fit_choice_bias_offsets(fit_rows, fit_scores)
    calibrated_dev = [_predict_calibrated_label(scores, offsets) for scores in dev_scores]
    calibrated_eval = [_predict_calibrated_label(scores, offsets) for scores in eval_scores]
    calibrated_train = [_predict_calibrated_label(scores, offsets) for scores in train_scores]
    policy_payloads["trained_choice_bias_label_copy"] = {
        "kind": "trained_label_copy_control",
        "payload_bytes": 1,
        "offsets": offsets,
        "train": _evaluate_policy(selected_train_rows, calibrated_train),
        "internal_dev": _evaluate_policy(dev_rows, calibrated_dev),
        "eval": _evaluate_policy(eval_rows, calibrated_eval),
    }

    for name, (feature_fn, metadata) in feature_fns.items():
        max_rank = 2 if name.startswith("top2") or name.startswith("score_shape") else 4
        decoder = _fit_rank_decoder(rows=fit_rows, scores=fit_scores, feature_fn=feature_fn, max_rank=max_rank)
        dev_predictions = _predict_rank_decoder(dev_scores, decoder, feature_fn)
        eval_predictions = _predict_rank_decoder(eval_scores, decoder, feature_fn)
        train_predictions = _predict_rank_decoder(train_scores, decoder, feature_fn)
        payload_bytes = 2 if max_rank == 2 else 3
        policy_payloads[name] = {
            "kind": "train_source_score_rank_decoder",
            "payload_bytes": payload_bytes,
            "max_rank": max_rank,
            "decoder": decoder,
            "feature_metadata": metadata,
            "train": _evaluate_policy(selected_train_rows, train_predictions),
            "internal_dev": _evaluate_policy(dev_rows, dev_predictions),
            "eval": _evaluate_policy(eval_rows, eval_predictions),
        }

    candidate_policy_names = [
        name
        for name, policy in policy_payloads.items()
        if policy["kind"] == "train_source_score_rank_decoder"
    ]
    selected_policy_name = max(
        candidate_policy_names,
        key=lambda name: (
            policy_payloads[name]["internal_dev"]["accuracy"],
            policy_payloads[name]["train"]["accuracy"],
            -policy_payloads[name]["payload_bytes"],
            name,
        ),
    )
    selected_policy = policy_payloads[selected_policy_name]
    source_label_eval_accuracy = policy_payloads["source_label_copy"]["eval"]["accuracy"]
    trained_label_eval_accuracy = policy_payloads["trained_choice_bias_label_copy"]["eval"]["accuracy"]
    best_label_copy_accuracy = max(source_label_eval_accuracy, trained_label_eval_accuracy)
    source_top2_contains = [
        row.answer_index in _ranked_indices(scores)[:2] for row, scores in zip(eval_rows, eval_scores, strict=True)
    ]
    source_top4_contains = [
        row.answer_index in _ranked_indices(scores)[:4] for row, scores in zip(eval_rows, eval_scores, strict=True)
    ]

    payload = {
        "gate": "source_private_hellaswag_train_source_score_repair_probe",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "eval_score_cache": _display_path(eval_score_cache),
        "eval_score_cache_sha256": _sha256_file(eval_score_cache),
        "train_score_cache": _display_path(train_score_cache),
        "train_score_cache_sha256": train_score_cache_sha256,
        "all_train_rows": len(all_train_rows),
        "scored_train_rows": len(selected_train_rows),
        "internal_fit_rows": len(fit_rows),
        "internal_dev_rows": len(dev_rows),
        "eval_rows": len(eval_rows),
        "selection_seed": selection_seed,
        "dev_fraction": dev_fraction,
        "source_model": {
            "train": train_source_model,
            "eval": eval_source_model,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS)
            + ["label", "activity_label", "source_id", "split", "split_type", "ind"],
        },
        "policy_readouts": policy_payloads,
        "selected_policy": selected_policy_name,
        "headline": {
            "selected_policy": selected_policy_name,
            "selected_policy_payload_bytes": selected_policy["payload_bytes"],
            "selected_internal_dev_accuracy": selected_policy["internal_dev"]["accuracy"],
            "selected_eval_accuracy": selected_policy["eval"]["accuracy"],
            "source_label_copy_eval_accuracy": source_label_eval_accuracy,
            "trained_choice_bias_label_copy_eval_accuracy": trained_label_eval_accuracy,
            "best_label_copy_eval_accuracy": best_label_copy_accuracy,
            "selected_minus_source_label_copy": selected_policy["eval"]["accuracy"] - source_label_eval_accuracy,
            "selected_minus_trained_choice_bias_label_copy": selected_policy["eval"]["accuracy"] - trained_label_eval_accuracy,
            "selected_minus_best_label_copy": selected_policy["eval"]["accuracy"] - best_label_copy_accuracy,
            "source_top2_oracle_accuracy": float(sum(source_top2_contains) / len(source_top2_contains)),
            "source_top4_oracle_accuracy": float(sum(source_top4_contains) / len(source_top4_contains)),
        },
        "pass_rule": {
            "selected_policy_selected_only_on_internal_train_dev": True,
            "selected_policy_must_beat_best_label_copy_by": 0.02,
            "minimum_selected_eval_accuracy_for_1024_rows": 494 / 1024,
            "claim_boundary": (
                "This gate is promoted only if a train-source-score repair decoder beats both raw source-label "
                "copy and a trained choice-bias label-copy control by at least 0.02 on frozen validation."
            ),
        },
        "packet_contract": {
            "selected_source_payload": "source rank order plus quantized score-shape code",
            "raw_payload_bytes": selected_policy["payload_bytes"],
            "framed_record_bytes": selected_policy["payload_bytes"] + 3,
            "source_text_exposed": False,
            "source_kv_exposed": False,
        },
        "timing": {
            "total_seconds": time.perf_counter() - started,
        },
    }
    payload["pass_gate"] = bool(payload["headline"]["selected_minus_best_label_copy"] >= 0.02)

    (output_dir / "hellaswag_train_source_score_repair_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "policy_readouts.jsonl").open("w", encoding="utf-8") as handle:
        for name, policy in sorted(policy_payloads.items()):
            handle.write(json.dumps({"policy": name, **policy}, sort_keys=True) + "\n")

    lines = [
        "# HellaSwag Train Source-Score Repair Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- scored train rows: `{len(selected_train_rows)}`",
        f"- selected policy: `{selected_policy_name}`",
        f"- selected internal dev accuracy: `{selected_policy['internal_dev']['accuracy']:.3f}`",
        f"- selected eval accuracy: `{selected_policy['eval']['accuracy']:.3f}`",
        f"- source-label copy eval accuracy: `{source_label_eval_accuracy:.3f}`",
        f"- trained choice-bias label-copy eval accuracy: `{trained_label_eval_accuracy:.3f}`",
        f"- selected minus best label-copy: `{payload['headline']['selected_minus_best_label_copy']:.3f}`",
        f"- source top-2 oracle accuracy: `{payload['headline']['source_top2_oracle_accuracy']:.3f}`",
        "",
        "## Interpretation",
        "",
        "This gate tests whether source-score calibration learned only from train rows can identify when the",
        "source model's top answer should be replaced by a lower-ranked source candidate. Promotion requires",
        "beating both raw source-label copy and trained label-copy controls on frozen validation.",
        "",
    ]
    (output_dir / "hellaswag_train_source_score_repair_probe.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe train-source-score repair for HellaSwag.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--train-score-cache", type=pathlib.Path)
    parser.add_argument("--train-score-rows", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--source-lm-model", required=True)
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_probe(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        train_score_cache=args.train_score_cache,
        train_score_rows=args.train_score_rows,
        selection_seed=args.selection_seed,
        dev_fraction=args.dev_fraction,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=args.local_files_only,
        run_date=args.run_date,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
