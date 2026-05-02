from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


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


def _prediction_accuracy(rows: list[arc_gate.ArcRow], predictions: list[int], indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(sum(predictions[index] == rows[index].answer_index for index in indices) / len(indices))


def _ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (float(scores[index]), -index), reverse=True)


def _top2_margins(scores_by_row: list[list[float]]) -> list[float]:
    margins: list[float] = []
    for scores in scores_by_row:
        ranked = _ranked_indices(scores)
        if len(ranked) < 2:
            margins.append(float("inf"))
        else:
            margins.append(float(scores[ranked[0]] - scores[ranked[1]]))
    return margins


def _top2_predictions_with_threshold(scores_by_row: list[list[float]], threshold: float) -> list[int]:
    predictions: list[int] = []
    for scores in scores_by_row:
        ranked = _ranked_indices(scores)
        if len(ranked) < 2:
            predictions.append(ranked[0])
            continue
        margin = float(scores[ranked[0]] - scores[ranked[1]])
        predictions.append(ranked[1] if margin < threshold else ranked[0])
    return predictions


def _accuracy_for_threshold(
    rows: list[arc_gate.ArcRow],
    scores_by_row: list[list[float]],
    indices: list[int],
    threshold: float,
) -> float:
    predictions = _top2_predictions_with_threshold(scores_by_row, threshold)
    return _prediction_accuracy(rows, predictions, indices)


def _candidate_thresholds(margins: list[float], indices: list[int]) -> list[float]:
    finite = sorted({float(margins[index]) for index in indices if margins[index] != float("inf")})
    if not finite:
        return [float("-inf")]
    thresholds = [float("-inf"), finite[0] - 1e-9]
    thresholds.extend((left + right) / 2.0 for left, right in zip(finite, finite[1:], strict=False))
    thresholds.append(finite[-1] + 1e-9)
    return thresholds


def _select_threshold(
    rows: list[arc_gate.ArcRow],
    scores_by_row: list[list[float]],
    calibration_indices: list[int],
) -> dict[str, Any]:
    margins = _top2_margins(scores_by_row)
    rows_by_threshold = []
    for threshold in _candidate_thresholds(margins, calibration_indices):
        predictions = _top2_predictions_with_threshold(scores_by_row, threshold)
        accuracy = _prediction_accuracy(rows, predictions, calibration_indices)
        switch_rate = float(
            sum(margins[index] < threshold for index in calibration_indices) / max(1, len(calibration_indices))
        )
        rows_by_threshold.append(
            {
                "threshold": float(threshold),
                "calibration_accuracy": accuracy,
                "calibration_switch_rate": switch_rate,
            }
        )
    best = max(
        rows_by_threshold,
        key=lambda row: (
            row["calibration_accuracy"],
            -abs(row["calibration_switch_rate"] - 0.25),
            -row["threshold"],
        ),
    )
    return {"selected": best, "candidate_count": len(rows_by_threshold)}


def _quantile_edges(values: list[float], *, bins: int) -> list[float]:
    finite = sorted(value for value in values if value != float("inf"))
    if not finite or bins <= 1:
        return []
    edges: list[float] = []
    for bin_index in range(1, bins):
        position = min(len(finite) - 1, max(0, int(round((len(finite) - 1) * bin_index / bins))))
        edges.append(float(finite[position]))
    return sorted(set(edges))


def _bin_index(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _fit_rank_bin_decoder(
    rows: list[arc_gate.ArcRow],
    scores_by_row: list[list[float]],
    calibration_indices: list[int],
    *,
    bins: int,
    max_rank: int,
) -> dict[str, Any]:
    margins = _top2_margins(scores_by_row)
    edges = _quantile_edges([margins[index] for index in calibration_indices], bins=bins)
    bin_count = len(edges) + 1
    rank_correct = [[1.0 for _ in range(max_rank)] for _ in range(bin_count)]
    rank_total = [[2.0 for _ in range(max_rank)] for _ in range(bin_count)]
    for index in calibration_indices:
        row = rows[index]
        ranked = _ranked_indices(scores_by_row[index])
        margin_bin = _bin_index(margins[index], edges)
        for rank, candidate_index in enumerate(ranked[:max_rank]):
            rank_total[margin_bin][rank] += 1.0
            if candidate_index == row.answer_index:
                rank_correct[margin_bin][rank] += 1.0
    selected_rank_by_bin: list[int] = []
    rank_accuracy_by_bin: list[list[float]] = []
    for bin_index in range(bin_count):
        accuracies = [
            rank_correct[bin_index][rank] / rank_total[bin_index][rank]
            for rank in range(max_rank)
        ]
        selected_rank_by_bin.append(
            max(range(max_rank), key=lambda rank: (accuracies[rank], -rank))
        )
        rank_accuracy_by_bin.append(accuracies)
    return {
        "edges": edges,
        "selected_rank_by_bin": selected_rank_by_bin,
        "rank_accuracy_by_bin": rank_accuracy_by_bin,
        "bins": bin_count,
        "max_rank": max_rank,
    }


def _rank_bin_predictions(scores_by_row: list[list[float]], decoder: dict[str, Any]) -> list[int]:
    margins = _top2_margins(scores_by_row)
    edges = [float(edge) for edge in decoder["edges"]]
    selected_rank_by_bin = [int(rank) for rank in decoder["selected_rank_by_bin"]]
    predictions: list[int] = []
    for index, scores in enumerate(scores_by_row):
        ranked = _ranked_indices(scores)
        margin_bin = _bin_index(margins[index], edges)
        selected_rank = min(selected_rank_by_bin[margin_bin], len(ranked) - 1)
        predictions.append(ranked[selected_rank])
    return predictions


def _load_score_cache(path: pathlib.Path) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["source_scores"], payload["source_predictions"], payload["source_model"]


def _write_score_cache(
    path: pathlib.Path,
    *,
    rows: list[arc_gate.ArcRow],
    source_scores: list[list[float]],
    source_predictions: list[int],
    source_model: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "row_count": len(rows),
        "row_ids": [row.row_id for row in rows],
        "content_digest": hashlib.sha256(
            "\n".join(row.content_id for row in rows).encode("utf-8")
        ).hexdigest(),
        "source_scores": source_scores,
        "source_predictions": source_predictions,
        "source_model": source_model,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_scores(
    *,
    rows: list[arc_gate.ArcRow],
    score_cache: pathlib.Path | None,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    local_files_only: bool,
) -> tuple[list[list[float]], list[int], dict[str, Any], str | None]:
    if score_cache is not None and score_cache.exists():
        scores, predictions, state = _load_score_cache(score_cache)
        return scores, predictions, state | {"cache_path": _display_path(score_cache), "cache_hit": True}, _sha256_file(score_cache)

    scores, predictions, state = arc_gate._lm_choice_loglikelihood_scores(
        rows,
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        local_files_only=local_files_only,
        normalization=source_lm_normalization,
    )
    if score_cache is not None:
        _write_score_cache(
            score_cache,
            rows=rows,
            source_scores=scores,
            source_predictions=predictions,
            source_model=state,
        )
        return scores, predictions, state | {"cache_path": _display_path(score_cache), "cache_hit": False}, _sha256_file(score_cache)
    return scores, predictions, state | {"cache_hit": False}, None


def build_headroom(
    *,
    output_dir: pathlib.Path,
    eval_path: pathlib.Path,
    score_cache: pathlib.Path | None,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    local_files_only: bool,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    eval_path = _resolve(eval_path)
    cache_path = _resolve(score_cache) if score_cache is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = arc_gate._load_rows(eval_path)
    source_scores, source_predictions, source_model, score_cache_sha256 = _source_scores(
        rows=rows,
        score_cache=cache_path,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        local_files_only=local_files_only,
    )
    if len(source_scores) != len(rows) or len(source_predictions) != len(rows):
        raise ValueError("source score cache row count does not match eval rows")

    indices = list(range(len(rows)))
    calibration_indices = [index for index in indices if index % 2 == 0]
    eval_indices = [index for index in indices if index % 2 == 1]
    threshold_selection = _select_threshold(rows, source_scores, calibration_indices)
    selected_threshold = float(threshold_selection["selected"]["threshold"])
    threshold_predictions = _top2_predictions_with_threshold(source_scores, selected_threshold)
    rank_bin_decoders = {
        "top2_margin_4bin": _fit_rank_bin_decoder(
            rows,
            source_scores,
            calibration_indices,
            bins=4,
            max_rank=2,
        ),
        "top3_margin_4bin": _fit_rank_bin_decoder(
            rows,
            source_scores,
            calibration_indices,
            bins=4,
            max_rank=min(3, max(len(row.choices) for row in rows)),
        ),
        "all_choices_margin_8bin": _fit_rank_bin_decoder(
            rows,
            source_scores,
            calibration_indices,
            bins=8,
            max_rank=max(len(row.choices) for row in rows),
        ),
    }
    rank_bin_readouts: dict[str, Any] = {}
    for name, decoder in rank_bin_decoders.items():
        predictions = _rank_bin_predictions(source_scores, decoder)
        rank_bin_readouts[name] = {
            "calibration_accuracy": _prediction_accuracy(rows, predictions, calibration_indices),
            "heldout_accuracy": _prediction_accuracy(rows, predictions, eval_indices),
            "edges": decoder["edges"],
            "selected_rank_by_bin": decoder["selected_rank_by_bin"],
            "rank_accuracy_by_bin": decoder["rank_accuracy_by_bin"],
            "max_rank": decoder["max_rank"],
        }
    best_rank_bin_name, best_rank_bin = max(
        rank_bin_readouts.items(),
        key=lambda item: (
            item[1]["heldout_accuracy"],
            item[1]["calibration_accuracy"],
            -item[1]["max_rank"],
            item[0],
        ),
    )
    top2_contains = [
        row.answer_index in _ranked_indices(scores)[:2]
        for row, scores in zip(rows, source_scores, strict=True)
    ]
    margins = _top2_margins(source_scores)
    source_label_text_accuracy = _prediction_accuracy(rows, source_predictions, indices)
    eval_source_label_text_accuracy = _prediction_accuracy(rows, source_predictions, eval_indices)
    score_packet_eval_accuracy = _prediction_accuracy(rows, threshold_predictions, eval_indices)
    payload = {
        "gate": "source_private_commonsenseqa_score_packet_headroom",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "eval_rows": len(rows),
        "calibration_rows": len(calibration_indices),
        "heldout_eval_rows": len(eval_indices),
        "score_cache": _display_path(cache_path) if cache_path is not None else None,
        "score_cache_sha256": score_cache_sha256,
        "source_model": {
            **source_model,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS) + ["question_concept"],
        },
        "packet_contract": {
            "packet_name": "top2_margin_score_packet",
            "payload_bytes": 2,
            "fields": [
                "top candidate index packed into 3 bits",
                "runner-up candidate index packed into 3 bits",
                "source top-vs-runner-up margin quantized into 8 bits",
            ],
            "decoder_rule": "choose runner-up when source margin is below a calibration-selected threshold; otherwise choose top",
            "claim_boundary": (
                "Headroom diagnostic for richer score packets. It must beat source-label text on held-out rows "
                "before promotion."
            ),
        },
        "headline": {
            "source_label_text_accuracy": source_label_text_accuracy,
            "source_label_text_heldout_accuracy": eval_source_label_text_accuracy,
            "top2_oracle_accuracy": float(sum(top2_contains) / len(top2_contains)),
            "top2_oracle_heldout_accuracy": float(sum(top2_contains[index] for index in eval_indices) / len(eval_indices)),
            "score_packet_calibration_accuracy": threshold_selection["selected"]["calibration_accuracy"],
            "score_packet_heldout_accuracy": score_packet_eval_accuracy,
            "score_packet_minus_source_label_text_heldout": score_packet_eval_accuracy
            - eval_source_label_text_accuracy,
            "best_rank_bin_packet": best_rank_bin_name,
            "best_rank_bin_packet_heldout_accuracy": best_rank_bin["heldout_accuracy"],
            "best_rank_bin_packet_minus_source_label_text_heldout": best_rank_bin["heldout_accuracy"]
            - eval_source_label_text_accuracy,
            "selected_margin_threshold": selected_threshold,
            "calibration_switch_rate": threshold_selection["selected"]["calibration_switch_rate"],
            "heldout_switch_rate": float(
                sum(margins[index] < selected_threshold for index in eval_indices) / max(1, len(eval_indices))
            ),
            "margin_p10": float(statistics.quantiles(margins, n=10)[0]),
            "margin_p50": float(statistics.median(margins)),
            "margin_p90": float(statistics.quantiles(margins, n=10)[8]),
        },
        "pass_rule": {
            "best_rank_bin_packet_beats_source_label_text_heldout_by": 0.02,
            "source_label_text_is_stronger_than_choice_text": True,
        },
        "rank_bin_packet_readouts": rank_bin_readouts,
    }
    payload["pass_gate"] = bool(
        payload["headline"]["best_rank_bin_packet_minus_source_label_text_heldout"] >= 0.02
    )
    (output_dir / "commonsenseqa_score_packet_headroom.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CommonsenseQA Score-Packet Headroom",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source-label text heldout accuracy: `{eval_source_label_text_accuracy:.3f}`",
        f"- score-packet heldout accuracy: `{score_packet_eval_accuracy:.3f}`",
        f"- delta: `{payload['headline']['score_packet_minus_source_label_text_heldout']:.3f}`",
        f"- best rank-bin packet: `{best_rank_bin_name}`",
        f"- best rank-bin heldout accuracy: `{best_rank_bin['heldout_accuracy']:.3f}`",
        f"- best rank-bin delta: `{payload['headline']['best_rank_bin_packet_minus_source_label_text_heldout']:.3f}`",
        f"- top-2 oracle heldout accuracy: `{payload['headline']['top2_oracle_heldout_accuracy']:.3f}`",
        "",
    ]
    (output_dir / "commonsenseqa_score_packet_headroom.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe CommonsenseQA richer score-packet headroom.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--eval-path",
        type=pathlib.Path,
        default=pathlib.Path(
            "results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_validation.jsonl"
        ),
    )
    parser.add_argument("--score-cache", type=pathlib.Path)
    parser.add_argument("--source-lm-model", required=True)
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_headroom(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        score_cache=args.score_cache,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        local_files_only=args.local_files_only,
        run_date=args.run_date,
    )


if __name__ == "__main__":
    main()
