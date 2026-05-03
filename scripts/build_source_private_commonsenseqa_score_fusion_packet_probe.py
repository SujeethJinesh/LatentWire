from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_commonsenseqa_score_packet_headroom as headroom
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in slug.split("_") if part)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (float(scores[index]), -index), reverse=True)


def _top_predictions(scores_by_row: list[list[float]]) -> list[int]:
    return [_ranked_indices(scores)[0] for scores in scores_by_row]


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int], indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(sum(predictions[index] == rows[index].answer_index for index in indices) / len(indices))


def _row_zscores(scores: list[float]) -> list[float]:
    finite = [float(score) for score in scores if math.isfinite(float(score))]
    if not finite:
        return [0.0 for _ in scores]
    mean = statistics.fmean(finite)
    variance = statistics.fmean((score - mean) ** 2 for score in finite)
    std = math.sqrt(max(variance, 1e-12))
    return [0.0 if not math.isfinite(float(score)) else (float(score) - mean) / std for score in scores]


def _zscore_rows(scores_by_row: list[list[float]]) -> list[list[float]]:
    return [_row_zscores(scores) for scores in scores_by_row]


def _quantize_4bit_zscore(scores: list[float], *, clip: float = 2.0) -> list[float]:
    if clip <= 0:
        raise ValueError("clip must be positive")
    quantized: list[float] = []
    for score in scores:
        clipped = min(clip, max(-clip, float(score)))
        level = round((clipped + clip) * 15.0 / (2.0 * clip))
        level = max(0, min(15, int(level)))
        quantized.append((level * (2.0 * clip) / 15.0) - clip)
    return quantized


def _quantized_source_rows(scores_by_row: list[list[float]]) -> list[list[float]]:
    return [_quantize_4bit_zscore(row) for row in _zscore_rows(scores_by_row)]


def _fusion_predictions(
    source_scores: list[list[float]],
    receiver_scores: list[list[float]],
    *,
    source_weight: float,
) -> list[int]:
    predictions: list[int] = []
    for source_row, receiver_row in zip(source_scores, receiver_scores, strict=True):
        if len(source_row) != len(receiver_row):
            raise ValueError("source and receiver score rows must have matching candidate counts")
        combined = [
            source_weight * float(source_score) + (1.0 - source_weight) * float(receiver_score)
            for source_score, receiver_score in zip(source_row, receiver_row, strict=True)
        ]
        predictions.append(_ranked_indices(combined)[0])
    return predictions


def _select_fusion_weight(
    rows: list[arc_gate.ArcRow],
    source_scores: list[list[float]],
    receiver_scores: list[list[float]],
    calibration_indices: list[int],
) -> dict[str, Any]:
    candidates = [step / 40.0 for step in range(41)]
    readouts = []
    for weight in candidates:
        predictions = _fusion_predictions(source_scores, receiver_scores, source_weight=weight)
        readouts.append(
            {
                "source_weight": float(weight),
                "calibration_accuracy": _accuracy(rows, predictions, calibration_indices),
            }
        )
    selected = max(readouts, key=lambda row: (row["calibration_accuracy"], -abs(row["source_weight"] - 0.5)))
    return {"selected": selected, "grid": readouts}


def _select_label_pair_rule(
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    receiver_predictions: list[int],
    calibration_indices: list[int],
) -> dict[str, Any]:
    predictions_by_rule: dict[str, list[int]] = {
        "always_source": source_predictions,
        "always_receiver": receiver_predictions,
        "receiver_on_disagreement": [
            receiver if source != receiver else source
            for source, receiver in zip(source_predictions, receiver_predictions, strict=True)
        ],
        "source_on_disagreement": [
            source if source != receiver else receiver
            for source, receiver in zip(source_predictions, receiver_predictions, strict=True)
        ],
    }
    readouts = [
        {
            "rule": rule,
            "calibration_accuracy": _accuracy(rows, predictions, calibration_indices),
        }
        for rule, predictions in predictions_by_rule.items()
    ]
    selected = max(readouts, key=lambda row: (row["calibration_accuracy"], row["rule"] == "always_source"))
    return {
        "selected": selected,
        "grid": readouts,
        "predictions": predictions_by_rule[selected["rule"]],
    }


def _union_top2_oracle(rows: list[arc_gate.ArcRow], source_scores: list[list[float]], receiver_scores: list[list[float]], indices: list[int]) -> float:
    hits = 0
    for index in indices:
        candidates = set(_ranked_indices(source_scores[index])[:2]) | set(_ranked_indices(receiver_scores[index])[:2])
        hits += int(rows[index].answer_index in candidates)
    return float(hits / len(indices)) if indices else 0.0


def build_probe(
    *,
    output_dir: pathlib.Path,
    benchmark_name: str,
    eval_path: pathlib.Path,
    source_score_cache: pathlib.Path,
    receiver_score_cache: pathlib.Path,
    source_lm_model: str,
    receiver_lm_model: str,
    source_lm_device: str,
    receiver_lm_device: str,
    lm_dtype: str,
    lm_max_length: int,
    lm_normalization: str,
    local_files_only: bool,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    eval_path = _resolve(eval_path)
    source_score_cache = _resolve(source_score_cache)
    receiver_score_cache = _resolve(receiver_score_cache)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = arc_gate._load_rows(eval_path)
    source_scores, source_predictions, source_state, source_cache_sha256 = headroom._source_scores(
        rows=rows,
        score_cache=source_score_cache,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=lm_dtype,
        source_lm_max_length=lm_max_length,
        source_lm_normalization=lm_normalization,
        local_files_only=local_files_only,
    )
    receiver_scores, receiver_predictions, receiver_state, receiver_cache_sha256 = headroom._source_scores(
        rows=rows,
        score_cache=receiver_score_cache,
        source_lm_model=receiver_lm_model,
        source_lm_device=receiver_lm_device,
        source_lm_dtype=lm_dtype,
        source_lm_max_length=lm_max_length,
        source_lm_normalization=lm_normalization,
        local_files_only=local_files_only,
    )
    if len(source_scores) != len(rows) or len(receiver_scores) != len(rows):
        raise ValueError("score caches must match eval row count")

    indices = list(range(len(rows)))
    calibration_indices = [index for index in indices if index % 2 == 0]
    eval_indices = [index for index in indices if index % 2 == 1]
    quantized_source_scores = _quantized_source_rows(source_scores)
    receiver_zscores = _zscore_rows(receiver_scores)
    fusion_selection = _select_fusion_weight(rows, quantized_source_scores, receiver_zscores, calibration_indices)
    fusion_weight = float(fusion_selection["selected"]["source_weight"])
    fusion_predictions = _fusion_predictions(
        quantized_source_scores,
        receiver_zscores,
        source_weight=fusion_weight,
    )
    label_pair = _select_label_pair_rule(rows, source_predictions, receiver_predictions, calibration_indices)
    label_pair_predictions = label_pair["predictions"]

    source_heldout = _accuracy(rows, source_predictions, eval_indices)
    receiver_heldout = _accuracy(rows, receiver_predictions, eval_indices)
    label_pair_heldout = _accuracy(rows, label_pair_predictions, eval_indices)
    fusion_heldout = _accuracy(rows, fusion_predictions, eval_indices)
    benchmark_slug = _slugify(benchmark_name)
    payload = {
        "gate": f"source_private_{benchmark_slug}_score_fusion_packet_probe",
        "benchmark": benchmark_name,
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "eval_rows": len(rows),
        "calibration_rows": len(calibration_indices),
        "heldout_eval_rows": len(eval_indices),
        "source_score_cache": _display_path(source_score_cache),
        "source_score_cache_sha256": source_cache_sha256,
        "receiver_score_cache": _display_path(receiver_score_cache),
        "receiver_score_cache_sha256": receiver_cache_sha256,
        "source_model": source_state
        | {
            "role": "source_sender",
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS) + ["question_concept"],
        },
        "receiver_model": receiver_state
        | {
            "role": "receiver_side_information",
            "receiver_visible_fields": ["question", "choices"],
            "forbidden_receiver_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS) + ["question_concept"],
        },
        "packet_contract": {
            "packet_name": "quantized_source_score_distribution",
            "raw_payload_bytes": 3,
            "record_bytes_with_header_crc": 6,
            "candidate_count": max(len(row.choices) for row in rows),
            "source_payload": "one 4-bit clipped row-zscore per candidate, packed in public candidate order",
            "receiver_side_information": "receiver local row-zscores over the same public candidates",
            "decoder_rule": "calibrate a global source/receiver score-fusion weight on even validation rows, evaluate on odd rows",
            "claim_boundary": (
                "Headroom probe for score-distribution communication. Promotion requires held-out improvement over "
                "source-label text and a top-label-only two-model rule."
            ),
        },
        "headline": {
            "source_label_text_heldout_accuracy": source_heldout,
            "receiver_label_text_heldout_accuracy": receiver_heldout,
            "best_top_label_pair_rule": label_pair["selected"]["rule"],
            "best_top_label_pair_heldout_accuracy": label_pair_heldout,
            "fusion_source_weight": fusion_weight,
            "fusion_calibration_accuracy": fusion_selection["selected"]["calibration_accuracy"],
            "fusion_heldout_accuracy": fusion_heldout,
            "fusion_minus_source_label_text_heldout": fusion_heldout - source_heldout,
            "fusion_minus_best_top_label_pair_heldout": fusion_heldout - label_pair_heldout,
            "source_top2_oracle_heldout_accuracy": _union_top2_oracle(rows, source_scores, source_scores, eval_indices),
            "source_receiver_union_top2_oracle_heldout_accuracy": _union_top2_oracle(rows, source_scores, receiver_scores, eval_indices),
        },
        "fusion_weight_grid": fusion_selection["grid"],
        "top_label_pair_grid": label_pair["grid"],
        "pass_rule": {
            "fusion_beats_source_label_text_heldout_by": 0.02,
            "fusion_beats_best_top_label_pair_heldout_by": 0.01,
        },
    }
    payload["pass_gate"] = bool(
        payload["headline"]["fusion_minus_source_label_text_heldout"] >= 0.02
        and payload["headline"]["fusion_minus_best_top_label_pair_heldout"] >= 0.01
    )
    (output_dir / f"{benchmark_slug}_score_fusion_packet_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {benchmark_name} Score-Fusion Packet Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source-label heldout accuracy: `{source_heldout:.3f}`",
        f"- receiver-label heldout accuracy: `{receiver_heldout:.3f}`",
        f"- best top-label pair heldout accuracy: `{label_pair_heldout:.3f}`",
        f"- fusion heldout accuracy: `{fusion_heldout:.3f}`",
        f"- fusion minus source-label: `{payload['headline']['fusion_minus_source_label_text_heldout']:.3f}`",
        f"- fusion minus best top-label pair: `{payload['headline']['fusion_minus_best_top_label_pair_heldout']:.3f}`",
        f"- source/receiver union top-2 oracle heldout accuracy: `{payload['headline']['source_receiver_union_top2_oracle_heldout_accuracy']:.3f}`",
        "",
    ]
    (output_dir / f"{benchmark_slug}_score_fusion_packet_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a quantized source-score packet fused with receiver scores.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-name", default="CommonsenseQA")
    parser.add_argument(
        "--eval-path",
        type=pathlib.Path,
        default=pathlib.Path(
            "results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_validation.jsonl"
        ),
    )
    parser.add_argument("--source-score-cache", type=pathlib.Path, required=True)
    parser.add_argument("--receiver-score-cache", type=pathlib.Path, required=True)
    parser.add_argument("--source-lm-model", required=True)
    parser.add_argument("--receiver-lm-model", required=True)
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--receiver-lm-device", default="auto")
    parser.add_argument("--lm-dtype", default="float32")
    parser.add_argument("--lm-max-length", type=int, default=256)
    parser.add_argument("--lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_probe(
        output_dir=args.output_dir,
        benchmark_name=args.benchmark_name,
        eval_path=args.eval_path,
        source_score_cache=args.source_score_cache,
        receiver_score_cache=args.receiver_score_cache,
        source_lm_model=args.source_lm_model,
        receiver_lm_model=args.receiver_lm_model,
        source_lm_device=args.source_lm_device,
        receiver_lm_device=args.receiver_lm_device,
        lm_dtype=args.lm_dtype,
        lm_max_length=args.lm_max_length,
        lm_normalization=args.lm_normalization,
        local_files_only=args.local_files_only,
        run_date=args.run_date,
    )


if __name__ == "__main__":
    main()
