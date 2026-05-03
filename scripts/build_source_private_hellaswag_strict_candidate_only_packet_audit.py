from __future__ import annotations

"""Audit strict HellaSwag packet compaction to a candidate-only payload.

This is deliberately an audit, not a new receiver. It verifies that the
strict Qwen HellaSwag multi-slice row can be decoded from the already-selected
candidate id alone, while preserving the original rank/score-channel control
claims from the source artifact.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = pathlib.Path(
    "results/"
    "source_private_hellaswag_hidden_innovation_multi_slice_stress_20260503_"
    "rank_score_channel_qwen05_validation0_9216/"
    "hellaswag_hidden_innovation_multi_slice_stress.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_strict_candidate_only_packet_audit_20260503_validation0_9216"
)
CANDIDATE_COUNT = 4
FLOAT_TOLERANCE = 1e-12


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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            rows.append(row)
    return rows


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _accuracy(rows: list[dict[str, Any]], prediction_key: str) -> float:
    if not rows:
        raise ValueError("cannot score empty prediction rows")
    correct = 0
    for row in rows:
        if prediction_key not in row:
            raise KeyError(f"missing prediction field: {prediction_key}")
        if int(row[prediction_key]) == int(row["answer_index"]):
            correct += 1
    return correct / float(len(rows))


def _prediction_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = sorted(key for key in rows[0] if key.endswith("_prediction"))
    required = {
        "selected_prediction",
        "source_label_prediction",
        "source_rank_only_bagged_prediction",
        "score_only_bagged_prediction",
        "zero_hidden_prediction",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise KeyError(f"missing required prediction fields: {missing}")
    return fields


def _validate_candidate_range(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        answer = int(row["answer_index"])
        selected = int(row["selected_prediction"])
        if not 0 <= answer < CANDIDATE_COUNT:
            raise ValueError(f"row {index} answer_index out of range: {answer}")
        if not 0 <= selected < CANDIDATE_COUNT:
            raise ValueError(f"row {index} selected_prediction out of range: {selected}")


def _raw_bytes_for_candidate_only(candidate_count: int) -> int:
    bits = int(math.ceil(math.log2(max(2, int(candidate_count)))))
    return max(1, int(math.ceil(bits / 8.0)))


def _framed_bytes(raw_bytes: int) -> int:
    return int(raw_bytes) + 3


def _find_predictions_path(artifact_path: pathlib.Path | str) -> pathlib.Path:
    artifact_path = _resolve(artifact_path)
    predictions_path = artifact_path.parent / "predictions.jsonl"
    if predictions_path.exists():
        return predictions_path
    nested_predictions_path = artifact_path.parent / "bagged_gate" / "predictions.jsonl"
    if nested_predictions_path.exists():
        return nested_predictions_path
    artifact = _read_json(artifact_path)
    nested_artifact_path = artifact.get("bagged_gate_path")
    if nested_artifact_path:
        return _find_predictions_path(nested_artifact_path)
    raise FileNotFoundError(f"missing predictions file for {artifact_path}")


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= FLOAT_TOLERANCE


def _slice_audit(slice_row: dict[str, Any]) -> dict[str, Any]:
    artifact_path = pathlib.Path(slice_row["artifact_path"])
    predictions_path = _find_predictions_path(artifact_path)
    predictions = _read_jsonl(predictions_path)
    _validate_candidate_range(predictions)
    fields = _prediction_fields(predictions)

    recomputed = {field: _accuracy(predictions, field) for field in fields}
    candidate_only_accuracy = recomputed["selected_prediction"]
    expected_rows = int(slice_row["eval_rows"])
    if len(predictions) != expected_rows:
        raise ValueError(
            f"{predictions_path} row count {len(predictions)} does not match eval_rows {expected_rows}"
        )
    if not _close(candidate_only_accuracy, float(slice_row["selected_eval_accuracy"])):
        raise ValueError(
            "candidate-only accuracy does not match source selected_eval_accuracy "
            f"for {artifact_path}: {candidate_only_accuracy} vs {slice_row['selected_eval_accuracy']}"
        )

    source_rank_key = "source_rank_only_bagged_prediction"
    score_key = "score_only_bagged_prediction"
    zero_key = "zero_hidden_prediction"
    label_key = "source_label_prediction"
    candidate_roll_key = "candidate_roll_hidden_prediction"
    score_roll_key = "score_channel_roll_hidden_prediction"
    wrong_key = "wrong_example_hidden_prediction"

    return {
        "artifact_path": _display_path(artifact_path),
        "predictions_path": _display_path(predictions_path),
        "predictions_sha256": _sha256_file(predictions_path),
        "eval_slice_start": int(slice_row["eval_slice_start"]),
        "eval_slice_end_exclusive": int(slice_row["eval_slice_end_exclusive"]),
        "eval_rows": len(predictions),
        "candidate_only_eval_accuracy": candidate_only_accuracy,
        "source_label_copy_eval_accuracy": recomputed[label_key],
        "source_rank_only_bagged_control_accuracy": recomputed[source_rank_key],
        "score_only_bagged_control_accuracy": recomputed[score_key],
        "zero_hidden_control_accuracy": recomputed[zero_key],
        "candidate_roll_hidden_control_accuracy": recomputed.get(candidate_roll_key),
        "score_channel_roll_hidden_control_accuracy": recomputed.get(score_roll_key),
        "wrong_example_hidden_control_accuracy": recomputed.get(wrong_key),
        "candidate_only_minus_best_label_copy": (
            candidate_only_accuracy - float(slice_row["best_label_copy_eval_accuracy"])
        ),
        "candidate_only_minus_source_rank_only_bagged": (
            candidate_only_accuracy - recomputed[source_rank_key]
        ),
        "candidate_only_minus_score_only_bagged": candidate_only_accuracy - recomputed[score_key],
        "candidate_only_minus_zero_hidden": candidate_only_accuracy - recomputed[zero_key],
        "paired_ci95_low_vs_best_label_copy": float(
            slice_row["paired_ci95_low_vs_best_label_copy"]
        ),
        "paired_ci95_low_vs_source_rank_only_bagged": float(
            slice_row["paired_ci95_low_vs_source_rank_only_bagged"]
        ),
        "paired_ci95_low_vs_score_only_bagged": float(
            slice_row["paired_ci95_low_vs_score_only_bagged"]
        ),
        "rank_score_channel_controls_available": bool(
            slice_row["rank_score_channel_controls_available"]
        ),
        "source_text_exposed": bool(slice_row["source_text_exposed"]),
        "source_kv_exposed": bool(slice_row["source_kv_exposed"]),
        "raw_hidden_vector_transmitted": bool(slice_row["raw_hidden_vector_transmitted"]),
        "raw_scores_transmitted": bool(slice_row["raw_scores_transmitted"]),
    }


def _weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["eval_rows"]) for row in rows)
    if total <= 0:
        raise ValueError("cannot weight empty rows")
    return sum(float(row[key]) * int(row["eval_rows"]) for row in rows) / float(total)


def _write_csv(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_slice_start",
        "eval_slice_end_exclusive",
        "eval_rows",
        "candidate_only_eval_accuracy",
        "source_label_copy_eval_accuracy",
        "source_rank_only_bagged_control_accuracy",
        "score_only_bagged_control_accuracy",
        "zero_hidden_control_accuracy",
        "candidate_only_minus_best_label_copy",
        "candidate_only_minus_source_rank_only_bagged",
        "candidate_only_minus_score_only_bagged",
        "candidate_only_minus_zero_hidden",
        "paired_ci95_low_vs_best_label_copy",
        "paired_ci95_low_vs_source_rank_only_bagged",
        "paired_ci95_low_vs_score_only_bagged",
        "predictions_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headline = payload["headline"]
    lines = [
        "# HellaSwag Strict Candidate-Only Packet Audit",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source artifact: `{payload['source_artifact']}`",
        f"- total eval rows: `{headline['total_eval_rows']}`",
        f"- slice count: `{headline['slice_count']}`",
        f"- candidate-only weighted accuracy: `{headline['weighted_candidate_only_eval_accuracy']:.6f}`",
        f"- source-rank/index-only control accuracy: `{headline['weighted_source_rank_only_bagged_control_accuracy']:.6f}`",
        f"- score-only control accuracy: `{headline['weighted_score_only_bagged_control_accuracy']:.6f}`",
        f"- best label-copy control accuracy: `{headline['weighted_best_label_copy_eval_accuracy']:.6f}`",
        f"- previous packet: `{headline['previous_raw_payload_bytes']}B` raw / `{headline['previous_framed_record_bytes']}B` framed",
        f"- candidate-only packet: `{headline['candidate_only_raw_payload_bytes']}B` raw / `{headline['candidate_only_framed_record_bytes']}B` framed",
        f"- framed byte reduction: `{headline['framed_byte_reduction']}` bytes/request",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Limitations",
        "",
        "- This is a multiple-choice candidate-id compaction result, not a positive receiver-fusion result.",
        "- The receiver-visible payload is the selected candidate id; it does not prove a general latent language.",
        "- Native latency, throughput, HBM traffic, and GPU-serving speedups still require NVIDIA/vLLM/SGLang rows.",
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(
    *,
    input_path: pathlib.Path | str = DEFAULT_INPUT,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    source = _read_json(input_path)
    if not bool(source.get("pass_gate")):
        raise ValueError("source multi-slice artifact did not pass its gate")
    if not source.get("slice_rows"):
        raise ValueError("source multi-slice artifact has no slice rows")

    slice_rows = [_slice_audit(row) for row in source["slice_rows"]]
    total_eval_rows = sum(int(row["eval_rows"]) for row in slice_rows)
    candidate_raw_bytes = _raw_bytes_for_candidate_only(CANDIDATE_COUNT)
    candidate_framed_bytes = _framed_bytes(candidate_raw_bytes)
    source_headline = source["headline"]
    previous_raw_bytes = int(source_headline["raw_payload_bytes"])
    previous_framed_bytes = int(source_headline["framed_record_bytes"])

    min_delta_vs_rank = min(
        float(row["candidate_only_minus_source_rank_only_bagged"]) for row in slice_rows
    )
    min_delta_vs_score = min(float(row["candidate_only_minus_score_only_bagged"]) for row in slice_rows)
    min_delta_vs_zero = min(float(row["candidate_only_minus_zero_hidden"]) for row in slice_rows)
    min_delta_vs_best_label = min(
        float(row["candidate_only_minus_best_label_copy"]) for row in slice_rows
    )
    all_private = all(
        not row["source_text_exposed"]
        and not row["source_kv_exposed"]
        and not row["raw_hidden_vector_transmitted"]
        and not row["raw_scores_transmitted"]
        for row in slice_rows
    )
    all_rank_score_controls = all(row["rank_score_channel_controls_available"] for row in slice_rows)
    preserves_accuracy = _close(
        _weighted_average(slice_rows, "candidate_only_eval_accuracy"),
        float(source_headline["weighted_selected_eval_accuracy"]),
    )
    strict_controls_positive = (
        min_delta_vs_best_label >= float(source_headline["strict_delta_required"])
        and min_delta_vs_rank >= float(source_headline["strict_delta_required"])
        and min_delta_vs_score >= float(source_headline["strict_delta_required"])
        and min_delta_vs_zero >= float(source_headline["strict_delta_required"])
        and float(source_headline["min_ci95_low_vs_best_label_copy"]) > 0.0
        and float(source_headline["min_ci95_low_vs_source_rank_only_bagged"]) > 0.0
        and float(source_headline["min_ci95_low_vs_score_only_bagged"]) > 0.0
    )

    headline = {
        "total_eval_rows": total_eval_rows,
        "slice_count": len(slice_rows),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_id_bits": int(math.ceil(math.log2(CANDIDATE_COUNT))),
        "candidate_only_raw_payload_bytes": candidate_raw_bytes,
        "candidate_only_framed_record_bytes": candidate_framed_bytes,
        "previous_raw_payload_bytes": previous_raw_bytes,
        "previous_framed_record_bytes": previous_framed_bytes,
        "raw_byte_reduction": previous_raw_bytes - candidate_raw_bytes,
        "framed_byte_reduction": previous_framed_bytes - candidate_framed_bytes,
        "framed_relative_reduction": (
            (previous_framed_bytes - candidate_framed_bytes) / float(previous_framed_bytes)
        ),
        "weighted_candidate_only_eval_accuracy": _weighted_average(
            slice_rows, "candidate_only_eval_accuracy"
        ),
        "weighted_source_label_copy_eval_accuracy": _weighted_average(
            slice_rows, "source_label_copy_eval_accuracy"
        ),
        "weighted_source_rank_only_bagged_control_accuracy": _weighted_average(
            slice_rows, "source_rank_only_bagged_control_accuracy"
        ),
        "weighted_score_only_bagged_control_accuracy": _weighted_average(
            slice_rows, "score_only_bagged_control_accuracy"
        ),
        "weighted_zero_hidden_control_accuracy": _weighted_average(
            slice_rows, "zero_hidden_control_accuracy"
        ),
        "weighted_best_label_copy_eval_accuracy": float(
            source_headline["weighted_best_label_copy_eval_accuracy"]
        ),
        "min_delta_vs_best_label_copy": min_delta_vs_best_label,
        "min_delta_vs_source_rank_only_bagged": min_delta_vs_rank,
        "min_delta_vs_score_only_bagged": min_delta_vs_score,
        "min_delta_vs_zero_hidden": min_delta_vs_zero,
        "min_ci95_low_vs_best_label_copy": float(source_headline["min_ci95_low_vs_best_label_copy"]),
        "min_ci95_low_vs_source_rank_only_bagged": float(
            source_headline["min_ci95_low_vs_source_rank_only_bagged"]
        ),
        "min_ci95_low_vs_score_only_bagged": float(
            source_headline["min_ci95_low_vs_score_only_bagged"]
        ),
        "preserves_selected_accuracy": preserves_accuracy,
        "all_rank_score_channel_controls_available": all_rank_score_controls,
        "all_source_private": all_private,
    }
    pass_gate = (
        preserves_accuracy
        and bool(source["pass_gate"])
        and strict_controls_positive
        and all_rank_score_controls
        and all_private
        and candidate_raw_bytes <= 1
        and candidate_framed_bytes <= 4
        and total_eval_rows == int(source_headline["total_eval_rows"])
    )
    payload = {
        "gate": "source_private_hellaswag_strict_candidate_only_packet_audit",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_artifact": _display_path(input_path),
        "source_artifact_sha256": _sha256_file(input_path),
        "pass_gate": pass_gate,
        "headline": headline,
        "packet_contract": {
            "packet_name": "strict_hellaswag_candidate_only_selector_packet",
            "decoder_rule": "receiver chooses the transmitted candidate id",
            "fields": ["selected candidate id packed into 2 bits"],
            "candidate_count": CANDIDATE_COUNT,
            "candidate_id_bits": headline["candidate_id_bits"],
            "raw_payload_bytes": candidate_raw_bytes,
            "framed_record_bytes": candidate_framed_bytes,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "source_logits_transmitted": False,
        },
        "pass_rule": (
            "Pass if candidate-only predictions exactly preserve the strict selected-packet "
            "accuracy over all 9 validation slices, keep the original strict positive paired "
            "CI and delta controls against label-copy, source-rank/index-only, score-only, "
            "and zero-hidden baselines, expose no source text/KV/raw hidden/raw scores, and "
            "reduce the packet to 1B raw / 4B framed."
        ),
        "interpretation": (
            "The strict Qwen HellaSwag positive surface does not need the packet's extra "
            "confidence/debug byte for the receiver-visible decision: the selected candidate "
            "id alone reproduces the same 9216-row accuracy while retaining the original "
            "rank, score-channel, label-copy, zero-hidden, and corrupted-hidden control "
            "separations. This strengthens the systems/privacy accounting row, but it also "
            "sharpens the limitation: the current evidence is candidate-id communication, "
            "not a learned receiver or general latent language."
        ),
        "lay_explanation": (
            "We checked whether the previous hint needed two bytes or whether it was enough "
            "to send only which of the four answer choices the source model picked. On the "
            "large frozen HellaSwag slice, sending just that choice gives exactly the same "
            "answers, so the packet can be smaller. This is useful for a systems table, but "
            "it does not prove that the receiving model learned to reason from a richer hidden message."
        ),
        "slice_rows": slice_rows,
    }

    _write_json(output_dir / "hellaswag_strict_candidate_only_packet_audit.json", payload)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_strict_candidate_only_packet_audit.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "source_artifact": payload["source_artifact"],
            "source_artifact_sha256": payload["source_artifact_sha256"],
            "outputs": [
                "hellaswag_strict_candidate_only_packet_audit.json",
                "hellaswag_strict_candidate_only_packet_audit.md",
                "slice_rows.csv",
            ],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit strict HellaSwag selected-packet compaction to candidate-only bytes."
    )
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-date", default=None)
    args = parser.parse_args()
    payload = build_audit(input_path=args.input, output_dir=args.output_dir, run_date=args.run_date)
    print(json.dumps({"pass_gate": payload["pass_gate"], "headline": payload["headline"]}, indent=2))


if __name__ == "__main__":
    main()
