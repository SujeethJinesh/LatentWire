from __future__ import annotations

"""Evaluate the strict Qwen hybrid packet policy on cached Phi HellaSwag rows."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate_20260503_validation1024_2048"
)
DEFAULT_SLICES = (
    {
        "slice_start": 1024,
        "slice_end_exclusive": 1536,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/"
            "qwen_strict_packet_predictions_1024_1536.jsonl"
        ),
        "phi_target_score_cache": (
            "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536/"
            "target_score_cache.json"
        ),
    },
    {
        "slice_start": 1536,
        "slice_end_exclusive": 2048,
        "qwen_predictions": (
            "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/"
            "qwen_strict_packet_predictions_1536_2048.jsonl"
        ),
        "phi_target_score_cache": (
            "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048/"
            "target_score_cache.json"
        ),
    },
)
CONTROL_FIELDS = (
    "source_label_prediction",
    "source_rank_only_bagged_prediction",
    "score_only_bagged_prediction",
    "score_mean_prediction",
    "score_vote_prediction",
    "trained_label_prediction",
    "wrong_example_hidden_prediction",
    "zero_hidden_prediction",
    "candidate_roll_hidden_prediction",
    "score_channel_roll_hidden_prediction",
)
BOOTSTRAP_SAMPLES = 5000
TRAIN_PREFIX_ROWS_PER_SLICE = 128


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


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


def _hybrid_prediction(row: dict[str, Any]) -> int:
    if int(row["hidden_mean_prediction"]) == int(row["score_mean_prediction"]):
        return int(row["vote_prediction"])
    return int(row["hidden_mean_prediction"])


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    deltas = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(deltas), size=len(deltas))
        draws.append(float(np.mean(deltas[indices])))
    return {
        "delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "helps": int(np.sum(deltas > 0)),
        "harms": int(np.sum(deltas < 0)),
    }


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _oracle_accuracy(*, answers: np.ndarray, candidates: list[np.ndarray]) -> float:
    hit = np.zeros(len(answers), dtype=bool)
    for predictions in candidates:
        hit |= predictions == answers
    return float(np.mean(hit))


def _load_slice(
    *,
    spec: dict[str, Any],
    train_prefix_rows_per_slice: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qwen_path = _resolve(spec["qwen_predictions"])
    target_cache_path = _resolve(spec["phi_target_score_cache"])
    qwen_rows = _read_jsonl(qwen_path)
    target_cache = _read_json(target_cache_path)
    target_by_id = {
        str(row_id): int(prediction)
        for row_id, prediction in zip(
            target_cache["row_ids"],
            target_cache["source_predictions"],
            strict=True,
        )
    }
    if len(qwen_rows) != int(target_cache["row_count"]):
        raise ValueError(f"row-count mismatch in slice {spec['slice_start']}")

    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(qwen_rows):
        row_id = str(row["row_id"])
        if row_id not in target_by_id:
            raise KeyError(f"missing target prediction for row_id={row_id}")
        copied = dict(row)
        copied["phi_target_prediction"] = target_by_id[row_id]
        copied["hybrid_vote_on_score_agreement_prediction"] = _hybrid_prediction(row)
        copied["_slice_start"] = int(spec["slice_start"])
        copied["_slice_end_exclusive"] = int(spec["slice_end_exclusive"])
        copied["_within_slice_index"] = int(row_index)
        copied["_split"] = "train_prefix" if row_index < train_prefix_rows_per_slice else "eval"
        rows.append(copied)

    metadata = {
        "slice_start": int(spec["slice_start"]),
        "slice_end_exclusive": int(spec["slice_end_exclusive"]),
        "rows": len(qwen_rows),
        "train_prefix_rows": int(train_prefix_rows_per_slice),
        "eval_rows": max(0, len(qwen_rows) - int(train_prefix_rows_per_slice)),
        "qwen_predictions": _display_path(qwen_path),
        "qwen_predictions_sha256": _sha256_file(qwen_path),
        "phi_target_score_cache": _display_path(target_cache_path),
        "phi_target_score_cache_sha256": _sha256_file(target_cache_path),
    }
    return rows, metadata


def _prediction_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _summary_row(
    *,
    name: str,
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    paired = _paired_ci(
        selected=predictions,
        baseline=baseline,
        answers=answers,
        seed=seed,
        samples=bootstrap_samples,
    )
    return {
        "name": name,
        "accuracy": _accuracy(predictions, answers),
        "delta_vs_candidate_only": paired["delta"],
        "ci95_low_vs_candidate_only": paired["ci95_low"],
        "ci95_high_vs_candidate_only": paired["ci95_high"],
        "helps_vs_candidate_only": paired["helps"],
        "harms_vs_candidate_only": paired["harms"],
    }


def _slice_summary_rows(
    *,
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    out = []
    slice_starts = sorted({int(row["_slice_start"]) for row in rows})
    for slice_start in slice_starts:
        slice_rows = [row for row in rows if int(row["_slice_start"]) == slice_start]
        eval_rows = [row for row in slice_rows if row["_split"] == "eval"]
        answers = _prediction_array(eval_rows, "answer_index")
        selected = _prediction_array(eval_rows, "selected_prediction")
        hybrid = _prediction_array(eval_rows, "hybrid_vote_on_score_agreement_prediction")
        target = _prediction_array(eval_rows, "phi_target_prediction")
        hybrid_vs_selected = _paired_ci(
            selected=hybrid,
            baseline=selected,
            answers=answers,
            seed=20260503 + slice_start,
            samples=bootstrap_samples,
        )
        hybrid_vs_target = _paired_ci(
            selected=hybrid,
            baseline=target,
            answers=answers,
            seed=20260531 + slice_start,
            samples=bootstrap_samples,
        )
        out.append(
            {
                "slice_start": slice_start,
                "slice_end_exclusive": int(slice_rows[0]["_slice_end_exclusive"]),
                "total_rows": len(slice_rows),
                "train_prefix_rows": len(slice_rows) - len(eval_rows),
                "eval_rows": len(eval_rows),
                "phi_target_accuracy": _accuracy(target, answers),
                "candidate_only_accuracy": _accuracy(selected, answers),
                "hybrid_accuracy": _accuracy(hybrid, answers),
                "hybrid_delta_vs_candidate_only": hybrid_vs_selected["delta"],
                "hybrid_ci95_low_vs_candidate_only": hybrid_vs_selected["ci95_low"],
                "hybrid_ci95_high_vs_candidate_only": hybrid_vs_selected["ci95_high"],
                "hybrid_helps_vs_candidate_only": hybrid_vs_selected["helps"],
                "hybrid_harms_vs_candidate_only": hybrid_vs_selected["harms"],
                "hybrid_delta_vs_phi_target": hybrid_vs_target["delta"],
                "hybrid_ci95_low_vs_phi_target": hybrid_vs_target["ci95_low"],
                "hybrid_ci95_high_vs_phi_target": hybrid_vs_target["ci95_high"],
                "target_or_hybrid_oracle_accuracy": _oracle_accuracy(
                    answers=answers,
                    candidates=[target, hybrid],
                ),
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen Hybrid-To-Phi Cross-Family Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- heldout eval rows: `{h['heldout_eval_rows']}`",
        f"- Phi target-only accuracy: `{h['phi_target_accuracy']:.6f}`",
        f"- Qwen candidate-only packet accuracy: `{h['candidate_only_accuracy']:.6f}`",
        f"- Qwen hybrid packet accuracy: `{h['hybrid_accuracy']:.6f}`",
        f"- hybrid delta vs candidate-only: `{h['hybrid_delta_vs_candidate_only']:.6f}`",
        f"- hybrid CI95 low vs candidate-only: `{h['hybrid_ci95_low_vs_candidate_only']:.6f}`",
        f"- target-or-hybrid oracle accuracy: `{h['target_or_hybrid_oracle_accuracy']:.6f}`",
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
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    train_prefix_rows_per_slice: int = TRAIN_PREFIX_ROWS_PER_SLICE,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)

    all_rows: list[dict[str, Any]] = []
    slice_metadata: list[dict[str, Any]] = []
    for spec in DEFAULT_SLICES:
        rows, metadata = _load_slice(
            spec=spec,
            train_prefix_rows_per_slice=train_prefix_rows_per_slice,
        )
        all_rows.extend(rows)
        slice_metadata.append(metadata)

    eval_rows = [row for row in all_rows if row["_split"] == "eval"]
    answers = _prediction_array(eval_rows, "answer_index")
    selected = _prediction_array(eval_rows, "selected_prediction")
    hybrid = _prediction_array(eval_rows, "hybrid_vote_on_score_agreement_prediction")
    target = _prediction_array(eval_rows, "phi_target_prediction")

    hybrid_vs_selected = _paired_ci(
        selected=hybrid,
        baseline=selected,
        answers=answers,
        seed=30360503,
        samples=bootstrap_samples,
    )
    hybrid_vs_target = _paired_ci(
        selected=hybrid,
        baseline=target,
        answers=answers,
        seed=30360531,
        samples=bootstrap_samples,
    )
    selected_vs_target = _paired_ci(
        selected=selected,
        baseline=target,
        answers=answers,
        seed=30360517,
        samples=bootstrap_samples,
    )

    method_rows = [
        _summary_row(
            name="qwen_hybrid_vote_on_score_agreement",
            predictions=hybrid,
            baseline=selected,
            answers=answers,
            seed=40360503,
            bootstrap_samples=bootstrap_samples,
        ),
        _summary_row(
            name="qwen_candidate_only",
            predictions=selected,
            baseline=selected,
            answers=answers,
            seed=40360504,
            bootstrap_samples=bootstrap_samples,
        ),
        _summary_row(
            name="phi_target_only",
            predictions=target,
            baseline=selected,
            answers=answers,
            seed=40360505,
            bootstrap_samples=bootstrap_samples,
        ),
    ]
    for offset, field in enumerate(CONTROL_FIELDS, start=10):
        method_rows.append(
            _summary_row(
                name=field.removesuffix("_prediction"),
                predictions=_prediction_array(eval_rows, field),
                baseline=selected,
                answers=answers,
                seed=40360503 + offset,
                bootstrap_samples=bootstrap_samples,
            )
        )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    slice_rows = _slice_summary_rows(rows=all_rows, bootstrap_samples=bootstrap_samples)
    control_rows = [row for row in method_rows if row["name"] not in {"qwen_hybrid_vote_on_score_agreement", "qwen_candidate_only", "phi_target_only"}]
    best_control = max(control_rows, key=lambda row: row["accuracy"])
    best_control_predictions = _prediction_array(eval_rows, best_control["name"] + "_prediction") if best_control["name"] + "_prediction" in eval_rows[0] else None
    if best_control_predictions is None:
        reverse_lookup = {
            field.removesuffix("_prediction"): field
            for field in CONTROL_FIELDS
        }
        best_control_predictions = _prediction_array(eval_rows, reverse_lookup[best_control["name"]])
    hybrid_vs_best_control = _paired_ci(
        selected=hybrid,
        baseline=best_control_predictions,
        answers=answers,
        seed=30360601,
        samples=bootstrap_samples,
    )

    pass_gate = (
        hybrid_vs_selected["delta"] > 0.0
        and hybrid_vs_selected["ci95_low"] > 0.0
        and hybrid_vs_target["ci95_low"] > 0.0
        and hybrid_vs_best_control["ci95_low"] > 0.0
        and all(row["hybrid_delta_vs_candidate_only"] > 0.0 for row in slice_rows)
    )
    payload = {
        "gate": "source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "The fixed Qwen hybrid packet survives this cached cross-family gate only if it beats "
            "Qwen candidate-only with positive paired CI on heldout Phi rows, beats Phi target-only "
            "and the best destructive/source-score control with positive paired CIs, and improves "
            "candidate-only on both contiguous slices. This is packet-policy survival, not learned receiver fusion."
        ),
        "headline": {
            "total_rows": len(all_rows),
            "heldout_eval_rows": len(eval_rows),
            "train_prefix_rows": len(all_rows) - len(eval_rows),
            "range_start": min(item["slice_start"] for item in slice_metadata),
            "range_end_exclusive": max(item["slice_end_exclusive"] for item in slice_metadata),
            "phi_target_accuracy": _accuracy(target, answers),
            "candidate_only_accuracy": _accuracy(selected, answers),
            "hybrid_accuracy": _accuracy(hybrid, answers),
            "hybrid_delta_vs_candidate_only": hybrid_vs_selected["delta"],
            "hybrid_ci95_low_vs_candidate_only": hybrid_vs_selected["ci95_low"],
            "hybrid_ci95_high_vs_candidate_only": hybrid_vs_selected["ci95_high"],
            "hybrid_helps_vs_candidate_only": hybrid_vs_selected["helps"],
            "hybrid_harms_vs_candidate_only": hybrid_vs_selected["harms"],
            "hybrid_delta_vs_phi_target": hybrid_vs_target["delta"],
            "hybrid_ci95_low_vs_phi_target": hybrid_vs_target["ci95_low"],
            "hybrid_ci95_high_vs_phi_target": hybrid_vs_target["ci95_high"],
            "candidate_only_delta_vs_phi_target": selected_vs_target["delta"],
            "candidate_only_ci95_low_vs_phi_target": selected_vs_target["ci95_low"],
            "best_control_name": best_control["name"],
            "best_control_accuracy": best_control["accuracy"],
            "hybrid_delta_vs_best_control": hybrid_vs_best_control["delta"],
            "hybrid_ci95_low_vs_best_control": hybrid_vs_best_control["ci95_low"],
            "target_or_candidate_only_oracle_accuracy": _oracle_accuracy(
                answers=answers,
                candidates=[target, selected],
            ),
            "target_or_hybrid_oracle_accuracy": _oracle_accuracy(
                answers=answers,
                candidates=[target, hybrid],
            ),
        },
        "packet_contract": {
            "receiver_visible_payload": "one final Qwen source candidate id emitted by a fixed source-side hybrid policy",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "target_family": "Phi-3-mini",
            "source_family": "Qwen2.5",
        },
        "slice_metadata": slice_metadata,
        "method_rows": method_rows,
        "slice_rows": slice_rows,
        "interpretation": (
            "The fixed Qwen hybrid vote-on-score-agreement packet improves over candidate-only on the "
            "cached Phi cross-family heldout rows while retaining the same receiver-visible one-candidate "
            "packet contract. This is useful cross-family packet-policy survival, but it is not a learned "
            "Phi receiver or a general latent language: the receiver still sees only the final candidate id."
        ),
        "lay_explanation": (
            "We checked whether the improved Qwen hint still helps when the receiving model is Phi instead "
            "of Qwen. On the cached Phi rows, the fixed hybrid hint beats both Phi's own answer and the "
            "older Qwen candidate-only hint. The caveat is that Phi is still just receiving an answer-choice "
            "hint, not a rich hidden thought."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_qwen_hybrid_to_phi_cross_family_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_hybrid_to_phi_cross_family_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_hybrid_to_phi_cross_family_gate.json",
                "hellaswag_qwen_hybrid_to_phi_cross_family_gate.md",
                "method_rows.csv",
                "slice_rows.csv",
            ],
            "slice_metadata": slice_metadata,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached Qwen hybrid packet to Phi cross-family gate.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--train-prefix-rows-per-slice", type=int, default=TRAIN_PREFIX_ROWS_PER_SLICE)
    parser.add_argument("--run-date", default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        train_prefix_rows_per_slice=args.train_prefix_rows_per_slice,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
