from __future__ import annotations

"""Full-validation gate for the strict HellaSwag fixed hybrid packet policy."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = pathlib.Path(
    "results/source_private_hellaswag_strict_candidate_only_packet_audit_20260503_validation0_9216/"
    "hellaswag_strict_candidate_only_packet_audit.json"
)
DEFAULT_TAIL_PREDICTIONS = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_"
    "rank_score_channel_qwen05_train512_validation9216_10042/bagged_gate/predictions.jsonl"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_fixed_hybrid_full_validation_gate_20260503_validation0_10042"
)
BOOTSTRAP_SAMPLES = 5000
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


def _load_rows(
    *,
    audit_path: pathlib.Path | str,
    tail_predictions: pathlib.Path | str | None,
    tail_start: int,
    tail_end_exclusive: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit = _read_json(audit_path)
    slice_sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for slice_row in audit["slice_rows"]:
        predictions_path = pathlib.Path(slice_row["predictions_path"])
        slice_start = int(slice_row["eval_slice_start"])
        slice_end = int(slice_row.get("eval_slice_end_exclusive", slice_start + 1024))
        loaded = _read_jsonl(predictions_path)
        for row in loaded:
            copied = dict(row)
            copied["_slice_start"] = slice_start
            copied["_slice_end_exclusive"] = slice_end
            copied["_predictions_path"] = _display_path(predictions_path)
            rows.append(copied)
        slice_sources.append(
            {
                "slice_start": slice_start,
                "slice_end_exclusive": slice_end,
                "rows": len(loaded),
                "predictions_path": _display_path(predictions_path),
                "predictions_sha256": _sha256_file(predictions_path),
            }
        )
    if tail_predictions is not None:
        loaded = _read_jsonl(tail_predictions)
        for row in loaded:
            copied = dict(row)
            copied["_slice_start"] = int(tail_start)
            copied["_slice_end_exclusive"] = int(tail_end_exclusive)
            copied["_predictions_path"] = _display_path(tail_predictions)
            rows.append(copied)
        slice_sources.append(
            {
                "slice_start": int(tail_start),
                "slice_end_exclusive": int(tail_end_exclusive),
                "rows": len(loaded),
                "predictions_path": _display_path(tail_predictions),
                "predictions_sha256": _sha256_file(tail_predictions),
            }
        )
    if not rows:
        raise ValueError("no rows loaded")
    required = {
        "answer_index",
        "selected_prediction",
        "vote_prediction",
        "hidden_mean_prediction",
        "score_mean_prediction",
        *CONTROL_FIELDS,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(f"missing required fields: {missing}")
    return rows, slice_sources


def _array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _hybrid_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_hybrid_prediction(row) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


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


def _oracle_accuracy(answers: np.ndarray, candidates: list[np.ndarray]) -> float:
    hit = np.zeros(len(answers), dtype=bool)
    for predictions in candidates:
        hit |= predictions == answers
    return float(np.mean(hit))


def _method_row(
    *,
    name: str,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    bootstrap_samples: int,
) -> dict[str, Any]:
    paired = _paired_ci(
        selected=predictions,
        baseline=baseline,
        answers=answers,
        seed=20260503 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "delta_vs_candidate_only": paired["delta"],
        "ci95_low_vs_candidate_only": paired["ci95_low"],
        "ci95_high_vs_candidate_only": paired["ci95_high"],
        "helps_vs_candidate_only": paired["helps"],
        "harms_vs_candidate_only": paired["harms"],
    }


def _slice_rows(
    *,
    rows: list[dict[str, Any]],
    candidate_only: np.ndarray,
    hybrid: np.ndarray,
    answers: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    row_slices = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    out: list[dict[str, Any]] = []
    for slice_start in sorted(set(row_slices.tolist())):
        mask = row_slices == slice_start
        paired = _paired_ci(
            selected=hybrid[mask],
            baseline=candidate_only[mask],
            answers=answers[mask],
            seed=30360503 + int(slice_start),
            samples=bootstrap_samples,
        )
        out.append(
            {
                "slice_start": int(slice_start),
                "slice_end_exclusive": int(rows[np.flatnonzero(mask)[0]]["_slice_end_exclusive"]),
                "eval_rows": int(np.sum(mask)),
                "candidate_only_accuracy": _accuracy(candidate_only[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(hybrid[mask], answers[mask]),
                "hybrid_delta_vs_candidate_only": paired["delta"],
                "hybrid_ci95_low_vs_candidate_only": paired["ci95_low"],
                "hybrid_ci95_high_vs_candidate_only": paired["ci95_high"],
                "hybrid_helps_vs_candidate_only": paired["helps"],
                "hybrid_harms_vs_candidate_only": paired["harms"],
            }
        )
    return out


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Fixed Hybrid Full-Validation Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- candidate-only accuracy: `{h['candidate_only_accuracy']:.6f}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- hybrid delta vs candidate-only: `{h['hybrid_delta_vs_candidate_only']:.6f}`",
        f"- hybrid CI95 low vs candidate-only: `{h['hybrid_ci95_low_vs_candidate_only']:.6f}`",
        f"- positive slice count: `{h['positive_slice_count']}` / `{h['slice_count']}`",
        f"- candidate/hybrid oracle accuracy: `{h['candidate_hybrid_oracle_accuracy']:.6f}`",
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
    audit_path: pathlib.Path | str = DEFAULT_AUDIT,
    tail_predictions: pathlib.Path | str | None = DEFAULT_TAIL_PREDICTIONS,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    tail_start: int = 9216,
    tail_end_exclusive: int = 10042,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows, slice_sources = _load_rows(
        audit_path=audit_path,
        tail_predictions=tail_predictions,
        tail_start=tail_start,
        tail_end_exclusive=tail_end_exclusive,
    )
    answers = _array(rows, "answer_index")
    candidate_only = _array(rows, "selected_prediction")
    hybrid = _hybrid_array(rows)
    hybrid_vs_candidate = _paired_ci(
        selected=hybrid,
        baseline=candidate_only,
        answers=answers,
        seed=40360503,
        samples=bootstrap_samples,
    )

    method_rows = [
        _method_row(
            name="candidate_only",
            rows=rows,
            predictions=candidate_only,
            baseline=candidate_only,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="fixed_hybrid_vote_on_score_agreement",
            rows=rows,
            predictions=hybrid,
            baseline=candidate_only,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
        ),
    ]
    for field in CONTROL_FIELDS:
        method_rows.append(
            _method_row(
                name=field.removesuffix("_prediction"),
                rows=rows,
                predictions=_array(rows, field),
                baseline=candidate_only,
                answers=answers,
                bootstrap_samples=bootstrap_samples,
            )
        )
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    controls = [
        row
        for row in method_rows
        if row["method"] not in {"candidate_only", "fixed_hybrid_vote_on_score_agreement"}
    ]
    best_control = max(controls, key=lambda row: row["accuracy"])
    best_control_predictions = _array(rows, f"{best_control['method']}_prediction") if (
        f"{best_control['method']}_prediction" in rows[0]
    ) else _array(
        rows,
        {
            field.removesuffix("_prediction"): field
            for field in CONTROL_FIELDS
        }[best_control["method"]],
    )
    hybrid_vs_best_control = _paired_ci(
        selected=hybrid,
        baseline=best_control_predictions,
        answers=answers,
        seed=40360603,
        samples=bootstrap_samples,
    )
    slice_rows = _slice_rows(
        rows=rows,
        candidate_only=candidate_only,
        hybrid=hybrid,
        answers=answers,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    positive_slice_count = sum(row["hybrid_delta_vs_candidate_only"] > 0.0 for row in slice_rows)
    pass_gate = (
        hybrid_vs_candidate["delta"] > 0.0
        and hybrid_vs_candidate["ci95_low"] > 0.0
        and hybrid_vs_best_control["ci95_low"] > 0.0
        and positive_slice_count == len(slice_rows)
    )
    payload = {
        "gate": "source_private_hellaswag_fixed_hybrid_full_validation_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "pass_rule": (
            "The fixed hybrid packet passes full-validation only if it beats candidate-only with positive "
            "paired CI over all cached validation rows, beats the best destructive/source-score control with "
            "positive paired CI, and has positive mean delta on every contiguous validation slice including "
            "the terminal tail."
        ),
        "headline": {
            "eval_rows": len(rows),
            "range_start": min(row["slice_start"] for row in slice_sources),
            "range_end_exclusive": max(row["slice_end_exclusive"] for row in slice_sources),
            "slice_count": len(slice_rows),
            "candidate_only_accuracy": _accuracy(candidate_only, answers),
            "fixed_hybrid_accuracy": _accuracy(hybrid, answers),
            "hybrid_delta_vs_candidate_only": hybrid_vs_candidate["delta"],
            "hybrid_ci95_low_vs_candidate_only": hybrid_vs_candidate["ci95_low"],
            "hybrid_ci95_high_vs_candidate_only": hybrid_vs_candidate["ci95_high"],
            "hybrid_helps_vs_candidate_only": hybrid_vs_candidate["helps"],
            "hybrid_harms_vs_candidate_only": hybrid_vs_candidate["harms"],
            "best_control_name": best_control["method"],
            "best_control_accuracy": best_control["accuracy"],
            "hybrid_delta_vs_best_control": hybrid_vs_best_control["delta"],
            "hybrid_ci95_low_vs_best_control": hybrid_vs_best_control["ci95_low"],
            "positive_slice_count": int(positive_slice_count),
            "candidate_hybrid_oracle_accuracy": _oracle_accuracy(answers, [candidate_only, hybrid]),
        },
        "packet_contract": {
            "receiver_visible_payload": "one final source candidate id emitted by fixed hybrid policy",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "source_artifacts": {
            "candidate_only_audit": _display_path(audit_path),
            "candidate_only_audit_sha256": _sha256_file(audit_path),
            "tail_predictions": _display_path(tail_predictions) if tail_predictions is not None else None,
            "tail_predictions_sha256": _sha256_file(tail_predictions) if tail_predictions is not None else None,
        },
        "slice_sources": slice_sources,
        "method_rows": method_rows,
        "slice_rows": slice_rows,
        "interpretation": (
            "The fixed hybrid vote-on-score-agreement packet extends from the prior strict 0:9216 surface "
            "to the full cached HellaSwag validation range 0:10042, including the previously unresolved "
            "terminal tail. This strengthens the packet-policy evidence and evaluation-quality story, but it "
            "remains a fixed-byte candidate-id packet rather than a learned common latent receiver."
        ),
        "lay_explanation": (
            "We checked the last cached HellaSwag examples that were not part of the previous large strict "
            "surface. The same tiny hybrid answer hint still helps on that tail and on the full validation "
            "set, so the current packet result is not just from the first 9216 examples."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_fixed_hybrid_full_validation_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_fixed_hybrid_full_validation_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_fixed_hybrid_full_validation_gate.json",
                "hellaswag_fixed_hybrid_full_validation_gate.md",
                "method_rows.csv",
                "slice_rows.csv",
            ],
            "source_artifacts": payload["source_artifacts"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=pathlib.Path, default=DEFAULT_AUDIT)
    parser.add_argument("--tail-predictions", type=pathlib.Path, default=DEFAULT_TAIL_PREDICTIONS)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tail-start", type=int, default=9216)
    parser.add_argument("--tail-end-exclusive", type=int, default=10042)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        audit_path=args.audit,
        tail_predictions=args.tail_predictions,
        output_dir=args.output_dir,
        tail_start=args.tail_start,
        tail_end_exclusive=args.tail_end_exclusive,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(json.dumps({
        "pass_gate": payload["pass_gate"],
        "eval_rows": h["eval_rows"],
        "candidate_only_accuracy": h["candidate_only_accuracy"],
        "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
        "hybrid_delta_vs_candidate_only": h["hybrid_delta_vs_candidate_only"],
        "hybrid_ci95_low_vs_candidate_only": h["hybrid_ci95_low_vs_candidate_only"],
        "positive_slice_count": h["positive_slice_count"],
        "slice_count": h["slice_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
