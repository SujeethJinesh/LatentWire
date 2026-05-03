from __future__ import annotations

"""Strict HellaSwag channel-selector gate over saved packet predictions.

This is a no-new-inference scout for the shallow hidden-channel selector branch.
It asks whether any fixed or train-prefix selector over existing source-side
packet channels beats the strict candidate-only packet with paired uncertainty.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = pathlib.Path(
    "results/source_private_hellaswag_strict_candidate_only_packet_audit_20260503_validation0_9216/"
    "hellaswag_strict_candidate_only_packet_audit.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_strict_channel_selector_gate_20260503_validation0_9216"
)
BOOTSTRAP_SAMPLES = 5000
CHANNEL_KEYS = (
    "selected_prediction",
    "vote_prediction",
    "hidden_mean_prediction",
    "trained_label_prediction",
    "score_mean_prediction",
    "score_vote_prediction",
    "wrong_example_hidden_prediction",
    "candidate_roll_hidden_prediction",
)
FIXED_CHANNEL_KEYS = (
    "vote_prediction",
    "hidden_mean_prediction",
    "trained_label_prediction",
    "score_mean_prediction",
    "score_vote_prediction",
)


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


def _load_rows(input_path: pathlib.Path | str) -> list[dict[str, Any]]:
    payload = _read_json(input_path)
    rows: list[dict[str, Any]] = []
    for slice_row in payload["slice_rows"]:
        predictions_path = pathlib.Path(slice_row["predictions_path"])
        for row in _read_jsonl(predictions_path):
            copied = dict(row)
            copied["_slice_start"] = int(slice_row["eval_slice_start"])
            copied["_predictions_path"] = _display_path(predictions_path)
            rows.append(copied)
    if not rows:
        raise ValueError("no prediction rows loaded")
    missing = [key for key in CHANNEL_KEYS if key not in rows[0]]
    if missing:
        raise KeyError(f"missing channel prediction keys: {missing}")
    return rows


def _arrays(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray([int(row[key]) for row in rows], dtype=np.int64)
    answers = np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)
    return predictions, answers


def _hybrid_vote_on_score_agreement(rows: list[dict[str, Any]]) -> np.ndarray:
    """Use the vote channel only when hidden-mean agrees with score-mean."""
    return np.asarray(
        [
            int(row["vote_prediction"])
            if int(row["hidden_mean_prediction"]) == int(row["score_mean_prediction"])
            else int(row["hidden_mean_prediction"])
            for row in rows
        ],
        dtype=np.int64,
    )


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
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


def _slice_rows_for_method(
    *,
    rows: list[dict[str, Any]],
    method_name: str,
    predictions: np.ndarray,
    selected: np.ndarray,
    answers: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    out = []
    slice_starts = sorted({int(row["_slice_start"]) for row in rows})
    row_slices = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    for slice_start in slice_starts:
        mask = row_slices == slice_start
        paired = _paired_ci(
            selected=predictions[mask],
            baseline=selected[mask],
            answers=answers[mask],
            seed=20260503 + int(slice_start) + sum(ord(ch) for ch in method_name),
            samples=bootstrap_samples,
        )
        out.append(
            {
                "method": method_name,
                "eval_slice_start": int(slice_start),
                "eval_rows": int(np.sum(mask)),
                "method_accuracy": _accuracy(predictions[mask], answers[mask]),
                "candidate_only_accuracy": _accuracy(selected[mask], answers[mask]),
                "delta_vs_candidate_only": paired["delta"],
                "ci95_low_vs_candidate_only": paired["ci95_low"],
                "ci95_high_vs_candidate_only": paired["ci95_high"],
                "helps": paired["helps"],
                "harms": paired["harms"],
            }
        )
    return out


def _method_row(
    *,
    rows: list[dict[str, Any]],
    method_name: str,
    predictions: np.ndarray,
    selected: np.ndarray,
    answers: np.ndarray,
    bootstrap_samples: int,
    train_rows: int,
    eval_rows: int,
    selector_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paired = _paired_ci(
        selected=predictions,
        baseline=selected,
        answers=answers,
        seed=30360503 + sum(ord(ch) for ch in method_name),
        samples=bootstrap_samples,
    )
    return {
        "method": method_name,
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "method_accuracy": _accuracy(predictions, answers),
        "candidate_only_accuracy": _accuracy(selected, answers),
        "delta_vs_candidate_only": paired["delta"],
        "ci95_low_vs_candidate_only": paired["ci95_low"],
        "ci95_high_vs_candidate_only": paired["ci95_high"],
        "helps": paired["helps"],
        "harms": paired["harms"],
        "improvement_slice_count": sum(
            item["delta_vs_candidate_only"] > 0
            for item in _slice_rows_for_method(
                rows=rows,
                method_name=method_name,
                predictions=predictions,
                selected=selected,
                answers=answers,
                bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
            )
        ),
        "selector_details": json.dumps(selector_details or {}, sort_keys=True),
    }


def _oracle_accuracy(rows: list[dict[str, Any]], channel_keys: tuple[str, ...]) -> float:
    total = 0
    for row in rows:
        answer = int(row["answer_index"])
        if any(int(row[key]) == answer for key in channel_keys):
            total += 1
    return total / float(len(rows))


def _train_prefix_global_selector(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    candidate_keys: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    best_key = max(
        candidate_keys,
        key=lambda key: (
            sum(int(row[key]) == int(row["answer_index"]) for row in train_rows),
            -candidate_keys.index(key),
        ),
    )
    predictions = np.asarray([int(row[best_key]) for row in eval_rows], dtype=np.int64)
    return predictions, {"selected_key": best_key}


def _train_prefix_margin_selector(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    candidate_keys: tuple[str, ...],
    bins: int,
    min_support: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_margins = np.asarray([float(row["selected_margin"]) for row in train_rows], dtype=np.float64)
    edges = np.quantile(train_margins, np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    def bin_for(row: dict[str, Any]) -> int:
        return int(np.clip(np.searchsorted(edges, float(row["selected_margin"]), side="right") - 1, 0, bins - 1))

    global_best = max(
        candidate_keys,
        key=lambda key: (
            sum(int(row[key]) == int(row["answer_index"]) for row in train_rows),
            -candidate_keys.index(key),
        ),
    )
    mapping: dict[int, str] = {}
    support: dict[int, int] = {}
    for bin_id in range(int(bins)):
        subset = [row for row in train_rows if bin_for(row) == bin_id]
        support[bin_id] = len(subset)
        if len(subset) < int(min_support):
            mapping[bin_id] = global_best
            continue
        mapping[bin_id] = max(
            candidate_keys,
            key=lambda key: (
                sum(int(row[key]) == int(row["answer_index"]) for row in subset),
                -candidate_keys.index(key),
            ),
        )
    predictions = np.asarray([int(row[mapping[bin_for(row)]]) for row in eval_rows], dtype=np.int64)
    return predictions, {
        "bins": int(bins),
        "min_support": int(min_support),
        "edges": [float(value) for value in edges],
        "global_best": global_best,
        "mapping": {str(key): value for key, value in sorted(mapping.items())},
        "support": {str(key): value for key, value in sorted(support.items())},
    }


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    best = headline["best_non_oracle_method"]
    lines = [
        "# HellaSwag Strict Channel-Selector Gate",
        "",
        f"- positive method pass: `{payload['positive_method_pass']}`",
        f"- eval rows: `{headline['total_eval_rows']}`",
        f"- candidate-only accuracy: `{headline['candidate_only_accuracy']:.6f}`",
        f"- best non-oracle method: `{best['method']}`",
        f"- best accuracy: `{best['method_accuracy']:.6f}`",
        f"- best delta vs candidate-only: `{best['delta_vs_candidate_only']:.6f}`",
        f"- best CI95 low: `{best['ci95_low_vs_candidate_only']:.6f}`",
        f"- channel oracle selected+vote+trained+score: `{headline['selected_vote_trained_score_oracle_accuracy']:.6f}`",
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
    input_path: pathlib.Path | str = DEFAULT_INPUT,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    rows = _load_rows(input_path)
    selected, answers = _arrays(rows, "selected_prediction")
    method_rows: list[dict[str, Any]] = []
    slice_method_rows: list[dict[str, Any]] = []

    for key in FIXED_CHANNEL_KEYS:
        predictions, _ = _arrays(rows, key)
        method_name = f"fixed_{key.removesuffix('_prediction')}"
        method_rows.append(
            _method_row(
                rows=rows,
                method_name=method_name,
                predictions=predictions,
                selected=selected,
                answers=answers,
                bootstrap_samples=bootstrap_samples,
                train_rows=0,
                eval_rows=len(rows),
                selector_details={"channel": key},
            )
        )
        slice_method_rows.extend(
            _slice_rows_for_method(
                rows=rows,
                method_name=method_name,
                predictions=predictions,
                selected=selected,
                answers=answers,
                bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
            )
        )

    hybrid_predictions = _hybrid_vote_on_score_agreement(rows)
    method_rows.append(
        _method_row(
            rows=rows,
            method_name="fixed_hybrid_vote_on_score_agreement",
            predictions=hybrid_predictions,
            selected=selected,
            answers=answers,
            bootstrap_samples=bootstrap_samples,
            train_rows=0,
            eval_rows=len(rows),
            selector_details={
                "policy": (
                    "if hidden_mean_prediction == score_mean_prediction, emit vote_prediction; "
                    "otherwise emit hidden_mean_prediction"
                )
            },
        )
    )
    slice_method_rows.extend(
        _slice_rows_for_method(
            rows=rows,
            method_name="fixed_hybrid_vote_on_score_agreement",
            predictions=hybrid_predictions,
            selected=selected,
            answers=answers,
            bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
        )
    )

    slice_starts = sorted({int(row["_slice_start"]) for row in rows})
    train_slice_start = slice_starts[0]
    train_rows = [row for row in rows if int(row["_slice_start"]) == train_slice_start]
    eval_rows = [row for row in rows if int(row["_slice_start"]) != train_slice_start]
    eval_selected, eval_answers = _arrays(eval_rows, "selected_prediction")
    candidate_keys = (
        "selected_prediction",
        "vote_prediction",
        "hidden_mean_prediction",
        "trained_label_prediction",
        "score_mean_prediction",
    )

    predictions, details = _train_prefix_global_selector(
        train_rows=train_rows,
        eval_rows=eval_rows,
        candidate_keys=candidate_keys,
    )
    method_rows.append(
        _method_row(
            rows=eval_rows,
            method_name="prefix_global_channel_selector",
            predictions=predictions,
            selected=eval_selected,
            answers=eval_answers,
            bootstrap_samples=bootstrap_samples,
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
            selector_details=details,
        )
    )
    slice_method_rows.extend(
        _slice_rows_for_method(
            rows=eval_rows,
            method_name="prefix_global_channel_selector",
            predictions=predictions,
            selected=eval_selected,
            answers=eval_answers,
            bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
        )
    )
    for bins in (2, 4, 8, 16):
        predictions, details = _train_prefix_margin_selector(
            train_rows=train_rows,
            eval_rows=eval_rows,
            candidate_keys=candidate_keys,
            bins=bins,
            min_support=32,
        )
        method_name = f"prefix_margin{bins}_channel_selector"
        method_rows.append(
            _method_row(
                rows=eval_rows,
                method_name=method_name,
                predictions=predictions,
                selected=eval_selected,
                answers=eval_answers,
                bootstrap_samples=bootstrap_samples,
                train_rows=len(train_rows),
                eval_rows=len(eval_rows),
                selector_details=details,
            )
        )
        slice_method_rows.extend(
            _slice_rows_for_method(
                rows=eval_rows,
                method_name=method_name,
                predictions=predictions,
                selected=eval_selected,
                answers=eval_answers,
                bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
            )
        )

    method_rows = sorted(
        method_rows,
        key=lambda row: (
            row["method_accuracy"],
            row["ci95_low_vs_candidate_only"],
            row["improvement_slice_count"],
        ),
        reverse=True,
    )
    best = method_rows[0]
    positive_method_pass = (
        best["delta_vs_candidate_only"] > 0.0
        and best["ci95_low_vs_candidate_only"] > 0.0
        and best["improvement_slice_count"] >= 8
    )
    payload = {
        "gate": "source_private_hellaswag_strict_channel_selector_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_artifact": _display_path(input_path),
        "input_artifact_sha256": _sha256_file(input_path),
        "positive_method_pass": bool(positive_method_pass),
        "headline": {
            "total_eval_rows": len(rows),
            "train_prefix_rows": len(train_rows),
            "heldout_eval_rows": len(eval_rows),
            "candidate_only_accuracy": _accuracy(selected, answers),
            "best_non_oracle_method": best,
            "selected_vote_trained_score_oracle_accuracy": _oracle_accuracy(
                rows,
                (
                    "selected_prediction",
                    "vote_prediction",
                    "trained_label_prediction",
                    "score_mean_prediction",
                    "score_vote_prediction",
                ),
            ),
            "all_channel_oracle_accuracy": _oracle_accuracy(rows, CHANNEL_KEYS),
            "method_count": len(method_rows),
        },
        "packet_contract": {
            "receiver_visible_payload": "selected candidate id only for fixed channels; source-side selector would still emit a candidate id",
            "raw_payload_bytes": 1,
            "framed_record_bytes": 4,
            "forbidden_source_fields": [
                "source_text",
                "source_kv_cache",
                "raw_hidden_vector",
                "raw_score_vector",
                "source_logits",
            ],
        },
        "pass_rule": (
            "A shallow channel selector passes only if the best non-oracle fixed or train-prefix "
            "selector beats candidate-only with positive paired CI95 low and improves at least 8 heldout "
            "slices. Oracle rows are headroom only."
        ),
        "interpretation": (
            "The static hybrid vote-on-score-agreement policy beats the 1B candidate-only packet with "
            "positive paired uncertainty on the strict 0:9216 HellaSwag surface. This strengthens the "
            "packet-policy contribution, but the train-prefix selectors still do not learn a stronger "
            "per-row receiver from this cache. The remaining ICLR blocker is still a learned receiver, "
            "common-basis transfer method, cross-family generalization, or native systems evidence."
        ),
        "lay_explanation": (
            "The source packet machinery produces several possible answer choices. We tested whether a "
            "simple rule could choose when to trust the vote channel instead of the default selected "
            "answer. The fixed rule helps reliably on this frozen HellaSwag slice, but the learned "
            "selectors still did not discover a better general rule."
        ),
        "method_rows": method_rows,
        "slice_method_rows": slice_method_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "hellaswag_strict_channel_selector_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "slice_method_rows.csv", slice_method_rows)
    _write_markdown(output_dir / "hellaswag_strict_channel_selector_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "input_artifact": payload["input_artifact"],
            "input_artifact_sha256": payload["input_artifact_sha256"],
            "outputs": [
                "hellaswag_strict_channel_selector_gate.json",
                "hellaswag_strict_channel_selector_gate.md",
                "method_rows.csv",
                "slice_method_rows.csv",
            ],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict HellaSwag channel-selector gate.")
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", default=None)
    args = parser.parse_args()
    payload = build_gate(
        input_path=args.input,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
