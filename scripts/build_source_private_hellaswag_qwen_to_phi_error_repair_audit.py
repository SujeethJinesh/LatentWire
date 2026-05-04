from __future__ import annotations

"""Held-out target-error repair audit for Qwen-to-Phi HellaSwag packets.

This audit is not a promoted method.  It measures the decision surface that a
source-private repair packet would have to solve: when Phi is wrong, how often
does a compact Qwen packet contain the right answer, and how much of that
headroom is already available from Phi-local candidate uncertainty?
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_error_repair_audit_20260504_validation1024_2048"
)
BOOTSTRAP_SAMPLES = 5000


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


def _answers(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == np.asarray(answers, dtype=np.int64)))


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64), axis=1)[:, :k].astype(np.int64)


def _oracle_from_candidates(fallback: np.ndarray, candidate_sets: np.ndarray, answers: np.ndarray) -> np.ndarray:
    predictions = np.asarray(fallback, dtype=np.int64).copy()
    for index, answer in enumerate(np.asarray(answers, dtype=np.int64)):
        if int(answer) in {int(item) for item in candidate_sets[index]}:
            predictions[index] = int(answer)
    return predictions


def _roll_candidate_labels(predictions: np.ndarray, shift: int = 1) -> np.ndarray:
    return (np.asarray(predictions, dtype=np.int64) + int(shift)) % 4


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return denoise._paired_ci(
        selected=np.asarray(selected, dtype=np.int64),
        baseline=np.asarray(baseline, dtype=np.int64),
        answers=np.asarray(answers, dtype=np.int64),
        seed=seed,
        samples=samples,
    )


def _method_row(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    fixed_hybrid: np.ndarray,
    phi_target: np.ndarray,
    candidate_only: np.ndarray,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    source_private: bool,
    oracle_diagnostic: bool,
    details: dict[str, Any] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.int64)
    vs_fixed = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=1729,
        samples=bootstrap_samples,
    )
    vs_phi = _paired_ci(
        selected=predictions,
        baseline=phi_target,
        answers=answers,
        seed=2718,
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=3141,
        samples=bootstrap_samples,
    )
    diff_fixed = (predictions == answers).astype(np.int64) - (fixed_hybrid == answers).astype(np.int64)
    return {
        "method": name,
        "accuracy": _accuracy(predictions, answers),
        "correct": int(np.sum(predictions == answers)),
        "rows": int(len(answers)),
        "delta_vs_fixed_hybrid": float(vs_fixed["delta"]),
        "ci95_low_vs_fixed_hybrid": float(vs_fixed["ci95_low"]),
        "ci95_high_vs_fixed_hybrid": float(vs_fixed["ci95_high"]),
        "helps_vs_fixed_hybrid": int(vs_fixed["helps"]),
        "harms_vs_fixed_hybrid": int(vs_fixed["harms"]),
        "delta_vs_phi_target": float(vs_phi["delta"]),
        "ci95_low_vs_phi_target": float(vs_phi["ci95_low"]),
        "delta_vs_candidate_only": float(vs_candidate["delta"]),
        "ci95_low_vs_candidate_only": float(vs_candidate["ci95_low"]),
        "override_count_vs_fixed_hybrid": int(np.sum(predictions != fixed_hybrid)),
        "net_help_vs_fixed_hybrid": int(np.sum(diff_fixed)),
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_private": bool(source_private),
        "oracle_diagnostic": bool(oracle_diagnostic),
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _partition_rows(
    *,
    answers: np.ndarray,
    phi_target: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    qwen_top2: np.ndarray,
    phi_top2: np.ndarray,
) -> list[dict[str, Any]]:
    masks = {
        "all_eval": np.ones(len(answers), dtype=bool),
        "phi_target_wrong": phi_target != answers,
        "phi_target_wrong_fixed_hybrid_correct": (phi_target != answers) & (fixed_hybrid == answers),
        "phi_target_correct_fixed_hybrid_wrong": (phi_target == answers) & (fixed_hybrid != answers),
        "fixed_hybrid_wrong": fixed_hybrid != answers,
        "fixed_hybrid_wrong_qwen_top2_contains_gold": (fixed_hybrid != answers)
        & np.any(qwen_top2 == answers[:, None], axis=1),
        "fixed_hybrid_wrong_phi_top2_contains_gold": (fixed_hybrid != answers)
        & np.any(phi_top2 == answers[:, None], axis=1),
        "fixed_hybrid_wrong_source_unique_top2": (fixed_hybrid != answers)
        & np.any(qwen_top2 == answers[:, None], axis=1)
        & ~np.any(phi_top2 == answers[:, None], axis=1),
        "phi_wrong_fixed_wrong_qwen_top2_contains_gold": (phi_target != answers)
        & (fixed_hybrid != answers)
        & np.any(qwen_top2 == answers[:, None], axis=1),
        "candidate_wrong_fixed_correct": (candidate_only != answers) & (fixed_hybrid == answers),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        count = int(np.sum(mask))
        rows.append(
            {
                "partition": name,
                "rows": count,
                "rate": float(count / len(answers)),
                "phi_target_accuracy": _accuracy(phi_target[mask], answers[mask]) if count else None,
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]) if count else None,
                "candidate_only_accuracy": _accuracy(candidate_only[mask], answers[mask]) if count else None,
                "qwen_top2_contains_gold_rate": float(np.mean(np.any(qwen_top2[mask] == answers[mask, None], axis=1)))
                if count
                else None,
                "phi_top2_contains_gold_rate": float(np.mean(np.any(phi_top2[mask] == answers[mask, None], axis=1)))
                if count
                else None,
            }
        )
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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_audit(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    source_score_cache: pathlib.Path | str = oracle.DEFAULT_SOURCE_SCORE_CACHE,
    fit_rows_per_slice: int = denoise.FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = denoise.SELECT_ROWS_PER_SLICE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_date = run_date or dt.date.today().isoformat()
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    answers = _answers(eval_rows)
    phi_target = _field_array(eval_rows, "phi_target_prediction")
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    qwen_scores = np.asarray([row["qwen_source_scores"] for row in eval_rows], dtype=np.float64)
    phi_scores = np.asarray([row["phi_target_scores"] for row in eval_rows], dtype=np.float64)

    qwen_top2 = _topk(qwen_scores, 2)
    phi_top2 = _topk(phi_scores, 2)
    qwen_top1 = qwen_top2[:, 0]
    phi_top1 = phi_top2[:, 0]
    qwen_top2_oracle = _oracle_from_candidates(qwen_top1, qwen_top2, answers)
    fixed_or_qwen_top2_oracle = _oracle_from_candidates(fixed_hybrid, qwen_top2, answers)
    fixed_or_phi_top2_oracle = _oracle_from_candidates(fixed_hybrid, phi_top2, answers)
    fixed_or_union_top2_oracle = _oracle_from_candidates(
        fixed_hybrid,
        np.concatenate([qwen_top2, phi_top2], axis=1),
        answers,
    )

    shuffled_qwen_top2 = np.roll(qwen_top2, 1, axis=0)
    rolled_qwen_top2 = _roll_candidate_labels(qwen_top2)
    method_rows = [
        _method_row(
            name="phi_target_only",
            predictions=phi_target,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            source_private=True,
            oracle_diagnostic=False,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="fixed_qwen_hybrid_packet",
            predictions=fixed_hybrid,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            source_private=True,
            oracle_diagnostic=False,
            details={"packet": "existing fixed hybrid vote-on-score-agreement"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="qwen_candidate_only",
            predictions=candidate_only,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            source_private=True,
            oracle_diagnostic=False,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="qwen_source_score_top1",
            predictions=qwen_top1,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            source_private=True,
            oracle_diagnostic=False,
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="qwen_source_top2_oracle_diagnostic",
            predictions=qwen_top2_oracle,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            source_private=True,
            oracle_diagnostic=True,
            details={"not_promotable": "uses gold label to choose within source top2"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="fixed_hybrid_or_qwen_top2_oracle_diagnostic",
            predictions=fixed_or_qwen_top2_oracle,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            source_private=True,
            oracle_diagnostic=True,
            details={"not_promotable": "upper bound for source top2 repair over fixed hybrid"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="fixed_hybrid_or_phi_top2_oracle_diagnostic",
            predictions=fixed_or_phi_top2_oracle,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=0,
            framed_record_bytes=0,
            source_private=True,
            oracle_diagnostic=True,
            details={"not_promotable": "target-side top2 upper bound, no source packet"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="fixed_hybrid_or_union_top2_oracle_diagnostic",
            predictions=fixed_or_union_top2_oracle,
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            source_private=True,
            oracle_diagnostic=True,
            details={"not_promotable": "union of source and target top2 candidates"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="source_row_shuffle_top1_control",
            predictions=shuffled_qwen_top2[:, 0],
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            source_private=True,
            oracle_diagnostic=False,
            details={"condition": "source row order rolled before decoding"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="source_row_shuffle_top2_oracle_control",
            predictions=_oracle_from_candidates(shuffled_qwen_top2[:, 0], shuffled_qwen_top2, answers),
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=2,
            framed_record_bytes=5,
            source_private=True,
            oracle_diagnostic=True,
            details={"condition": "source row order rolled before oracle; chance overlap diagnostic"},
            bootstrap_samples=bootstrap_samples,
        ),
        _method_row(
            name="candidate_roll_top1_control",
            predictions=rolled_qwen_top2[:, 0],
            answers=answers,
            fixed_hybrid=fixed_hybrid,
            phi_target=phi_target,
            candidate_only=candidate_only,
            raw_payload_bytes=1,
            framed_record_bytes=4,
            source_private=True,
            oracle_diagnostic=False,
            details={"condition": "candidate ids shifted by one"},
            bootstrap_samples=bootstrap_samples,
        ),
    ]
    partition_rows = _partition_rows(
        answers=answers,
        phi_target=phi_target,
        fixed_hybrid=fixed_hybrid,
        candidate_only=candidate_only,
        qwen_top2=qwen_top2,
        phi_top2=phi_top2,
    )
    source_unique = next(row for row in partition_rows if row["partition"] == "fixed_hybrid_wrong_source_unique_top2")
    fixed_wrong_source = next(row for row in partition_rows if row["partition"] == "fixed_hybrid_wrong_qwen_top2_contains_gold")
    phi_wrong = next(row for row in partition_rows if row["partition"] == "phi_target_wrong")
    headline = {
        "eval_rows": int(len(eval_rows)),
        "phi_target_accuracy": _accuracy(phi_target, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "qwen_candidate_only_accuracy": _accuracy(candidate_only, answers),
        "qwen_source_score_top1_accuracy": _accuracy(qwen_top1, answers),
        "qwen_source_top2_oracle_accuracy": _accuracy(qwen_top2_oracle, answers),
        "fixed_hybrid_or_qwen_top2_oracle_accuracy": _accuracy(fixed_or_qwen_top2_oracle, answers),
        "fixed_hybrid_or_phi_top2_oracle_accuracy": _accuracy(fixed_or_phi_top2_oracle, answers),
        "fixed_hybrid_or_union_top2_oracle_accuracy": _accuracy(fixed_or_union_top2_oracle, answers),
        "fixed_hybrid_wrong_qwen_top2_repair_rows": int(fixed_wrong_source["rows"]),
        "fixed_hybrid_wrong_source_unique_top2_rows": int(source_unique["rows"]),
        "fixed_hybrid_wrong_source_unique_top2_rate": float(source_unique["rate"]),
        "phi_target_wrong_rows": int(phi_wrong["rows"]),
        "source_unique_share_of_fixed_hybrid_errors": float(
            int(source_unique["rows"]) / max(1, int(np.sum(fixed_hybrid != answers)))
        ),
        "target_error_branch_alive": bool(int(source_unique["rows"]) >= 12),
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_error_repair_audit",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "headline": headline,
        "pass_gate": False,
        "audit_only": True,
        "slices": metadata,
        "source_score_metadata": source_score_metadata,
        "packet_contract": {
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "source_hidden_vector_exposed": False,
            "raw_scores_or_logits_transmitted": False,
            "smallest_source_top2_raw_payload_bytes": 2,
            "smallest_source_top2_framed_record_bytes": 5,
        },
        "fit_rows_per_slice": int(fit_rows_per_slice),
        "select_rows_per_slice": int(select_rows_per_slice),
        "eval_policy": "held-out rows only after fit/select prefix of each cached slice",
        "method_rows": method_rows,
        "partition_rows": partition_rows,
        "decision": {
            "promoted": [
                "target-error-conditioned source top2 repair remains alive"
            ]
            if headline["target_error_branch_alive"]
            else [],
            "weakened": [
                "unconditioned source-score/top2 receivers remain saturated",
                "fixed hybrid still beats Phi but is not a sufficient ICLR method without source-specific repair",
            ],
            "next_gate": (
                "Train a harm-controlled receiver only on fixed-hybrid-error/source-top2-headroom rows, "
                "and require it to beat fixed hybrid, Phi-top2 target-only oracle controls, row-shuffle, "
                "candidate-roll, and source-index/rank controls."
            ),
        },
    }
    json_path = output_dir / "hellaswag_qwen_to_phi_error_repair_audit.json"
    method_path = output_dir / "method_rows.csv"
    partition_path = output_dir / "partition_rows.csv"
    md_path = output_dir / "hellaswag_qwen_to_phi_error_repair_audit.md"
    _write_json(json_path, payload)
    _write_csv(method_path, method_rows)
    _write_csv(partition_path, partition_rows)
    md_lines = [
        "# Qwen-to-Phi Target-Error Repair Audit",
        "",
        f"- audit only: `{payload['audit_only']}`",
        f"- eval rows: `{headline['eval_rows']}`",
        f"- Phi target-only accuracy: `{headline['phi_target_accuracy']:.6f}`",
        f"- fixed Qwen-hybrid packet accuracy: `{headline['fixed_hybrid_accuracy']:.6f}`",
        f"- Qwen source top-2 oracle accuracy: `{headline['qwen_source_top2_oracle_accuracy']:.6f}`",
        f"- fixed-hybrid or Qwen top-2 oracle accuracy: `{headline['fixed_hybrid_or_qwen_top2_oracle_accuracy']:.6f}`",
        f"- fixed-hybrid or Phi top-2 oracle accuracy: `{headline['fixed_hybrid_or_phi_top2_oracle_accuracy']:.6f}`",
        f"- fixed-hybrid errors with source-unique top-2 repair: `{headline['fixed_hybrid_wrong_source_unique_top2_rows']}`",
        f"- target-error branch alive: `{headline['target_error_branch_alive']}`",
        "",
        "## Interpretation",
        "",
        "The useful next branch is not another unconditioned score receiver. The audit asks whether a tiny source",
        "packet can act like an error-correcting syndrome for the subset of rows where the target-side decision",
        "is wrong and source top-2 still contains the gold candidate. A future method must convert this oracle",
        "headroom into held-out overrides while beating Phi-local top-2 and source-destroying controls.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {
                "path": _display_path(path),
                "sha256": _sha256_file(path),
                "bytes": _resolve(path).stat().st_size,
            }
            for path in (json_path, md_path, method_path, partition_path)
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen-to-Phi target-error repair audit.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=oracle.DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--fit-rows-per-slice", type=int, default=denoise.FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=denoise.SELECT_ROWS_PER_SLICE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_audit(
        output_dir=args.output_dir,
        source_score_cache=args.source_score_cache,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
