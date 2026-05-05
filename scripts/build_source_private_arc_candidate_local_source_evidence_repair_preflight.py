from __future__ import annotations

"""ARC candidate-local source-evidence repair preflight.

This is the follow-up to the row-level token-pool readout failure.  It preserves
one source hidden vector per candidate answer, removes train-only public-text
predictable structure, and asks whether candidate-local source innovations
repair held-out ARC target errors beyond source-choice and destructive controls.
"""

import argparse
import datetime as dt
import json
import math
import pathlib
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_tokenwise_repair_readout_preflight as row_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_candidate_local_source_evidence_repair_preflight_20260505_"
    "n32_qwen05_to_qwen3"
)
DEFAULT_SOURCE_SCORE_CACHE = pathlib.Path(
    "results/source_private_arc_score_fusion_packet_probe_20260503_qwen05_qwen3_validation/"
    "source_scores.json"
)
MATCHED_CONDITION = "matched_candidate_local_source_repair"
STRICT_CONTROLS = (
    "target_only",
    "public_candidate_readout",
    "zero_source_control",
    "wrong_row_source_control",
    "source_row_shuffle_control",
    "same_source_choice_wrong_row_control",
    "candidate_source_roll_control",
    "atom_shuffle_control",
    "coefficient_shuffle_control",
    "candidate_roll_control",
    "packet_only_source_index",
    "source_rank_control",
    "source_score_control",
    "same_byte_visible_text",
)


def _candidate_design(
    rows: Sequence[arc_gate.ArcRow],
    *,
    source_candidates: np.ndarray,
    public_candidates: np.ndarray,
    use_source: bool,
    use_public: bool,
    max_choices: int,
) -> np.ndarray:
    source = np.asarray(source_candidates, dtype=np.float64)
    public = np.asarray(public_candidates, dtype=np.float64)
    offsets = row_repair._row_offsets(rows)
    if source.ndim != 2 or public.ndim != 2:
        raise ValueError("source_candidates and public_candidates must be rank-2")
    if source.shape[0] != public.shape[0] or source.shape[0] != offsets[-1][1]:
        raise ValueError("source/public candidate feature counts must match rows")

    features: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        start, end = offsets[row_index]
        for candidate_index in range(len(row.choices)):
            flat_index = start + candidate_index
            parts = [
                source[flat_index] if use_source else np.zeros(source.shape[1], dtype=np.float64),
                public[flat_index] if use_public else np.zeros(public.shape[1], dtype=np.float64),
                row_repair._one_hot(candidate_index, max_choices),
            ]
            features.append(np.concatenate(parts))
        if end - start != len(row.choices):
            raise AssertionError("candidate offset accounting failed")
    return np.asarray(features, dtype=np.float64)


def _fit_eval_scores(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    return row_repair._predict_ridge_scores(train_x, train_y, eval_x, ridge=ridge)


def _resize_candidate_block(block: np.ndarray, choice_count: int) -> np.ndarray:
    values = np.asarray(block, dtype=np.float64)
    if values.shape[0] == int(choice_count):
        return values.copy()
    if values.shape[0] > int(choice_count):
        return values[: int(choice_count)].copy()
    repeats = int(math.ceil(float(choice_count) / max(values.shape[0], 1)))
    return np.tile(values, (repeats, 1))[: int(choice_count)].copy()


def _replace_row_block(
    rows: Sequence[arc_gate.ArcRow],
    source_candidates: np.ndarray,
    *,
    target_row_index: int,
    donor_block: np.ndarray,
) -> None:
    offsets = row_repair._row_offsets(rows)
    start, end = offsets[int(target_row_index)]
    source_candidates[start:end] = _resize_candidate_block(donor_block, end - start)


def _row_block(
    rows: Sequence[arc_gate.ArcRow],
    source_candidates: np.ndarray,
    row_index: int,
) -> np.ndarray:
    offsets = row_repair._row_offsets(rows)
    start, end = offsets[int(row_index)]
    return np.asarray(source_candidates[start:end], dtype=np.float64)


def _candidate_source_controls(
    rows: Sequence[arc_gate.ArcRow],
    source_candidates: np.ndarray,
    *,
    eval_indices: Sequence[int],
    source_predictions: Sequence[int],
    seed: int,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    source = np.asarray(source_candidates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    offsets = row_repair._row_offsets(rows)
    eval_indices = [int(index) for index in eval_indices]

    controls: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    zero = source.copy()
    for row_index in eval_indices:
        start, end = offsets[row_index]
        zero[start:end] = 0.0
    controls["zero_source_control"] = (zero, {"kind": "zero_candidate_source"})

    wrong = source.copy()
    for pos, row_index in enumerate(eval_indices):
        donor_index = eval_indices[(pos - 1) % len(eval_indices)]
        _replace_row_block(rows, wrong, target_row_index=row_index, donor_block=_row_block(rows, source, donor_index))
    controls["wrong_row_source_control"] = (wrong, {"kind": "rolled_eval_row_candidate_source", "shift": 1})

    shuffled = source.copy()
    donors = eval_indices.copy()
    rng.shuffle(donors)
    for row_index, donor_index in zip(eval_indices, donors, strict=True):
        _replace_row_block(rows, shuffled, target_row_index=row_index, donor_block=_row_block(rows, source, donor_index))
    controls["source_row_shuffle_control"] = (shuffled, {"kind": "rng_eval_row_candidate_shuffle", "seed": int(seed)})

    same_choice = source.copy()
    same_choice_hits = 0
    for pos, row_index in enumerate(eval_indices):
        source_choice = int(source_predictions[row_index])
        donor_index = None
        for candidate_index in eval_indices:
            if candidate_index == row_index:
                continue
            if int(source_predictions[candidate_index]) == source_choice:
                donor_index = candidate_index
                break
        if donor_index is None:
            donor_index = eval_indices[(pos - 1) % len(eval_indices)]
        else:
            same_choice_hits += 1
        _replace_row_block(
            rows,
            same_choice,
            target_row_index=row_index,
            donor_block=_row_block(rows, source, donor_index),
        )
    controls["same_source_choice_wrong_row_control"] = (
        same_choice,
        {
            "kind": "same_source_choice_candidate_row_where_possible",
            "covered_rows": int(same_choice_hits),
            "eval_rows": int(len(eval_indices)),
        },
    )

    rolled = source.copy()
    for row_index in eval_indices:
        start, end = offsets[row_index]
        rolled[start:end] = np.roll(rolled[start:end], shift=1, axis=0)
    controls["candidate_source_roll_control"] = (
        rolled,
        {"kind": "within_row_candidate_source_roll", "shift": 1},
    )

    atom = source.copy()
    eval_flat = row_repair._candidate_indices_for_rows(rows, eval_indices)
    atom[eval_flat] = atom[eval_flat][:, rng.permutation(atom.shape[1])]
    controls["atom_shuffle_control"] = (
        atom,
        {"kind": "shared_candidate_source_dimension_permutation", "seed": int(seed)},
    )

    coeff = source.copy()
    for flat_index in eval_flat:
        rng.shuffle(coeff[int(flat_index)])
    controls["coefficient_shuffle_control"] = (
        coeff,
        {"kind": "candidatewise_source_coefficient_shuffle", "seed": int(seed)},
    )
    return controls


def _load_source_score_rows(path: pathlib.Path | None, rows: Sequence[arc_gate.ArcRow]) -> list[list[float]] | None:
    if path is None:
        return None
    resolved = row_repair._resolve(path)
    if not resolved.exists():
        return None
    data = json.loads(resolved.read_text())
    by_row_id = {
        str(row_id): [float(value) for value in scores]
        for row_id, scores in zip(data.get("row_ids", ()), data.get("source_scores", ()), strict=True)
    }
    score_rows: list[list[float]] = []
    for row in rows:
        scores = by_row_id.get(row.row_id)
        if scores is None:
            return None
        if len(scores) != len(row.choices):
            raise ValueError(f"source score length mismatch row_id={row.row_id}")
        score_rows.append(scores)
    return score_rows


def _source_score_condition_rows(
    rows: Sequence[arc_gate.ArcRow],
    eval_indices: Sequence[int],
    source_predictions: Sequence[int],
    source_score_rows: Sequence[Sequence[float]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    score_rows: list[list[float]] = []
    for row_index in eval_indices:
        raw_scores = source_score_rows[int(row_index)]
        if condition == "source_rank_control":
            score_rows.append(preflight._source_rank_scores(raw_scores))
        elif condition == "source_score_control":
            score_rows.append(preflight._centered_source_score_control(raw_scores))
        else:
            raise ValueError(f"unknown source score condition {condition!r}")
    return row_repair._prediction_rows(rows, eval_indices, condition, score_rows, source_predictions)


def evaluate_candidate_local_readout(
    rows: Sequence[arc_gate.ArcRow],
    *,
    source_candidates: np.ndarray,
    public_candidates: np.ndarray,
    source_predictions: Sequence[int],
    source_score_rows: Sequence[Sequence[float]] | None = None,
    fit_indices: Sequence[int],
    eval_indices: Sequence[int],
    audit_rows: dict[str, dict[str, dict[str, Any]]],
    ridge: float,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_choices = max(len(row.choices) for row in rows)
    labels = row_repair._candidate_labels(rows)
    fit_flat = row_repair._candidate_indices_for_rows(rows, fit_indices)
    eval_flat = row_repair._candidate_indices_for_rows(rows, eval_indices)

    matched_x = _candidate_design(
        rows,
        source_candidates=source_candidates,
        public_candidates=public_candidates,
        use_source=True,
        use_public=True,
        max_choices=max_choices,
    )
    public_x = _candidate_design(
        rows,
        source_candidates=source_candidates,
        public_candidates=public_candidates,
        use_source=False,
        use_public=True,
        max_choices=max_choices,
    )
    source_x = _candidate_design(
        rows,
        source_candidates=source_candidates,
        public_candidates=public_candidates,
        use_source=True,
        use_public=False,
        max_choices=max_choices,
    )

    condition_rows: dict[str, list[dict[str, Any]]] = {}
    matched_scores = _fit_eval_scores(
        train_x=matched_x[fit_flat],
        train_y=labels[fit_flat],
        eval_x=matched_x[eval_flat],
        ridge=ridge,
    )
    condition_rows[MATCHED_CONDITION] = row_repair._prediction_rows(
        rows,
        eval_indices,
        MATCHED_CONDITION,
        row_repair._scores_by_eval_row(rows, eval_indices, matched_scores),
        source_predictions,
    )

    public_scores = _fit_eval_scores(
        train_x=public_x[fit_flat],
        train_y=labels[fit_flat],
        eval_x=public_x[eval_flat],
        ridge=ridge,
    )
    condition_rows["public_candidate_readout"] = row_repair._prediction_rows(
        rows,
        eval_indices,
        "public_candidate_readout",
        row_repair._scores_by_eval_row(rows, eval_indices, public_scores),
        source_predictions,
    )

    source_scores = _fit_eval_scores(
        train_x=source_x[fit_flat],
        train_y=labels[fit_flat],
        eval_x=source_x[eval_flat],
        ridge=ridge,
    )
    condition_rows["source_only_readout"] = row_repair._prediction_rows(
        rows,
        eval_indices,
        "source_only_readout",
        row_repair._scores_by_eval_row(rows, eval_indices, source_scores),
        source_predictions,
    )

    control_meta: dict[str, dict[str, Any]] = {}
    for condition, (controlled_source, metadata) in _candidate_source_controls(
        rows,
        source_candidates,
        eval_indices=eval_indices,
        source_predictions=source_predictions,
        seed=seed,
    ).items():
        controlled_x = _candidate_design(
            rows,
            source_candidates=controlled_source,
            public_candidates=public_candidates,
            use_source=True,
            use_public=True,
            max_choices=max_choices,
        )
        scores = _fit_eval_scores(
            train_x=matched_x[fit_flat],
            train_y=labels[fit_flat],
            eval_x=controlled_x[eval_flat],
            ridge=ridge,
        )
        condition_rows[condition] = row_repair._prediction_rows(
            rows,
            eval_indices,
            condition,
            row_repair._scores_by_eval_row(rows, eval_indices, scores),
            source_predictions,
        )
        control_meta[condition] = metadata

    rolled_scores = []
    for row in condition_rows[MATCHED_CONDITION]:
        scores = list(row["scores"])
        rolled_scores.append(scores[-1:] + scores[:-1])
    condition_rows["candidate_roll_control"] = row_repair._prediction_rows(
        rows,
        eval_indices,
        "candidate_roll_control",
        rolled_scores,
        source_predictions,
    )
    condition_rows["target_only"] = row_repair._audit_condition_rows(
        rows, eval_indices, source_predictions, audit_rows, condition="target_only"
    )
    condition_rows["same_byte_visible_text"] = row_repair._audit_condition_rows(
        rows, eval_indices, source_predictions, audit_rows, condition="same_byte_visible_text"
    )
    condition_rows["packet_only_source_index"] = row_repair._source_index_rows(
        rows, eval_indices, source_predictions
    )
    if source_score_rows is not None:
        condition_rows["source_rank_control"] = _source_score_condition_rows(
            rows,
            eval_indices,
            source_predictions,
            source_score_rows,
            condition="source_rank_control",
        )
        condition_rows["source_score_control"] = _source_score_condition_rows(
            rows,
            eval_indices,
            source_predictions,
            source_score_rows,
            condition="source_score_control",
        )

    prediction_rows: list[dict[str, Any]] = []
    for condition in [MATCHED_CONDITION, *sorted(k for k in condition_rows if k != MATCHED_CONDITION)]:
        prediction_rows.extend(condition_rows[condition])

    condition_metrics = {
        condition: row_repair._metrics(rows_) for condition, rows_ in sorted(condition_rows.items())
    }
    matched = condition_rows[MATCHED_CONDITION]
    paired = {
        condition: row_repair._paired_bootstrap(
            matched,
            rows_,
            seed=seed + 41,
            samples=bootstrap_samples,
        )
        for condition, rows_ in sorted(condition_rows.items())
        if condition != MATCHED_CONDITION
    }
    strict_available = [condition for condition in STRICT_CONTROLS if condition in condition_metrics]
    best_control = max(strict_available, key=lambda name: condition_metrics[name]["accuracy"])
    pass_gate = all(paired[condition]["ci95_low"] > 0.0 for condition in strict_available)
    headline = {
        "fit_candidate_count": int(len(fit_flat)),
        "eval_candidate_count": int(len(eval_flat)),
        "max_choices": int(max_choices),
        "source_control_metadata": control_meta,
        "source_score_controls_available": bool(source_score_rows is not None),
        "strict_controls": strict_available,
        "best_control_by_accuracy": best_control,
        "best_control_accuracy": float(condition_metrics[best_control]["accuracy"]),
        "matched_accuracy": float(condition_metrics[MATCHED_CONDITION]["accuracy"]),
        "matched_minus_best_control_accuracy": float(
            condition_metrics[MATCHED_CONDITION]["accuracy"] - condition_metrics[best_control]["accuracy"]
        ),
        "pass_gate": bool(pass_gate),
        "paired_deltas": paired,
    }
    return {"condition_metrics": condition_metrics, "headline": headline}, prediction_rows


def _load_candidate_source_features(
    rows: list[arc_gate.ArcRow],
    *,
    fit_indices: Sequence[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_source, source_meta = preflight._hf_choice_hidden_features(
        rows,
        model_path=str(row_repair._resolve(args.source_model)),
        device=str(args.source_device),
        dtype=str(args.source_dtype),
        max_length=int(args.source_max_length),
        local_files_only=not bool(args.allow_downloads),
        hidden_layer=int(args.source_hidden_layer),
    )
    public_features, public_meta = preflight._public_candidate_hashed_features(
        rows,
        feature_dim=int(args.candidate_feature_dim),
    )
    source_features = np.asarray(raw_source, dtype=np.float64)
    innovation_meta: dict[str, Any] = {"enabled": False}
    if not bool(args.no_public_innovation):
        source_features, innovation_meta = preflight._public_candidate_innovation_features(
            source_features,
            public_features,
            fit_flat_indices=row_repair._candidate_indices_for_rows(rows, fit_indices),
            ridge=float(args.innovation_ridge),
        )
        innovation_meta = {"enabled": True, **innovation_meta}
    if bool(args.normalize_source_rows):
        norms = np.linalg.norm(source_features, axis=1, keepdims=True)
        source_features = np.divide(
            source_features,
            np.maximum(norms, 1e-12),
            out=np.zeros_like(source_features),
            where=norms > 0,
        )
    return source_features.astype(np.float64, copy=False), public_features, {
        "raw_source": source_meta,
        "public": public_meta,
        "innovation": innovation_meta,
        "normalize_source_rows": bool(args.normalize_source_rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = row_repair._resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_predictions, audit = row_repair._load_inputs(args)
    source_score_rows = _load_source_score_rows(args.source_score_cache, rows)
    fit_indices = list(range(int(args.fit_rows)))
    eval_indices = list(range(int(args.fit_rows), int(args.fit_rows) + int(args.eval_rows)))
    source_candidates, public_candidates, feature_meta = _load_candidate_source_features(
        rows,
        fit_indices=fit_indices,
        args=args,
    )
    evaluation, prediction_rows = evaluate_candidate_local_readout(
        rows,
        source_candidates=source_candidates,
        public_candidates=public_candidates,
        source_predictions=source_predictions,
        source_score_rows=source_score_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        audit_rows=audit,
        ridge=float(args.ridge),
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
    )

    average_choices = float(np.mean([len(row.choices) for row in rows])) if rows else 0.0
    dense_bytes_per_row = float(average_choices * source_candidates.shape[1] * 4)
    atom_id_bits = int(math.ceil(math.log2(max(source_candidates.shape[1], 2))))
    sparse_bits_per_candidate = int(args.sparse_packet_top_k) * (
        atom_id_bits + int(args.sparse_packet_bits)
    )
    systems = {
        "diagnostic_not_final_packet": True,
        "candidate_source_feature_dim": int(source_candidates.shape[1]),
        "candidate_count": int(source_candidates.shape[0]),
        "average_choices_per_row": average_choices,
        "dense_source_fp32_bytes_per_row": dense_bytes_per_row,
        "hypothetical_topk_sparse_proxy_bits_per_candidate": int(sparse_bits_per_candidate),
        "hypothetical_topk_sparse_proxy_bytes_per_row": float(
            sparse_bits_per_candidate * average_choices / 8.0
        ),
        "sparse_proxy_top_k": int(args.sparse_packet_top_k),
        "sparse_proxy_coeff_bits": int(args.sparse_packet_bits),
        "sparse_proxy_atom_id_bits": int(atom_id_bits),
    }
    result = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "implementation_gate_only": True,
        "benchmark": "ARC-Challenge validation",
        "method": MATCHED_CONDITION,
        "fit_rows": int(args.fit_rows),
        "eval_rows": int(args.eval_rows),
        "row_limit": int(args.row_limit),
        "inputs": {
            "eval_path": row_repair._display(args.eval_path),
            "source_cache_path": row_repair._display(args.source_cache_path),
            "source_score_cache": row_repair._display(args.source_score_cache)
            if args.source_score_cache is not None
            else None,
            "audit_jsonl": row_repair._display(args.audit_jsonl),
        },
        "config": {
            "source_feature_mode": "hf_choice_hidden_public_innovation_candidate_local"
            if not bool(args.no_public_innovation)
            else "hf_choice_hidden_candidate_local",
            "candidate_feature_dim": int(args.candidate_feature_dim),
            "innovation_ridge": float(args.innovation_ridge),
            "ridge": float(args.ridge),
            "seed": int(args.seed),
            "bootstrap_samples": int(args.bootstrap_samples),
            "source_score_controls_available": bool(source_score_rows is not None),
        },
        "feature_metadata": {
            **feature_meta,
            "source_candidate_shape": [int(value) for value in source_candidates.shape],
            "public_candidate_shape": [int(value) for value in public_candidates.shape],
        },
        "systems": systems,
        **evaluation,
        "runtime": {
            "wall_s": float(time.perf_counter() - start),
            "peak_rss_mib": float(preflight._peak_rss_mib()),
        },
        "interpretation": (
            "Pass only if candidate-local source evidence beats all strict controls "
            "with positive paired CI. This is a signal gate, not a final low-byte packet."
        ),
    }

    result_path = output_dir / "arc_candidate_local_source_evidence_repair_preflight.json"
    prediction_path = output_dir / "prediction_audit.jsonl"
    report_path = output_dir / "arc_candidate_local_source_evidence_repair_preflight.md"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row_repair._write_jsonl(prediction_path, prediction_rows)
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    manifest = {
        "created_utc": result["created_utc"],
        "files": {
            "result": {"path": row_repair._display(result_path), "sha256": row_repair._sha256_file(result_path)},
            "prediction_audit": {
                "path": row_repair._display(prediction_path),
                "sha256": row_repair._sha256_file(prediction_path),
            },
            "report": {"path": row_repair._display(report_path), "sha256": row_repair._sha256_file(report_path)},
        },
        "inputs": result["inputs"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _markdown_report(result: dict[str, Any]) -> str:
    headline = result["headline"]
    metrics = result["condition_metrics"]
    paired = headline["paired_deltas"]
    lines = [
        "# ARC Candidate-Local Source-Evidence Repair Preflight",
        "",
        "This is an implementation gate, not a paper-positive result. It preserves candidate-local source hidden evidence after the row-level token-pool readout failed.",
        "",
        "## Headline",
        "",
        f"- pass_gate: `{headline['pass_gate']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.4f}`",
        f"- best strict control: `{headline['best_control_by_accuracy']}` at `{headline['best_control_accuracy']:.4f}`",
        f"- matched minus best control: `{headline['matched_minus_best_control_accuracy']:.4f}`",
        "",
        "## Condition Metrics",
        "",
        "| condition | accuracy | mean margin | matched delta CI |",
        "|---|---:|---:|---:|",
    ]
    for condition, row in sorted(metrics.items()):
        if condition == MATCHED_CONDITION:
            delta = "-"
        else:
            ci = paired.get(condition, {})
            delta = f"{ci.get('mean', 0.0):+.4f} [{ci.get('ci95_low', 0.0):+.4f}, {ci.get('ci95_high', 0.0):+.4f}]"
        lines.append(
            f"| {condition} | {row['accuracy']:.4f} | {row['mean_margin']:.4f} | {delta} |"
        )
    lines.extend(
        [
            "",
            "## Systems Diagnostic",
            "",
            f"- dense candidate source bytes per row: `{result['systems']['dense_source_fp32_bytes_per_row']:.1f}`",
            f"- hypothetical top-k sparse proxy bytes per row: `{result['systems']['hypothetical_topk_sparse_proxy_bytes_per_row']:.2f}`",
            "- This run does not claim a final low-byte packet; it probes whether candidate-local source signal is worth compressing.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-path", type=pathlib.Path, default=preflight.DEFAULT_ARC_VALIDATION)
    parser.add_argument("--source-cache-path", type=pathlib.Path, default=preflight.DEFAULT_ARC_SOURCE_CACHE)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--audit-jsonl", type=pathlib.Path, default=row_repair.DEFAULT_AUDIT)
    parser.add_argument("--row-limit", type=int, default=32)
    parser.add_argument("--fit-rows", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--source-model", type=pathlib.Path, default=pathlib.Path(preflight.DEFAULT_QWEN_SOURCE))
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--source-dtype", default="float32")
    parser.add_argument("--source-max-length", type=int, default=160)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--candidate-feature-dim", type=int, default=128)
    parser.add_argument("--innovation-ridge", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--sparse-packet-top-k", type=int, default=8)
    parser.add_argument("--sparse-packet-bits", type=int, default=8)
    parser.add_argument("--normalize-source-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-public-innovation", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
