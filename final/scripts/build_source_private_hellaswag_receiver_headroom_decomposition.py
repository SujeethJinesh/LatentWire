from __future__ import annotations

"""Decompose cross-family receiver headroom for HellaSwag fixed-byte packets."""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_receiver_headroom_decomposition_20260502")
DEFAULT_TINY_PACKET_JSONL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "bagged_gate/predictions.jsonl"
)
DEFAULT_TINY_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "hellaswag_hidden_innovation_eval_slice_stress.json"
)
DEFAULT_QWEN_PACKET_JSONL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/predictions.jsonl"
)
DEFAULT_QWEN_GLOBAL_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/"
    "hellaswag_hidden_innovation_global_stability.json"
)

DEFAULT_TINY_FIELD = "selected_prediction"
DEFAULT_TINY_MARGIN_FIELD = "selected_margin"
DEFAULT_QWEN_FIELDS = (
    "mean_zscore_prediction",
    "hybrid_vote_on_score_agreement_prediction",
)
DEFAULT_CONTROL_FIELDS = (
    "wrong_example_hidden_prediction",
    "candidate_roll_hidden_prediction",
    "zero_hidden_prediction",
    "source_label_prediction",
)

HEADROOM_DELTA = 0.02
SELECTOR_DELTA = 0.005
RAW_PACKET_BYTES = 2
FRAMED_PACKET_BYTES = 5


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_target_score_cache(global_artifact: pathlib.Path | str) -> dict[str, Any]:
    artifact = _read_json(global_artifact)
    row_ids: list[str] = []
    scores: list[list[float]] = []
    predictions: list[int] = []
    slices: list[dict[str, Any]] = []
    for item in artifact.get("eval_slices", []):
        cache_path = _resolve(item["score_cache"])
        cache = _read_json(cache_path)
        row_ids.extend(str(value) for value in cache["row_ids"])
        scores.extend(cache["source_scores"])
        predictions.extend(int(value) for value in cache["source_predictions"])
        slices.append(
            {
                "name": item.get("name"),
                "start": item.get("start"),
                "end": item.get("end"),
                "score_cache": _display_path(cache_path),
                "score_cache_sha256": _sha256_file(cache_path),
                "score_cache_bytes": cache_path.stat().st_size,
                "rows": cache.get("row_count"),
            }
        )
    if not scores:
        raise ValueError(f"no eval_slices score caches found in {global_artifact}")
    return {
        "row_ids": row_ids,
        "scores": np.asarray(scores, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "slices": slices,
        "artifact_path": _display_path(global_artifact),
        "artifact_sha256": _sha256_file(global_artifact),
    }


def _accuracy(predictions: np.ndarray, answers: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) == 0:
        raise ValueError("cannot score an empty split")
    return float(np.mean(predictions[indices] == answers[indices]))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    indices: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    delta = (selected[indices] == answers[indices]).astype(np.float64) - (
        baseline[indices] == answers[indices]
    ).astype(np.float64)
    if int(samples) <= 0:
        return {
            "delta": float(np.mean(delta)),
            "ci95_low": float(np.mean(delta)),
            "ci95_high": float(np.mean(delta)),
        }
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    boot = np.mean(delta[boot_indices], axis=1)
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _target_margin(scores: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def _oracle_predictions(predictions: list[np.ndarray], answers: np.ndarray) -> np.ndarray:
    correct = np.zeros_like(answers, dtype=bool)
    for pred in predictions:
        correct |= pred == answers
    return np.where(correct, answers, predictions[0]).astype(np.int64)


def _read_prediction(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    if field not in rows[0]:
        raise ValueError(f"missing prediction field: {field}")
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _baseline_row(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    packet_predictions: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ci_packet = _paired_ci(
        selected=predictions,
        baseline=packet_predictions,
        answers=answers,
        indices=eval_indices,
        seed=seed,
        samples=bootstrap_samples,
    )
    return {
        "name": name,
        "train_accuracy": _accuracy(predictions, answers, train_indices),
        "eval_accuracy": _accuracy(predictions, answers, eval_indices),
        "delta_vs_packet_only": ci_packet["delta"],
        "ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "ci95_high_vs_packet_only": ci_packet["ci95_high"],
    }


def _overlap_row(
    *,
    name_a: str,
    pred_a: np.ndarray,
    name_b: str,
    pred_b: np.ndarray,
    answers: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, Any]:
    a = pred_a[eval_indices] == answers[eval_indices]
    b = pred_b[eval_indices] == answers[eval_indices]
    both = a & b
    a_only = a & ~b
    b_only = b & ~a
    neither = ~a & ~b
    a_wrong = ~a
    b_wrong = ~b
    return {
        "name_a": name_a,
        "name_b": name_b,
        "eval_rows": int(len(eval_indices)),
        "both_correct_count": int(np.sum(both)),
        "a_only_correct_count": int(np.sum(a_only)),
        "b_only_correct_count": int(np.sum(b_only)),
        "both_wrong_count": int(np.sum(neither)),
        "both_correct_rate": float(np.mean(both)),
        "a_only_correct_rate": float(np.mean(a_only)),
        "b_only_correct_rate": float(np.mean(b_only)),
        "both_wrong_rate": float(np.mean(neither)),
        "b_correct_given_a_wrong": float(np.sum(b_only) / max(1, int(np.sum(a_wrong)))),
        "a_correct_given_b_wrong": float(np.sum(a_only) / max(1, int(np.sum(b_wrong)))),
    }


def _selector_candidates(
    *,
    alternatives: dict[str, np.ndarray],
    target_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> list[dict[str, Any]]:
    target_predictions = np.argmax(target_scores, axis=1).astype(np.int64)
    target_margins = _target_margin(target_scores)
    features = {
        "packet_margin": packet_margins,
        "target_margin": target_margins,
    }
    candidates: list[dict[str, Any]] = []
    for alt_name, alt_predictions in alternatives.items():
        for feature_name, values in features.items():
            thresholds = np.unique(np.quantile(values[train_indices], np.linspace(0.0, 1.0, 101)))
            for threshold in thresholds:
                for direction in ("low", "high"):
                    if direction == "low":
                        use_alt = values <= float(threshold)
                    else:
                        use_alt = values >= float(threshold)
                    use_alt = use_alt & (alt_predictions != packet_predictions)
                    predictions = np.where(use_alt, alt_predictions, packet_predictions).astype(np.int64)
                    candidates.append(
                        {
                            "kind": "threshold_override",
                            "alternative": alt_name,
                            "feature": feature_name,
                            "direction": direction,
                            "threshold": float(threshold),
                            "train_accuracy": _accuracy(predictions, answers, train_indices),
                            "eval_accuracy": _accuracy(predictions, answers, eval_indices),
                            "train_override_rate": float(np.mean(use_alt[train_indices])),
                            "eval_override_rate": float(np.mean(use_alt[eval_indices])),
                            "predictions": predictions,
                        }
                    )
        agree_with_target = (alt_predictions == target_predictions) & (alt_predictions != packet_predictions)
        agree_predictions = np.where(agree_with_target, alt_predictions, packet_predictions).astype(np.int64)
        candidates.append(
            {
                "kind": "target_agreement_override",
                "alternative": alt_name,
                "feature": "alternative_equals_target_top1",
                "direction": "boolean",
                "threshold": None,
                "train_accuracy": _accuracy(agree_predictions, answers, train_indices),
                "eval_accuracy": _accuracy(agree_predictions, answers, eval_indices),
                "train_override_rate": float(np.mean(agree_with_target[train_indices])),
                "eval_override_rate": float(np.mean(agree_with_target[eval_indices])),
                "predictions": agree_predictions,
            }
        )
    selected_train = max(
        candidates,
        key=lambda item: (
            item["train_accuracy"],
            -item["train_override_rate"],
            item["kind"] == "target_agreement_override",
            item["alternative"],
        ),
    )
    selected_eval = max(
        candidates,
        key=lambda item: (
            item["eval_accuracy"],
            item["train_accuracy"],
            -item["eval_override_rate"],
            item["alternative"],
        ),
    )
    for item in candidates:
        item["selected_by_train_prefix"] = item is selected_train
        item["selected_by_eval_diagnostic"] = item is selected_eval
    return candidates


def _strip_predictions(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "predictions"}


def _top_rows(rows: list[dict[str, Any]], *, key: str, limit: int) -> list[dict[str, Any]]:
    return [_strip_predictions(row) for row in sorted(rows, key=lambda item: item[key], reverse=True)[:limit]]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Receiver Headroom Decomposition",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- receiver headroom gate: `{payload['receiver_headroom_gate']}`",
        f"- train-selected selector improvement gate: `{payload['train_selected_selector_improvement_gate']}`",
        f"- train/eval split: validation `0:{h['train_rows']}` train, `{h['train_rows']}:{h['row_count']}` eval",
        f"- TinyLlama packet-only eval accuracy: `{h['tiny_packet_eval_accuracy']:.6f}`",
        f"- Qwen target-score eval accuracy: `{h['qwen_target_score_eval_accuracy']:.6f}`",
        f"- best Tiny+Qwen oracle eval accuracy: `{h['best_oracle_eval_accuracy']:.6f}`",
        f"- best oracle delta vs packet-only: `{h['best_oracle_delta_vs_packet_only']:.6f}`",
        f"- train-selected simple selector eval accuracy: `{h['train_selected_selector_eval_accuracy']:.6f}`",
        f"- train-selected simple selector delta vs packet-only: `{h['train_selected_selector_delta_vs_packet_only']:.6f}`",
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_decomposition(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    tiny_packet_jsonl: pathlib.Path = DEFAULT_TINY_PACKET_JSONL,
    tiny_artifact: pathlib.Path = DEFAULT_TINY_ARTIFACT,
    qwen_packet_jsonl: pathlib.Path = DEFAULT_QWEN_PACKET_JSONL,
    qwen_global_artifact: pathlib.Path = DEFAULT_QWEN_GLOBAL_ARTIFACT,
    tiny_field: str = DEFAULT_TINY_FIELD,
    tiny_margin_field: str = DEFAULT_TINY_MARGIN_FIELD,
    qwen_fields: tuple[str, ...] = DEFAULT_QWEN_FIELDS,
    control_fields: tuple[str, ...] = DEFAULT_CONTROL_FIELDS,
    train_prefix_rows: int = 1024,
    bootstrap_samples: int = 1000,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    load_start = time.perf_counter()
    tiny_rows = _read_jsonl(tiny_packet_jsonl)
    qwen_rows = _read_jsonl(qwen_packet_jsonl)
    target = _load_target_score_cache(qwen_global_artifact)
    load_wall_time_s = time.perf_counter() - load_start

    if len(tiny_rows) != len(qwen_rows):
        raise ValueError("TinyLlama and Qwen prediction rows differ in length")
    row_ids = [str(row["row_id"]) for row in tiny_rows]
    qwen_row_ids = [str(row["row_id"]) for row in qwen_rows]
    if row_ids != qwen_row_ids:
        raise ValueError("TinyLlama and Qwen prediction rows are not aligned")
    if row_ids != target["row_ids"]:
        raise ValueError("packet prediction rows and Qwen target-score rows are not aligned")

    answers = np.asarray([int(row["answer_index"]) for row in tiny_rows], dtype=np.int64)
    qwen_answers = np.asarray([int(row["answer_index"]) for row in qwen_rows], dtype=np.int64)
    if not np.array_equal(answers, qwen_answers):
        raise ValueError("TinyLlama and Qwen answer rows disagree")
    if not 0 < int(train_prefix_rows) < len(tiny_rows):
        raise ValueError("train_prefix_rows must leave at least one heldout eval row")

    train_indices = np.arange(int(train_prefix_rows), dtype=np.int64)
    eval_indices = np.arange(int(train_prefix_rows), len(tiny_rows), dtype=np.int64)

    tiny_packet = _read_prediction(tiny_rows, tiny_field)
    tiny_margins = np.asarray([float(row.get(tiny_margin_field, 0.0)) for row in tiny_rows], dtype=np.float64)
    qwen_target_score = np.argmax(target["scores"], axis=1).astype(np.int64)
    qwen_packets = {field: _read_prediction(qwen_rows, field) for field in qwen_fields}
    alternatives = {"qwen_target_score": qwen_target_score, **qwen_packets}

    selector_start = time.perf_counter()
    selector_candidates = _selector_candidates(
        alternatives=alternatives,
        target_scores=target["scores"],
        packet_predictions=tiny_packet,
        packet_margins=tiny_margins,
        answers=answers,
        train_indices=train_indices,
        eval_indices=eval_indices,
    )
    selector_wall_time_s = time.perf_counter() - selector_start
    train_selected = next(row for row in selector_candidates if row["selected_by_train_prefix"])
    eval_selected = next(row for row in selector_candidates if row["selected_by_eval_diagnostic"])

    prediction_rows: list[tuple[str, np.ndarray]] = [
        ("tiny_packet_only", tiny_packet),
        ("qwen_target_score", qwen_target_score),
    ]
    prediction_rows.extend((field, pred) for field, pred in qwen_packets.items())
    for field in control_fields:
        if field in tiny_rows[0]:
            prediction_rows.append((field, _read_prediction(tiny_rows, field)))

    baselines = [
        _baseline_row(
            name=name,
            predictions=predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
            packet_predictions=tiny_packet,
            seed=7000 + index,
            bootstrap_samples=bootstrap_samples,
        )
        for index, (name, predictions) in enumerate(prediction_rows)
    ]

    oracle_pairs = []
    for index, (name, predictions) in enumerate([("qwen_target_score", qwen_target_score), *qwen_packets.items()]):
        oracle = _oracle_predictions([tiny_packet, predictions], answers)
        row = _baseline_row(
            name=f"tiny_packet_or_{name}_oracle",
            predictions=oracle,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
            packet_predictions=tiny_packet,
            seed=7100 + index,
            bootstrap_samples=bootstrap_samples,
        )
        row["alternative"] = name
        oracle_pairs.append(row)
    best_oracle = max(oracle_pairs, key=lambda row: row["eval_accuracy"])

    selector_train_ci = _paired_ci(
        selected=train_selected["predictions"],
        baseline=tiny_packet,
        answers=answers,
        indices=eval_indices,
        seed=7201,
        samples=bootstrap_samples,
    )
    selector_eval_ci = _paired_ci(
        selected=eval_selected["predictions"],
        baseline=tiny_packet,
        answers=answers,
        indices=eval_indices,
        seed=7202,
        samples=bootstrap_samples,
    )
    best_oracle_ci = {
        "delta": best_oracle["delta_vs_packet_only"],
        "ci95_low": best_oracle["ci95_low_vs_packet_only"],
        "ci95_high": best_oracle["ci95_high_vs_packet_only"],
    }

    receiver_headroom_gate = bool(
        best_oracle_ci["delta"] >= HEADROOM_DELTA and best_oracle_ci["ci95_low"] > 0.0
    )
    train_selected_selector_improvement_gate = bool(
        selector_train_ci["delta"] >= SELECTOR_DELTA and selector_train_ci["ci95_low"] > 0.0
    )
    pass_gate = bool(receiver_headroom_gate and train_selected_selector_improvement_gate)

    overlap_rows = [
        _overlap_row(
            name_a="tiny_packet_only",
            pred_a=tiny_packet,
            name_b=name,
            pred_b=predictions,
            answers=answers,
            eval_indices=eval_indices,
        )
        for name, predictions in [("qwen_target_score", qwen_target_score), *qwen_packets.items()]
    ]

    packet_cache_records = {
        "raw_payload_bytes_per_request": RAW_PACKET_BYTES,
        "framed_record_bytes_per_request": FRAMED_PACKET_BYTES,
        "logical_raw_payload_bytes_total": int(RAW_PACKET_BYTES * len(tiny_rows)),
        "logical_framed_record_bytes_total": int(FRAMED_PACKET_BYTES * len(tiny_rows)),
        "single_request_cacheline_bytes": 64,
        "batch64_packed_bytes_per_request": FRAMED_PACKET_BYTES,
        "tiny_packet_jsonl_bytes": _resolve(tiny_packet_jsonl).stat().st_size,
        "qwen_packet_jsonl_bytes": _resolve(qwen_packet_jsonl).stat().st_size,
        "qwen_score_cache_bytes_total": int(sum(item["score_cache_bytes"] for item in target["slices"])),
    }
    headline = {
        "row_count": len(tiny_rows),
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "tiny_packet_eval_accuracy": _accuracy(tiny_packet, answers, eval_indices),
        "qwen_target_score_eval_accuracy": _accuracy(qwen_target_score, answers, eval_indices),
        "qwen_mean_zscore_eval_accuracy": _accuracy(
            qwen_packets.get("mean_zscore_prediction", qwen_target_score), answers, eval_indices
        ),
        "qwen_hybrid_eval_accuracy": _accuracy(
            qwen_packets.get("hybrid_vote_on_score_agreement_prediction", qwen_target_score),
            answers,
            eval_indices,
        ),
        "best_oracle_name": best_oracle["name"],
        "best_oracle_eval_accuracy": best_oracle["eval_accuracy"],
        "best_oracle_delta_vs_packet_only": best_oracle_ci["delta"],
        "best_oracle_ci95_low_vs_packet_only": best_oracle_ci["ci95_low"],
        "train_selected_selector_kind": train_selected["kind"],
        "train_selected_selector_alternative": train_selected["alternative"],
        "train_selected_selector_eval_accuracy": train_selected["eval_accuracy"],
        "train_selected_selector_delta_vs_packet_only": selector_train_ci["delta"],
        "train_selected_selector_ci95_low_vs_packet_only": selector_train_ci["ci95_low"],
        "eval_only_best_selector_eval_accuracy": eval_selected["eval_accuracy"],
        "eval_only_best_selector_delta_vs_packet_only": selector_eval_ci["delta"],
        "eval_only_best_selector_not_promotable": True,
        "packet_raw_bytes": RAW_PACKET_BYTES,
        "packet_framed_bytes": FRAMED_PACKET_BYTES,
        "native_gpu_claims_allowed": False,
    }
    lay_explanation = (
        "This experiment asks a simple question: if TinyLlama sends a tiny answer hint and Qwen also "
        "has its own guess, are their mistakes different enough that a receiver could combine them? "
        "The oracle row is the unrealistic best case that peeks at the answer and picks whichever "
        "model was right. The train-selected selector row is the fair cheap version: it learns a rule "
        "only on the first validation prefix and then applies that frozen rule to the heldout rows."
    )
    interpretation = (
        "The decomposition is diagnostic, not a promoted positive method. A positive oracle gap means "
        "Qwen and TinyLlama contain complementary candidate information, so the receiver branch is "
        "alive. Failure of the train-prefix selector means simple confidence thresholds do not recover "
        "that complementarity. The next live branch should therefore learn a common-basis or selective "
        "residual receiver under train-only selection, while retaining the fixed 2B raw / 5B framed "
        "packet boundary and destructive controls."
    )

    payload = {
        "gate": "source_private_hellaswag_receiver_headroom_decomposition",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "receiver_headroom_gate": receiver_headroom_gate,
        "train_selected_selector_improvement_gate": train_selected_selector_improvement_gate,
        "claim_boundary": (
            "This artifact supports receiver headroom and packet-sideband systems accounting. It does "
            "not claim a receiver improvement over packet-only unless the train-selected selector "
            "subgate passes; it makes no native GPU throughput claim."
        ),
        "pass_rule": (
            "Headroom requires best Tiny+Qwen oracle delta >= 0.02 over Tiny packet-only with positive "
            "paired CI95 low. Receiver promotion additionally requires the train-prefix-selected "
            "simple selector to beat Tiny packet-only by >= 0.005 with positive paired CI95 low."
        ),
        "headline": headline,
        "baselines": baselines,
        "oracle_pairs": oracle_pairs,
        "overlap_rows": overlap_rows,
        "train_selected_selector": {
            **_strip_predictions(train_selected),
            "delta_vs_packet_only": selector_train_ci["delta"],
            "ci95_low_vs_packet_only": selector_train_ci["ci95_low"],
            "ci95_high_vs_packet_only": selector_train_ci["ci95_high"],
        },
        "eval_only_best_selector_diagnostic": {
            **_strip_predictions(eval_selected),
            "not_promotable": True,
            "delta_vs_packet_only": selector_eval_ci["delta"],
            "ci95_low_vs_packet_only": selector_eval_ci["ci95_low"],
            "ci95_high_vs_packet_only": selector_eval_ci["ci95_high"],
        },
        "selector_candidates_top_by_train": _top_rows(selector_candidates, key="train_accuracy", limit=12),
        "selector_candidates_top_by_eval_diagnostic": _top_rows(selector_candidates, key="eval_accuracy", limit=12),
        "systems_packet_sideband": {
            **packet_cache_records,
            "load_wall_time_s": float(load_wall_time_s),
            "selector_wall_time_s": float(selector_wall_time_s),
            "selector_examples_per_second": float(len(tiny_rows) / max(selector_wall_time_s, 1e-12)),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
        },
        "inputs": {
            "tiny_packet_jsonl": _display_path(tiny_packet_jsonl),
            "tiny_packet_jsonl_sha256": _sha256_file(tiny_packet_jsonl),
            "tiny_artifact": _display_path(tiny_artifact),
            "tiny_artifact_sha256": _sha256_file(tiny_artifact),
            "qwen_packet_jsonl": _display_path(qwen_packet_jsonl),
            "qwen_packet_jsonl_sha256": _sha256_file(qwen_packet_jsonl),
            "qwen_global_artifact": target["artifact_path"],
            "qwen_global_artifact_sha256": target["artifact_sha256"],
            "qwen_score_slices": target["slices"],
        },
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hellaswag_receiver_headroom_decomposition.json"
    md_path = output_dir / "hellaswag_receiver_headroom_decomposition.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path)
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tiny-packet-jsonl", type=pathlib.Path, default=DEFAULT_TINY_PACKET_JSONL)
    parser.add_argument("--tiny-artifact", type=pathlib.Path, default=DEFAULT_TINY_ARTIFACT)
    parser.add_argument("--qwen-packet-jsonl", type=pathlib.Path, default=DEFAULT_QWEN_PACKET_JSONL)
    parser.add_argument("--qwen-global-artifact", type=pathlib.Path, default=DEFAULT_QWEN_GLOBAL_ARTIFACT)
    parser.add_argument("--tiny-field", default=DEFAULT_TINY_FIELD)
    parser.add_argument("--tiny-margin-field", default=DEFAULT_TINY_MARGIN_FIELD)
    parser.add_argument("--qwen-fields", type=_parse_tuple, default=DEFAULT_QWEN_FIELDS)
    parser.add_argument("--control-fields", type=_parse_tuple, default=DEFAULT_CONTROL_FIELDS)
    parser.add_argument("--train-prefix-rows", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_decomposition(
        output_dir=args.output_dir,
        tiny_packet_jsonl=args.tiny_packet_jsonl,
        tiny_artifact=args.tiny_artifact,
        qwen_packet_jsonl=args.qwen_packet_jsonl,
        qwen_global_artifact=args.qwen_global_artifact,
        tiny_field=args.tiny_field,
        tiny_margin_field=args.tiny_margin_field,
        qwen_fields=args.qwen_fields,
        control_fields=args.control_fields,
        train_prefix_rows=args.train_prefix_rows,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
