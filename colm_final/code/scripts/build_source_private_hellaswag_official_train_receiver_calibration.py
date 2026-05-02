from __future__ import annotations

"""Official-train receiver calibration for HellaSwag cross-family packets."""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_hidden_innovation_bagged_gate as bagged  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_repair_probe as repair  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_innovation_train_sample_stress as stress  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_acceptance_gate as accept  # noqa: E402
from scripts import build_source_private_hellaswag_receiver_headroom_decomposition as decomp  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_official_train_receiver_calibration_20260502")
DEFAULT_TRAIN_PATH = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
DEFAULT_TINY_TRAIN_CACHE_DIR = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260502_tinyllama_train512"
)
DEFAULT_QWEN_TRAIN_CACHE_DIR = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024"
)
DEFAULT_TINY_EVAL_PACKET_JSONL = decomp.DEFAULT_TINY_PACKET_JSONL
DEFAULT_TINY_EVAL_ARTIFACT = decomp.DEFAULT_TINY_ARTIFACT
DEFAULT_QWEN_EVAL_PACKET_JSONL = decomp.DEFAULT_QWEN_PACKET_JSONL
DEFAULT_QWEN_GLOBAL_ARTIFACT = decomp.DEFAULT_QWEN_GLOBAL_ARTIFACT
DEFAULT_SAMPLE_SEEDS = (2027, 2039, 2053)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733)
DEFAULT_RIDGES = (1000.0, 10000.0, 100000.0)
DEFAULT_RECEIVER_RIDGES = accept.DEFAULT_RIDGES
DEFAULT_K_VALUES = accept.DEFAULT_K_VALUES
STRICT_DELTA = 0.005
STRICT_TARGET_DELTA = 0.02


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


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def _cache_paths(cache_dir: pathlib.Path, sample_seed: int) -> tuple[pathlib.Path, pathlib.Path]:
    root = _resolve(cache_dir) / "caches" / f"train_sample_seed_{sample_seed}"
    return root / "source_train_score_cache.json", root / "source_train_hidden_cache.npz"


def _load_sample_cache(
    *,
    cache_dir: pathlib.Path,
    all_train_rows: list[arc_gate.ArcRow],
    sample_seed: int,
    train_hidden_rows: int,
) -> dict[str, Any]:
    rows = top2._select_train_rows(all_train_rows, count=train_hidden_rows, seed=sample_seed)
    score_path, hidden_path = _cache_paths(cache_dir, sample_seed)
    scores, _, score_model = headroom._load_score_cache(score_path, rows=rows)
    hidden, hidden_model = top2._load_hidden_cache(hidden_path, rows=rows)
    hidden = np.asarray(hidden, dtype=np.float32)
    receiver_hidden = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
    return {
        "sample_seed": int(sample_seed),
        "rows": rows,
        "row_ids": [row.row_id for row in rows],
        "answers": np.asarray([row.answer_index for row in rows], dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "hidden": hidden,
        "receiver_hidden": np.asarray(receiver_hidden, dtype=np.float32),
        "score_path": _display_path(score_path),
        "score_sha256": _sha256_file(score_path),
        "hidden_path": _display_path(hidden_path),
        "hidden_sha256": _sha256_file(hidden_path),
        "score_cache_hit": bool(score_model.get("cache_hit", True)),
        "hidden_cache_hit": bool(hidden_model.get("cache_hit", True)),
        "content_digest": headroom._content_digest(rows),
    }


def _fit_family_model_bank(
    *,
    samples: dict[int, dict[str, Any]],
    split_seeds: tuple[int, ...],
    ridges: tuple[float, ...],
    dev_fraction: float,
) -> dict[int, dict[str, Any]]:
    banks: dict[int, dict[str, Any]] = {}
    for sample_seed, sample in samples.items():
        rows = sample["rows"]
        scores = sample["scores"].tolist()
        hidden = np.asarray(sample["hidden"], dtype=np.float32)
        train_feature_by_view = {
            "score_only": repair._candidate_feature_tensor(scores=scores, hidden=hidden, view="score_only"),
            "score_hidden_residual": repair._candidate_feature_tensor(
                scores=scores,
                hidden=hidden,
                view="score_hidden_residual",
            ),
        }
        banks[sample_seed] = {
            "score_models": [],
            "hidden_models": [],
            "component_rows": [],
        }
        for split_seed in split_seeds:
            fit_indices, dev_indices = top2._split_indices(
                len(rows),
                dev_fraction=dev_fraction,
                seed=int(split_seed) + 17,
            )
            for view in ("score_only", "score_hidden_residual"):
                model, component = bagged._select_component_model(
                    view=view,
                    train_features=train_feature_by_view[view],
                    eval_features=train_feature_by_view[view],
                    fit_indices=fit_indices,
                    dev_indices=dev_indices,
                    train_rows=rows,
                    train_scores=scores,
                    eval_rows=rows,
                    ridges=ridges,
                )
                banks[sample_seed]["component_rows"].append(
                    {
                        "train_sample_seed": int(sample_seed),
                        "split_seed": int(split_seed),
                        "fit_rows": len(fit_indices),
                        "dev_rows": len(dev_indices),
                        **component,
                    }
                )
                if view == "score_only":
                    banks[sample_seed]["score_models"].append(model)
                else:
                    banks[sample_seed]["hidden_models"].append(model)
    return banks


def _packet_predictions_for_sample(
    *,
    sample: dict[str, Any],
    model_bank: dict[int, dict[str, Any]],
    included_seeds: tuple[int, ...],
    aggregation_policy: str,
) -> dict[str, Any]:
    scores = sample["scores"].tolist()
    hidden = np.asarray(sample["hidden"], dtype=np.float32)
    score_features = repair._candidate_feature_tensor(scores=scores, hidden=hidden, view="score_only")
    hidden_features = repair._candidate_feature_tensor(
        scores=scores,
        hidden=hidden,
        view="score_hidden_residual",
    )
    score_models = [model for seed in included_seeds for model in model_bank[int(seed)]["score_models"]]
    hidden_models = [model for seed in included_seeds for model in model_bank[int(seed)]["hidden_models"]]
    if not score_models or not hidden_models:
        raise ValueError("out-of-bag prediction requires at least one included model seed")
    score_mean_predictions, _ = bagged._aggregate_model_scores(
        features=score_features,
        models=score_models,
        policy="mean_zscore",
    )
    score_vote_predictions, _ = bagged._aggregate_model_scores(
        features=score_features,
        models=score_models,
        policy="vote",
    )
    hidden_mean_predictions, hidden_mean_scores = bagged._aggregate_model_scores(
        features=hidden_features,
        models=hidden_models,
        policy="mean_zscore",
    )
    vote_predictions, _ = bagged._aggregate_model_scores(
        features=hidden_features,
        models=hidden_models,
        policy="vote",
    )
    hybrid_predictions = bagged._hybrid_vote_on_score_agreement(
        mean_predictions=hidden_mean_predictions,
        vote_predictions=vote_predictions,
        score_mean_predictions=score_mean_predictions,
    )
    if aggregation_policy == "mean_zscore":
        selected = hidden_mean_predictions
    elif aggregation_policy == "mean_zscore_vote_on_score_agreement":
        selected = hybrid_predictions
    else:
        raise ValueError(f"unsupported aggregation policy: {aggregation_policy}")
    margins = np.partition(hidden_mean_scores, -2, axis=1)[:, -1] - np.partition(hidden_mean_scores, -2, axis=1)[:, -2]
    return {
        "selected_prediction": np.asarray(selected, dtype=np.int64),
        "selected_margin": margins.astype(np.float64),
        "mean_zscore_prediction": np.asarray(hidden_mean_predictions, dtype=np.int64),
        "hybrid_vote_on_score_agreement_prediction": np.asarray(hybrid_predictions, dtype=np.int64),
        "score_mean_prediction": np.asarray(score_mean_predictions, dtype=np.int64),
        "score_vote_prediction": np.asarray(score_vote_predictions, dtype=np.int64),
        "vote_prediction": np.asarray(vote_predictions, dtype=np.int64),
    }


def _build_oob_calibration_rows(
    *,
    tiny_samples: dict[int, dict[str, Any]],
    qwen_samples: dict[int, dict[str, Any]],
    tiny_bank: dict[int, dict[str, Any]],
    qwen_bank: dict[int, dict[str, Any]],
    sample_seeds: tuple[int, ...],
    tiny_aggregation_policy: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "answers": [],
        "tiny_packet": [],
        "tiny_margin": [],
        "qwen_scores": [],
        "qwen_hidden": [],
        "qwen_target": [],
        "qwen_mean": [],
        "qwen_hybrid": [],
    }
    seen: set[str] = set()
    duplicate_count = 0
    oob_overlap_drop_count = 0
    for seed in sample_seeds:
        tiny = tiny_samples[int(seed)]
        qwen = qwen_samples[int(seed)]
        if tiny["row_ids"] != qwen["row_ids"]:
            raise ValueError(f"TinyLlama/Qwen sample seed {seed} row IDs are not aligned")
        if tiny["content_digest"] != qwen["content_digest"]:
            raise ValueError(f"TinyLlama/Qwen sample seed {seed} content digests differ")
        included = tuple(item for item in sample_seeds if int(item) != int(seed))
        included_train_ids = {
            row_id
            for included_seed in included
            for row_id in tiny_samples[int(included_seed)]["row_ids"]
        }
        tiny_pred = _packet_predictions_for_sample(
            sample=tiny,
            model_bank=tiny_bank,
            included_seeds=included,
            aggregation_policy=tiny_aggregation_policy,
        )
        qwen_pred = _packet_predictions_for_sample(
            sample=qwen,
            model_bank=qwen_bank,
            included_seeds=included,
            aggregation_policy="mean_zscore",
        )
        keep_indices: list[int] = []
        for local_index, row_id in enumerate(tiny["row_ids"]):
            if row_id in included_train_ids:
                oob_overlap_drop_count += 1
                continue
            if row_id in seen:
                duplicate_count += 1
                continue
            seen.add(row_id)
            keep_indices.append(local_index)
            rows.append(
                {
                    "row_id": row_id,
                    "answer_index": int(tiny["answers"][local_index]),
                    "sample_seed": int(seed),
                    "oob_model_seeds": list(included),
                }
            )
        if not keep_indices:
            continue
        ids = np.asarray(keep_indices, dtype=np.int64)
        arrays["answers"].append(tiny["answers"][ids])
        arrays["tiny_packet"].append(tiny_pred["selected_prediction"][ids])
        arrays["tiny_margin"].append(tiny_pred["selected_margin"][ids])
        arrays["qwen_scores"].append(qwen["scores"][ids])
        arrays["qwen_hidden"].append(qwen["receiver_hidden"][ids])
        arrays["qwen_target"].append(np.argmax(qwen["scores"][ids], axis=1).astype(np.int64))
        arrays["qwen_mean"].append(qwen_pred["mean_zscore_prediction"][ids])
        arrays["qwen_hybrid"].append(qwen_pred["hybrid_vote_on_score_agreement_prediction"][ids])
    return {
        "rows": rows,
        "duplicate_row_count": int(duplicate_count),
        "oob_overlap_drop_count": int(oob_overlap_drop_count),
        "answers": np.concatenate(arrays["answers"], axis=0).astype(np.int64),
        "tiny_packet": np.concatenate(arrays["tiny_packet"], axis=0).astype(np.int64),
        "tiny_margin": np.concatenate(arrays["tiny_margin"], axis=0).astype(np.float64),
        "qwen_scores": np.concatenate(arrays["qwen_scores"], axis=0).astype(np.float64),
        "qwen_hidden": np.concatenate(arrays["qwen_hidden"], axis=0).astype(np.float32),
        "qwen_target": np.concatenate(arrays["qwen_target"], axis=0).astype(np.int64),
        "qwen_mean": np.concatenate(arrays["qwen_mean"], axis=0).astype(np.int64),
        "qwen_hybrid": np.concatenate(arrays["qwen_hybrid"], axis=0).astype(np.int64),
    }


def _official_split_indices(row_count: int, *, dev_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(row_count)
    dev_count = max(1, int(round(row_count * dev_fraction)))
    dev = np.sort(order[:dev_count]).astype(np.int64)
    fit = np.sort(order[dev_count:]).astype(np.int64)
    if len(fit) == 0:
        raise ValueError("official train calibration split has no fit rows")
    return fit, dev


def _baseline_accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions == answers))


def _oracle_accuracy(predictions: list[np.ndarray], answers: np.ndarray) -> float:
    correct = np.zeros_like(answers, dtype=bool)
    for item in predictions:
        correct |= item == answers
    return float(np.mean(correct))


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Official-Train Receiver Calibration",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- predeclared default pass gate: `{payload['predeclared_default_pass_gate']}`",
        f"- scout pass gate: `{payload['scout_pass_gate']}`",
        f"- official train calibration rows: `{h['official_train_calibration_rows']}`",
        f"- validation rows: `{h['validation_rows']}`",
        f"- default eval accuracy: `{h['default_eval_accuracy']:.6f}`",
        f"- default delta vs packet-only: `{h['default_delta_vs_packet_only']:.6f}`",
        f"- best scout eval accuracy: `{h['best_scout_eval_accuracy']:.6f}`",
        f"- best scout delta vs packet-only: `{h['best_scout_delta_vs_packet_only']:.6f}`",
        f"- full-validation Tiny+Qwen oracle: `{h['validation_tiny_qwen_hybrid_oracle_accuracy']:.6f}`",
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


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = DEFAULT_TRAIN_PATH,
    tiny_train_cache_dir: pathlib.Path = DEFAULT_TINY_TRAIN_CACHE_DIR,
    qwen_train_cache_dir: pathlib.Path = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    tiny_eval_packet_jsonl: pathlib.Path = DEFAULT_TINY_EVAL_PACKET_JSONL,
    tiny_eval_artifact: pathlib.Path = DEFAULT_TINY_EVAL_ARTIFACT,
    qwen_eval_packet_jsonl: pathlib.Path = DEFAULT_QWEN_EVAL_PACKET_JSONL,
    qwen_global_artifact: pathlib.Path = DEFAULT_QWEN_GLOBAL_ARTIFACT,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    receiver_ridges: tuple[float, ...] = DEFAULT_RECEIVER_RIDGES,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    bootstrap_samples: int = 1000,
    tiny_aggregation_policy: str = "mean_zscore",
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_train_rows = arc_gate._load_rows(_resolve(train_path))

    tiny_samples = {
        int(seed): _load_sample_cache(
            cache_dir=tiny_train_cache_dir,
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for seed in sample_seeds
    }
    qwen_samples = {
        int(seed): _load_sample_cache(
            cache_dir=qwen_train_cache_dir,
            all_train_rows=all_train_rows,
            sample_seed=int(seed),
            train_hidden_rows=train_hidden_rows,
        )
        for seed in sample_seeds
    }
    tiny_bank = _fit_family_model_bank(
        samples=tiny_samples,
        split_seeds=split_seeds,
        ridges=ridges,
        dev_fraction=dev_fraction,
    )
    qwen_bank = _fit_family_model_bank(
        samples=qwen_samples,
        split_seeds=split_seeds,
        ridges=ridges,
        dev_fraction=dev_fraction,
    )
    calibration = _build_oob_calibration_rows(
        tiny_samples=tiny_samples,
        qwen_samples=qwen_samples,
        tiny_bank=tiny_bank,
        qwen_bank=qwen_bank,
        sample_seeds=sample_seeds,
        tiny_aggregation_policy=tiny_aggregation_policy,
    )
    fit_indices, dev_indices = _official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )

    tiny_eval_rows = _read_jsonl(tiny_eval_packet_jsonl)
    qwen_eval_rows = _read_jsonl(qwen_eval_packet_jsonl)
    qwen_eval = accept._load_qwen_bundle(qwen_global_artifact)
    row_ids = [str(row["row_id"]) for row in tiny_eval_rows]
    if row_ids != [str(row["row_id"]) for row in qwen_eval_rows]:
        raise ValueError("validation TinyLlama/Qwen packet rows are not aligned")
    if row_ids != qwen_eval["row_ids"]:
        raise ValueError("validation packet rows and Qwen eval scores are not aligned")
    validation_answers = np.asarray([int(row["answer_index"]) for row in tiny_eval_rows], dtype=np.int64)
    validation_packet = np.asarray([int(row["selected_prediction"]) for row in tiny_eval_rows], dtype=np.int64)
    validation_margin = np.asarray(
        [float(row.get("selected_margin", 0.0)) for row in tiny_eval_rows],
        dtype=np.float64,
    )
    validation_alternatives = {
        "qwen_target_score": np.argmax(qwen_eval["scores"], axis=1).astype(np.int64),
        "mean_zscore_prediction": np.asarray(
            [int(row["mean_zscore_prediction"]) for row in qwen_eval_rows],
            dtype=np.int64,
        ),
        "hybrid_vote_on_score_agreement_prediction": np.asarray(
            [int(row["hybrid_vote_on_score_agreement_prediction"]) for row in qwen_eval_rows],
            dtype=np.int64,
        ),
    }
    train_alternatives = {
        "qwen_target_score": calibration["qwen_target"],
        "mean_zscore_prediction": calibration["qwen_mean"],
        "hybrid_vote_on_score_agreement_prediction": calibration["qwen_hybrid"],
    }
    eval_indices = np.arange(len(validation_answers), dtype=np.int64)
    frontier_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str, str], np.ndarray] = {}
    feature_build_start = time.perf_counter()
    train_features: dict[tuple[str, str], np.ndarray] = {}
    eval_features: dict[tuple[str, str], np.ndarray] = {}
    for alt_name in train_alternatives:
        for view in ("score_only", "score_hidden_confidence"):
            train_features[(alt_name, view)] = accept._feature_matrix(
                scores=calibration["qwen_scores"],
                hidden=calibration["qwen_hidden"],
                packet_predictions=calibration["tiny_packet"],
                packet_margins=calibration["tiny_margin"],
                alt_predictions=train_alternatives[alt_name],
                view=view,
            )
            eval_features[(alt_name, view)] = accept._feature_matrix(
                scores=qwen_eval["scores"],
                hidden=qwen_eval["hidden"],
                packet_predictions=validation_packet,
                packet_margins=validation_margin,
                alt_predictions=validation_alternatives[alt_name],
                view=view,
            )
    feature_build_wall_time_s = time.perf_counter() - feature_build_start
    selector_start = time.perf_counter()
    for alt_index, alt_name in enumerate(train_alternatives):
        benefit = accept._benefit_values(
            alt_predictions=train_alternatives[alt_name],
            packet_predictions=calibration["tiny_packet"],
            answers=calibration["answers"],
        )
        for view in ("score_only", "score_hidden_confidence"):
            for method_index, receiver in enumerate(
                [
                    accept._run_ridge_receiver(
                        features=train_features[(alt_name, view)],
                        benefit=benefit,
                        packet_predictions=calibration["tiny_packet"],
                        alt_predictions=train_alternatives[alt_name],
                        answers=calibration["answers"],
                        fit_indices=fit_indices,
                        dev_indices=dev_indices,
                        eval_indices=dev_indices,
                        ridges=receiver_ridges,
                    ),
                    accept._run_relative_knn_receiver(
                        features=train_features[(alt_name, view)],
                        benefit=benefit,
                        packet_predictions=calibration["tiny_packet"],
                        alt_predictions=train_alternatives[alt_name],
                        answers=calibration["answers"],
                        fit_indices=fit_indices,
                        dev_indices=dev_indices,
                        eval_indices=dev_indices,
                        k_values=k_values,
                    ),
                ]
            ):
                if receiver["method"] == "benefit_ridge":
                    model = accept._fit_benefit_ridge(
                        features=train_features[(alt_name, view)],
                        benefit=benefit,
                        fit_indices=fit_indices,
                        ridges=(float(receiver["ridge"]),),
                    )[0]
                    eval_scores = accept._score_benefit_ridge(model, eval_features[(alt_name, view)])
                    use_alt = (eval_scores > receiver["threshold"]) & (
                        validation_alternatives[alt_name] != validation_packet
                    )
                else:
                    # The relative receiver uses train-only anchors. Evaluate validation rows as queries.
                    eval_scores = accept._knn_scores(
                        features=np.concatenate(
                            [train_features[(alt_name, view)], eval_features[(alt_name, view)]],
                            axis=0,
                        ),
                        benefit=np.concatenate(
                            [benefit, np.zeros(len(validation_answers), dtype=np.float64)],
                            axis=0,
                        ),
                        fit_indices=fit_indices,
                        query_indices=np.arange(
                            len(calibration["answers"]),
                            len(calibration["answers"]) + len(validation_answers),
                            dtype=np.int64,
                        ),
                        k=int(receiver["k"]),
                    )
                    use_alt = (eval_scores > receiver["threshold"]) & (
                        validation_alternatives[alt_name] != validation_packet
                    )
                predictions = accept._task_predictions(
                    packet_predictions=validation_packet,
                    alt_predictions=validation_alternatives[alt_name],
                    use_alt=use_alt,
                )
                stats = accept._receiver_stats(
                    predictions=predictions,
                    packet_predictions=validation_packet,
                    target_predictions=validation_alternatives["qwen_target_score"],
                    answers=validation_answers,
                    indices=eval_indices,
                    seed=9100 + 31 * alt_index + method_index,
                    samples=bootstrap_samples,
                )
                prediction_cache[(alt_name, view, receiver["method"])] = predictions
                frontier_rows.append(
                    {
                        "alternative": alt_name,
                        "feature_view": view,
                        "method": receiver["method"],
                        "hyperparameter": {
                            "ridge": receiver.get("ridge"),
                            "k": receiver.get("k"),
                            "threshold": receiver["threshold"],
                        },
                        "official_fit_rows": int(len(fit_indices)),
                        "official_dev_rows": int(len(dev_indices)),
                        "official_dev_accuracy": receiver["dev_accuracy"],
                        "official_dev_override_rate": receiver["dev_override_rate"],
                        "official_dev_help_count": receiver["dev_help_count"],
                        "official_dev_harm_count": receiver["dev_harm_count"],
                        "validation_override_rate": float(np.mean(use_alt)),
                        **stats,
                    }
                )
    selector_wall_time_s = time.perf_counter() - selector_start
    best_scout = max(
        frontier_rows,
        key=lambda row: (
            row["delta_vs_packet_only"],
            row["ci95_low_vs_packet_only"],
            row["official_dev_accuracy"],
            -row["validation_override_rate"],
        ),
    )
    default_rows = [
        row
        for row in frontier_rows
        if row["alternative"] == "qwen_target_score"
        and row["feature_view"] == "score_hidden_confidence"
        and row["method"] == "benefit_ridge"
    ]
    if not default_rows:
        raise ValueError("missing predeclared default row")
    default_row = default_rows[0]
    default_predictions = prediction_cache[
        (default_row["alternative"], default_row["feature_view"], default_row["method"])
    ]
    default_blocks = accept._contiguous_block_deltas(
        predictions=default_predictions,
        packet_predictions=validation_packet,
        answers=validation_answers,
        eval_indices=eval_indices,
    )
    validation_oracle = _oracle_accuracy(
        [validation_packet, validation_alternatives["hybrid_vote_on_score_agreement_prediction"]],
        validation_answers,
    )
    train_oracle = _oracle_accuracy(
        [calibration["tiny_packet"], calibration["qwen_hybrid"]],
        calibration["answers"],
    )
    scout_pass_gate = bool(
        best_scout["delta_vs_packet_only"] >= STRICT_DELTA
        and best_scout["ci95_low_vs_packet_only"] > 0.0
    )
    predeclared_default_pass_gate = bool(
        default_row["delta_vs_packet_only"] >= STRICT_DELTA
        and default_row["ci95_low_vs_packet_only"] > 0.0
    )
    target_transfer_gate = bool(
        default_row["delta_vs_target_only"] >= STRICT_TARGET_DELTA
        and default_row["ci95_low_vs_target_only"] > 0.0
    )
    block_stability_gate = bool(all(row["delta_vs_packet_only"] > 0.0 for row in default_blocks))
    pass_gate = bool(predeclared_default_pass_gate and target_transfer_gate and block_stability_gate)
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "validation_rows": int(len(validation_answers)),
        "sample_seeds": list(sample_seeds),
        "default_method": default_row["method"],
        "default_alternative": default_row["alternative"],
        "default_feature_view": default_row["feature_view"],
        "default_eval_accuracy": default_row["accuracy"],
        "default_packet_only_accuracy": default_row["packet_only_accuracy"],
        "default_target_only_accuracy": default_row["target_only_accuracy"],
        "default_delta_vs_packet_only": default_row["delta_vs_packet_only"],
        "default_ci95_low_vs_packet_only": default_row["ci95_low_vs_packet_only"],
        "default_delta_vs_target_only": default_row["delta_vs_target_only"],
        "default_ci95_low_vs_target_only": default_row["ci95_low_vs_target_only"],
        "best_scout_method": best_scout["method"],
        "best_scout_alternative": best_scout["alternative"],
        "best_scout_feature_view": best_scout["feature_view"],
        "best_scout_eval_accuracy": best_scout["accuracy"],
        "best_scout_packet_only_accuracy": best_scout["packet_only_accuracy"],
        "best_scout_delta_vs_packet_only": best_scout["delta_vs_packet_only"],
        "best_scout_ci95_low_vs_packet_only": best_scout["ci95_low_vs_packet_only"],
        "official_train_tiny_packet_accuracy": _baseline_accuracy(
            calibration["tiny_packet"],
            calibration["answers"],
        ),
        "official_train_qwen_hybrid_accuracy": _baseline_accuracy(
            calibration["qwen_hybrid"],
            calibration["answers"],
        ),
        "official_train_tiny_qwen_hybrid_oracle_accuracy": train_oracle,
        "validation_tiny_packet_accuracy": _baseline_accuracy(validation_packet, validation_answers),
        "validation_qwen_hybrid_accuracy": _baseline_accuracy(
            validation_alternatives["hybrid_vote_on_score_agreement_prediction"],
            validation_answers,
        ),
        "validation_tiny_qwen_hybrid_oracle_accuracy": validation_oracle,
        "strict_delta_required": STRICT_DELTA,
        "packet_raw_bytes": decomp.RAW_PACKET_BYTES,
        "packet_framed_bytes": decomp.FRAMED_PACKET_BYTES,
        "native_gpu_claims_allowed": False,
    }
    lay_explanation = (
        "The earlier receiver used early validation rows for calibration. This experiment moves receiver "
        "training to official HellaSwag train rows. To avoid teaching the receiver from a packet that saw "
        "the same row during packet training, each train row is scored by packet models trained on other "
        "official-train samples only. The frozen receiver is then evaluated on full validation."
    )
    interpretation = (
        "This is the cleanest cached test of whether more legitimate receiver supervision fixes the "
        "packet-only blocker. If it passes, it promotes an official-train calibrated receiver. If it fails, "
        "the next branch should move beyond scalar acceptance into a richer learned common-basis or query "
        "bottleneck receiver. Because it reuses cached 512-row packet-training samples, even a pass should "
        "be repeated on a new disjoint official-train calibration sample before becoming the paper claim."
    )
    payload = {
        "gate": "source_private_hellaswag_official_train_receiver_calibration",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "scout_pass_gate": scout_pass_gate,
        "predeclared_default_pass_gate": predeclared_default_pass_gate,
        "target_transfer_gate": target_transfer_gate,
        "block_stability_gate": block_stability_gate,
        "pass_rule": (
            "Strict promotion requires the predeclared official-train calibrated receiver to beat "
            "TinyLlama packet-only on full validation by >=0.005 with positive paired CI95 low, beat "
            "Qwen target-only by >=0.02, and remain positive across contiguous validation blocks. The "
            "best-scout row is diagnostic because it is chosen after reading the frontier."
        ),
        "headline": headline,
        "frontier_rows": frontier_rows,
        "default_block_rows": default_blocks,
        "sample_cache_rows": [
            {
                "sample_seed": int(seed),
                "row_count": int(len(tiny_samples[int(seed)]["row_ids"])),
                "content_digest": tiny_samples[int(seed)]["content_digest"],
                "tiny_score_cache": tiny_samples[int(seed)]["score_path"],
                "tiny_score_cache_sha256": tiny_samples[int(seed)]["score_sha256"],
                "tiny_hidden_cache": tiny_samples[int(seed)]["hidden_path"],
                "tiny_hidden_cache_sha256": tiny_samples[int(seed)]["hidden_sha256"],
                "qwen_score_cache": qwen_samples[int(seed)]["score_path"],
                "qwen_score_cache_sha256": qwen_samples[int(seed)]["score_sha256"],
                "qwen_hidden_cache": qwen_samples[int(seed)]["hidden_path"],
                "qwen_hidden_cache_sha256": qwen_samples[int(seed)]["hidden_sha256"],
            }
            for seed in sample_seeds
        ],
        "component_rows": [
            *[row for seed in sample_seeds for row in tiny_bank[int(seed)]["component_rows"]],
            *[
                {"family": "Qwen2.5", **row}
                for seed in sample_seeds
                for row in qwen_bank[int(seed)]["component_rows"]
            ],
        ],
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": decomp.RAW_PACKET_BYTES,
            "framed_record_bytes_per_request": decomp.FRAMED_PACKET_BYTES,
            "logical_validation_raw_payload_bytes_total": int(decomp.RAW_PACKET_BYTES * len(validation_answers)),
            "logical_validation_framed_record_bytes_total": int(
                decomp.FRAMED_PACKET_BYTES * len(validation_answers)
            ),
            "feature_build_wall_time_s": float(feature_build_wall_time_s),
            "selector_wall_time_s": float(selector_wall_time_s),
            "total_wall_time_s": float(time.perf_counter() - started),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
        },
        "inputs": {
            "train_path": _display_path(train_path),
            "train_sha256": _sha256_file(train_path),
            "tiny_train_cache_dir": _display_path(tiny_train_cache_dir),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "tiny_eval_packet_jsonl": _display_path(tiny_eval_packet_jsonl),
            "tiny_eval_packet_jsonl_sha256": _sha256_file(tiny_eval_packet_jsonl),
            "tiny_eval_artifact": _display_path(tiny_eval_artifact),
            "tiny_eval_artifact_sha256": _sha256_file(tiny_eval_artifact),
            "qwen_eval_packet_jsonl": _display_path(qwen_eval_packet_jsonl),
            "qwen_eval_packet_jsonl_sha256": _sha256_file(qwen_eval_packet_jsonl),
            "qwen_global_artifact": _display_path(qwen_global_artifact),
            "qwen_global_artifact_sha256": _sha256_file(qwen_global_artifact),
        },
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_official_train_receiver_calibration.json"
    md_path = output_dir / "hellaswag_official_train_receiver_calibration.md"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--tiny-train-cache-dir", type=pathlib.Path, default=DEFAULT_TINY_TRAIN_CACHE_DIR)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--tiny-eval-packet-jsonl", type=pathlib.Path, default=DEFAULT_TINY_EVAL_PACKET_JSONL)
    parser.add_argument("--tiny-eval-artifact", type=pathlib.Path, default=DEFAULT_TINY_EVAL_ARTIFACT)
    parser.add_argument("--qwen-eval-packet-jsonl", type=pathlib.Path, default=DEFAULT_QWEN_EVAL_PACKET_JSONL)
    parser.add_argument("--qwen-global-artifact", type=pathlib.Path, default=DEFAULT_QWEN_GLOBAL_ARTIFACT)
    parser.add_argument("--sample-seeds", type=_parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--receiver-ridges", type=_parse_float_tuple, default=DEFAULT_RECEIVER_RIDGES)
    parser.add_argument("--k-values", type=_parse_int_tuple, default=DEFAULT_K_VALUES)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--tiny-aggregation-policy", default="mean_zscore")
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        tiny_train_cache_dir=args.tiny_train_cache_dir,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        tiny_eval_packet_jsonl=args.tiny_eval_packet_jsonl,
        tiny_eval_artifact=args.tiny_eval_artifact,
        qwen_eval_packet_jsonl=args.qwen_eval_packet_jsonl,
        qwen_global_artifact=args.qwen_global_artifact,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        receiver_ridges=args.receiver_ridges,
        k_values=args.k_values,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        bootstrap_samples=args.bootstrap_samples,
        tiny_aggregation_policy=args.tiny_aggregation_policy,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
