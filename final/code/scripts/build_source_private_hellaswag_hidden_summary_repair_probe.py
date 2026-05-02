from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl"
)
DEFAULT_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json"
)
DEFAULT_TRAIN_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/source_train_score_cache.json"
)
DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024"
)


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


def _content_digest(rows: list[arc_gate.ArcRow]) -> str:
    return hashlib.sha256("\n".join(row.content_id for row in rows).encode("utf-8")).hexdigest()


def _ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    return float(sum(row.answer_index == pred for row, pred in zip(rows, predictions, strict=True)) / len(rows))


def _select_train_rows(rows: list[arc_gate.ArcRow], *, count: int, seed: int) -> list[arc_gate.ArcRow]:
    if count >= len(rows):
        return list(rows)
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:count]


def _split_indices(n: int, *, dev_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    dev_count = max(1, min(n - 1, int(round(n * dev_fraction))))
    dev = sorted(indices[:dev_count])
    fit = sorted(indices[dev_count:])
    return fit, dev


def _take_rows(rows: list[arc_gate.ArcRow], indices: list[int]) -> list[arc_gate.ArcRow]:
    return [rows[index] for index in indices]


def _take_scores(scores: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [scores[index] for index in indices]


def _take_features(features: np.ndarray, indices: list[int]) -> np.ndarray:
    return features[np.asarray(indices, dtype=np.int64)]


def _hidden_cache_metadata(
    *,
    rows: list[arc_gate.ArcRow],
    model_path: str,
    dtype: str,
    max_length: int,
    prompt_mode: str,
    layers: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "row_ids": [row.row_id for row in rows],
        "content_digest": _content_digest(rows),
        "model_path": model_path,
        "dtype": dtype,
        "max_length": max_length,
        "prompt_mode": prompt_mode,
        "layers": list(layers),
    }


def _load_hidden_cache(
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    expected: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not npz_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("row_count", "row_ids", "content_digest", "model_path", "dtype", "max_length", "prompt_mode", "layers"):
        if metadata.get(key) != expected.get(key):
            return None
    with np.load(npz_path) as data:
        features = np.asarray(data["features"], dtype=np.float64)
    return features, metadata | {"cache_hit": True, "cache_npz": _display_path(npz_path), "cache_meta": _display_path(meta_path)}


def _write_hidden_cache(
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    features: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, features=features.astype(np.float32))
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_source_hidden_features(
    rows: list[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    prompt_mode: str,
    layers: tuple[int, ...],
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=arc_gate._torch_dtype(dtype),
    ).to(resolved_device)
    model.eval()
    base_model = getattr(model, "model", None)

    started = time.perf_counter()
    row_features: list[list[list[np.ndarray]]] = []
    hidden_dim: int | None = None
    max_choices = max(len(row.choices) for row in rows)
    with torch.inference_mode():
        for row in rows:
            prompt = arc_gate._lm_choice_prompt(row, prompt_mode=prompt_mode)
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            texts = [prompt + " " + choice for choice in row.choices]
            padding_mode: bool | str = True
            tokenizer_max_length = max_length
            if str(resolved_device).startswith("mps"):
                raw_lengths = tokenizer(texts, padding=False, truncation=True, max_length=max_length)["input_ids"]
                row_max_length = max(len(input_ids) for input_ids in raw_lengths)
                tokenizer_max_length = min(max_length, int(math.ceil(row_max_length / 32.0) * 32))
                padding_mode = "max_length"
            encoded = tokenizer(
                texts,
                padding=padding_mode,
                truncation=True,
                max_length=tokenizer_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            if base_model is not None:
                output = base_model(**encoded, output_hidden_states=True, return_dict=True)
            else:
                output = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
            attention = encoded["attention_mask"].bool()
            choice_features: list[list[np.ndarray]] = []
            for choice_index in range(len(row.choices)):
                layer_features: list[np.ndarray] = []
                choice_mask = attention[choice_index].clone()
                choice_mask[: min(prompt_len, choice_mask.shape[0])] = False
                for layer in layers:
                    states = output.hidden_states[layer][choice_index]
                    values = states[choice_mask]
                    if values.numel() == 0:
                        values = states[attention[choice_index]]
                    feature = values.mean(dim=0).detach().cpu().numpy().astype(np.float64)
                    norm = float(np.linalg.norm(feature))
                    if norm > 0:
                        feature = feature / norm
                    hidden_dim = int(feature.shape[0])
                    layer_features.append(feature)
                choice_features.append(layer_features)
            row_features.append(choice_features)

    if hidden_dim is None:
        raise ValueError("no hidden features were extracted")
    features = np.zeros((len(rows), max_choices, len(layers), hidden_dim), dtype=np.float64)
    for row_index, choice_features in enumerate(row_features):
        for choice_index, layer_features in enumerate(choice_features):
            for layer_index, feature in enumerate(layer_features):
                features[row_index, choice_index, layer_index] = feature
    return features, {
        "kind": "local_causal_lm_hidden_choice_mean",
        "model_path": model_path,
        "device": resolved_device,
        "dtype": dtype,
        "max_length": max_length,
        "prompt_mode": prompt_mode,
        "layers": list(layers),
        "hidden_dim": hidden_dim,
        "latency_s": float(time.perf_counter() - started),
        "cache_hit": False,
    }


def _source_hidden_features(
    rows: list[arc_gate.ArcRow],
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    prompt_mode: str,
    layers: tuple[int, ...],
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected = _hidden_cache_metadata(
        rows=rows,
        model_path=model_path,
        dtype=dtype,
        max_length=max_length,
        prompt_mode=prompt_mode,
        layers=layers,
    )
    cached = _load_hidden_cache(npz_path=npz_path, meta_path=meta_path, expected=expected)
    if cached is not None:
        return cached
    features, metadata = _extract_source_hidden_features(
        rows,
        model_path=model_path,
        device=device,
        dtype=dtype,
        max_length=max_length,
        prompt_mode=prompt_mode,
        layers=layers,
        local_files_only=local_files_only,
    )
    metadata = expected | metadata | {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write_hidden_cache(npz_path=npz_path, meta_path=meta_path, features=features, metadata=metadata)
    return features, metadata | {"cache_npz": _display_path(npz_path), "cache_meta": _display_path(meta_path)}


def _fit_hidden_pair_scorer(
    rows: list[arc_gate.ArcRow],
    features: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    labels: list[float] = []
    flat_features: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        for choice_index in range(len(row.choices)):
            labels.append(1.0 if choice_index == row.answer_index else -1.0)
            flat_features.append(features[row_index, choice_index])
    x_raw = np.asarray(flat_features, dtype=np.float64)
    mean = np.mean(x_raw, axis=0)
    scale = np.std(x_raw, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_body = (x_raw - mean) / scale
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    y = np.asarray(labels, dtype=np.float64)
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    xtx[0, 0] -= float(ridge)
    weights = np.linalg.solve(xtx, x.T @ y)
    train_scores = x @ weights
    return {
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "ridge": float(ridge),
        "feature_dim": int(x_raw.shape[1]),
        "train_pair_rows": int(x_raw.shape[0]),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
    }


def _predict_hidden_rows(rows: list[arc_gate.ArcRow], features: np.ndarray, scorer: dict[str, Any]) -> tuple[list[list[float]], list[int]]:
    flat: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    start = 0
    for row_index, row in enumerate(rows):
        for choice_index in range(len(row.choices)):
            flat.append(features[row_index, choice_index])
        end = start + len(row.choices)
        offsets.append((start, end))
        start = end
    x_raw = np.asarray(flat, dtype=np.float64)
    x_body = (x_raw - scorer["mean"]) / scorer["scale"]
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    flat_scores = x @ scorer["weights"]
    scores_by_row: list[list[float]] = []
    predictions: list[int] = []
    for start, end in offsets:
        scores = [float(value) for value in flat_scores[start:end]]
        scores_by_row.append(scores)
        predictions.append(int(max(range(len(scores)), key=lambda index: (scores[index], -index))))
    return scores_by_row, predictions


def _select_hidden_model(
    *,
    layers: tuple[int, ...],
    ridges: tuple[float, ...],
    fit_rows: list[arc_gate.ArcRow],
    fit_features: np.ndarray,
    dev_rows: list[arc_gate.ArcRow],
    dev_features: np.ndarray,
    all_rows: list[arc_gate.ArcRow],
    all_features: np.ndarray,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        fit_layer = fit_features[:, :, layer_index, :]
        dev_layer = dev_features[:, :, layer_index, :]
        all_layer = all_features[:, :, layer_index, :]
        for ridge in ridges:
            scorer = _fit_hidden_pair_scorer(fit_rows, fit_layer, ridge=ridge)
            _, dev_predictions = _predict_hidden_rows(dev_rows, dev_layer, scorer)
            _, fit_predictions = _predict_hidden_rows(fit_rows, fit_layer, scorer)
            _, all_predictions = _predict_hidden_rows(all_rows, all_layer, scorer)
            candidates.append(
                {
                    "layer": layer,
                    "layer_index": layer_index,
                    "ridge": ridge,
                    "scorer": scorer,
                    "fit_accuracy": _accuracy(fit_rows, fit_predictions),
                    "dev_accuracy": _accuracy(dev_rows, dev_predictions),
                    "all_train_accuracy": _accuracy(all_rows, all_predictions),
                    "fit_correct": int(sum(row.answer_index == pred for row, pred in zip(fit_rows, fit_predictions, strict=True))),
                    "dev_correct": int(sum(row.answer_index == pred for row, pred in zip(dev_rows, dev_predictions, strict=True))),
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["dev_accuracy"],
            item["fit_accuracy"],
            -float(item["ridge"]),
            str(item["layer"]),
        ),
    )
    readouts = []
    for item in candidates:
        readouts.append(
            {
                "layer": item["layer"],
                "ridge": item["ridge"],
                "fit_accuracy": item["fit_accuracy"],
                "dev_accuracy": item["dev_accuracy"],
                "all_train_accuracy": item["all_train_accuracy"],
                "fit_correct": item["fit_correct"],
                "dev_correct": item["dev_correct"],
            }
        )
    return selected | {"candidate_readouts": readouts}


def _packet_rows(
    *,
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    budget_bytes: int,
    feature_dim: int,
    code_dim: int,
    seed: int,
    index_prior_rows: list[arc_gate.ArcRow],
) -> list[dict[str, Any]]:
    features = arc_gate._features(
        arc_gate._choice_pair_texts(rows),
        dim=feature_dim,
        feature_mode="hashed",
        feature_model="unused",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
    )
    residuals = arc_gate._candidate_residuals(rows, features)
    projection = arc_gate._projection_matrix(feature_dim, code_dim, seed=seed + 171)
    return arc_gate._rows_for_predictions(
        eval_rows=rows,
        residuals=residuals,
        source_predictions=source_predictions,
        projection=projection,
        budget_bytes=budget_bytes,
        index_prior=arc_gate._index_prior(index_prior_rows),
        seed=seed + 911 + budget_bytes,
    )


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return arc_gate._condition_metrics(rows)


def _paired_ci_predictions(
    rows: list[arc_gate.ArcRow],
    candidate: list[int],
    baseline: list[int],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    deltas = [
        float(row.answer_index == cand) - float(row.answer_index == base)
        for row, cand, base in zip(rows, candidate, baseline, strict=True)
    ]
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _source_label_predictions(scores: list[list[float]]) -> list[int]:
    return [_ranked_indices(row_scores)[0] for row_scores in scores]


def _topk_oracle(rows: list[arc_gate.ArcRow], scores: list[list[float]], *, k: int) -> float:
    return float(
        sum(row.answer_index in _ranked_indices(row_scores)[:k] for row, row_scores in zip(rows, scores, strict=True))
        / len(rows)
    )


def build_probe(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    eval_score_cache: pathlib.Path,
    train_score_cache: pathlib.Path,
    train_hidden_cache: pathlib.Path | None,
    eval_hidden_cache: pathlib.Path | None,
    train_hidden_rows: int,
    selection_seed: int,
    dev_fraction: float,
    hidden_layers: tuple[int, ...],
    hidden_ridges: tuple[float, ...],
    packet_budgets: tuple[int, ...],
    packet_feature_dim: int,
    packet_code_dim: int,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
    bootstrap_samples: int,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    train_path = _resolve(train_path)
    eval_path = _resolve(eval_path)
    eval_score_cache = _resolve(eval_score_cache)
    train_score_cache = _resolve(train_score_cache)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    all_train_rows = arc_gate._load_rows(train_path)
    selected_train_rows = _select_train_rows(all_train_rows, count=train_hidden_rows, seed=selection_seed)
    eval_rows = arc_gate._load_rows(eval_path)
    fit_indices, dev_indices = _split_indices(len(selected_train_rows), dev_fraction=dev_fraction, seed=selection_seed + 17)
    fit_rows = _take_rows(selected_train_rows, fit_indices)
    dev_rows = _take_rows(selected_train_rows, dev_indices)

    train_scores, _, train_source_model, train_score_cache_sha256 = headroom._source_scores(
        rows=selected_train_rows,
        score_cache=train_score_cache,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)

    train_hidden_npz = _resolve(train_hidden_cache) if train_hidden_cache is not None else output_dir / "source_train_hidden_cache.npz"
    train_hidden_meta = train_hidden_npz.with_suffix(".json")
    eval_hidden_npz = _resolve(eval_hidden_cache) if eval_hidden_cache is not None else output_dir / "source_eval_hidden_cache.npz"
    eval_hidden_meta = eval_hidden_npz.with_suffix(".json")
    train_hidden, train_hidden_model = _source_hidden_features(
        selected_train_rows,
        npz_path=train_hidden_npz,
        meta_path=train_hidden_meta,
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        prompt_mode=source_lm_prompt_mode,
        layers=hidden_layers,
        local_files_only=local_files_only,
    )
    eval_hidden, eval_hidden_model = _source_hidden_features(
        eval_rows,
        npz_path=eval_hidden_npz,
        meta_path=eval_hidden_meta,
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        prompt_mode=source_lm_prompt_mode,
        layers=hidden_layers,
        local_files_only=local_files_only,
    )

    fit_hidden = _take_features(train_hidden, fit_indices)
    dev_hidden = _take_features(train_hidden, dev_indices)
    selected_model = _select_hidden_model(
        layers=hidden_layers,
        ridges=hidden_ridges,
        fit_rows=fit_rows,
        fit_features=fit_hidden,
        dev_rows=dev_rows,
        dev_features=dev_hidden,
        all_rows=selected_train_rows,
        all_features=train_hidden,
    )
    layer_index = int(selected_model["layer_index"])
    scorer = selected_model["scorer"]
    _, fit_hidden_predictions = _predict_hidden_rows(fit_rows, fit_hidden[:, :, layer_index, :], scorer)
    _, dev_hidden_predictions = _predict_hidden_rows(dev_rows, dev_hidden[:, :, layer_index, :], scorer)
    _, train_hidden_predictions = _predict_hidden_rows(selected_train_rows, train_hidden[:, :, layer_index, :], scorer)
    _, eval_hidden_predictions = _predict_hidden_rows(eval_rows, eval_hidden[:, :, layer_index, :], scorer)

    source_label_train = _source_label_predictions(train_scores)
    source_label_fit = [source_label_train[index] for index in fit_indices]
    source_label_dev = [source_label_train[index] for index in dev_indices]
    source_label_eval = _source_label_predictions(eval_scores)

    dev_packet_readouts: dict[str, Any] = {}
    for budget in packet_budgets:
        rows = _packet_rows(
            rows=dev_rows,
            source_predictions=dev_hidden_predictions,
            budget_bytes=budget,
            feature_dim=packet_feature_dim,
            code_dim=packet_code_dim,
            seed=selection_seed,
            index_prior_rows=fit_rows,
        )
        metrics = _condition_metrics(rows)
        dev_packet_readouts[str(budget)] = {
            "budget_bytes": budget,
            "matched_accuracy": metrics[arc_gate.MATCHED_CONDITION]["accuracy"],
            "same_byte_text_accuracy": metrics["same_byte_structured_text"]["accuracy"],
            "best_destructive_accuracy": max(metrics[name]["accuracy"] for name in arc_gate.STRICT_DESTRUCTIVE_CONTROLS),
            "condition_metrics": metrics,
        }
    selected_budget = max(
        packet_budgets,
        key=lambda budget: (
            dev_packet_readouts[str(budget)]["matched_accuracy"],
            dev_packet_readouts[str(budget)]["matched_accuracy"] - dev_packet_readouts[str(budget)]["same_byte_text_accuracy"],
            -budget,
        ),
    )
    eval_packet_rows = _packet_rows(
        rows=eval_rows,
        source_predictions=eval_hidden_predictions,
        budget_bytes=selected_budget,
        feature_dim=packet_feature_dim,
        code_dim=packet_code_dim,
        seed=selection_seed,
        index_prior_rows=selected_train_rows,
    )
    eval_packet_metrics = _condition_metrics(eval_packet_rows)
    matched = eval_packet_metrics[arc_gate.MATCHED_CONDITION]["accuracy"]
    same_byte_text = eval_packet_metrics["same_byte_structured_text"]["accuracy"]
    best_destructive_name = max(arc_gate.STRICT_DESTRUCTIVE_CONTROLS, key=lambda name: eval_packet_metrics[name]["accuracy"])
    best_destructive = eval_packet_metrics[best_destructive_name]["accuracy"]

    packet_predictions = [
        int(row["prediction_index"])
        for row in eval_packet_rows
        if row["condition"] == arc_gate.MATCHED_CONDITION
    ]
    same_byte_ci = arc_gate._paired_bootstrap(
        eval_packet_rows,
        condition=arc_gate.MATCHED_CONDITION,
        baseline="same_byte_structured_text",
        seed=selection_seed + 2001,
        samples=bootstrap_samples,
    )
    source_label_ci = _paired_ci_predictions(
        eval_rows,
        packet_predictions,
        source_label_eval,
        seed=selection_seed + 2002,
        samples=bootstrap_samples,
    )
    hidden_label_ci = _paired_ci_predictions(
        eval_rows,
        eval_hidden_predictions,
        source_label_eval,
        seed=selection_seed + 2003,
        samples=bootstrap_samples,
    )
    control_ci = arc_gate._paired_bootstrap(
        eval_packet_rows,
        condition=arc_gate.MATCHED_CONDITION,
        baseline=best_destructive_name,
        seed=selection_seed + 2004,
        samples=bootstrap_samples,
    )

    source_label_eval_accuracy = _accuracy(eval_rows, source_label_eval)
    hidden_label_eval_accuracy = _accuracy(eval_rows, eval_hidden_predictions)
    train_source_label_accuracy = _accuracy(selected_train_rows, source_label_train)
    fit_source_label_accuracy = _accuracy(fit_rows, source_label_fit)
    dev_source_label_accuracy = _accuracy(dev_rows, source_label_dev)

    hidden_in_source_top2 = [
        pred in _ranked_indices(scores)[:2]
        for pred, scores in zip(eval_hidden_predictions, eval_scores, strict=True)
    ]
    packet_contract = {
        "selected_source_payload": "source hidden-ridge selected candidate encoded as public hashed residual sketch",
        "raw_payload_bytes": int(selected_budget),
        "framed_record_bytes": int(selected_budget) + 3,
        "packet_feature_dim": packet_feature_dim,
        "packet_code_dim": packet_code_dim,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
    }
    payload = {
        "gate": "source_private_hellaswag_hidden_summary_repair_probe",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display_path(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "eval_score_cache": _display_path(eval_score_cache),
        "eval_score_cache_sha256": _sha256_file(eval_score_cache),
        "train_score_cache": _display_path(train_score_cache),
        "train_score_cache_sha256": train_score_cache_sha256,
        "all_train_rows": len(all_train_rows),
        "scored_train_rows": len(selected_train_rows),
        "internal_fit_rows": len(fit_rows),
        "internal_dev_rows": len(dev_rows),
        "eval_rows": len(eval_rows),
        "selection_seed": selection_seed,
        "dev_fraction": dev_fraction,
        "source_model": {
            "score_train": train_source_model,
            "score_eval": eval_source_model,
            "hidden_train": train_hidden_model,
            "hidden_eval": eval_hidden_model,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS)
            + ["label", "activity_label", "source_id", "split", "split_type", "ind"],
        },
        "hidden_model_selection": {
            "selected_layer": selected_model["layer"],
            "selected_ridge": selected_model["ridge"],
            "selected_fit_accuracy": selected_model["fit_accuracy"],
            "selected_dev_accuracy": selected_model["dev_accuracy"],
            "selected_all_train_accuracy": selected_model["all_train_accuracy"],
            "candidate_readouts": selected_model["candidate_readouts"],
        },
        "dev_packet_readouts": dev_packet_readouts,
        "condition_metrics": eval_packet_metrics,
        "headline": {
            "selected_budget_bytes": int(selected_budget),
            "hidden_label_copy_fit_accuracy": _accuracy(fit_rows, fit_hidden_predictions),
            "hidden_label_copy_internal_dev_accuracy": _accuracy(dev_rows, dev_hidden_predictions),
            "hidden_label_copy_train_accuracy": _accuracy(selected_train_rows, train_hidden_predictions),
            "hidden_label_copy_eval_accuracy": hidden_label_eval_accuracy,
            "source_label_copy_fit_accuracy": fit_source_label_accuracy,
            "source_label_copy_internal_dev_accuracy": dev_source_label_accuracy,
            "source_label_copy_train_accuracy": train_source_label_accuracy,
            "source_label_copy_eval_accuracy": source_label_eval_accuracy,
            "hidden_label_copy_minus_source_label_copy_eval": hidden_label_eval_accuracy - source_label_eval_accuracy,
            "hidden_packet_eval_accuracy": matched,
            "hidden_packet_minus_source_label_copy": matched - source_label_eval_accuracy,
            "hidden_packet_minus_hidden_label_copy": matched - hidden_label_eval_accuracy,
            "hidden_packet_minus_same_byte_text": matched - same_byte_text,
            "hidden_packet_minus_best_destructive": matched - best_destructive,
            "same_byte_text_accuracy": same_byte_text,
            "best_destructive_control": best_destructive_name,
            "best_destructive_control_accuracy": best_destructive,
            "source_top2_oracle_accuracy": _topk_oracle(eval_rows, eval_scores, k=2),
            "source_top4_oracle_accuracy": _topk_oracle(eval_rows, eval_scores, k=4),
            "hidden_prediction_in_source_top2_rate": float(sum(hidden_in_source_top2) / len(hidden_in_source_top2)),
            "paired_ci95_hidden_packet_vs_source_label_copy": source_label_ci,
            "paired_ci95_hidden_label_vs_source_label_copy": hidden_label_ci,
            "paired_ci95_hidden_packet_vs_same_byte_text": same_byte_ci,
            "paired_ci95_hidden_packet_vs_best_destructive": control_ci,
        },
        "packet_contract": packet_contract,
        "pass_rule": {
            "selected_hidden_model_and_budget_use_train_dev_only": True,
            "hidden_packet_must_beat_source_label_copy_by": 0.02,
            "hidden_packet_must_beat_same_byte_text_by": 0.02,
            "paired_ci95_low_vs_source_label_copy_must_be_positive": True,
            "claim_boundary": (
                "This is a source-hidden-summary repair gate. The source may use train-fitted hidden features "
                "to select a candidate, but it transmits only a fixed public-basis residual sketch. A passing "
                "row still needs seed stability and full-validation expansion before ICLR promotion."
            ),
        },
        "timing": {
            "total_seconds": float(time.perf_counter() - started),
        },
    }
    payload["pass_gate"] = bool(
        payload["headline"]["hidden_packet_minus_source_label_copy"] >= 0.02
        and payload["headline"]["hidden_packet_minus_same_byte_text"] >= 0.02
        and source_label_ci["ci95_low"] > 0.0
    )

    (output_dir / "hellaswag_hidden_summary_repair_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arc_gate._write_jsonl(output_dir / "predictions.jsonl", eval_packet_rows)
    with (output_dir / "hidden_model_readouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_model["candidate_readouts"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# HellaSwag Source-Hidden Summary Repair Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval rows: `{len(selected_train_rows)}` / `{len(eval_rows)}`",
        f"- selected layer/ridge: `{selected_model['layer']}` / `{selected_model['ridge']}`",
        f"- selected packet budget: `{selected_budget}B` raw / `{selected_budget + 3}B` framed",
        f"- source-label copy eval accuracy: `{source_label_eval_accuracy:.3f}`",
        f"- hidden-label copy eval accuracy: `{hidden_label_eval_accuracy:.3f}`",
        f"- hidden packet eval accuracy: `{matched:.3f}`",
        f"- hidden packet minus source-label copy: `{payload['headline']['hidden_packet_minus_source_label_copy']:.3f}`",
        f"- hidden packet minus same-byte text: `{payload['headline']['hidden_packet_minus_same_byte_text']:.3f}`",
        f"- source top-2 oracle accuracy: `{payload['headline']['source_top2_oracle_accuracy']:.3f}`",
        "",
        "## Interpretation",
        "",
        "This gate asks whether source hidden summaries contain repair signal that source score shape did not. "
        "The transmitted object is still a fixed-byte public residual sketch, not the raw hidden state.",
        "",
    ]
    (output_dir / "hellaswag_hidden_summary_repair_probe.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe source-hidden-summary repair packets for HellaSwag.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--train-score-cache", type=pathlib.Path, default=DEFAULT_TRAIN_SCORE_CACHE)
    parser.add_argument("--train-hidden-cache", type=pathlib.Path)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-layers", type=_parse_int_tuple, default=(-1,))
    parser.add_argument("--hidden-ridges", type=_parse_float_tuple, default=(0.1, 1.0, 10.0, 100.0))
    parser.add_argument("--packet-budgets", type=_parse_int_tuple, default=(2, 3, 4))
    parser.add_argument("--packet-feature-dim", type=int, default=384)
    parser.add_argument("--packet-code-dim", type=int, default=96)
    parser.add_argument(
        "--source-lm-model",
        default="/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    )
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_probe(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        train_score_cache=args.train_score_cache,
        train_hidden_cache=args.train_hidden_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        selection_seed=args.selection_seed,
        dev_fraction=args.dev_fraction,
        hidden_layers=args.hidden_layers,
        hidden_ridges=args.hidden_ridges,
        packet_budgets=args.packet_budgets,
        packet_feature_dim=args.packet_feature_dim,
        packet_code_dim=args.packet_code_dim,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=args.local_files_only,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
