from __future__ import annotations

import argparse
import dataclasses
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

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_source_latent_endpoint_gate_20260501")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)

MATCHED_CONDITION = "matched_source_latent_packet"
STRICT_DESTRUCTIVE_CONTROLS = (
    "zero_source",
    "shuffled_source_packet",
    "source_feature_permutation_packet",
    "random_same_byte_packet",
    "target_derived_sidecar",
    "candidate_derangement",
)
REPORT_CONDITIONS = (
    "target_only",
    MATCHED_CONDITION,
    *STRICT_DESTRUCTIVE_CONTROLS,
    "label_permutation",
    "same_byte_structured_text",
    "answer_only_text_forbidden_oracle",
)


@dataclasses.dataclass(frozen=True)
class SourceFeatureState:
    pair_features: np.ndarray
    scores_by_row: list[list[float]] | None
    predictions: list[int] | None
    metadata: dict[str, Any]


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


def _standardizer(features: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean, "scale": scale}


def _apply_standardizer(features: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    return (features - state["mean"]) / state["scale"]


def _fit_ridge_map(source_features: np.ndarray, target_features: np.ndarray, *, ridge: float) -> dict[str, Any]:
    if source_features.shape[0] != target_features.shape[0]:
        raise ValueError("source and target feature rows must match")
    standard = _standardizer(source_features)
    x_body = _apply_standardizer(source_features, standard)
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    y = target_features.astype(np.float64)
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    xtx[0, 0] -= float(ridge)
    weights = np.linalg.solve(xtx, x.T @ y)
    pred = x @ weights
    cos = _rowwise_cosine(pred, y)
    return {
        "weights": weights,
        "standard_mean": standard["mean"],
        "standard_scale": standard["scale"],
        "ridge": float(ridge),
        "train_rows": int(source_features.shape[0]),
        "source_dim": int(source_features.shape[1]),
        "target_dim": int(target_features.shape[1]),
        "train_alignment_cosine_mean": float(np.mean(cos)),
        "train_alignment_cosine_p10": float(np.percentile(cos, 10)),
        "train_alignment_cosine_p90": float(np.percentile(cos, 90)),
    }


def _apply_ridge_map(source_features: np.ndarray, mapper: dict[str, Any]) -> np.ndarray:
    x_body = (source_features - mapper["standard_mean"]) / mapper["standard_scale"]
    x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
    return x @ mapper["weights"]


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numer = np.sum(left * right, axis=1)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numer, np.maximum(denom, 1e-12), out=np.zeros_like(numer), where=denom > 0)


def _source_hidden_features_and_scores(
    rows: list[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    local_files_only: bool,
    normalization: str,
    hidden_layer: int,
    need_scores: bool,
) -> SourceFeatureState:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if normalization not in {"mean", "sum"}:
        raise ValueError(f"unknown normalization {normalization!r}")
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

    flat_features: list[np.ndarray] = []
    scores_by_row: list[list[float]] | None = [] if need_scores else None
    predictions: list[int] | None = [] if need_scores else None
    start = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            prompt = arc_gate._lm_choice_prompt(row)
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            texts = [prompt + " " + choice for choice in row.choices]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            if need_scores:
                output = model(**encoded, output_hidden_states=True)
                logits = output.logits[:, :-1, :]
                labels = encoded["input_ids"][:, 1:]
                token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            else:
                base_model = getattr(model, "model", None)
                if base_model is not None:
                    output = base_model(**encoded, output_hidden_states=True, return_dict=True)
                else:
                    output = model(**encoded, output_hidden_states=True)
                token_logp = None
            hidden_states = output.hidden_states[hidden_layer]
            attention = encoded["attention_mask"].bool()
            row_scores: list[float] = []
            for choice_index in range(len(row.choices)):
                choice_mask = attention[choice_index].clone()
                choice_mask[:prompt_len] = False
                values = hidden_states[choice_index][choice_mask]
                if values.numel() == 0:
                    values = hidden_states[choice_index][attention[choice_index]]
                feature = values.mean(dim=0).detach().cpu().numpy().astype(np.float64)
                norm = np.linalg.norm(feature)
                flat_features.append(feature / max(norm, 1e-12))
                if need_scores and token_logp is not None:
                    valid = encoded["attention_mask"][choice_index, 1:].bool().clone()
                    valid[: max(0, prompt_len - 1)] = False
                    logp_values = token_logp[choice_index][valid]
                    if logp_values.numel() == 0:
                        row_scores.append(float("-inf"))
                    elif normalization == "sum":
                        row_scores.append(float(logp_values.sum().detach().cpu()))
                    else:
                        row_scores.append(float(logp_values.mean().detach().cpu()))
            if need_scores and scores_by_row is not None and predictions is not None:
                scores_by_row.append(row_scores)
                predictions.append(int(max(range(len(row_scores)), key=lambda index: (row_scores[index], -index))))

    return SourceFeatureState(
        pair_features=np.asarray(flat_features, dtype=np.float64),
        scores_by_row=scores_by_row,
        predictions=predictions,
        metadata={
            "kind": "local_causal_lm_hidden_choice_features",
            "model_path": model_path,
            "device": resolved_device,
            "dtype": dtype,
            "max_length": max_length,
            "normalization": normalization if need_scores else None,
            "hidden_layer": hidden_layer,
            "need_scores": need_scores,
            "latency_s": float(time.perf_counter() - start),
        },
    )


def _source_features(
    rows: list[arc_gate.ArcRow],
    *,
    source_feature_mode: str,
    source_feature_dim: int,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_hidden_layer: int,
    local_files_only: bool,
    need_scores: bool,
) -> SourceFeatureState:
    if source_feature_mode == "hashed_pair":
        features = arc_gate._hashed_features(arc_gate._choice_pair_texts(rows), dim=source_feature_dim)
        return SourceFeatureState(
            pair_features=features,
            scores_by_row=None,
            predictions=None,
            metadata={"kind": "hashed_pair_features", "feature_dim": source_feature_dim, "need_scores": need_scores},
        )
    if source_feature_mode == "lm_hidden_last_mean":
        return _source_hidden_features_and_scores(
            rows,
            model_path=source_lm_model,
            device=source_lm_device,
            dtype=source_lm_dtype,
            max_length=source_lm_max_length,
            local_files_only=local_files_only,
            normalization=source_lm_normalization,
            hidden_layer=source_hidden_layer,
            need_scores=need_scores,
        )
    raise ValueError(f"unknown source feature mode {source_feature_mode!r}")


def _flat_to_rows(rows: list[arc_gate.ArcRow], flat: np.ndarray) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for start, end in arc_gate._row_offsets(rows):
        output.append(flat[start:end])
    return output


def _fit_source_score_predictions(
    train_rows: list[arc_gate.ArcRow],
    eval_rows: list[arc_gate.ArcRow],
    train_source_features: np.ndarray,
    eval_source_features: np.ndarray,
    *,
    ridge: float,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    scorer = arc_gate._fit_ridge_pair_scorer(train_rows, train_source_features, ridge=ridge)
    scores, predictions = arc_gate._score_rows(eval_rows, eval_source_features, scorer)
    return scores, predictions, {
        "kind": "train_split_ridge_on_source_features",
        "train_state": {key: value for key, value in scorer.items() if key != "weights"},
    }


def _endpoint_packet_rows(
    *,
    eval_rows: list[arc_gate.ArcRow],
    target_residuals: list[np.ndarray],
    mapped_source_vectors: list[np.ndarray],
    source_predictions: list[int],
    projection: np.ndarray,
    budget_bytes: int,
    index_prior: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    matched_packets: list[tuple[bytes, dict[str, Any]]] = []
    permuted_packets: list[tuple[bytes, dict[str, Any]]] = []
    for row, mapped_rows, selected_index in zip(eval_rows, mapped_source_vectors, source_predictions, strict=True):
        matched_packets.append(
            arc_gate._encode_packet(mapped_rows[selected_index], projection, budget_bytes=budget_bytes)
        )
        alternate = (selected_index + 1) % len(row.choices)
        permuted_packets.append(
            arc_gate._encode_packet(mapped_rows[alternate], projection, budget_bytes=budget_bytes)
        )

    output: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for row_index, row in enumerate(eval_rows):
        for condition in REPORT_CONDITIONS:
            start = time.perf_counter()
            source_index = int(source_predictions[row_index])
            payload: bytes | None = None
            meta: dict[str, Any] = {
                "source_visible_fields": ["question", "choices"],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
                "source_selected_index": source_index,
                "source_selected_label": row.choice_labels[source_index],
                "source_selected_choice_sha256": arc_gate._sha256_text(row.choices[source_index]),
                "source_packet_origin": "source_lm_hidden_feature_ridge_to_target_candidate_residual",
            }

            if condition == "target_only":
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=None,
                    projection=projection,
                    index_prior=index_prior,
                )
            elif condition == MATCHED_CONDITION:
                payload, packet_meta = matched_packets[row_index]
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta
            elif condition == "zero_source":
                payload, packet_meta = arc_gate._encode_packet(
                    np.zeros(projection.shape[0], dtype=np.float64),
                    projection,
                    budget_bytes=budget_bytes,
                )
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta
            elif condition == "shuffled_source_packet":
                other_index = arc_gate._nonself_index(eval_rows, row_index)
                payload, packet_meta = matched_packets[other_index]
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"shuffled_source_row_id": eval_rows[other_index].row_id}
            elif condition == "source_feature_permutation_packet":
                payload, packet_meta = permuted_packets[row_index]
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"permuted_source_choice": (source_index + 1) % len(row.choices)}
            elif condition == "random_same_byte_packet":
                payload = bytes(rng.randrange(256) for _ in range(budget_bytes))
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
            elif condition == "target_derived_sidecar":
                prior = arc_gate._target_prior_index(row, index_prior)
                payload, packet_meta = arc_gate._encode_packet(
                    target_residuals[row_index][prior],
                    projection,
                    budget_bytes=budget_bytes,
                )
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"target_prior_index": prior}
            elif condition == "candidate_derangement":
                payload, packet_meta = matched_packets[row_index]
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=row,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                    derange_candidates=True,
                )
                meta |= packet_meta
            elif condition == "label_permutation":
                permuted = arc_gate._permuted_row(row)
                payload, packet_meta = matched_packets[row_index]
                prediction_index, decode_meta = arc_gate._predict_from_code(
                    row=permuted,
                    residuals=target_residuals[row_index],
                    payload=payload,
                    projection=projection,
                    index_prior=index_prior,
                )
                meta |= packet_meta | {"label_permutation": True}
                output.append(
                    arc_gate._prediction_row(
                        condition=condition,
                        row=permuted,
                        original_row=row,
                        prediction_index=prediction_index,
                        payload=payload,
                        meta=meta,
                        decode_meta=decode_meta,
                        latency_ms=(time.perf_counter() - start) * 1000.0,
                    )
                )
                continue
            elif condition == "same_byte_structured_text":
                payload = row.choices[source_index].encode("utf-8")[:budget_bytes]
                prediction_index, decode_meta = arc_gate._text_payload_prediction(row, payload, index_prior)
            elif condition == "answer_only_text_forbidden_oracle":
                payload = row.answer_label.encode("utf-8")[:budget_bytes]
                prediction_index, decode_meta = arc_gate._answer_text_oracle_prediction(row, payload, index_prior)
            else:
                raise ValueError(f"unknown condition {condition!r}")

            output.append(
                arc_gate._prediction_row(
                    condition=condition,
                    row=row,
                    original_row=row,
                    prediction_index=prediction_index,
                    payload=payload,
                    meta=meta,
                    decode_meta=decode_meta,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            )
    return output


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {condition: arc_gate._summarize([row for row in rows if row["condition"] == condition]) for condition in REPORT_CONDITIONS}


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private ARC-Challenge Source-Latent Endpoint Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval rows: `{payload['train_rows']}` / `{payload['eval_rows']}`",
        f"- packet budget: `{payload['budget_bytes']}B`",
        f"- source feature mode: `{payload['source_feature_mode']}`",
        f"- target feature mode: `{payload['target_feature_mode']}`",
        "",
        "| Condition | Accuracy | Correct / N | Mean bytes |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in payload["condition_metrics"].items():
        lines.append(
            f"| `{condition}` | {metrics['accuracy']:.3f} | {metrics['correct']} / {metrics['n']} | "
            f"{metrics['mean_payload_bytes']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Readout",
            "",
            f"- best destructive control: `{payload['headline']['best_destructive_control']}`",
            f"- matched minus target: `{payload['headline']['matched_minus_target']:.3f}`",
            f"- matched minus best destructive control: `{payload['headline']['matched_minus_best_destructive']:.3f}`",
            f"- matched minus same-byte structured text: `{payload['headline']['matched_minus_same_byte_text']:.3f}`",
            f"- paired CI95 vs target: `{payload['headline']['paired_ci95_vs_target']}`",
            "",
            "This gate is stricter than the ARC fixed-packet bridge because the transmitted vector is emitted "
            "from source-model features through a train-only alignment map rather than copied from the target "
            "candidate residual space.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    train_limit: int | None,
    eval_limit: int | None,
    budget_bytes: int,
    target_feature_dim: int,
    code_dim: int,
    target_feature_mode: str,
    target_feature_model: str,
    target_feature_device: str,
    target_feature_dtype: str,
    target_feature_max_length: int,
    alignment_target_mode: str,
    decoder_target_mode: str,
    source_feature_mode: str,
    source_feature_dim: int,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_hidden_layer: int,
    source_score_mode: str,
    source_score_ridge: float,
    alignment_ridge: float,
    local_files_only: bool,
    seed: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = arc_gate._load_rows(train_path, limit=train_limit)
    eval_rows = arc_gate._load_rows(eval_path, limit=eval_limit)
    overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in eval_rows})

    target_feature_start = time.perf_counter()
    train_target_features = arc_gate._features(
        arc_gate._choice_pair_texts(train_rows),
        dim=target_feature_dim,
        feature_mode=target_feature_mode,
        feature_model=target_feature_model,
        feature_device=target_feature_device,
        feature_dtype=target_feature_dtype,
        feature_max_length=target_feature_max_length,
        local_files_only=local_files_only,
    )
    eval_target_features = arc_gate._features(
        arc_gate._choice_pair_texts(eval_rows),
        dim=target_feature_dim,
        feature_mode=target_feature_mode,
        feature_model=target_feature_model,
        feature_device=target_feature_device,
        feature_dtype=target_feature_dtype,
        feature_max_length=target_feature_max_length,
        local_files_only=local_files_only,
    )
    target_feature_latency_s = float(time.perf_counter() - target_feature_start)
    train_target_residuals = arc_gate._candidate_residuals(train_rows, train_target_features)
    eval_target_residuals = arc_gate._candidate_residuals(eval_rows, eval_target_features)
    if alignment_target_mode == "target_residual":
        train_alignment_target_flat = np.concatenate(train_target_residuals, axis=0)
    elif alignment_target_mode == "target_absolute":
        train_alignment_target_flat = train_target_features
    else:
        raise ValueError(f"unknown alignment target mode {alignment_target_mode!r}")

    if decoder_target_mode == "target_residual":
        eval_decoder_features = eval_target_residuals
    elif decoder_target_mode == "target_absolute":
        eval_decoder_features = _flat_to_rows(eval_rows, eval_target_features)
    else:
        raise ValueError(f"unknown decoder target mode {decoder_target_mode!r}")

    train_source = _source_features(
        train_rows,
        source_feature_mode=source_feature_mode,
        source_feature_dim=source_feature_dim,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_hidden_layer=source_hidden_layer,
        local_files_only=local_files_only,
        need_scores=False,
    )
    eval_source = _source_features(
        eval_rows,
        source_feature_mode=source_feature_mode,
        source_feature_dim=source_feature_dim,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_hidden_layer=source_hidden_layer,
        local_files_only=local_files_only,
        need_scores=source_score_mode == "lm_choice_loglikelihood",
    )

    mapper = _fit_ridge_map(train_source.pair_features, train_alignment_target_flat, ridge=alignment_ridge)
    eval_mapped_flat = _apply_ridge_map(eval_source.pair_features, mapper)
    eval_mapped = _flat_to_rows(eval_rows, eval_mapped_flat)

    if source_score_mode == "source_feature_ridge":
        source_scores, source_predictions, score_state = _fit_source_score_predictions(
            train_rows,
            eval_rows,
            train_source.pair_features,
            eval_source.pair_features,
            ridge=source_score_ridge,
        )
    elif source_score_mode == "lm_choice_loglikelihood":
        if eval_source.scores_by_row is None or eval_source.predictions is None:
            raise ValueError("LM source-score mode requires eval source scores")
        source_scores = eval_source.scores_by_row
        source_predictions = eval_source.predictions
        score_state = {
            "kind": "source_lm_choice_loglikelihood",
            "normalization": source_lm_normalization,
        }
    else:
        raise ValueError(f"unknown source score mode {source_score_mode!r}")

    projection = arc_gate._projection_matrix(target_feature_dim, code_dim, seed=seed + 171)
    priors = arc_gate._index_prior(train_rows)
    prediction_rows = _endpoint_packet_rows(
        eval_rows=eval_rows,
        target_residuals=eval_decoder_features,
        mapped_source_vectors=eval_mapped,
        source_predictions=source_predictions,
        projection=projection,
        budget_bytes=budget_bytes,
        index_prior=priors,
        seed=seed + 911,
    )
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)

    metrics = _condition_metrics(prediction_rows)
    matched = metrics[MATCHED_CONDITION]["accuracy"]
    target = metrics["target_only"]["accuracy"]
    same_byte_text = metrics["same_byte_structured_text"]["accuracy"]
    best_control_name = max(STRICT_DESTRUCTIVE_CONTROLS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    target_ci = arc_gate._paired_bootstrap(
        prediction_rows,
        condition=MATCHED_CONDITION,
        baseline="target_only",
        seed=seed + 1001,
        samples=bootstrap_samples,
    )
    control_ci = arc_gate._paired_bootstrap(
        prediction_rows,
        condition=MATCHED_CONDITION,
        baseline=best_control_name,
        seed=seed + 1002,
        samples=bootstrap_samples,
    )
    source_accuracy = float(
        sum(int(prediction == row.answer_index) for prediction, row in zip(source_predictions, eval_rows, strict=True))
        / len(eval_rows)
    )
    pass_gate = (
        not overlap
        and matched >= target + min_lift_over_target
        and matched >= best_control + min_gap_over_control
        and matched >= same_byte_text + min_gap_over_text
        and target_ci["ci95_low"] > 0.0
        and metrics["candidate_derangement"]["accuracy"] <= target + 0.05
        and metrics["source_feature_permutation_packet"]["accuracy"] <= max(target + 0.08, same_byte_text + 0.02)
    )
    payload = {
        "gate": "source_private_arc_challenge_source_latent_endpoint_gate",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "eval_path": _display_path(eval_path),
        "train_sha256": _sha256_file(train_path),
        "eval_sha256": _sha256_file(eval_path),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_eval_content_overlap_count": len(overlap),
        "train_eval_content_overlap_sha256": hashlib.sha256("\n".join(overlap).encode("utf-8")).hexdigest(),
        "budget_bytes": budget_bytes,
        "target_feature_dim": target_feature_dim,
        "code_dim": code_dim,
        "target_feature_mode": target_feature_mode,
        "alignment_target_mode": alignment_target_mode,
        "decoder_target_mode": decoder_target_mode,
        "target_feature_model": target_feature_model if target_feature_mode.startswith("hf_") else None,
        "target_feature_device": (
            arc_gate.syn._resolve_torch_device(target_feature_device)
            if target_feature_mode.startswith("hf_")
            else None
        ),
        "target_feature_dtype": target_feature_dtype if target_feature_mode.startswith("hf_") else None,
        "target_feature_max_length": target_feature_max_length if target_feature_mode.startswith("hf_") else None,
        "target_feature_latency_s": target_feature_latency_s,
        "source_feature_mode": source_feature_mode,
        "source_feature_dim": int(train_source.pair_features.shape[1]),
        "source_model": {
            **train_source.metadata,
            "eval_feature_metadata": eval_source.metadata,
            "source_score_mode": source_score_mode,
            "source_score_state": score_state,
            "source_eval_accuracy_before_packet": source_accuracy,
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "source_score_digest": hashlib.sha256(
                json.dumps(source_scores, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "alignment": {
            "kind": "train_only_ridge_source_features_to_target_candidate_space",
            "alignment_target_mode": alignment_target_mode,
            "decoder_target_mode": decoder_target_mode,
            "ridge": alignment_ridge,
            "source_dim": mapper["source_dim"],
            "target_dim": mapper["target_dim"],
            "train_pair_rows": mapper["train_rows"],
            "train_alignment_cosine_mean": mapper["train_alignment_cosine_mean"],
            "train_alignment_cosine_p10": mapper["train_alignment_cosine_p10"],
            "train_alignment_cosine_p90": mapper["train_alignment_cosine_p90"],
        },
        "method_contract": {
            "fixed_packet_budget_bytes": budget_bytes,
            "packet_format": "top-k signed random-projection residual sketch; two bytes per selected dimension",
            "source_packet_inputs_at_eval": ["source LM hidden/logit features from question and choices"],
            "forbidden_eval_source_inputs": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "target_side_information": "candidate answer texts and public projection matrix",
            "claim_boundary": (
                "Mac-local source-latent endpoint bridge. The packet vector is emitted from source-model "
                "features through a train-only alignment map; it is not native GPU systems evidence."
            ),
        },
        "condition_metrics": metrics,
        "headline": {
            "matched_accuracy": matched,
            "target_accuracy": target,
            "same_byte_structured_text_accuracy": same_byte_text,
            "best_destructive_control": best_control_name,
            "best_destructive_control_accuracy": best_control,
            "source_feature_permutation_accuracy": metrics["source_feature_permutation_packet"]["accuracy"],
            "candidate_derangement_accuracy": metrics["candidate_derangement"]["accuracy"],
            "matched_minus_target": matched - target,
            "matched_minus_best_destructive": matched - best_control,
            "matched_minus_same_byte_text": matched - same_byte_text,
            "paired_ci95_vs_target": target_ci,
            "paired_ci95_vs_best_destructive": control_ci,
        },
        "pass_gate": bool(pass_gate),
        "public_benchmark_source_latent_endpoint_ready": bool(pass_gate),
        "pass_rule": (
            "Pass requires no train/eval content overlap; source-latent packet beats target-only, the best "
            "strict destructive control, and same-byte structured text; paired CI95 lower bound versus target is "
            "positive; candidate derangement collapses; source-feature permutation is not a stronger route."
        ),
    }
    (output_dir / "source_latent_endpoint_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "source_latent_endpoint_gate.md", payload)
    manifest = {
        "artifacts": [
            "source_latent_endpoint_gate.json",
            "source_latent_endpoint_gate.md",
            "predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ("source_latent_endpoint_gate.json", "source_latent_endpoint_gate.md", "predictions.jsonl")
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private ARC-Challenge Source-Latent Endpoint Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- public benchmark source-latent endpoint ready: `{payload['public_benchmark_source_latent_endpoint_ready']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--target-feature-dim", type=int, default=384)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--target-feature-mode", choices=("hashed", "hf_last_mean", "hf_mid_last_mean"), default="hashed")
    parser.add_argument("--target-feature-model", default="BAAI/bge-small-en")
    parser.add_argument("--target-feature-device", default="auto")
    parser.add_argument("--target-feature-dtype", default="float32")
    parser.add_argument("--target-feature-max-length", type=int, default=128)
    parser.add_argument(
        "--alignment-target-mode",
        choices=("target_residual", "target_absolute"),
        default="target_residual",
    )
    parser.add_argument(
        "--decoder-target-mode",
        choices=("target_residual", "target_absolute"),
        default="target_residual",
    )
    parser.add_argument("--source-feature-mode", choices=("hashed_pair", "lm_hidden_last_mean"), default="hashed_pair")
    parser.add_argument("--source-feature-dim", type=int, default=384)
    parser.add_argument(
        "--source-lm-model",
        default="/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    )
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument(
        "--source-score-mode",
        choices=("source_feature_ridge", "lm_choice_loglikelihood"),
        default="source_feature_ridge",
    )
    parser.add_argument("--source-score-ridge", type=float, default=0.25)
    parser.add_argument("--alignment-ridge", type=float, default=1.0)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lift-over-target", type=float, default=0.03)
    parser.add_argument("--min-gap-over-control", type=float, default=0.03)
    parser.add_argument("--min-gap-over-text", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_gate(
        output_dir=args.output_dir,
        train_path=_resolve(args.train_path),
        eval_path=_resolve(args.eval_path),
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        budget_bytes=args.budget_bytes,
        target_feature_dim=args.target_feature_dim,
        code_dim=args.code_dim,
        target_feature_mode=args.target_feature_mode,
        target_feature_model=args.target_feature_model,
        target_feature_device=args.target_feature_device,
        target_feature_dtype=args.target_feature_dtype,
        target_feature_max_length=args.target_feature_max_length,
        alignment_target_mode=args.alignment_target_mode,
        decoder_target_mode=args.decoder_target_mode,
        source_feature_mode=args.source_feature_mode,
        source_feature_dim=args.source_feature_dim,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_hidden_layer=args.source_hidden_layer,
        source_score_mode=args.source_score_mode,
        source_score_ridge=args.source_score_ridge,
        alignment_ridge=args.alignment_ridge,
        local_files_only=not args.allow_downloads,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
    )


if __name__ == "__main__":
    main()
