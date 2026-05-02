from __future__ import annotations

"""Build a cross-family receiver scout for HellaSwag source-private packets."""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_receiver_family_packet_gate_20260502")
DEFAULT_SOURCE_PACKET_JSONL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "bagged_gate/predictions.jsonl"
)
DEFAULT_TARGET_GLOBAL_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/"
    "hellaswag_hidden_innovation_global_stability.json"
)
DEFAULT_SOURCE_PACKET_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "hellaswag_hidden_innovation_eval_slice_stress.json"
)
DEFAULT_SOURCE_FAMILY = "TinyLlama"
DEFAULT_TARGET_FAMILY = "Qwen2.5"
DEFAULT_PACKET_FIELD = "selected_prediction"
DEFAULT_PACKET_MARGIN_FIELD = "selected_margin"
DEFAULT_CONTROL_FIELDS = (
    "wrong_example_hidden_prediction",
    "candidate_roll_hidden_prediction",
    "zero_hidden_prediction",
    "source_label_prediction",
)
DEFAULT_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
STRICT_TARGET_DELTA = 0.02
STRICT_PACKET_DELTA = 0.005


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
    rng = np.random.default_rng(seed)
    boot = [
        float(np.mean(delta[rng.integers(0, len(delta), len(delta))]))
        for _ in range(int(samples))
    ]
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _target_margin(scores: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    ranks = np.argsort(-scores, axis=1)
    rank_positions = np.zeros_like(ranks)
    for row_index in range(scores.shape[0]):
        for rank, candidate in enumerate(ranks[row_index]):
            rank_positions[row_index, candidate] = rank
    return rank_positions


def _candidate_features(
    *,
    target_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
) -> np.ndarray:
    centered = target_scores - np.mean(target_scores, axis=1, keepdims=True)
    scale = np.std(target_scores, axis=1, keepdims=True)
    target_z = centered / np.where(scale < 1e-6, 1.0, scale)
    probs = _row_softmax(target_scores)
    ranks = _rank_positions(target_scores)
    target_predictions = np.argmax(target_scores, axis=1)
    feature_rows: list[list[list[float]]] = []
    for row_index in range(target_scores.shape[0]):
        row_features: list[list[float]] = []
        for candidate in range(4):
            candidate_one_hot = [1.0 if candidate == item else 0.0 for item in range(4)]
            rank_one_hot = [1.0 if int(ranks[row_index, candidate]) == item else 0.0 for item in range(4)]
            packet_match = float(candidate == int(packet_predictions[row_index]))
            row_features.append(
                [
                    1.0,
                    float(target_scores[row_index, candidate]),
                    float(target_z[row_index, candidate]),
                    float(probs[row_index, candidate]),
                    float(ranks[row_index, candidate]),
                    packet_match,
                    float(packet_margins[row_index]) if packet_match else 0.0,
                    float(packet_margins[row_index]),
                    float(candidate == int(target_predictions[row_index])),
                    *candidate_one_hot,
                    *rank_one_hot,
                ]
            )
        feature_rows.append(row_features)
    return np.asarray(feature_rows, dtype=np.float64)


def _fit_ridge_receiver(
    *,
    target_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
    ridges: tuple[float, ...],
) -> dict[str, Any]:
    features = _candidate_features(
        target_scores=target_scores,
        packet_predictions=packet_predictions,
        packet_margins=packet_margins,
    )

    def fit(ridge: float) -> dict[str, Any]:
        x_fit = features[train_indices].reshape(-1, features.shape[-1])
        labels: list[float] = []
        weights: list[float] = []
        target_predictions = np.argmax(target_scores, axis=1)
        for index in train_indices:
            for candidate in range(4):
                labels.append(1.0 if int(answers[index]) == candidate else -1.0)
                weights.append(
                    1.0
                    if candidate
                    in {
                        int(answers[index]),
                        int(target_predictions[index]),
                        int(packet_predictions[index]),
                    }
                    else 0.5
                )
        y = np.asarray(labels, dtype=np.float64)
        sample_weights = np.asarray(weights, dtype=np.float64)
        mean = np.mean(x_fit, axis=0)
        scale = np.std(x_fit, axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        x_body = (x_fit - mean) / scale
        x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
        weighted_x = x * sample_weights[:, None]
        xtx = x.T @ weighted_x + float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        xtx[0, 0] -= float(ridge)
        beta = np.linalg.solve(xtx, weighted_x.T @ y)
        return {"ridge": float(ridge), "mean": mean, "scale": scale, "beta": beta}

    def predict(model: dict[str, Any], packet: np.ndarray) -> np.ndarray:
        packet_features = _candidate_features(
            target_scores=target_scores,
            packet_predictions=packet,
            packet_margins=packet_margins,
        )
        x_body = (packet_features.reshape(-1, packet_features.shape[-1]) - model["mean"]) / model["scale"]
        x = np.concatenate([np.ones((x_body.shape[0], 1), dtype=np.float64), x_body], axis=1)
        scores = (x @ model["beta"]).reshape(target_scores.shape[0], 4)
        return np.argmax(scores, axis=1).astype(np.int64)

    models = []
    for ridge in ridges:
        model = fit(ridge)
        predictions = predict(model, packet_predictions)
        models.append(
            {
                "model": model,
                "train_accuracy": _accuracy(predictions, answers, train_indices),
            }
        )
    selected = max(models, key=lambda item: (item["train_accuracy"], -item["model"]["ridge"]))
    selected_model = selected["model"]
    return {
        "kind": "candidate_ridge_receiver",
        "ridge": selected_model["ridge"],
        "train_accuracy": selected["train_accuracy"],
        "predictions": predict(selected_model, packet_predictions),
        "control_predictor": lambda packet: predict(selected_model, packet),
    }


def _fit_margin_receiver(
    *,
    target_scores: np.ndarray,
    packet_predictions: np.ndarray,
    packet_margins: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    target_predictions = np.argmax(target_scores, axis=1).astype(np.int64)
    margins = _target_margin(target_scores)
    thresholds = np.quantile(margins[train_indices], np.linspace(0.0, 1.0, 101))

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accept_packet = margins <= float(threshold)
        predictions = np.where(accept_packet, packet_predictions, target_predictions).astype(np.int64)
        rows.append(
            {
                "threshold": float(threshold),
                "train_accuracy": _accuracy(predictions, answers, train_indices),
                "train_accept_rate": float(np.mean(accept_packet[train_indices])),
            }
        )
    selected = max(rows, key=lambda item: (item["train_accuracy"], -item["train_accept_rate"]))

    def predict(packet: np.ndarray) -> np.ndarray:
        accept_packet = margins <= selected["threshold"]
        return np.where(accept_packet, packet, target_predictions).astype(np.int64)

    return {
        "kind": "target_margin_accept_packet",
        "threshold": selected["threshold"],
        "train_accuracy": selected["train_accuracy"],
        "train_accept_rate": selected["train_accept_rate"],
        "predictions": predict(packet_predictions),
        "control_predictor": predict,
    }


def _receiver_readout(
    *,
    name: str,
    predictions: np.ndarray,
    answers: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": name,
        "train_accuracy": _accuracy(predictions, answers, train_indices),
        "eval_accuracy": _accuracy(predictions, answers, eval_indices),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Receiver-Family Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- target-family transfer gate: `{payload['target_family_transfer_gate']}`",
        f"- receiver-improvement gate: `{payload['receiver_improvement_gate']}`",
        f"- source family: `{h['source_family']}`",
        f"- target family: `{h['target_family']}`",
        f"- train/eval split: validation `0:{h['train_rows']}` train, `{h['train_rows']}:{h['row_count']}` eval",
        f"- selected receiver: `{h['selected_receiver_kind']}`",
        f"- target-only eval accuracy: `{h['target_only_eval_accuracy']:.6f}`",
        f"- packet-only eval accuracy: `{h['packet_only_eval_accuracy']:.6f}`",
        f"- receiver eval accuracy: `{h['receiver_eval_accuracy']:.6f}`",
        f"- delta vs target-only: `{h['receiver_minus_target_only']:.6f}`",
        f"- delta vs packet-only: `{h['receiver_minus_packet_only']:.6f}`",
        f"- oracle target-or-packet eval accuracy: `{h['target_or_packet_oracle_eval_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    source_packet_jsonl: pathlib.Path = DEFAULT_SOURCE_PACKET_JSONL,
    target_global_artifact: pathlib.Path = DEFAULT_TARGET_GLOBAL_ARTIFACT,
    source_packet_artifact: pathlib.Path = DEFAULT_SOURCE_PACKET_ARTIFACT,
    source_family: str = DEFAULT_SOURCE_FAMILY,
    target_family: str = DEFAULT_TARGET_FAMILY,
    packet_field: str = DEFAULT_PACKET_FIELD,
    packet_margin_field: str = DEFAULT_PACKET_MARGIN_FIELD,
    control_fields: tuple[str, ...] = DEFAULT_CONTROL_FIELDS,
    train_prefix_rows: int = 1024,
    bootstrap_samples: int = 1000,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    packet_rows = _read_jsonl(source_packet_jsonl)
    target = _load_target_score_cache(target_global_artifact)
    row_ids = [str(row["row_id"]) for row in packet_rows]
    if row_ids != target["row_ids"]:
        raise ValueError("source packet rows and target score rows are not aligned")
    if not 0 < int(train_prefix_rows) < len(packet_rows):
        raise ValueError("train_prefix_rows must leave at least one heldout eval row")

    answers = np.asarray([int(row["answer_index"]) for row in packet_rows], dtype=np.int64)
    target_scores = target["scores"]
    target_predictions = np.argmax(target_scores, axis=1).astype(np.int64)
    packet_predictions = np.asarray([int(row[packet_field]) for row in packet_rows], dtype=np.int64)
    packet_margins = np.asarray(
        [float(row.get(packet_margin_field, 0.0)) for row in packet_rows],
        dtype=np.float64,
    )
    train_indices = np.arange(int(train_prefix_rows), dtype=np.int64)
    eval_indices = np.arange(int(train_prefix_rows), len(packet_rows), dtype=np.int64)

    ridge_receiver = _fit_ridge_receiver(
        target_scores=target_scores,
        packet_predictions=packet_predictions,
        packet_margins=packet_margins,
        answers=answers,
        train_indices=train_indices,
        ridges=ridges,
    )
    margin_receiver = _fit_margin_receiver(
        target_scores=target_scores,
        packet_predictions=packet_predictions,
        packet_margins=packet_margins,
        answers=answers,
        train_indices=train_indices,
    )
    receiver_candidates = (ridge_receiver, margin_receiver)
    selected_receiver = max(
        receiver_candidates,
        key=lambda item: (
            _accuracy(item["predictions"], answers, train_indices),
            _accuracy(item["predictions"], answers, eval_indices),
            item["kind"] == "target_margin_accept_packet",
        ),
    )
    receiver_predictions = selected_receiver["predictions"]
    packet_only = packet_predictions
    oracle = np.where(packet_predictions == answers, packet_predictions, target_predictions)

    baseline_rows = [
        _receiver_readout(
            name="target_only",
            predictions=target_predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _receiver_readout(
            name="packet_only",
            predictions=packet_only,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _receiver_readout(
            name="receiver_selected",
            predictions=receiver_predictions,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
        _receiver_readout(
            name="target_or_packet_oracle",
            predictions=oracle,
            answers=answers,
            train_indices=train_indices,
            eval_indices=eval_indices,
        ),
    ]

    control_rows: list[dict[str, Any]] = []
    for offset, field in enumerate(control_fields):
        if field not in packet_rows[0]:
            continue
        control_packet = np.asarray([int(row[field]) for row in packet_rows], dtype=np.int64)
        control_predictions = selected_receiver["control_predictor"](control_packet)
        ci = _paired_ci(
            selected=receiver_predictions,
            baseline=control_predictions,
            answers=answers,
            indices=eval_indices,
            seed=9100 + offset,
            samples=bootstrap_samples,
        )
        control_rows.append(
            {
                **_receiver_readout(
                    name=field,
                    predictions=control_predictions,
                    answers=answers,
                    train_indices=train_indices,
                    eval_indices=eval_indices,
                ),
                "receiver_minus_control": ci["delta"],
                "ci95_low_vs_receiver": ci["ci95_low"],
                "ci95_high_vs_receiver": ci["ci95_high"],
            }
        )

    ci_target = _paired_ci(
        selected=receiver_predictions,
        baseline=target_predictions,
        answers=answers,
        indices=eval_indices,
        seed=9001,
        samples=bootstrap_samples,
    )
    ci_packet = _paired_ci(
        selected=receiver_predictions,
        baseline=packet_only,
        answers=answers,
        indices=eval_indices,
        seed=9002,
        samples=bootstrap_samples,
    )
    target_family_transfer_gate = bool(
        ci_target["delta"] >= STRICT_TARGET_DELTA
        and ci_target["ci95_low"] > 0.0
        and all(row["receiver_minus_control"] >= STRICT_TARGET_DELTA for row in control_rows)
    )
    receiver_improvement_gate = bool(
        ci_packet["delta"] >= STRICT_PACKET_DELTA
        and ci_packet["ci95_low"] > 0.0
    )
    pass_gate = bool(target_family_transfer_gate and receiver_improvement_gate)
    headline = {
        "source_family": source_family,
        "target_family": target_family,
        "row_count": len(packet_rows),
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "packet_field": packet_field,
        "packet_margin_field": packet_margin_field,
        "selected_receiver_kind": selected_receiver["kind"],
        "selected_receiver_train_accuracy": _accuracy(receiver_predictions, answers, train_indices),
        "receiver_eval_accuracy": _accuracy(receiver_predictions, answers, eval_indices),
        "target_only_eval_accuracy": _accuracy(target_predictions, answers, eval_indices),
        "packet_only_eval_accuracy": _accuracy(packet_only, answers, eval_indices),
        "target_or_packet_oracle_eval_accuracy": _accuracy(oracle, answers, eval_indices),
        "receiver_minus_target_only": ci_target["delta"],
        "receiver_ci95_low_vs_target_only": ci_target["ci95_low"],
        "receiver_ci95_high_vs_target_only": ci_target["ci95_high"],
        "receiver_minus_packet_only": ci_packet["delta"],
        "receiver_ci95_low_vs_packet_only": ci_packet["ci95_low"],
        "receiver_ci95_high_vs_packet_only": ci_packet["ci95_high"],
        "strict_target_delta_required": STRICT_TARGET_DELTA,
        "strict_packet_delta_required": STRICT_PACKET_DELTA,
    }
    interpretation = (
        "This scout tests whether a different-family target score receiver can use the TinyLlama "
        "hidden-innovation packet. It shows target-family utility when the receiver is compared "
        "with Qwen target-only scoring, but it does not promote a true receiver-improvement claim "
        "unless the target-aware receiver beats packet-only. The remaining ICLR gap is therefore a "
        "receiver/common-language method that closes the target-or-packet oracle headroom rather "
        "than merely trusting the source-side packet."
    )
    payload = {
        "gate": "source_private_hellaswag_receiver_family_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "target_family_transfer_gate": target_family_transfer_gate,
        "receiver_improvement_gate": receiver_improvement_gate,
        "pass_rule": (
            "ICLR strict pass requires a different-family target-score receiver that beats target-only "
            "by >=0.02 with positive paired CI, keeps destructive packet controls below the selected "
            "receiver by >=0.02, and also beats packet-only by >=0.005 with positive paired CI. "
            "The target-family transfer subgate alone is not enough because a trivial receiver can "
            "trust the source packet without learning a target-specific communication interface."
        ),
        "headline": headline,
        "selected_receiver": {
            key: value
            for key, value in selected_receiver.items()
            if key not in {"predictions", "control_predictor", "model", "mean", "scale", "beta"}
        },
        "receiver_candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"predictions", "control_predictor", "model", "mean", "scale", "beta"}
            }
            for candidate in receiver_candidates
        ],
        "baseline_rows": baseline_rows,
        "control_rows": control_rows,
        "target_ci": ci_target,
        "packet_ci": ci_packet,
        "source_packet": {
            "predictions_jsonl": _display_path(source_packet_jsonl),
            "predictions_sha256": _sha256_file(source_packet_jsonl),
            "artifact_path": _display_path(source_packet_artifact),
            "artifact_sha256": _sha256_file(source_packet_artifact),
        },
        "target_scores": {
            "artifact_path": target["artifact_path"],
            "artifact_sha256": target["artifact_sha256"],
            "row_count": len(target["row_ids"]),
            "score_slice_count": len(target["slices"]),
            "slices": target["slices"],
        },
        "interpretation": interpretation,
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hellaswag_receiver_family_packet_gate.json"
    md_path = output_dir / "hellaswag_receiver_family_packet_gate.md"
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
        "inputs": [
            payload["source_packet"],
            {
                "artifact_path": target["artifact_path"],
                "artifact_sha256": target["artifact_sha256"],
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one ridge value is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-packet-jsonl", type=pathlib.Path, default=DEFAULT_SOURCE_PACKET_JSONL)
    parser.add_argument("--target-global-artifact", type=pathlib.Path, default=DEFAULT_TARGET_GLOBAL_ARTIFACT)
    parser.add_argument("--source-packet-artifact", type=pathlib.Path, default=DEFAULT_SOURCE_PACKET_ARTIFACT)
    parser.add_argument("--source-family", default=DEFAULT_SOURCE_FAMILY)
    parser.add_argument("--target-family", default=DEFAULT_TARGET_FAMILY)
    parser.add_argument("--packet-field", default=DEFAULT_PACKET_FIELD)
    parser.add_argument("--packet-margin-field", default=DEFAULT_PACKET_MARGIN_FIELD)
    parser.add_argument("--control-fields", type=_parse_tuple, default=DEFAULT_CONTROL_FIELDS)
    parser.add_argument("--train-prefix-rows", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        source_packet_jsonl=args.source_packet_jsonl,
        target_global_artifact=args.target_global_artifact,
        source_packet_artifact=args.source_packet_artifact,
        source_family=args.source_family,
        target_family=args.target_family,
        packet_field=args.packet_field,
        packet_margin_field=args.packet_margin_field,
        control_fields=args.control_fields,
        train_prefix_rows=args.train_prefix_rows,
        bootstrap_samples=args.bootstrap_samples,
        ridges=args.ridges,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
