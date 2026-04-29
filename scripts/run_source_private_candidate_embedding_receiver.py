from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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

from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _deterministic_nonself_index,
    _mask_log_components,
    _mask_repair_diag,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_learned_syndrome import (  # noqa: E402
    _augment,
    _bitpack,
    _bytes_to_bits,
    _candidate_matrix,
    _fit_ridge_encoder,
    _hashed_text_features,
    _normalize_rows,
    _source_vector,
    _token_count,
)


CONDITIONS = [
    "target_only",
    "matched_candidate_embedding_receiver",
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "random_same_byte",
    "target_derived_sidecar",
    "answer_only",
    "structured_text_matched",
    "wrong_projection_source",
    "full_diag_oracle",
]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_text(example: Example, *, mode: str) -> str:
    if mode == "matched":
        return example.private_test_log
    if mode == "answer_masked":
        masked = _mask_repair_diag(example.private_test_log)
        masked = _mask_log_components(masked, mask_expected_actual=True, mask_test_name=True)
        lines = []
        for line in masked.splitlines():
            if line.startswith("hidden_input="):
                lines.append("hidden_input=<MASKED>")
            elif "repair_family=" in line:
                lines.append("repair_family=<MASKED>")
            elif line.startswith("failure_status="):
                lines.append("failure_status=<MASKED>")
            else:
                lines.append(line)
        return "\n".join(lines)
    if mode == "zero":
        return ""
    raise ValueError(f"unknown source mode {mode!r}")


def _source_vector_mode(example: Example, feature_dim: int, *, mode: str) -> np.ndarray:
    if mode in {"matched", "answer_masked", "zero"}:
        return _hashed_text_features(_source_text(example, mode=mode), feature_dim, namespace="source").astype(np.float32)
    raise ValueError(f"unknown source mode {mode!r}")


def _packet_from_vector(vec: np.ndarray, code_projection: np.ndarray, budget_bytes: int) -> bytes:
    bit_count = budget_bytes * 8
    logits = code_projection[:bit_count] @ vec
    return _bitpack((logits >= 0).astype(np.uint8), budget_bytes)


def _source_packet(
    example: Example,
    *,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    mode: str,
) -> bytes:
    predicted = _augment(_source_vector_mode(example, feature_dim, mode=mode)) @ encoder
    return _packet_from_vector(predicted, code_projection, budget_bytes)


def _candidate_codes(example: Example, code_projection: np.ndarray, feature_dim: int, bit_count: int) -> np.ndarray:
    candidates = _candidate_matrix(example, feature_dim)
    logits = candidates @ code_projection[:bit_count].T
    return (logits >= 0).astype(np.uint8)


def _packet_bits(payload: bytes | None, bit_count: int) -> np.ndarray:
    return _bytes_to_bits(payload, bit_count).astype(np.float32)


def _receiver_features(example: Example, payload: bytes | None, *, code_projection: np.ndarray, feature_dim: int, budget_bytes: int) -> np.ndarray:
    bit_count = budget_bytes * 8
    bits = _packet_bits(payload, bit_count)
    signed_packet = bits * 2.0 - 1.0
    codes = _candidate_codes(example, code_projection, feature_dim, bit_count).astype(np.float32)
    signed_codes = codes * 2.0 - 1.0
    interaction = signed_codes * signed_packet[None, :]
    similarity = interaction.mean(axis=1, keepdims=True)
    hamming = (codes != bits[None, :]).mean(axis=1, keepdims=True)
    priors = np.array([[float(candidate.prior_score)] for candidate in example.candidates], dtype=np.float32)
    candidate_feats = _candidate_matrix(example, feature_dim)[:, : min(32, feature_dim)]
    return np.concatenate([np.ones((len(example.candidates), 1), dtype=np.float32), priors, similarity, hamming, interaction, candidate_feats], axis=1)


def _fit_receiver(
    train_examples: list[Example],
    *,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    ridge: float,
) -> np.ndarray:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for example in train_examples:
        payload = _source_packet(
            example,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        features = _receiver_features(example, payload, code_projection=code_projection, feature_dim=feature_dim, budget_bytes=budget_bytes)
        for idx, candidate in enumerate(example.candidates):
            xs.append(features[idx])
            ys.append(1.0 if candidate.label == example.answer_label else 0.0)
    x = np.stack(xs, axis=0).astype(np.float64)
    y = np.array(ys, dtype=np.float64)
    xtx = x.T @ x
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= ridge
    return np.linalg.solve(xtx, x.T @ y).astype(np.float32)


def _predict_with_receiver(
    example: Example,
    payload: bytes | None,
    *,
    receiver: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    margin_threshold: float,
) -> tuple[str, dict[str, Any]]:
    if payload is None:
        return _prior_prediction(example), {"decoder": "prior"}
    features = _receiver_features(example, payload, code_projection=code_projection, feature_dim=feature_dim, budget_bytes=budget_bytes)
    scores = features @ receiver
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    prior_index = labels.index(prior)
    best_score = float(np.max(scores))
    tied = [idx for idx, score in enumerate(scores) if abs(float(score) - best_score) <= 1e-8]
    margin_vs_prior = best_score - float(scores[prior_index])
    if labels[tied[0]] != prior and margin_vs_prior < margin_threshold:
        return prior, {
            "decoder": "learned_candidate_embedding_target_preserve",
            "scores": [float(score) for score in scores],
            "ties": tied,
            "margin_vs_prior": margin_vs_prior,
            "margin_threshold": margin_threshold,
            "preserved_prior": True,
        }
    if any(labels[idx] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[tied[0]]
    return prediction, {
        "decoder": "learned_candidate_embedding_target_preserve",
        "scores": [float(score) for score in scores],
        "ties": tied,
        "margin_vs_prior": margin_vs_prior,
        "margin_threshold": margin_threshold,
        "preserved_prior": False,
    }


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    rng: random.Random,
    random_encoder: np.ndarray,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {}
    if condition == "matched_candidate_embedding_receiver":
        return (
            _source_packet(
                example,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                mode="matched",
            ),
            {"source": example.example_id},
        )
    if condition == "shuffled_source":
        other = eval_examples[_deterministic_nonself_index(index, len(eval_examples))]
        return (
            _source_packet(
                other,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                mode="matched",
            ),
            {"source": other.example_id},
        )
    if condition == "answer_masked_source":
        return (
            _source_packet(
                example,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                mode="answer_masked",
            ),
            {"source": "answer_masked"},
        )
    if condition == "random_same_byte":
        return rng.randbytes(budget_bytes), {"source": "random"}
    if condition == "target_derived_sidecar":
        prior = _prior_prediction(example)
        prior_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)
        return _packet_from_vector(_candidate_matrix(example, feature_dim)[prior_index], code_projection, budget_bytes), {"source": "target_prior"}
    if condition == "answer_only":
        return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_hidden_log"}
    if condition == "wrong_projection_source":
        predicted = _augment(_source_vector(example, feature_dim, mode="matched")) @ random_encoder
        return _packet_from_vector(predicted, code_projection, budget_bytes), {"source": "wrong_encoder"}
    if condition == "full_diag_oracle":
        answer_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)
        return _packet_from_vector(_candidate_matrix(example, feature_dim)[answer_index], code_projection, budget_bytes), {"source": "candidate_oracle"}
    raise ValueError(f"unknown condition {condition!r}")


def _evaluate_for_threshold(
    examples: list[Example],
    *,
    conditions: list[str],
    encoder: np.ndarray,
    receiver: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    margin_threshold: float,
    random_encoder: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    by_condition: dict[str, list[bool]] = {condition: [] for condition in conditions}
    for row_index, example in enumerate(examples):
        for condition in conditions:
            payload, _ = _payload_for_condition(
                condition=condition,
                example=example,
                eval_examples=examples,
                index=row_index,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                rng=rng,
                random_encoder=random_encoder,
            )
            prediction, _ = _predict_with_receiver(
                example,
                payload,
                receiver=receiver,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                margin_threshold=margin_threshold,
            )
            by_condition[condition].append(prediction == example.answer_label)
    return {condition: sum(values) / len(values) for condition, values in by_condition.items()}


def _calibrate_margin_threshold(
    calibration_examples: list[Example],
    *,
    encoder: np.ndarray,
    receiver: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    random_encoder: np.ndarray,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    controls = [
        "zero_source",
        "shuffled_source",
        "answer_masked_source",
        "random_same_byte",
        "target_derived_sidecar",
        "answer_only",
        "structured_text_matched",
        "wrong_projection_source",
    ]
    candidate_thresholds = [0.0]
    for example in calibration_examples:
        payload = _source_packet(
            example,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        scores = _receiver_features(example, payload, code_projection=code_projection, feature_dim=feature_dim, budget_bytes=budget_bytes) @ receiver
        labels = [candidate.label for candidate in example.candidates]
        prior_index = labels.index(_prior_prediction(example))
        candidate_thresholds.append(max(0.0, float(np.max(scores) - scores[prior_index])))
    ordered_thresholds = sorted(candidate_thresholds)
    quantiles = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
    thresholds = sorted(
        set(
            round(ordered_thresholds[min(len(ordered_thresholds) - 1, max(0, int(q * (len(ordered_thresholds) - 1))))], 6)
            for q in quantiles
        )
        | {0.0}
    )
    best_threshold = 0.0
    best_record: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = _evaluate_for_threshold(
            calibration_examples,
            conditions=["target_only", "matched_candidate_embedding_receiver", *controls],
            encoder=encoder,
            receiver=receiver,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            margin_threshold=threshold,
            random_encoder=random_encoder,
            seed=seed,
        )
        target = metrics["target_only"]
        best_control = max(metrics[name] for name in controls)
        control_ok = all(metrics[name] <= target + 0.05 for name in controls)
        record = {
            "threshold": threshold,
            "matched_accuracy": metrics["matched_candidate_embedding_receiver"],
            "target_accuracy": target,
            "best_control_accuracy": best_control,
            "control_ok": control_ok,
        }
        if best_record is None:
            best_record = record
            best_threshold = threshold
        elif control_ok and (
            not best_record["control_ok"]
            or record["matched_accuracy"] > best_record["matched_accuracy"]
            or (
                record["matched_accuracy"] == best_record["matched_accuracy"]
                and record["best_control_accuracy"] < best_record["best_control_accuracy"]
            )
        ):
            best_record = record
            best_threshold = threshold
        elif not best_record["control_ok"] and record["best_control_accuracy"] < best_record["best_control_accuracy"]:
            best_record = record
            best_threshold = threshold
    assert best_record is not None
    return best_threshold, best_record


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    receiver: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    margin_threshold: float,
    rng: random.Random,
    random_encoder: np.ndarray,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, metadata = _payload_for_condition(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        encoder=encoder,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        rng=rng,
        random_encoder=random_encoder,
    )
    prediction, decode_meta = _predict_with_receiver(
        example,
        payload,
        receiver=receiver,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        margin_threshold=0.0 if condition == "full_diag_oracle" else margin_threshold,
    )
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "payload_hex": payload_hex,
        "metadata": {**metadata, **decode_meta},
    }


def _summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in predictions]
    correct = sum(1 for row in predictions if row["correct"])
    return {
        "n": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in predictions),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in predictions),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    budgets: list[int],
    train_seed: int,
    eval_seed: int,
    ridge: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(examples=train_examples, candidates=candidates, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=candidates, seed=eval_seed, family_set=eval_family_set)
    encoder = _fit_ridge_encoder(train_rows, feature_dim=feature_dim, ridge=ridge)
    rng_np = np.random.default_rng(train_seed * 1009 + eval_seed)
    code_projection = _normalize_rows(rng_np.normal(size=(max(budgets) * 8, feature_dim))).astype(np.float32)
    random_encoder = rng_np.normal(0.0, 1.0 / np.sqrt(feature_dim), size=encoder.shape).astype(np.float32)
    rng = random.Random(train_seed * 2003 + eval_seed)
    prediction_files: dict[str, str] = {}
    budget_summaries: list[dict[str, Any]] = []
    for budget in budgets:
        receiver = _fit_receiver(
            train_rows,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget,
            ridge=ridge,
        )
        margin_threshold, calibration = _calibrate_margin_threshold(
            train_rows,
            encoder=encoder,
            receiver=receiver,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget,
            random_encoder=random_encoder,
            seed=train_seed * 3011 + eval_seed + budget,
        )
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
        for row_index, example in enumerate(eval_rows):
            for condition in CONDITIONS:
                by_condition[condition].append(
                    _predict_condition(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        encoder=encoder,
                        receiver=receiver,
                        code_projection=code_projection,
                        feature_dim=feature_dim,
                        budget_bytes=budget,
                        margin_threshold=margin_threshold,
                        rng=rng,
                        random_encoder=random_encoder,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        metrics = {condition: _summarize(rows) for condition, rows in by_condition.items()}
        controls = [
            "zero_source",
            "shuffled_source",
            "answer_masked_source",
            "random_same_byte",
            "target_derived_sidecar",
            "answer_only",
            "structured_text_matched",
            "wrong_projection_source",
        ]
        best_no_source = max(metrics[name]["accuracy"] for name in ["target_only", *controls])
        best_destructive = max(metrics[name]["accuracy"] for name in controls)
        matched = metrics["matched_candidate_embedding_receiver"]["accuracy"]
        target = metrics["target_only"]["accuracy"]
        pass_gate = (
            matched >= target + 0.15
            and matched >= best_destructive + 0.15
            and all(metrics[name]["accuracy"] <= target + 0.05 for name in controls)
            and metrics["full_diag_oracle"]["accuracy"] >= 0.95
        )
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for condition in CONDITIONS:
                for row in by_condition[condition]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_files[str(budget)] = predictions_path.name
        budget_summaries.append(
            {
                "budget_bytes": budget,
                "pass_gate": pass_gate,
                "matched_accuracy": matched,
                "target_only_accuracy": target,
                "best_no_source_accuracy": best_no_source,
                "best_destructive_control_accuracy": best_destructive,
                "matched_minus_target": matched - target,
                "matched_minus_best_destructive_control": matched - best_destructive,
                "full_diag_oracle_accuracy": metrics["full_diag_oracle"]["accuracy"],
                "margin_threshold": margin_threshold,
                "margin_calibration": calibration,
                "receiver_sha256": hashlib.sha256(receiver.tobytes()).hexdigest(),
                "metrics": metrics,
            }
        )
    exact_ids = [row.example_id for row in eval_rows]
    payload = {
        "gate": "source_private_candidate_embedding_receiver",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "budgets": budgets,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "ridge": ridge,
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
        "prediction_files": prediction_files,
        "pass_rule": (
            "Matched learned candidate-embedding receiver must beat target by >=0.15, beat every destructive "
            "control by >=0.15, keep all destructive controls within target+0.05, and keep full diagnostic oracle >=0.95."
        ),
        "interpretation": (
            "This is a learned target-side receiver smoke: source evidence is compressed into a bit packet, "
            "and a trained candidate scorer decodes the packet using public candidate side information."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", *prediction_files.values(), "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md", *prediction_files.values()]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Candidate-Embedding Receiver Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- train family set: `{train_family_set}`",
                f"- eval family set: `{eval_family_set}`",
                f"- budgets: `{budgets}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Candidate-Embedding Receiver",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{payload['train_family_set']}:{payload['train_examples']}` / `{payload['eval_family_set']}:{payload['eval_examples']}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget bytes | Pass | Matched | Target | Best destructive | Delta target | Delta destructive | Full diag oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_destructive_control_accuracy']:.3f} | "
            f"{row['matched_minus_target']:.3f} | {row['matched_minus_best_destructive_control']:.3f} | "
            f"{row['full_diag_oracle_accuracy']:.3f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_candidate_embedding_receiver_20260429"))
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        budgets=args.budgets,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        ridge=args.ridge,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
