from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
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
from scripts.run_source_private_shared_sparse_crosscoder_packet_gate import _stress_candidate_intent  # noqa: E402
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


CONDITIONS = (
    "target_only",
    "conditional_semantic_syndrome",
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "public_only_source",
    "target_derived_sidecar",
    "random_same_byte",
    "answer_only_text",
    "structured_text_matched",
    "wrong_projection_source",
    "oracle_candidate_residual",
)

SOURCE_DESTROYING_CONTROLS = (
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "public_only_source",
    "target_derived_sidecar",
    "random_same_byte",
    "answer_only_text",
    "structured_text_matched",
    "wrong_projection_source",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2s(value.encode("utf-8"), digest_size=8).digest(), "little")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-8)


def _features(text: str, dim: int, *, namespace: str) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9_]+|==|!=|<=|>=|[{}()[\],.:+-]", text.lower())
    grams: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        padded = f"<{token}>"
        for n in (3, 4, 5):
            grams.extend(padded[index : index + n] for index in range(max(0, len(padded) - n + 1)))
    features = [*tokens, *(f"{token}@{idx % 7}" for idx, token in enumerate(tokens)), *(f"ng:{gram}" for gram in grams)]
    for feature in features:
        h = _stable_hash(f"{namespace}:{feature}")
        vec[h % dim] += 1.0 if (h >> 63) == 0 else -1.0
    return vec / max(1.0, len(features) ** 0.5)


def _source_text(example: Example, *, mode: str) -> str:
    if mode == "zero":
        return ""
    if mode == "public_only":
        return example.public_issue
    if mode == "matched":
        log = _mask_repair_diag(example.private_test_log)
        log = re.sub(r"repair_family=[A-Za-z0-9_]+", "repair_family=<MASKED>", log)
        log = re.sub(r"hidden_tests/test_[A-Za-z0-9_]+\.py", "hidden_tests/test_<MASKED>.py", log)
        return log
    if mode == "answer_masked":
        masked = _mask_repair_diag(example.private_test_log)
        masked = _mask_log_components(masked, mask_expected_actual=True, mask_test_name=True)
        lines = []
        for line in masked.splitlines():
            if line.startswith(("hidden_input=", "expected=", "actual=", "failure_status=")):
                lines.append(f"{line.split('=', 1)[0]}=<MASKED>")
            else:
                lines.append(line)
        return "\n".join(lines)
    raise ValueError(f"unknown source mode {mode!r}")


def _candidate_texts(example: Example, *, candidate_view: str) -> list[str]:
    return [
        "\n".join(
            [
                f"intent={_stress_candidate_intent(candidate.patch_intent, candidate_atom_view=candidate_view)}",
                f"public_issue={example.public_issue}",
            ]
        )
        for candidate in example.candidates
    ]


def _candidate_matrix(example: Example, *, dim: int, candidate_view: str) -> np.ndarray:
    return _normalize_rows(
        np.stack([_features(text, dim, namespace="semantic") for text in _candidate_texts(example, candidate_view=candidate_view)])
    ).astype(np.float32)


def _source_vector(example: Example, *, dim: int, mode: str) -> np.ndarray:
    return _features(_source_text(example, mode=mode), dim, namespace="semantic").astype(np.float32)


def _answer_index(example: Example) -> int:
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)


def _target_residual(example: Example, *, dim: int, candidate_view: str) -> np.ndarray:
    candidates = _candidate_matrix(example, dim=dim, candidate_view=candidate_view)
    residual = candidates[_answer_index(example)] - candidates[_prior_index(example)]
    norm = float(np.linalg.norm(residual))
    return (residual / max(norm, 1e-8)).astype(np.float32)


def _fit_encoder(train: list[Example], *, dim: int, candidate_view: str, ridge: float) -> np.ndarray:
    x = np.stack([_source_vector(example, dim=dim, mode="matched") for example in train]).astype(np.float64)
    y = np.stack([_target_residual(example, dim=dim, candidate_view=candidate_view) for example in train]).astype(np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[-1, -1] -= ridge
    return np.linalg.solve(xtx, x_aug.T @ y).astype(np.float32)


def _augment(vec: np.ndarray) -> np.ndarray:
    return np.concatenate([vec.astype(np.float32), np.ones(1, dtype=np.float32)])


def _bitpack(bits: np.ndarray, budget_bytes: int) -> bytes:
    padded = np.zeros(budget_bytes * 8, dtype=np.uint8)
    padded[: min(bits.size, padded.size)] = bits[: padded.size].astype(np.uint8)
    return np.packbits(padded, bitorder="big").tobytes()


def _bytes_to_bits(payload: bytes | None, bit_count: int) -> np.ndarray:
    if not payload:
        return np.zeros(bit_count, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[:bit_count].astype(np.uint8)


def _packet_from_vector(vector: np.ndarray, projection: np.ndarray, *, budget_bytes: int) -> bytes:
    logits = projection[: budget_bytes * 8] @ vector
    return _bitpack((logits >= 0).astype(np.uint8), budget_bytes)


def _source_residual(example: Example, *, encoder: np.ndarray, dim: int, mode: str) -> np.ndarray:
    residual = _augment(_source_vector(example, dim=dim, mode=mode)) @ encoder
    norm = float(np.linalg.norm(residual))
    return (residual / max(norm, 1e-8)).astype(np.float32)


def _candidate_residual_codes(example: Example, *, projection: np.ndarray, dim: int, candidate_view: str, budget_bytes: int) -> np.ndarray:
    candidates = _candidate_matrix(example, dim=dim, candidate_view=candidate_view)
    prior = candidates[_prior_index(example)]
    residuals = candidates - prior[None, :]
    residuals = _normalize_rows(residuals).astype(np.float32)
    logits = residuals @ projection[: budget_bytes * 8].T
    return (logits >= 0).astype(np.uint8)


def _decode_packet(
    example: Example,
    payload: bytes | None,
    *,
    projection: np.ndarray,
    dim: int,
    candidate_view: str,
    budget_bytes: int,
) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    bits = _bytes_to_bits(payload, budget_bytes * 8)
    codes = _candidate_residual_codes(
        example,
        projection=projection,
        dim=dim,
        candidate_view=candidate_view,
        budget_bytes=budget_bytes,
    )
    signed_bits = bits * 2.0 - 1.0
    signed_codes = codes * 2.0 - 1.0
    scores = (signed_codes * signed_bits[None, :]).mean(axis=1)
    best = float(np.max(scores))
    tied = [idx for idx, score in enumerate(scores) if abs(float(score) - best) <= 1e-8]
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[idx] == prior for idx in tied):
        return prior, {"decoder": "conditional_semantic_syndrome", "scores": [float(score) for score in scores], "ties": tied}
    return labels[tied[0]], {"decoder": "conditional_semantic_syndrome", "scores": [float(score) for score in scores], "ties": tied}


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_rows: list[Example],
    index: int,
    encoder: np.ndarray,
    random_encoder: np.ndarray,
    projection: np.ndarray,
    dim: int,
    candidate_view: str,
    budget_bytes: int,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {"source": condition}
    if condition == "conditional_semantic_syndrome":
        return _packet_from_vector(
            _source_residual(example, encoder=encoder, dim=dim, mode="matched"), projection, budget_bytes=budget_bytes
        ), {"source": example.example_id}
    if condition == "shuffled_source":
        other = eval_rows[_deterministic_nonself_index(index, len(eval_rows))]
        return _packet_from_vector(
            _source_residual(other, encoder=encoder, dim=dim, mode="matched"), projection, budget_bytes=budget_bytes
        ), {"source": other.example_id}
    if condition == "answer_masked_source":
        return _packet_from_vector(
            _source_residual(example, encoder=encoder, dim=dim, mode="answer_masked"), projection, budget_bytes=budget_bytes
        ), {"source": "answer_masked"}
    if condition == "public_only_source":
        return _packet_from_vector(
            _source_residual(example, encoder=encoder, dim=dim, mode="public_only"), projection, budget_bytes=budget_bytes
        ), {"source": "public_only"}
    if condition == "target_derived_sidecar":
        prior_vector = _candidate_matrix(example, dim=dim, candidate_view=candidate_view)[_prior_index(example)]
        return _packet_from_vector(prior_vector, projection, budget_bytes=budget_bytes), {"source": "target_prior"}
    if condition == "random_same_byte":
        return rng.randbytes(budget_bytes), {"source": "random"}
    if condition == "answer_only_text":
        return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_text"}
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_log"}
    if condition == "wrong_projection_source":
        residual = _augment(_source_vector(example, dim=dim, mode="matched")) @ random_encoder
        return _packet_from_vector(residual, projection, budget_bytes=budget_bytes), {"source": "wrong_projection"}
    if condition == "oracle_candidate_residual":
        return _packet_from_vector(_target_residual(example, dim=dim, candidate_view=candidate_view), projection, budget_bytes=budget_bytes), {
            "source": "oracle_candidate_residual"
        }
    raise ValueError(f"unknown condition {condition!r}")


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_rows: list[Example],
    index: int,
    encoder: np.ndarray,
    random_encoder: np.ndarray,
    projection: np.ndarray,
    dim: int,
    candidate_view: str,
    budget_bytes: int,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, metadata = _payload_for_condition(
        condition=condition,
        example=example,
        eval_rows=eval_rows,
        index=index,
        encoder=encoder,
        random_encoder=random_encoder,
        projection=projection,
        dim=dim,
        candidate_view=candidate_view,
        budget_bytes=budget_bytes,
        rng=rng,
    )
    prediction, decode_meta = _decode_packet(
        example,
        payload,
        projection=projection,
        dim=dim,
        candidate_view=candidate_view,
        budget_bytes=budget_bytes,
    )
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "payload_hex": payload_hex,
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "metadata": {**metadata, **decode_meta},
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    return {
        "n": len(rows),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(latencies),
    }


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))]


def _paired_bootstrap(rows: list[dict[str, Any]], *, condition: str, baseline: str, seed: int, samples: int = 500) -> dict[str, float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = [
        float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"])
        for _, conditions in sorted(by_example.items())
    ]
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {"mean": statistics.fmean(deltas), "ci95_low": _percentile(means, 0.025), "ci95_high": _percentile(means, 0.975)}


def _budget_summary(rows: list[dict[str, Any]], *, budget: int, seed: int) -> dict[str, Any]:
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    metrics = {condition: _summarize(condition_rows) for condition, condition_rows in by_condition.items()}
    target = metrics["target_only"]["accuracy"]
    matched = metrics["conditional_semantic_syndrome"]["accuracy"]
    best_control_name = max(SOURCE_DESTROYING_CONTROLS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    oracle = metrics["oracle_candidate_residual"]["accuracy"]
    ci_target = _paired_bootstrap(rows, condition="conditional_semantic_syndrome", baseline="target_only", seed=seed)
    ci_control = _paired_bootstrap(rows, condition="conditional_semantic_syndrome", baseline=best_control_name, seed=seed + 1)
    pass_gate = (
        matched >= target + 0.15
        and matched >= best_control + 0.15
        and all(metrics[condition]["accuracy"] <= target + 0.05 for condition in SOURCE_DESTROYING_CONTROLS)
        and ci_target["ci95_low"] > 0.10
        and oracle >= 0.90
    )
    return {
        "budget_bytes": budget,
        "pass_gate": pass_gate,
        "target_accuracy": target,
        "conditional_semantic_syndrome_accuracy": matched,
        "best_control_name": best_control_name,
        "best_control_accuracy": best_control,
        "matched_minus_target": matched - target,
        "matched_minus_best_control": matched - best_control,
        "oracle_candidate_residual_accuracy": oracle,
        "paired_bootstrap_vs_target": ci_target,
        "paired_bootstrap_vs_best_control": ci_control,
        "controls_ok": all(metrics[condition]["accuracy"] <= target + 0.05 for condition in SOURCE_DESTROYING_CONTROLS),
        "metrics": metrics,
    }


def _run_direction(
    *,
    output_dir: pathlib.Path,
    direction: str,
    train_family_set: str,
    eval_family_set: str,
    train_seed: int,
    eval_seed: int,
    train_examples: int,
    eval_examples: int,
    budgets: list[int],
    dim: int,
    ridge: float,
    candidate_view: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = make_benchmark(examples=train_examples, candidates=4, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=4, seed=eval_seed, family_set=eval_family_set)
    encoder = _fit_encoder(train, dim=dim, candidate_view=candidate_view, ridge=ridge)
    rng_np = np.random.default_rng(train_seed * 1009 + eval_seed)
    projection = _normalize_rows(rng_np.normal(size=(max(budgets) * 8, dim))).astype(np.float32)
    random_encoder = rng_np.normal(0.0, 1.0 / max(1.0, dim**0.5), size=encoder.shape).astype(np.float32)
    budget_summaries = []
    prediction_files: dict[str, str] = {}
    for budget in budgets:
        rng = random.Random(train_seed * 1000003 + eval_seed * 9176 + budget)
        rows: list[dict[str, Any]] = []
        for row_index, example in enumerate(eval_rows):
            for condition in CONDITIONS:
                rows.append(
                    _predict_condition(
                        condition=condition,
                        example=example,
                        eval_rows=eval_rows,
                        index=row_index,
                        encoder=encoder,
                        random_encoder=random_encoder,
                        projection=projection,
                        dim=dim,
                        candidate_view=candidate_view,
                        budget_bytes=budget,
                        rng=rng,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        predictions_name = f"predictions_budget{budget}.jsonl"
        (output_dir / predictions_name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        prediction_files[str(budget)] = predictions_name
        budget_summaries.append(_budget_summary(rows, budget=budget, seed=train_seed + eval_seed + budget))
    payload = {
        "gate": "source_private_conditional_semantic_syndrome_direction",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "direction": direction,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "budgets": budgets,
        "feature_dim": dim,
        "candidate_view": candidate_view,
        "ridge": ridge,
        "budget_summaries": budget_summaries,
        "prediction_files": prediction_files,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_direction_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", *prediction_files.values(), "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name) for name in ["summary.json", "summary.md", *prediction_files.values()]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Conditional Semantic Syndrome Direction Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def run_gate(
    *,
    output_dir: pathlib.Path,
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    seed: int,
    dim: int,
    ridge: float,
    candidate_view: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("core_to_holdout", "core", "holdout", seed, seed + 1),
        ("holdout_to_core", "holdout", "core", seed + 1, seed),
        ("same_family_all", "all", "all", seed, seed + 2),
    ]
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for direction, train_family, eval_family, train_seed, eval_seed in specs:
        result = _run_direction(
            output_dir=output_dir / direction,
            direction=direction,
            train_family_set=train_family,
            eval_family_set=eval_family,
            train_seed=train_seed,
            eval_seed=eval_seed,
            train_examples=train_examples,
            eval_examples=eval_examples,
            budgets=budgets,
            dim=dim,
            ridge=ridge,
            candidate_view=candidate_view,
        )
        run_dirs.append(str((output_dir / direction).relative_to(ROOT)) if output_dir.is_relative_to(ROOT) else str(output_dir / direction))
        for summary in result["budget_summaries"]:
            rows.append(
                {
                    "direction": direction,
                    "budget_bytes": summary["budget_bytes"],
                    "pass_gate": summary["pass_gate"],
                    "target_accuracy": summary["target_accuracy"],
                    "conditional_semantic_syndrome_accuracy": summary["conditional_semantic_syndrome_accuracy"],
                    "best_control_accuracy": summary["best_control_accuracy"],
                    "best_control_name": summary["best_control_name"],
                    "matched_minus_target": summary["matched_minus_target"],
                    "matched_minus_best_control": summary["matched_minus_best_control"],
                    "oracle_candidate_residual_accuracy": summary["oracle_candidate_residual_accuracy"],
                    "paired_ci95_low_vs_target": summary["paired_bootstrap_vs_target"]["ci95_low"],
                    "controls_ok": summary["controls_ok"],
                }
            )
    direction_pass = {direction: any(row["pass_gate"] for row in rows if row["direction"] == direction) for direction, *_ in specs}
    payload = {
        "gate": "source_private_conditional_semantic_syndrome_gate",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "candidate_view": candidate_view,
        "feature_dim": dim,
        "budgets": budgets,
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "direction_pass": direction_pass,
            "cross_family_pass": direction_pass["core_to_holdout"] and direction_pass["holdout_to_core"],
            "pass_rows": sum(1 for row in rows if row["pass_gate"]),
            "max_accuracy": max(row["conditional_semantic_syndrome_accuracy"] for row in rows),
            "max_lift_vs_target": max(row["matched_minus_target"] for row in rows),
        },
        "pass_gate": direction_pass["core_to_holdout"] and direction_pass["holdout_to_core"],
    }
    (output_dir / "conditional_semantic_syndrome_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_gate_markdown(output_dir / "conditional_semantic_syndrome_gate.md", payload)
    manifest = {
        "artifacts": ["conditional_semantic_syndrome_gate.json", "conditional_semantic_syndrome_gate.md", "manifest.json", "manifest.md"],
        "artifact_sha256": {
            "conditional_semantic_syndrome_gate.json": _sha256_file(output_dir / "conditional_semantic_syndrome_gate.json"),
            "conditional_semantic_syndrome_gate.md": _sha256_file(output_dir / "conditional_semantic_syndrome_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Conditional Semantic Syndrome Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def _write_direction_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Conditional Semantic Syndrome Direction",
        "",
        f"- direction: `{payload['direction']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- candidate view: `{payload['candidate_view']}`",
        "",
        "| Budget | Pass | Syndrome | Target | Best control | Delta target | CI95 low | Oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['conditional_semantic_syndrome_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | {row['matched_minus_target']:.3f} | "
            f"{row['paired_bootstrap_vs_target']['ci95_low']:.3f} | {row['oracle_candidate_residual_accuracy']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Conditional Semantic Syndrome Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- candidate view: `{payload['candidate_view']}`",
        f"- direction pass: `{payload['headline']['direction_pass']}`",
        f"- max accuracy: `{payload['headline']['max_accuracy']:.3f}`",
        f"- max lift vs target: `{payload['headline']['max_lift_vs_target']:.3f}`",
        "",
        "| Direction | Budget | Pass | Syndrome | Target | Best control | Delta target | CI95 low | Oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['direction']} | {row['budget_bytes']} | `{row['pass_gate']}` | "
            f"{row['conditional_semantic_syndrome_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['matched_minus_target']:.3f} | "
            f"{row['paired_ci95_low_vs_target']:.3f} | {row['oracle_candidate_residual_accuracy']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_conditional_semantic_syndrome_gate_20260429"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", choices=["native", "synonym_stress"], default="synonym_stress")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=output_dir,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        seed=args.seed,
        dim=args.feature_dim,
        ridge=args.ridge,
        candidate_view=args.candidate_view,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
