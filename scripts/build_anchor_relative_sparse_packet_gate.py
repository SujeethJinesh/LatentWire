from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _constrained_nonself_label_index,
    _fit_ridge_encoder_for_view,
    _project_source,
)


CONDITIONS = (
    "target_only",
    "sparse_anchor_source",
    "sparse_anchor_constrained_shuffled_source",
    "sparse_anchor_answer_masked_source",
    "sparse_anchor_random_valid_same_byte",
    "sparse_anchor_target_derived_sidecar",
    "sparse_anchor_id_permutation",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_index(example: Example) -> int:
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)


def _prior_index(example: Example) -> int:
    prior = _prior_prediction(example)
    return next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)


def _score_candidates(
    example: Example,
    *,
    encoder: np.ndarray,
    feature_dim: int,
    candidate_view: str,
    mode: str,
) -> np.ndarray:
    predicted = _project_source(example, encoder=encoder, feature_dim=feature_dim, mode=mode)
    candidates = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view)
    return (candidates @ predicted).astype(np.float32)


def _fit_score_calibration(
    train_examples: list[Example],
    *,
    encoder: np.ndarray,
    feature_dim: int,
    candidate_view: str,
) -> tuple[float, float]:
    values: list[float] = []
    for example in train_examples:
        values.extend(
            float(value)
            for value in _score_candidates(
                example,
                encoder=encoder,
                feature_dim=feature_dim,
                candidate_view=candidate_view,
                mode="matched",
            )
        )
    arr = np.array(values, dtype=np.float32)
    lo = float(np.quantile(arr, 0.01))
    hi = float(np.quantile(arr, 0.99))
    return lo, max(hi, lo + 1e-4)


def _quantize_score(value: float, *, lo: float, hi: float) -> int:
    scaled = (value - lo) / max(hi - lo, 1e-6)
    return int(np.clip(round(scaled * 255.0), 0, 255))


def _dequantize_score(value: int, *, lo: float, hi: float) -> float:
    return lo + (float(value) / 255.0) * (hi - lo)


def _sparse_anchor_packet(
    example: Example,
    *,
    encoder: np.ndarray,
    feature_dim: int,
    candidate_view: str,
    budget_bytes: int,
    lo: float,
    hi: float,
    mode: str,
) -> bytes:
    atom_count = max(1, min(len(example.candidates), budget_bytes // 2))
    scores = _score_candidates(
        example,
        encoder=encoder,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
        mode=mode,
    )
    order = np.argsort(-scores, kind="stable")[:atom_count]
    payload = bytearray()
    for idx in order:
        payload.append(int(idx))
        payload.append(_quantize_score(float(scores[int(idx)]), lo=lo, hi=hi))
    return bytes(payload[:budget_bytes])


def _random_valid_packet(example: Example, *, budget_bytes: int, rng: random.Random) -> bytes:
    atom_count = max(1, min(len(example.candidates), budget_bytes // 2))
    order = list(range(len(example.candidates)))
    rng.shuffle(order)
    payload = bytearray()
    for idx in order[:atom_count]:
        payload.append(int(idx))
        payload.append(rng.randrange(0, 256))
    return bytes(payload[:budget_bytes])


def _target_derived_packet(example: Example, *, budget_bytes: int) -> bytes:
    atom_count = max(1, min(len(example.candidates), budget_bytes // 2))
    prior_idx = _prior_index(example)
    order = [prior_idx] + [idx for idx in range(len(example.candidates)) if idx != prior_idx]
    payload = bytearray()
    for rank, idx in enumerate(order[:atom_count]):
        payload.append(int(idx))
        payload.append(max(0, 255 - rank))
    return bytes(payload[:budget_bytes])


def _decode_sparse_anchor_packet(
    example: Example,
    payload: bytes | None,
    *,
    budget_bytes: int,
    lo: float,
    hi: float,
    permute_anchor_ids: bool = False,
) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    scores = np.full(len(example.candidates), -np.inf, dtype=np.float32)
    pairs = min(len(payload) // 2, max(1, budget_bytes // 2))
    decoded_atoms: list[dict[str, Any]] = []
    for pair_idx in range(pairs):
        anchor_idx = int(payload[2 * pair_idx])
        score_byte = int(payload[2 * pair_idx + 1])
        if permute_anchor_ids:
            anchor_idx = (anchor_idx + 1) % len(example.candidates)
        if 0 <= anchor_idx < len(example.candidates):
            score = _dequantize_score(score_byte, lo=lo, hi=hi)
            scores[anchor_idx] = max(scores[anchor_idx], score)
            decoded_atoms.append({"candidate_index": anchor_idx, "score_byte": score_byte, "score": score})
    if not np.any(np.isfinite(scores)):
        return _prior_prediction(example), {"decoder": "prior_invalid_packet", "decoded_atoms": decoded_atoms}
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score, rtol=1e-6, atol=1e-8))
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[int(idx)] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[int(tied[0])]
    sorted_scores = sorted((float(score), idx) for idx, score in enumerate(scores) if np.isfinite(score))
    margin = None if len(sorted_scores) < 2 else sorted_scores[-1][0] - sorted_scores[-2][0]
    return prediction, {
        "decoder": "sparse_candidate_anchor",
        "decoded_atoms": decoded_atoms,
        "score_margin": margin,
        "ties": [int(idx) for idx in tied.tolist()],
    }


def _condition_payload_and_prediction(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    feature_dim: int,
    candidate_view: str,
    budget_bytes: int,
    lo: float,
    hi: float,
    rng: random.Random,
) -> tuple[str, bytes | None, dict[str, Any]]:
    if condition == "target_only":
        prediction, meta = _decode_sparse_anchor_packet(example, None, budget_bytes=budget_bytes, lo=lo, hi=hi)
        return prediction, None, meta | {"packet_family": "none"}
    if condition == "sparse_anchor_source":
        payload = _sparse_anchor_packet(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            budget_bytes=budget_bytes,
            lo=lo,
            hi=hi,
            mode="matched",
        )
        prediction, meta = _decode_sparse_anchor_packet(example, payload, budget_bytes=budget_bytes, lo=lo, hi=hi)
        return prediction, payload, meta | {"packet_family": "sparse_anchor"}
    if condition == "sparse_anchor_constrained_shuffled_source":
        other = eval_examples[_constrained_nonself_label_index(index, eval_examples)]
        payload = _sparse_anchor_packet(
            other,
            encoder=encoder,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            budget_bytes=budget_bytes,
            lo=lo,
            hi=hi,
            mode="matched",
        )
        prediction, meta = _decode_sparse_anchor_packet(example, payload, budget_bytes=budget_bytes, lo=lo, hi=hi)
        return prediction, payload, meta | {"packet_family": "sparse_anchor", "source": other.example_id}
    if condition == "sparse_anchor_answer_masked_source":
        payload = _sparse_anchor_packet(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            budget_bytes=budget_bytes,
            lo=lo,
            hi=hi,
            mode="answer_masked",
        )
        prediction, meta = _decode_sparse_anchor_packet(example, payload, budget_bytes=budget_bytes, lo=lo, hi=hi)
        return prediction, payload, meta | {"packet_family": "sparse_anchor", "source": "answer_masked"}
    if condition == "sparse_anchor_random_valid_same_byte":
        payload = _random_valid_packet(example, budget_bytes=budget_bytes, rng=rng)
        prediction, meta = _decode_sparse_anchor_packet(example, payload, budget_bytes=budget_bytes, lo=lo, hi=hi)
        return prediction, payload, meta | {"packet_family": "sparse_anchor", "source": "random_valid"}
    if condition == "sparse_anchor_target_derived_sidecar":
        payload = _target_derived_packet(example, budget_bytes=budget_bytes)
        prediction, meta = _decode_sparse_anchor_packet(example, payload, budget_bytes=budget_bytes, lo=lo, hi=hi)
        return prediction, payload, meta | {"packet_family": "sparse_anchor", "source": "target_prior"}
    if condition == "sparse_anchor_id_permutation":
        payload = _sparse_anchor_packet(
            example,
            encoder=encoder,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            budget_bytes=budget_bytes,
            lo=lo,
            hi=hi,
            mode="matched",
        )
        prediction, meta = _decode_sparse_anchor_packet(
            example,
            payload,
            budget_bytes=budget_bytes,
            lo=lo,
            hi=hi,
            permute_anchor_ids=True,
        )
        return prediction, payload, meta | {"packet_family": "sparse_anchor", "source": "anchor_id_permutation"}
    raise ValueError(f"unknown condition {condition!r}")


def _summarize_rows(rows: list[dict[str, Any]], *, budget_bytes: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    example_ids = sorted({row["example_id"] for row in rows})
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        correct = [row["example_id"] for row in condition_rows if row["correct"]]
        payload_bytes = [row["payload_bytes"] for row in condition_rows]
        metrics[condition] = {
            "correct": len(correct),
            "accuracy": len(correct) / len(condition_rows),
            "correct_ids": correct,
            "mean_payload_bytes": float(np.mean(payload_bytes)),
        }
    target = metrics["target_only"]["accuracy"]
    source = metrics["sparse_anchor_source"]["accuracy"]
    controls = [
        "sparse_anchor_constrained_shuffled_source",
        "sparse_anchor_answer_masked_source",
        "sparse_anchor_random_valid_same_byte",
        "sparse_anchor_target_derived_sidecar",
        "sparse_anchor_id_permutation",
    ]
    best_control = max(metrics[name]["accuracy"] for name in controls)
    return {
        "budget_bytes": budget_bytes,
        "n": len(example_ids),
        "exact_id_count": len(example_ids),
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "exact_id_parity": len(rows) == len(example_ids) * len(CONDITIONS),
        "target_accuracy": target,
        "sparse_anchor_accuracy": source,
        "best_control_accuracy": best_control,
        "sparse_minus_target": source - target,
        "sparse_minus_best_control": source - best_control,
        "controls_ok": best_control <= target + 0.03,
        "pass_gate": source >= target + 0.10 and source - best_control >= 0.05 and best_control <= target + 0.03,
        "metrics": metrics,
    }


def _run_direction(
    *,
    output_dir: pathlib.Path,
    direction: str,
    train_family_set: str,
    eval_family_set: str,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    feature_dim: int,
    budgets: list[int],
    candidate_view: str,
    ridge: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = make_benchmark(examples=train_examples, candidates=4, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=4, seed=eval_seed, family_set=eval_family_set)
    encoder = _fit_ridge_encoder_for_view(
        train,
        feature_dim=feature_dim,
        ridge=ridge,
        candidate_view=candidate_view,
        fit_intercept=False,
    )
    lo, hi = _fit_score_calibration(train, encoder=encoder, feature_dim=feature_dim, candidate_view=candidate_view)
    budget_summaries: list[dict[str, Any]] = []
    for budget in budgets:
        rng = random.Random(train_seed * 1000003 + eval_seed * 9176 + budget)
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(eval_rows):
            for condition in CONDITIONS:
                prediction, payload, meta = _condition_payload_and_prediction(
                    condition=condition,
                    example=example,
                    eval_examples=eval_rows,
                    index=index,
                    encoder=encoder,
                    feature_dim=feature_dim,
                    candidate_view=candidate_view,
                    budget_bytes=budget,
                    lo=lo,
                    hi=hi,
                    rng=rng,
                )
                rows.append(
                    {
                        "example_id": example.example_id,
                        "condition": condition,
                        "answer_label": example.answer_label,
                        "prediction": prediction,
                        "correct": prediction == example.answer_label,
                        "payload_hex": "" if payload is None else payload.hex(),
                        "payload_bytes": 0 if payload is None else len(payload),
                        "answer_index": _answer_index(example),
                        "prior_index": _prior_index(example),
                        **meta,
                    }
                )
        predictions_name = f"predictions_budget{budget}.jsonl"
        (output_dir / predictions_name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = _summarize_rows(rows, budget_bytes=budget)
        summary["predictions_file"] = predictions_name
        budget_summaries.append(summary)
    payload = {
        "gate": "anchor_relative_sparse_packet_direction",
        "direction": direction,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "feature_dim": feature_dim,
        "candidate_view": candidate_view,
        "ridge": ridge,
        "score_calibration": {"lo": lo, "hi": hi},
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_direction_markdown(output_dir / "summary.md", payload)
    manifest = {
        "artifacts": ["summary.json", "summary.md", "manifest.json", "manifest.md"]
        + [f"predictions_budget{budget}.jsonl" for budget in budgets],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md"] + [f"predictions_budget{budget}.jsonl" for budget in budgets]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Anchor-Relative Sparse Packet Direction Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def build_anchor_relative_sparse_gate(
    *,
    output_dir: pathlib.Path,
    budgets: list[int],
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    seed: int,
    candidate_view: str,
    ridge: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("core_to_holdout", "core", "holdout", seed, seed + 1),
        ("holdout_to_core", "holdout", "core", seed + 1, seed),
    ]
    rows: list[dict[str, Any]] = []
    run_dirs: list[str] = []
    for direction, train_family, eval_family, train_seed, eval_seed in specs:
        run_dir = output_dir / direction
        result = _run_direction(
            output_dir=run_dir,
            direction=direction,
            train_family_set=train_family,
            eval_family_set=eval_family,
            train_examples=train_examples,
            eval_examples=eval_examples,
            train_seed=train_seed,
            eval_seed=eval_seed,
            feature_dim=feature_dim,
            budgets=budgets,
            candidate_view=candidate_view,
            ridge=ridge,
        )
        run_dirs.append(str(run_dir))
        for summary in result["budget_summaries"]:
            rows.append(
                {
                    "direction": direction,
                    "budget_bytes": summary["budget_bytes"],
                    "n": summary["n"],
                    "target_accuracy": summary["target_accuracy"],
                    "sparse_anchor_accuracy": summary["sparse_anchor_accuracy"],
                    "best_control_accuracy": summary["best_control_accuracy"],
                    "sparse_minus_target": summary["sparse_minus_target"],
                    "sparse_minus_best_control": summary["sparse_minus_best_control"],
                    "controls_ok": summary["controls_ok"],
                    "pass_gate": summary["pass_gate"],
                    "exact_id_parity": summary["exact_id_parity"],
                    "control_accuracies": {
                        name: summary["metrics"][name]["accuracy"]
                        for name in CONDITIONS
                        if name not in {"target_only", "sparse_anchor_source"}
                    },
                }
            )
    directions = sorted({row["direction"] for row in rows})
    direction_pass = {
        direction: any(row["pass_gate"] for row in rows if row["direction"] == direction)
        for direction in directions
    }
    payload = {
        "gate": "anchor_relative_sparse_packet_gate",
        "rows": rows,
        "run_dirs": run_dirs,
        "headline": {
            "directions": directions,
            "direction_pass": direction_pass,
            "pass_directions": sum(1 for ok in direction_pass.values() if ok),
            "budgets": budgets,
            "max_sparse_accuracy": max(row["sparse_anchor_accuracy"] for row in rows),
            "min_sparse_accuracy": min(row["sparse_anchor_accuracy"] for row in rows),
            "max_sparse_minus_target": max(row["sparse_minus_target"] for row in rows),
            "min_sparse_minus_control": min(row["sparse_minus_best_control"] for row in rows),
        },
        "pass_gate": all(direction_pass.values()),
        "pass_rule": (
            "At least one budget per direction must beat target by >=0.10, beat the best source-destroying "
            "control by >=0.05, and keep every control within target+0.03."
        ),
    }
    (output_dir / "anchor_relative_sparse_packet_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_gate_markdown(output_dir / "anchor_relative_sparse_packet_gate.md", payload)
    manifest = {
        "artifacts": [
            "anchor_relative_sparse_packet_gate.json",
            "anchor_relative_sparse_packet_gate.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            "anchor_relative_sparse_packet_gate.json": _sha256_file(output_dir / "anchor_relative_sparse_packet_gate.json"),
            "anchor_relative_sparse_packet_gate.md": _sha256_file(output_dir / "anchor_relative_sparse_packet_gate.md"),
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Anchor-Relative Sparse Packet Gate Manifest", "", f"- pass gate: `{payload['pass_gate']}`", ""]),
        encoding="utf-8",
    )
    return payload


def _write_direction_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Anchor-Relative Sparse Packet Direction",
        "",
        f"- direction: `{payload['direction']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval families: `{payload['train_family_set']} -> {payload['eval_family_set']}`",
        f"- candidate view: `{payload['candidate_view']}`",
        "",
        "| Budget | Sparse | Target | Best control | Sparse-target | Sparse-control | Controls ok | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | {row['sparse_anchor_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['sparse_minus_target']:.3f} | "
            f"{row['sparse_minus_best_control']:.3f} | `{row['controls_ok']}` | `{row['pass_gate']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Anchor-Relative Sparse Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- direction pass: `{h['direction_pass']}`",
        f"- budgets: `{h['budgets']}`",
        f"- sparse accuracy range: `{h['min_sparse_accuracy']:.3f}-{h['max_sparse_accuracy']:.3f}`",
        f"- max sparse-target delta: `{h['max_sparse_minus_target']:.3f}`",
        f"- min sparse-control delta: `{h['min_sparse_minus_control']:.3f}`",
        "",
        "## Rows",
        "",
        "| Direction | Budget | N | Sparse | Target | Best control | Sparse-target | Sparse-control | Controls ok | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['direction']} | {row['budget_bytes']} | {row['n']} | "
            f"{row['sparse_anchor_accuracy']:.3f} | {row['target_accuracy']:.3f} | "
            f"{row['best_control_accuracy']:.3f} | {row['sparse_minus_target']:.3f} | "
            f"{row['sparse_minus_best_control']:.3f} | `{row['controls_ok']}` | `{row['pass_gate']}` |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/anchor_relative_sparse_packet_gate_20260429"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--train-examples", type=int, default=768)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "slot"], default="semantic")
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_anchor_relative_sparse_gate(
        output_dir=output_dir,
        budgets=args.budgets,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        feature_dim=args.feature_dim,
        seed=args.seed,
        candidate_view=args.candidate_view,
        ridge=args.ridge,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
