from __future__ import annotations

"""Cache-only HellaSwag complementarity/headroom gate."""

import argparse
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


DEFAULT_SOURCE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "bagged_gate/predictions.jsonl"
)
DEFAULT_TARGET = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/predictions.jsonl"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_complementarity_headroom_gate_20260502")


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
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


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == answers.astype(np.int64)))


def _bootstrap_delta_ci(
    selected_correct: np.ndarray,
    baseline_correct: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    deltas = selected_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    n = len(deltas)
    for idx in range(int(samples)):
        sample = rng.integers(0, n, size=n)
        means[idx] = float(np.mean(deltas[sample]))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _block_rows(
    *,
    source_correct: np.ndarray,
    target_correct: np.ndarray,
    oracle_correct: np.ndarray,
    blocks: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    n = len(source_correct)
    edges = np.linspace(0, n, int(blocks) + 1, dtype=np.int64)
    for block in range(int(blocks)):
        start = int(edges[block])
        end = int(edges[block + 1])
        if end <= start:
            continue
        result.append(
            {
                "block": block,
                "start": start,
                "end": end,
                "rows": end - start,
                "source_accuracy": float(np.mean(source_correct[start:end])),
                "target_accuracy": float(np.mean(target_correct[start:end])),
                "oracle_accuracy": float(np.mean(oracle_correct[start:end])),
                "oracle_lift_vs_source": float(
                    np.mean(oracle_correct[start:end]) - np.mean(source_correct[start:end])
                ),
            }
        )
    return result


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    source_predictions: pathlib.Path = DEFAULT_SOURCE,
    target_predictions: pathlib.Path = DEFAULT_TARGET,
    source_field: str = "selected_prediction",
    target_field: str = "hybrid_vote_on_score_agreement_prediction",
    source_label_field: str = "source_label_prediction",
    blocks: int = 5,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 6201,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _read_jsonl(source_predictions)
    target_rows = _read_jsonl(target_predictions)
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target prediction row counts differ")
    source_ids = [str(row["row_id"]) for row in source_rows]
    target_ids = [str(row["row_id"]) for row in target_rows]
    if source_ids != target_ids:
        raise ValueError("source and target prediction row_id order differs")
    answers = np.asarray([int(row["answer_index"]) for row in source_rows], dtype=np.int64)
    target_answers = np.asarray([int(row["answer_index"]) for row in target_rows], dtype=np.int64)
    if not np.array_equal(answers, target_answers):
        raise ValueError("source and target answers differ")
    source = np.asarray([int(row[source_field]) for row in source_rows], dtype=np.int64)
    target = np.asarray([int(row[target_field]) for row in target_rows], dtype=np.int64)
    source_label = np.asarray([int(row[source_label_field]) for row in source_rows], dtype=np.int64)
    source_correct = source == answers
    target_correct = target == answers
    source_label_correct = source_label == answers
    oracle_correct = source_correct | target_correct
    source_label_oracle_correct = source_label_correct | target_correct
    low, high = _bootstrap_delta_ci(
        oracle_correct,
        source_correct,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    label_low, label_high = _bootstrap_delta_ci(
        source_label_oracle_correct,
        source_label_correct,
        seed=bootstrap_seed + 1,
        samples=bootstrap_samples,
    )
    target_only_source_wrong = (~source_correct) & target_correct
    source_only_target_wrong = source_correct & (~target_correct)
    both_wrong = (~source_correct) & (~target_correct)
    both_correct = source_correct & target_correct
    disagreements = source != target
    block_rows = _block_rows(
        source_correct=source_correct,
        target_correct=target_correct,
        oracle_correct=oracle_correct,
        blocks=blocks,
    )
    oracle_lift = _accuracy(oracle_correct.astype(np.int64), np.ones_like(answers)) - _accuracy(
        source_correct.astype(np.int64),
        np.ones_like(answers),
    )
    target_only_rate = float(np.mean(target_only_source_wrong))
    pass_gate = bool(
        oracle_lift >= 0.03
        and low > 0.01
        and target_only_rate >= 0.03
        and all(row["oracle_lift_vs_source"] > 0.0 for row in block_rows)
    )
    payload = {
        "gate": "source_private_hellaswag_complementarity_headroom_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if target-or-source oracle lift over source packet is >=0.03, paired CI95 low "
            "is >0.01, target-correct/source-wrong rows are >=3%, and all contiguous blocks have "
            "positive oracle lift. This is a headroom gate, not a method result."
        ),
        "inputs": {
            "source_predictions": _display(source_predictions),
            "source_predictions_sha256": _sha256_file(source_predictions),
            "target_predictions": _display(target_predictions),
            "target_predictions_sha256": _sha256_file(target_predictions),
            "source_field": source_field,
            "target_field": target_field,
            "source_label_field": source_label_field,
        },
        "headline": {
            "eval_rows": int(len(answers)),
            "source_packet_accuracy": _accuracy(source, answers),
            "target_side_accuracy": _accuracy(target, answers),
            "source_label_accuracy": _accuracy(source_label, answers),
            "target_or_source_oracle_accuracy": float(np.mean(oracle_correct)),
            "target_or_source_oracle_lift_vs_source": oracle_lift,
            "target_or_source_oracle_ci95_low_vs_source": low,
            "target_or_source_oracle_ci95_high_vs_source": high,
            "target_or_source_label_oracle_accuracy": float(np.mean(source_label_oracle_correct)),
            "target_or_source_label_oracle_lift_vs_source_label": float(
                np.mean(source_label_oracle_correct) - np.mean(source_label_correct)
            ),
            "target_or_source_label_oracle_ci95_low_vs_source_label": label_low,
            "target_or_source_label_oracle_ci95_high_vs_source_label": label_high,
            "source_correct_target_wrong_count": int(np.sum(source_only_target_wrong)),
            "target_correct_source_wrong_count": int(np.sum(target_only_source_wrong)),
            "both_correct_count": int(np.sum(both_correct)),
            "both_wrong_count": int(np.sum(both_wrong)),
            "disagreement_rate": float(np.mean(disagreements)),
            "target_correct_source_wrong_rate": target_only_rate,
            "source_correct_target_wrong_rate": float(np.mean(source_only_target_wrong)),
            "positive_block_count": int(sum(row["oracle_lift_vs_source"] > 0.0 for row in block_rows)),
            "block_count": len(block_rows),
        },
        "blocks": block_rows,
        "interpretation": (
            "This cache-only gate measures whether the existing source packet and target-side "
            "prediction have complementary errors. It justifies a conditional syndrome/selector "
            "method only if the oracle lift is stable; it does not itself transmit new information."
        ),
    }
    json_path = output_dir / "hellaswag_complementarity_headroom_gate.json"
    md_path = output_dir / "hellaswag_complementarity_headroom_gate.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": payload["headline"] | {"pass_gate": pass_gate},
        "files": [
            {"path": _display(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Complementarity Headroom Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- source packet accuracy: `{h['source_packet_accuracy']:.6f}`",
        f"- target-side accuracy: `{h['target_side_accuracy']:.6f}`",
        f"- target-or-source oracle accuracy: `{h['target_or_source_oracle_accuracy']:.6f}`",
        f"- oracle lift vs source: `{h['target_or_source_oracle_lift_vs_source']:.6f}`",
        f"- oracle CI95 low vs source: `{h['target_or_source_oracle_ci95_low_vs_source']:.6f}`",
        f"- target-correct/source-wrong rows: `{h['target_correct_source_wrong_count']}`",
        f"- disagreement rate: `{h['disagreement_rate']:.6f}`",
        f"- positive blocks: `{h['positive_block_count']}/{h['block_count']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-predictions", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-predictions", type=pathlib.Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-field", default="selected_prediction")
    parser.add_argument("--target-field", default="hybrid_vote_on_score_agreement_prediction")
    parser.add_argument("--source-label-field", default="source_label_prediction")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=6201)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        source_predictions=args.source_predictions,
        target_predictions=args.target_predictions,
        source_field=args.source_field,
        target_field=args.target_field,
        source_label_field=args.source_label_field,
        blocks=args.blocks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
