"""Aggregate stochastic route prediction files into verifier-style controls."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.evaluate import (  # noqa: E402
    add_paired_prediction_summary,
    write_prediction_records,
    write_prediction_sidecar,
)


def load_records(path: str | pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _rows_by_index(records: list[dict[str, Any]], method: str) -> dict[int, dict[str, Any]]:
    return {
        int(record["index"]): record
        for record in records
        if str(record.get("method")) == method
    }


def _copy_baseline_record(record: dict[str, Any], *, baseline_method: str) -> dict[str, Any]:
    copied = dict(record)
    copied["method"] = baseline_method
    return copied


def _prediction_key(record: dict[str, Any]) -> str:
    value = record.get("normalized_prediction")
    if value is None or value == "":
        value = record.get("prediction", "")
    return str(value)


def _majority_choice(seed_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    votes: collections.Counter[str] = collections.Counter(_prediction_key(row) for row in seed_rows)
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    winner, winner_votes = ranked[0]
    chosen = next(row for row in seed_rows if _prediction_key(row) == winner)
    telemetry = {
        "vote_prediction": winner,
        "vote_count": int(winner_votes),
        "vote_margin": int(winner_votes - (ranked[1][1] if len(ranked) > 1 else 0)),
        "vote_unique_predictions": int(len(votes)),
        "vote_entropy": _vote_entropy(votes),
        "source_random_salts": _salts(seed_rows),
        "source_seed_correct_count": int(sum(bool(row.get("correct")) for row in seed_rows)),
    }
    return chosen, telemetry


def _vote_entropy(votes: collections.Counter[str]) -> float:
    import math

    total = sum(votes.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in votes.values():
        p = count / total
        entropy -= p * math.log(max(p, 1e-12))
    return float(entropy)


def _salts(seed_rows: list[dict[str, Any]]) -> list[int]:
    salts: list[int] = []
    for row in seed_rows:
        salt = row.get("random_salt")
        if salt is not None:
            salts.append(int(salt))
    return salts


def _aggregate_record(
    *,
    method: str,
    chosen: dict[str, Any],
    baseline: dict[str, Any],
    telemetry: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    record = dict(chosen)
    record["method"] = method
    record["aggregation_policy"] = policy
    record["answer"] = baseline.get("answer", chosen.get("answer"))
    record["index"] = int(baseline["index"])
    if baseline.get("example_id") is not None:
        record["example_id"] = baseline.get("example_id")
    record.update(telemetry)
    return record


def aggregate_records(
    record_sets: list[list[dict[str, Any]]],
    *,
    method: str,
    baseline_method: str = "target_alone",
) -> list[dict[str, Any]]:
    if not record_sets:
        raise ValueError("At least one prediction record set is required")

    baseline_rows = _rows_by_index(record_sets[0], baseline_method)
    seed_method_rows = [_rows_by_index(records, method) for records in record_sets]
    indices = sorted(set(baseline_rows).intersection(*(set(rows) for rows in seed_method_rows)))
    if not indices:
        raise ValueError(f"No paired examples for method={method!r} and baseline={baseline_method!r}")

    output: list[dict[str, Any]] = [
        _copy_baseline_record(baseline_rows[idx], baseline_method=baseline_method)
        for idx in indices
    ]
    for idx in indices:
        baseline = baseline_rows[idx]
        seed_rows = [rows[idx] for rows in seed_method_rows]
        majority, majority_telemetry = _majority_choice(seed_rows)
        majority_telemetry["seed_count"] = len(seed_rows)
        output.append(
            _aggregate_record(
                method="stochastic_majority_vote",
                chosen=majority,
                baseline=baseline,
                telemetry=majority_telemetry,
                policy="majority_vote",
            )
        )

        if majority_telemetry["vote_count"] > len(seed_rows) / 2:
            target_tiebreak = majority
            target_tiebreak_telemetry = dict(majority_telemetry)
            target_tiebreak_telemetry["target_tiebreak_used"] = False
        else:
            target_tiebreak = baseline
            target_tiebreak_telemetry = dict(majority_telemetry)
            target_tiebreak_telemetry["target_tiebreak_used"] = True
        output.append(
            _aggregate_record(
                method="stochastic_target_tiebreak",
                chosen=target_tiebreak,
                baseline=baseline,
                telemetry=target_tiebreak_telemetry,
                policy="target_tiebreak_on_no_majority",
            )
        )

        correct_seed = next((row for row in seed_rows if bool(row.get("correct"))), None)
        oracle = correct_seed or majority
        oracle_telemetry = dict(majority_telemetry)
        oracle_telemetry["oracle_used_correct_seed"] = correct_seed is not None
        output.append(
            _aggregate_record(
                method="stochastic_any_seed_oracle",
                chosen=oracle,
                baseline=baseline,
                telemetry=oracle_telemetry,
                policy="oracle_any_correct_seed",
            )
        )

        target_or_seed = baseline if bool(baseline.get("correct")) else (correct_seed or majority)
        target_or_seed_telemetry = dict(majority_telemetry)
        target_or_seed_telemetry["oracle_used_target"] = bool(baseline.get("correct"))
        target_or_seed_telemetry["oracle_used_correct_seed"] = (
            not bool(baseline.get("correct")) and correct_seed is not None
        )
        output.append(
            _aggregate_record(
                method="stochastic_target_or_seed_oracle",
                chosen=target_or_seed,
                baseline=baseline,
                telemetry=target_or_seed_telemetry,
                policy="oracle_target_or_any_correct_seed",
            )
        )
    return output


def summarize_results(records: list[dict[str, Any]]) -> dict[str, float]:
    results: dict[str, float] = {}
    methods = sorted({str(record["method"]) for record in records})
    for method in methods:
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)
    add_paired_prediction_summary(results, records)
    return results


def write_markdown_summary(results: dict[str, float], output_md: str | pathlib.Path) -> None:
    methods = [
        "target_alone",
        "stochastic_majority_vote",
        "stochastic_target_tiebreak",
        "stochastic_any_seed_oracle",
        "stochastic_target_or_seed_oracle",
    ]
    lines = [
        "# Stochastic Route Aggregation Summary",
        "",
        "| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    target = float(results.get("target_alone", 0.0))
    for method in methods:
        if method not in results:
            continue
        slug = method.replace(".", "_")
        prefix = f"paired_{slug}_vs_target_alone"
        lines.append(
            "| {method} | {acc:.4f} | {delta:+.4f} | {method_only:.0f} | {baseline_only:.0f} | "
            "{both_correct:.0f} | {both_wrong:.0f} |".format(
                method=method,
                acc=float(results[method]),
                delta=float(results.get(f"{prefix}_delta_accuracy", float(results[method]) - target)),
                method_only=float(results.get(f"{prefix}_method_only", 0.0)),
                baseline_only=float(results.get(f"{prefix}_baseline_only", 0.0)),
                both_correct=float(results.get(f"{prefix}_both_correct", 0.0)),
                both_wrong=float(results.get(f"{prefix}_both_wrong", 0.0)),
            )
        )
    if "stochastic_target_or_seed_oracle" in results:
        lines.extend(
            [
                "",
                "Interpretation:",
                "",
                "The oracle rows measure candidate-set quality, while majority and target tie-break "
                "measure naive non-oracle selection. A large oracle gap with weak naive selection "
                "means the next blocker is verifier/reranker selection rather than raw sampling.",
            ]
        )
    pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction JSONLs for different stochastic salts.")
    parser.add_argument("--method", default="rotalign_kv_gate_0.10")
    parser.add_argument("--baseline-method", default="target_alone")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_sets = [load_records(path) for path in args.inputs]
    records = aggregate_records(record_sets, method=args.method, baseline_method=args.baseline_method)
    results = summarize_results(records)
    write_prediction_records(args.output_jsonl, records)
    write_prediction_sidecar(
        args.output_jsonl,
        records,
        results,
        {
            "inputs": [str(path) for path in args.inputs],
            "method": args.method,
            "baseline_method": args.baseline_method,
            "aggregation_methods": sorted({record["method"] for record in records if record["method"] != args.baseline_method}),
        },
    )
    if args.output_md:
        write_markdown_summary(results, args.output_md)
    for key, value in sorted(results.items()):
        if not key.startswith("paired_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
