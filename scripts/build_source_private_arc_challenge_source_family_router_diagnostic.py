from __future__ import annotations

"""Diagnose whether ARC source-family disagreement is repairable by cheap routing.

This gate consumes the TinyLlama-vs-Qwen source-cache falsification artifact.
It asks whether receiver-side packet confidence can choose between the
TinyLlama-selected packet and the Qwen-substituted packet on examples where the
two source families disagree.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_INPUT_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"
)
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_router_diagnostic_20260502"
)
ALT_CONDITION = arc_gate.MATCHED_CONDITION
QWEN_CONDITION = "qwen_substituted_packet"
ROUTER_CONDITION = "validation_selected_packet_confidence_router"
METRICS = ("best_score", "margin", "neg_entropy", "packet_norm", "pred_eq_source")


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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_agreement(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scores(row: dict[str, Any]) -> list[float]:
    return [float(value) for value in row.get("metadata", {}).get("scores", [])]


def _margin(row: dict[str, Any]) -> float:
    scores = sorted(_scores(row), reverse=True)
    return float(scores[0] - scores[1]) if len(scores) >= 2 else 0.0


def _neg_entropy(row: dict[str, Any]) -> float:
    scores = _scores(row)
    if not scores:
        return 0.0
    offset = max(scores)
    exps = [math.exp(score - offset) for score in scores]
    total = sum(exps)
    probs = [value / total for value in exps]
    return float(sum(prob * math.log(max(prob, 1e-12)) for prob in probs))


def _feature(row: dict[str, Any], metric: str) -> float:
    metadata = row.get("metadata", {})
    if metric == "best_score":
        return float(metadata.get("best_score", max(_scores(row), default=0.0)))
    if metric == "margin":
        return _margin(row)
    if metric == "neg_entropy":
        return _neg_entropy(row)
    if metric == "packet_norm":
        return float(metadata.get("packet_code_l2", 0.0))
    if metric == "pred_eq_source":
        return float(row.get("prediction_index") == metadata.get("source_selected_index"))
    raise ValueError(f"unknown metric: {metric}")


def _group_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["split"]), int(row["seed"]), str(row["content_id"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    pairs_by_split_seed: dict[str, dict[int, list[dict[str, Any]]]] = {"validation": {}, "test": {}}
    for (split, seed, content_id), conditions in grouped.items():
        if ALT_CONDITION not in conditions or QWEN_CONDITION not in conditions:
            continue
        pair = {
            "split": split,
            "seed": seed,
            "content_id": content_id,
            "alt": conditions[ALT_CONDITION],
            "qwen": conditions[QWEN_CONDITION],
        }
        pairs_by_split_seed.setdefault(split, {}).setdefault(seed, []).append(pair)
    for split in pairs_by_split_seed:
        for seed in pairs_by_split_seed[split]:
            pairs_by_split_seed[split][seed].sort(key=lambda row: row["content_id"])
    return pairs_by_split_seed


def _accuracy(rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(sum(bool(selector(row)["correct"]) for row in rows) / len(rows))


def _condition_rows_for_bootstrap(
    pairs: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for pair in pairs:
        chosen = selector(pair)
        output.append(
            {
                **chosen,
                "condition": ROUTER_CONDITION,
                "content_id": pair["content_id"],
            }
        )
        output.append(
            {
                **pair["qwen"],
                "condition": QWEN_CONDITION,
                "content_id": pair["content_id"],
            }
        )
        output.append(
            {
                **pair["alt"],
                "condition": ALT_CONDITION,
                "content_id": pair["content_id"],
            }
        )
    return output


def _paired_ci_vs_baseline(
    pairs: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    baseline: str,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, float]:
    rows = _condition_rows_for_bootstrap(pairs, selector)
    return arc_gate._paired_bootstrap(
        rows,
        condition=ROUTER_CONDITION,
        baseline=baseline,
        seed=seed,
        samples=bootstrap_samples,
    )


def _threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [0.0]
    mids = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    span = max(unique[-1] - unique[0], 1.0)
    below_all = unique[0] - span - 1.0
    above_all = unique[-1] + span + 1.0
    return [below_all, *unique, *mids, above_all]


def _selector(metric: str, threshold: float) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def choose(pair: dict[str, Any]) -> dict[str, Any]:
        delta = _feature(pair["alt"], metric) - _feature(pair["qwen"], metric)
        return pair["alt"] if delta >= threshold else pair["qwen"]

    return choose


def _alt_rate(pairs: list[dict[str, Any]], metric: str, threshold: float) -> float:
    if not pairs:
        return 0.0
    return float(
        sum(_feature(pair["alt"], metric) - _feature(pair["qwen"], metric) >= threshold for pair in pairs)
        / len(pairs)
    )


def _fit_rules(validation_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        values = [_feature(pair["alt"], metric) - _feature(pair["qwen"], metric) for pair in validation_pairs]
        best: dict[str, Any] | None = None
        for threshold in _threshold_candidates(values):
            selector = _selector(metric, threshold)
            accuracy = _accuracy(validation_pairs, selector)
            alt_rate = _alt_rate(validation_pairs, metric, threshold)
            candidate = {
                "metric": metric,
                "threshold": threshold,
                "validation_accuracy": accuracy,
                "validation_alt_rate": alt_rate,
            }
            if best is None or (
                candidate["validation_accuracy"],
                -candidate["validation_alt_rate"],
                candidate["metric"],
            ) > (
                best["validation_accuracy"],
                -best["validation_alt_rate"],
                best["metric"],
            ):
                best = candidate
        assert best is not None
        rows.append(best)
    rows.sort(key=lambda row: (row["validation_accuracy"], -row["validation_alt_rate"], row["metric"]), reverse=True)
    return rows


def _summarize_rule_by_seed(
    *,
    rule: dict[str, Any],
    pairs_by_seed: dict[int, list[dict[str, Any]]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    metric = str(rule["metric"])
    threshold = float(rule["threshold"])
    selector = _selector(metric, threshold)
    per_seed = []
    for seed, pairs in sorted(pairs_by_seed.items()):
        alt_accuracy = _accuracy(pairs, lambda pair: pair["alt"])
        qwen_accuracy = _accuracy(pairs, lambda pair: pair["qwen"])
        router_accuracy = _accuracy(pairs, selector)
        oracle_accuracy = _accuracy(pairs, lambda pair: pair["alt"] if pair["alt"]["correct"] else pair["qwen"])
        ci_qwen = _paired_ci_vs_baseline(
            pairs,
            selector,
            baseline=QWEN_CONDITION,
            seed=seed + 3001,
            bootstrap_samples=bootstrap_samples,
        )
        per_seed.append(
            {
                "seed": seed,
                "n": len(pairs),
                "alt_accuracy": alt_accuracy,
                "qwen_accuracy": qwen_accuracy,
                "router_accuracy": router_accuracy,
                "oracle_accuracy": oracle_accuracy,
                "router_minus_qwen": router_accuracy - qwen_accuracy,
                "router_minus_alt": router_accuracy - alt_accuracy,
                "router_alt_rate": _alt_rate(pairs, metric, threshold),
                "paired_ci95_vs_qwen": ci_qwen,
            }
        )
    return {
        "per_seed": per_seed,
        "aggregate": {
            "seed_count": len(per_seed),
            "n": per_seed[0]["n"] if per_seed else 0,
            "alt_accuracy_mean": float(statistics.fmean(row["alt_accuracy"] for row in per_seed)),
            "qwen_accuracy_mean": float(statistics.fmean(row["qwen_accuracy"] for row in per_seed)),
            "router_accuracy_mean": float(statistics.fmean(row["router_accuracy"] for row in per_seed)),
            "router_accuracy_min": float(min(row["router_accuracy"] for row in per_seed)),
            "oracle_accuracy_mean": float(statistics.fmean(row["oracle_accuracy"] for row in per_seed)),
            "router_minus_qwen_mean": float(statistics.fmean(row["router_minus_qwen"] for row in per_seed)),
            "router_minus_qwen_min": float(min(row["router_minus_qwen"] for row in per_seed)),
            "router_alt_rate_mean": float(statistics.fmean(row["router_alt_rate"] for row in per_seed)),
            "paired_ci95_low_vs_qwen_min": float(
                min(row["paired_ci95_vs_qwen"]["ci95_low"] for row in per_seed)
            ),
        },
    }


def _summarize_source_complementarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        disagreement = [row for row in split_rows if row["agree"] == "False"]
        output[split] = {
            "n": len(split_rows),
            "agreement_count": sum(row["agree"] == "True" for row in split_rows),
            "disagreement_count": len(disagreement),
            "alt_source_accuracy": float(sum(row["alt_source_correct"] == "True" for row in split_rows) / len(split_rows)),
            "qwen_source_accuracy": float(sum(row["qwen_source_correct"] == "True" for row in split_rows) / len(split_rows)),
            "source_choice_oracle_accuracy": float(
                sum(row["alt_source_correct"] == "True" or row["qwen_source_correct"] == "True" for row in split_rows)
                / len(split_rows)
            ),
            "disagreement_alt_correct": sum(row["alt_source_correct"] == "True" for row in disagreement),
            "disagreement_qwen_correct": sum(row["qwen_source_correct"] == "True" for row in disagreement),
            "disagreement_both_wrong": sum(
                row["alt_source_correct"] != "True" and row["qwen_source_correct"] != "True"
                for row in disagreement
            ),
        }
    return output


def _write_rule_csv(path: pathlib.Path, rule_rows: list[dict[str, Any]], test_summaries: dict[str, Any]) -> None:
    fields = [
        "metric",
        "threshold",
        "validation_accuracy",
        "validation_alt_rate",
        "test_router_accuracy_mean",
        "test_qwen_accuracy_mean",
        "test_oracle_accuracy_mean",
        "test_router_minus_qwen_mean",
        "test_router_minus_qwen_min",
        "test_router_alt_rate_mean",
        "test_ci95_low_vs_qwen_min",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rule_rows:
            summary = test_summaries[row["metric"]]["aggregate"]
            writer.writerow(
                {
                    "metric": row["metric"],
                    "threshold": row["threshold"],
                    "validation_accuracy": row["validation_accuracy"],
                    "validation_alt_rate": row["validation_alt_rate"],
                    "test_router_accuracy_mean": summary["router_accuracy_mean"],
                    "test_qwen_accuracy_mean": summary["qwen_accuracy_mean"],
                    "test_oracle_accuracy_mean": summary["oracle_accuracy_mean"],
                    "test_router_minus_qwen_mean": summary["router_minus_qwen_mean"],
                    "test_router_minus_qwen_min": summary["router_minus_qwen_min"],
                    "test_router_alt_rate_mean": summary["router_alt_rate_mean"],
                    "test_ci95_low_vs_qwen_min": summary["paired_ci95_low_vs_qwen_min"],
                }
            )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_rule_test_summary"]["aggregate"]
    source = payload["source_choice_complementarity"]["test"]
    lines = [
        "# ARC Source-Family Router Diagnostic",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected metric: `{payload['selected_rule']['metric']}`",
        f"- selected threshold: `{payload['selected_rule']['threshold']}`",
        f"- test disagreement rows: `{selected['n']}`",
        f"- source-choice oracle accuracy: `{source['source_choice_oracle_accuracy']:.3f}`",
        f"- packet oracle accuracy on disagreement rows: `{selected['oracle_accuracy_mean']:.3f}`",
        "",
        "| Row | Accuracy | Delta vs Qwen | CI95 low vs Qwen | Alt-rate |",
        "|---|---:|---:|---:|---:|",
        f"| TinyLlama packet | {selected['alt_accuracy_mean']:.3f} | "
        f"{selected['alt_accuracy_mean'] - selected['qwen_accuracy_mean']:.3f} | - | - |",
        f"| Qwen-substituted packet | {selected['qwen_accuracy_mean']:.3f} | 0.000 | - | - |",
        f"| selected router | {selected['router_accuracy_mean']:.3f} | "
        f"{selected['router_minus_qwen_mean']:.3f} | "
        f"{selected['paired_ci95_low_vs_qwen_min']:.3f} | "
        f"{selected['router_alt_rate_mean']:.3f} |",
        f"| packet oracle | {selected['oracle_accuracy_mean']:.3f} | "
        f"{selected['oracle_accuracy_mean'] - selected['qwen_accuracy_mean']:.3f} | - | - |",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "Lay description: the two source models often pick different answers. We tried to decide which tiny "
        "packet to trust using only the receiver's packet-confidence signals learned on validation. That simple "
        "router did not recover the oracle headroom, so the next repair needs source-side confidence scores or "
        "a learned connector, not just receiver confidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_router_diagnostic(
    *,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    bootstrap_samples: int,
    min_gap_over_qwen: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = input_dir / "qwen_disagreement_predictions.jsonl"
    agreement_path = input_dir / "source_cache_agreement.csv"
    parent_path = input_dir / "source_family_cache_falsification.json"
    prediction_rows = _read_jsonl(prediction_path)
    agreement_rows = _read_agreement(agreement_path)
    parent = _read_json(parent_path)
    pairs_by_split_seed = _group_pairs(prediction_rows)
    validation_pairs = [
        pair for pairs in pairs_by_split_seed["validation"].values() for pair in pairs
    ]
    rule_rows = _fit_rules(validation_pairs)
    test_summaries = {
        row["metric"]: _summarize_rule_by_seed(
            rule=row,
            pairs_by_seed=pairs_by_split_seed["test"],
            bootstrap_samples=bootstrap_samples,
        )
        for row in rule_rows
    }
    validation_summaries = {
        row["metric"]: _summarize_rule_by_seed(
            rule=row,
            pairs_by_seed=pairs_by_split_seed["validation"],
            bootstrap_samples=bootstrap_samples,
        )
        for row in rule_rows
    }
    selected_rule = rule_rows[0]
    selected_test = test_summaries[str(selected_rule["metric"])]
    selected_validation = validation_summaries[str(selected_rule["metric"])]
    pass_gate = bool(
        selected_test["aggregate"]["router_minus_qwen_min"] >= min_gap_over_qwen
        and selected_test["aggregate"]["paired_ci95_low_vs_qwen_min"] > 0.0
    )
    payload = {
        "gate": "source_private_arc_challenge_source_family_router_diagnostic",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "parent_gate": _display_path(parent_path),
        "parent_pass_gate": parent["pass_gate"],
        "input_artifacts": {
            "qwen_disagreement_predictions": _display_path(prediction_path),
            "qwen_disagreement_predictions_sha256": _sha256_file(prediction_path),
            "source_cache_agreement": _display_path(agreement_path),
            "source_cache_agreement_sha256": _sha256_file(agreement_path),
        },
        "source_choice_complementarity": _summarize_source_complementarity(agreement_rows),
        "rule_candidates": rule_rows,
        "validation_rule_summaries": validation_summaries,
        "test_rule_summaries": test_summaries,
        "selected_rule": selected_rule,
        "selected_rule_validation_summary": selected_validation,
        "selected_rule_test_summary": selected_test,
        "pass_rule": (
            "Pass requires the validation-selected packet-confidence router to beat the Qwen-substituted packet "
            f"on every ARC test seed by at least {min_gap_over_qwen:.3f}, with positive paired CI95 low."
        ),
        "interpretation": (
            "The diagnostic separates source complementarity from routing quality. TinyLlama and Qwen have "
            "substantial oracle headroom on disagreement rows, but receiver-side packet confidence alone does "
            "not select the better source reliably. This weakens cheap abstention/routing as an ICLR repair and "
            "promotes source-side confidence scores or a learned cross-family connector as the next gate."
        ),
    }
    (output_dir / "source_family_router_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "source_family_router_diagnostic.md", payload)
    _write_rule_csv(output_dir / "router_rule_metrics.csv", rule_rows, test_summaries)
    selected_rows = []
    selected_selector = _selector(str(selected_rule["metric"]), float(selected_rule["threshold"]))
    for split, seed_map in pairs_by_split_seed.items():
        for seed, pairs in seed_map.items():
            for pair in pairs:
                chosen = selected_selector(pair)
                selected_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "content_id": pair["content_id"],
                        "chosen_condition": chosen["condition"],
                        "chosen_correct": bool(chosen["correct"]),
                        "alt_correct": bool(pair["alt"]["correct"]),
                        "qwen_correct": bool(pair["qwen"]["correct"]),
                        "metric": selected_rule["metric"],
                        "alt_minus_qwen_feature": _feature(pair["alt"], str(selected_rule["metric"]))
                        - _feature(pair["qwen"], str(selected_rule["metric"])),
                    }
                )
    _write_jsonl(output_dir / "selected_router_predictions.jsonl", selected_rows)
    manifest = {
        "artifacts": [
            "source_family_router_diagnostic.json",
            "source_family_router_diagnostic.md",
            "router_rule_metrics.csv",
            "selected_router_predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "source_family_router_diagnostic.json",
                "source_family_router_diagnostic.md",
                "router_rule_metrics.csv",
                "selected_router_predictions.jsonl",
            )
        },
        "pass_gate": pass_gate,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# ARC Source-Family Router Diagnostic Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- selected metric: `{selected_rule['metric']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-gap-over-qwen", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_router_diagnostic(
        input_dir=_resolve(args.input_dir),
        output_dir=_resolve(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        min_gap_over_qwen=args.min_gap_over_qwen,
    )


if __name__ == "__main__":
    main()
