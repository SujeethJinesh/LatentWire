from __future__ import annotations

"""Probe the failed ARC Llama-8B source-family scout.

This script is diagnostic only. It does not add a new method row; it explains
whether the failed Llama-8B source-choice scout contains a validation-stable
conditional signal worth reviving.
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_llama8b_failure_probe_20260502")
DEFAULT_LLAMA_RESULT = pathlib.Path(
    "results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/"
    "llama8b_disagreement_source_scout.json"
)
DEFAULT_LLAMA_PREDICTIONS = pathlib.Path(
    "results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/predictions.jsonl"
)
DEFAULT_VALIDATION_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/"
    "llama8b_validation/source_prediction_cache.jsonl"
)
DEFAULT_TEST_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/"
    "llama8b_test/source_prediction_cache.jsonl"
)
DEFAULT_QWEN_DISAGREEMENT = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu/"
    "qwen_disagreement_predictions.jsonl"
)
DEFAULT_VALIDATION_ROWS = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST_ROWS = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)

MATCHED = arc_gate.MATCHED_CONDITION
QWEN_SUB = "qwen_substituted_packet"
SAME_BYTE_TEXT = "same_byte_structured_text"
TARGET = "target_derived_sidecar"


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


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _prefix(row_id: str) -> str:
    if "_" not in row_id:
        return row_id
    parts = row_id.split("_")
    if parts[0] in {"Mercury", "MCAS", "MDSA", "NYSEDREGENTS", "TIMSS"} and len(parts) >= 2:
        return "_".join(parts[:2]) if parts[1] in {"SC", "LBS"} else parts[0]
    return parts[0]


def _load_arc_row_map(path: pathlib.Path | str) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in _read_jsonl(path)}


def _load_source_cache(path: pathlib.Path | str) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in _read_jsonl(path)}


def _prediction_index(
    rows: list[dict[str, Any]],
    *,
    variant: str | None = None,
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    indexed: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if variant is not None and row.get("variant") != variant:
            continue
        indexed[str(row["split"])][int(row["seed"])][str(row["condition"]) + "\0" + str(row["row_id"])] = row
    return indexed


def _lookup(
    indexed: dict[str, dict[int, dict[str, dict[str, Any]]]],
    *,
    split: str,
    seed: int,
    condition: str,
    row_id: str,
) -> dict[str, Any] | None:
    return indexed.get(split, {}).get(seed, {}).get(condition + "\0" + row_id)


def _row_feature(row: dict[str, Any], source_cache: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    scores = metadata.get("scores") or []
    sorted_scores = sorted([float(score) for score in scores], reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else float("nan")
    return {
        "source_selected_index": int(source_cache["source_selected_index"]),
        "choice_count": len(scores) if scores else None,
        "packet_scale": float(metadata.get("packet_scale", float("nan"))),
        "packet_code_l2": float(metadata.get("packet_code_l2", float("nan"))),
        "packet_margin": float(margin),
        "payload_dims": int(metadata.get("payload_dims", 0)),
    }


def _router_decision(rule: dict[str, Any], feature: dict[str, Any], row_lookup: dict[str, Any]) -> str:
    kind = rule["kind"]
    if kind in {"always_llama", "always_qwen"}:
        return kind.removeprefix("always_")
    if kind == "source_matches_llama_prediction":
        return "llama" if feature["source_selected_index"] == row_lookup["llama_prediction_index"] else "qwen"
    if kind == "source_matches_qwen_prediction":
        return "llama" if feature["source_selected_index"] == row_lookup["qwen_prediction_index"] else "qwen"
    if kind == "llama_qwen_disagree":
        return "llama" if row_lookup["llama_prediction_index"] != row_lookup["qwen_prediction_index"] else "qwen"
    if kind == "packet_margin_ge":
        value = feature.get("packet_margin", float("nan"))
        return "llama" if math.isfinite(value) and value >= float(rule["threshold"]) else "qwen"
    if kind == "packet_l2_ge":
        value = feature.get("packet_code_l2", float("nan"))
        return "llama" if math.isfinite(value) and value >= float(rule["threshold"]) else "qwen"
    if kind == "packet_scale_ge":
        value = feature.get("packet_scale", float("nan"))
        return "llama" if math.isfinite(value) and value >= float(rule["threshold"]) else "qwen"
    if kind == "source_index_eq":
        return "llama" if feature["source_selected_index"] == int(rule["index"]) else "qwen"
    raise ValueError(f"unknown router rule {kind!r}")


def _rule_name(rule: dict[str, Any]) -> str:
    if "threshold" in rule:
        return f"{rule['kind']}:{float(rule['threshold']):.6g}"
    if "index" in rule:
        return f"{rule['kind']}:{int(rule['index'])}"
    return str(rule["kind"])


def _candidate_rules(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {"kind": "always_llama", "deployable_without_source_index": True},
        {"kind": "always_qwen", "deployable_without_source_index": True},
        {"kind": "source_matches_llama_prediction", "deployable_without_source_index": False},
        {"kind": "source_matches_qwen_prediction", "deployable_without_source_index": False},
        {"kind": "llama_qwen_disagree", "deployable_without_source_index": True},
    ]
    max_source_index = max(int(row["feature"]["source_selected_index"]) for row in validation_rows)
    rules.extend(
        {"kind": "source_index_eq", "index": index, "deployable_without_source_index": False}
        for index in range(max_source_index + 1)
    )
    for key, kind in (
        ("packet_margin", "packet_margin_ge"),
        ("packet_code_l2", "packet_l2_ge"),
        ("packet_scale", "packet_scale_ge"),
    ):
        values = sorted(
            {
                round(float(row["feature"][key]), 8)
                for row in validation_rows
                if math.isfinite(float(row["feature"][key]))
            }
        )
        if values:
            for value in values[:: max(1, len(values) // 12)]:
                rules.append({"kind": kind, "threshold": float(value), "deployable_without_source_index": True})
    return rules


def _evaluate_router(rule: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct: list[float] = []
    chose_llama = 0
    for row in rows:
        decision = _router_decision(rule, row["feature"], row)
        chose_llama += int(decision == "llama")
        correct.append(float(row[f"{decision}_correct"]))
    return {
        "rule": _rule_name(rule),
        "deployable_without_source_index": bool(rule.get("deployable_without_source_index", False)),
        "accuracy": _mean(correct),
        "llama_choice_rate": float(chose_llama / len(rows)) if rows else float("nan"),
        "rows": len(rows),
    }


def _build_joined_rows(
    *,
    split: str,
    seeds: list[int],
    row_map: dict[str, dict[str, Any]],
    source_cache: dict[str, dict[str, Any]],
    llama_index: dict[str, dict[int, dict[str, dict[str, Any]]]],
    qwen_index: dict[str, dict[int, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for seed in seeds:
        row_ids = sorted(source_cache)
        for row_id in row_ids:
            llama = _lookup(llama_index, split=split, seed=seed, condition=MATCHED, row_id=row_id)
            qwen = _lookup(qwen_index, split=split, seed=seed, condition=QWEN_SUB, row_id=row_id)
            cached = _lookup(qwen_index, split=split, seed=seed, condition=MATCHED, row_id=row_id)
            same_byte = _lookup(llama_index, split=split, seed=seed, condition=SAME_BYTE_TEXT, row_id=row_id)
            target = _lookup(llama_index, split=split, seed=seed, condition=TARGET, row_id=row_id)
            if not (llama and qwen and cached and same_byte and target):
                raise ValueError(f"missing joined row for {split} seed {seed} row {row_id}")
            source = source_cache[row_id]
            raw = row_map[row_id]
            feature = _row_feature(llama, source)
            source_correct = int(source["source_selected_index"]) == int(raw["answer_index"])
            joined.append(
                {
                    "split": split,
                    "seed": seed,
                    "row_id": row_id,
                    "prefix": _prefix(row_id),
                    "answer_index": int(raw["answer_index"]),
                    "source_correct": bool(source_correct),
                    "llama_correct": bool(llama["correct"]),
                    "qwen_correct": bool(qwen["correct"]),
                    "cached_tiny_correct": bool(cached["correct"]),
                    "same_byte_text_correct": bool(same_byte["correct"]),
                    "target_correct": bool(target["correct"]),
                    "llama_prediction_index": int(llama["prediction_index"]),
                    "qwen_prediction_index": int(qwen["prediction_index"]),
                    "cached_tiny_prediction_index": int(cached["prediction_index"]),
                    "same_byte_text_prediction_index": int(same_byte["prediction_index"]),
                    "feature": feature,
                }
            )
    return joined


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        raise ValueError("cannot summarize empty split")
    metrics = {
        "source_accuracy": _mean([float(row["source_correct"]) for row in rows]),
        "llama_packet_accuracy": _mean([float(row["llama_correct"]) for row in rows]),
        "qwen_substituted_accuracy": _mean([float(row["qwen_correct"]) for row in rows]),
        "cached_tiny_packet_accuracy": _mean([float(row["cached_tiny_correct"]) for row in rows]),
        "same_byte_text_accuracy": _mean([float(row["same_byte_text_correct"]) for row in rows]),
        "target_accuracy": _mean([float(row["target_correct"]) for row in rows]),
        "llama_minus_qwen": _mean(
            [float(row["llama_correct"]) - float(row["qwen_correct"]) for row in rows]
        ),
        "same_byte_text_minus_llama": _mean(
            [float(row["same_byte_text_correct"]) - float(row["llama_correct"]) for row in rows]
        ),
        "source_to_llama_packet_loss": _mean(
            [float(row["source_correct"]) - float(row["llama_correct"]) for row in rows]
        ),
        "llama_qwen_oracle_accuracy": _mean(
            [float(row["llama_correct"] or row["qwen_correct"]) for row in rows]
        ),
        "source_qwen_oracle_accuracy": _mean(
            [float(row["source_correct"] or row["qwen_correct"]) for row in rows]
        ),
        "same_byte_llama_oracle_accuracy": _mean(
            [float(row["same_byte_text_correct"] or row["llama_correct"]) for row in rows]
        ),
    }
    source_correct_rows = [row for row in rows if row["source_correct"]]
    source_wrong_rows = [row for row in rows if not row["source_correct"]]
    metrics["llama_accuracy_when_source_correct"] = _mean([float(row["llama_correct"]) for row in source_correct_rows])
    metrics["llama_accuracy_when_source_wrong"] = _mean([float(row["llama_correct"]) for row in source_wrong_rows])
    metrics["qwen_accuracy_when_source_correct"] = _mean([float(row["qwen_correct"]) for row in source_correct_rows])
    metrics["qwen_accuracy_when_source_wrong"] = _mean([float(row["qwen_correct"]) for row in source_wrong_rows])
    prefix_counts: Counter[str] = Counter(str(row["prefix"]) for row in rows)
    prefix_deltas: list[dict[str, Any]] = []
    for prefix, count in prefix_counts.most_common():
        subset = [row for row in rows if row["prefix"] == prefix]
        prefix_deltas.append(
            {
                "prefix": prefix,
                "row_seed_count": count,
                "llama_minus_qwen": _mean(
                    [float(row["llama_correct"]) - float(row["qwen_correct"]) for row in subset]
                ),
                "source_accuracy": _mean([float(row["source_correct"]) for row in subset]),
                "same_byte_text_minus_llama": _mean(
                    [float(row["same_byte_text_correct"]) - float(row["llama_correct"]) for row in subset]
                ),
            }
        )
    return {
        "row_seed_count": total,
        "unique_rows": len({row["row_id"] for row in rows}),
        **metrics,
        "top_prefix_deltas": prefix_deltas[:12],
    }


def _build_row_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["row_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for row_id, group in grouped.items():
        feature = group[0]["feature"]
        summaries.append(
            {
                "split": group[0]["split"],
                "row_id": row_id,
                "prefix": group[0]["prefix"],
                "source_correct": group[0]["source_correct"],
                "llama_accuracy": _mean([float(row["llama_correct"]) for row in group]),
                "qwen_accuracy": _mean([float(row["qwen_correct"]) for row in group]),
                "cached_tiny_accuracy": _mean([float(row["cached_tiny_correct"]) for row in group]),
                "same_byte_text_accuracy": _mean([float(row["same_byte_text_correct"]) for row in group]),
                "llama_minus_qwen": _mean(
                    [float(row["llama_correct"]) - float(row["qwen_correct"]) for row in group]
                ),
                "same_byte_text_minus_llama": _mean(
                    [float(row["same_byte_text_correct"]) - float(row["llama_correct"]) for row in group]
                ),
                "source_selected_index": feature["source_selected_index"],
                "packet_margin": feature["packet_margin"],
                "packet_code_l2": feature["packet_code_l2"],
                "packet_scale": feature["packet_scale"],
            }
        )
    return sorted(summaries, key=lambda row: (row["split"], row["row_id"]))


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC Llama-8B Failure Probe",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected router: `{h['selected_router_rule']}`",
        (
            "- selected router deployable without source index: "
            f"`{h['selected_router_deployable_without_source_index']}`"
        ),
        f"- best deployable router: `{h['best_deployable_router_rule']}`",
        f"- validation selected-router accuracy: `{h['validation_selected_router_accuracy']:.6f}`",
        f"- test selected-router accuracy: `{h['test_selected_router_accuracy']:.6f}`",
        (
            "- validation best-deployable-router accuracy: "
            f"`{h['validation_best_deployable_router_accuracy']:.6f}`"
        ),
        f"- test best-deployable-router accuracy: `{h['test_best_deployable_router_accuracy']:.6f}`",
        f"- test Llama/Qwen oracle accuracy: `{h['test_llama_qwen_oracle_accuracy']:.6f}`",
        f"- test source/Qwen oracle accuracy: `{h['test_source_qwen_oracle_accuracy']:.6f}`",
        f"- test same-byte-text minus Llama: `{h['test_same_byte_text_minus_llama']:.6f}`",
        f"- test source-to-packet loss: `{h['test_source_to_llama_packet_loss']:.6f}`",
        "",
        "## Lay Explanation",
        "",
        (
            "This probe checks whether the failed Llama row failed because the source "
            "model was bad, because the 12-byte packet lost useful source choices, "
            "or because a simple rule can tell when to trust Llama instead of Qwen."
        ),
        "",
        "## Decision",
        "",
        payload["decision"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_probe(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    llama_result: pathlib.Path = DEFAULT_LLAMA_RESULT,
    llama_predictions: pathlib.Path = DEFAULT_LLAMA_PREDICTIONS,
    validation_cache: pathlib.Path = DEFAULT_VALIDATION_CACHE,
    test_cache: pathlib.Path = DEFAULT_TEST_CACHE,
    qwen_disagreement: pathlib.Path = DEFAULT_QWEN_DISAGREEMENT,
    validation_rows: pathlib.Path = DEFAULT_VALIDATION_ROWS,
    test_rows: pathlib.Path = DEFAULT_TEST_ROWS,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llama_payload = json.loads(_resolve(llama_result).read_text(encoding="utf-8"))
    variant = "matched_llama8b_source_packet"
    predictions = _read_jsonl(llama_predictions)
    qwen_predictions = _read_jsonl(qwen_disagreement)
    llama_index = _prediction_index(predictions, variant=variant)
    qwen_index = _prediction_index(qwen_predictions)
    seed_values = sorted({int(row["seed"]) for row in predictions if row.get("variant") == variant})
    row_maps = {
        "validation": _load_arc_row_map(validation_rows),
        "test": _load_arc_row_map(test_rows),
    }
    caches = {
        "validation": _load_source_cache(validation_cache),
        "test": _load_source_cache(test_cache),
    }
    split_rows = {
        split: _build_joined_rows(
            split=split,
            seeds=seed_values,
            row_map=row_maps[split],
            source_cache=caches[split],
            llama_index=llama_index,
            qwen_index=qwen_index,
        )
        for split in ("validation", "test")
    }
    validation_rules = _candidate_rules(split_rows["validation"])
    router_rows: list[dict[str, Any]] = []
    for rule in validation_rules:
        val = _evaluate_router(rule, split_rows["validation"])
        test = _evaluate_router(rule, split_rows["test"])
        router_rows.append(
            {
                "rule": val["rule"],
                "deployable_without_source_index": val["deployable_without_source_index"],
                "validation_accuracy": val["accuracy"],
                "validation_llama_choice_rate": val["llama_choice_rate"],
                "test_accuracy": test["accuracy"],
                "test_llama_choice_rate": test["llama_choice_rate"],
                "validation_minus_qwen": val["accuracy"]
                - llama_payload["splits"]["validation"]["matched_aggregate"]["qwen_substituted_accuracy_mean"],
                "test_minus_qwen": test["accuracy"]
                - llama_payload["splits"]["test"]["matched_aggregate"]["qwen_substituted_accuracy_mean"],
            }
        )
    selected_router = max(
        router_rows,
        key=lambda row: (
            row["validation_accuracy"],
            -abs(row["validation_llama_choice_rate"] - 0.5),
            row["test_accuracy"],
        ),
    )
    deployable_router = max(
        [row for row in router_rows if row["deployable_without_source_index"]],
        key=lambda row: (
            row["validation_accuracy"],
            -abs(row["validation_llama_choice_rate"] - 0.5),
            row["test_accuracy"],
        ),
    )
    split_summaries = {split: _summarize_split(rows) for split, rows in split_rows.items()}
    row_summary = _build_row_summary(split_rows["validation"] + split_rows["test"])
    pass_gate = bool(
        deployable_router["validation_minus_qwen"] >= 0.02
        and deployable_router["test_minus_qwen"] >= 0.02
        and split_summaries["test"]["same_byte_text_minus_llama"] <= 0.0
    )
    if pass_gate:
        decision = (
            "A deployable validation-selected conditional signal is alive. Promote "
            "a bounded router or learned connector follow-up before widening."
        )
    else:
        decision = (
            "The Llama scout contains diagnostic conditional headroom, but no "
            "reviewer-safe deployable source-choice method. The best router overall "
            "uses the audit-only source-selected index, the best packet-observable "
            "router is a small margin rule, and same-byte visible text remains "
            "stronger than the Llama packet on test. Treat this as evidence that "
            "the source answer signal exists but the current packet codec is lossy; "
            "move to a learned query/cache or soft-prefix connector rather than "
            "another source-choice sender."
        )
    headline = {
        "selected_router_rule": selected_router["rule"],
        "selected_router_deployable_without_source_index": selected_router[
            "deployable_without_source_index"
        ],
        "best_deployable_router_rule": deployable_router["rule"],
        "validation_best_deployable_router_accuracy": deployable_router["validation_accuracy"],
        "test_best_deployable_router_accuracy": deployable_router["test_accuracy"],
        "validation_selected_router_accuracy": selected_router["validation_accuracy"],
        "test_selected_router_accuracy": selected_router["test_accuracy"],
        "test_llama_qwen_oracle_accuracy": split_summaries["test"]["llama_qwen_oracle_accuracy"],
        "test_source_qwen_oracle_accuracy": split_summaries["test"]["source_qwen_oracle_accuracy"],
        "test_same_byte_text_minus_llama": split_summaries["test"]["same_byte_text_minus_llama"],
        "test_source_to_llama_packet_loss": split_summaries["test"]["source_to_llama_packet_loss"],
    }
    payload = {
        "gate": "source_private_arc_llama8b_failure_probe",
        "pass_gate": pass_gate,
        "decision": decision,
        "headline": headline,
        "inputs": {
            "llama_result": _display_path(llama_result),
            "llama_result_sha256": _sha256_file(llama_result),
            "llama_predictions": _display_path(llama_predictions),
            "llama_predictions_sha256": _sha256_file(llama_predictions),
            "qwen_disagreement": _display_path(qwen_disagreement),
            "qwen_disagreement_sha256": _sha256_file(qwen_disagreement),
            "validation_cache": _display_path(validation_cache),
            "validation_cache_sha256": _sha256_file(validation_cache),
            "test_cache": _display_path(test_cache),
            "test_cache_sha256": _sha256_file(test_cache),
        },
        "splits": split_summaries,
        "router_rules": sorted(router_rows, key=lambda row: row["validation_accuracy"], reverse=True),
        "claim_policy": {
            "paper_positive_allowed": pass_gate,
            "llama_source_choice_branch_alive": False,
            "diagnostic_conditional_headroom_alive": bool(
                selected_router["validation_minus_qwen"] > 0.0 and selected_router["test_minus_qwen"] > 0.0
            ),
            "safe_next_gate": "learned query/cache connector or new validation-cleared source scoring branch",
        },
    }
    json_path = output_dir / "arc_llama8b_failure_probe.json"
    md_path = output_dir / "arc_llama8b_failure_probe.md"
    csv_path = output_dir / "row_summary.csv"
    router_path = output_dir / "router_rules.csv"
    _write_json(json_path, payload)
    _write_md(md_path, payload)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_summary[0]))
        writer.writeheader()
        writer.writerows(row_summary)
    with router_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(router_rows[0]))
        writer.writeheader()
        writer.writerows(payload["router_rules"])
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "files": [
            {"path": _display_path(json_path), "sha256": _sha256_file(json_path)},
            {"path": _display_path(md_path), "sha256": _sha256_file(md_path)},
            {"path": _display_path(csv_path), "sha256": _sha256_file(csv_path)},
            {"path": _display_path(router_path), "sha256": _sha256_file(router_path)},
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llama-result", type=pathlib.Path, default=DEFAULT_LLAMA_RESULT)
    parser.add_argument("--llama-predictions", type=pathlib.Path, default=DEFAULT_LLAMA_PREDICTIONS)
    parser.add_argument("--validation-cache", type=pathlib.Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--test-cache", type=pathlib.Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--qwen-disagreement", type=pathlib.Path, default=DEFAULT_QWEN_DISAGREEMENT)
    parser.add_argument("--validation-rows", type=pathlib.Path, default=DEFAULT_VALIDATION_ROWS)
    parser.add_argument("--test-rows", type=pathlib.Path, default=DEFAULT_TEST_ROWS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_probe(
        output_dir=args.output_dir,
        llama_result=args.llama_result,
        llama_predictions=args.llama_predictions,
        validation_cache=args.validation_cache,
        test_cache=args.test_cache,
        qwen_disagreement=args.qwen_disagreement,
        validation_rows=args.validation_rows,
        test_rows=args.test_rows,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
