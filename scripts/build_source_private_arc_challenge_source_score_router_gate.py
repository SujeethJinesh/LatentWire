from __future__ import annotations

"""Run a source-score routing gate on ARC source-family disagreement rows.

This gate extends the TinyLlama-vs-Qwen source-family falsification artifact by
materializing answer-key-forbidden source score caches for the disagreement
rows.  It then asks whether source-side confidence can choose between the
TinyLlama-selected packet and the Qwen-substituted packet better than always
falling back to Qwen.
"""

import argparse
import csv
import datetime as dt
import gc
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


DEFAULT_PARENT_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"
)
DEFAULT_OUTPUT_DIR = pathlib.Path("results/source_private_arc_challenge_source_score_router_gate_20260502")
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_TINY_MODEL = pathlib.Path(
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
    "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)
DEFAULT_QWEN_MODEL = pathlib.Path(
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)

ALT_CONDITION = arc_gate.MATCHED_CONDITION
QWEN_CONDITION = "qwen_substituted_packet"
ROUTER_CONDITION = "validation_selected_source_score_router"
SOURCE_INDEX_LOOKUP = "source_index_pair_lookup"
SCALAR_METRICS = (
    "alt_margin_minus_qwen_margin",
    "alt_neg_entropy_minus_qwen_neg_entropy",
    "alt_best_minus_qwen_best",
    "alt_std_minus_qwen_std",
    "alt_margin",
    "negative_qwen_margin",
    "alt_neg_entropy",
    "negative_qwen_neg_entropy",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_agreement(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_disagreement_ids(agreement_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    ids = {"validation": set(), "test": set()}
    for row in agreement_rows:
        if row["agree"] == "False":
            ids.setdefault(row["split"], set()).add(row["content_id"])
    return ids


def _select_rows(path: pathlib.Path, content_ids: set[str]) -> list[arc_gate.ArcRow]:
    rows = [row for row in arc_gate._load_rows(path) if row.content_id in content_ids]
    rows.sort(key=lambda row: row.content_id)
    if len(rows) != len(content_ids):
        missing = sorted(content_ids - {row.content_id for row in rows})
        raise ValueError(f"{path} missing disagreement rows: {missing[:5]}")
    return rows


def _score_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"best_score": 0.0, "margin": 0.0, "neg_entropy": 0.0, "score_std": 0.0}
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0
    offset = max(scores)
    exps = [math.exp(score - offset) for score in scores]
    total = sum(exps)
    probs = [value / total for value in exps]
    neg_entropy = sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
    mean_score = sum(scores) / len(scores)
    score_std = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
    return {
        "best_score": float(max(scores)),
        "margin": float(margin),
        "neg_entropy": float(neg_entropy),
        "score_std": float(score_std),
    }


def _materialize_score_cache(
    *,
    rows: list[arc_gate.ArcRow],
    output_path: pathlib.Path,
    source_family: str,
    source_model: pathlib.Path,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
) -> dict[str, Any]:
    scores_by_row, predictions, lm_state = arc_gate._lm_choice_loglikelihood_scores(
        rows,
        model_path=str(source_model),
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        local_files_only=local_files_only,
        normalization=source_lm_normalization,
        prompt_mode=source_lm_prompt_mode,
    )
    cache_rows = []
    for row, scores, prediction in zip(rows, scores_by_row, predictions, strict=True):
        stats = _score_stats([float(score) for score in scores])
        cache_rows.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "source_family": source_family,
                "source_model": _display_path(source_model),
                "source_score_mode": "lm_choice_loglikelihood",
                "source_lm_prompt_mode": source_lm_prompt_mode,
                "source_lm_normalization": source_lm_normalization,
                "source_scores": [float(score) for score in scores],
                "source_selected_index": int(prediction),
                "source_selected_choice_sha256": arc_gate._sha256_text(row.choices[int(prediction)]),
                "source_visible_fields": ["question", "choices"],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
                **stats,
            }
        )
    _write_jsonl(output_path, cache_rows)
    gc.collect()
    return {
        **lm_state,
        "rows": len(rows),
        "source_family": source_family,
        "source_model": _display_path(source_model),
        "source_score_cache": _display_path(output_path),
        "source_score_cache_sha256": _sha256_file(output_path),
    }


def _load_score_cache(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    return {str(row["content_id"]): row for row in _read_jsonl(path)}


def _audit_score_cache(
    *,
    rows: list[arc_gate.ArcRow],
    score_cache: dict[str, dict[str, Any]],
    source_prediction_cache: dict[str, dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    row_by_content = {row.content_id: row for row in rows}
    errors = {
        "missing_content_ids": sorted(set(row_by_content) - set(score_cache)),
        "extra_content_ids": sorted(set(score_cache) - set(row_by_content)),
        "leaked_payload_keys": [],
        "invalid_score_lengths": [],
        "selected_index_mismatches": [],
    }
    for content_id, cache_row in score_cache.items():
        row = row_by_content.get(content_id)
        if row is None:
            continue
        leaked = sorted(set(cache_row) & set(arc_gate.FORBIDDEN_SOURCE_KEYS))
        if leaked:
            errors["leaked_payload_keys"].append(f"{content_id}:{','.join(leaked)}")
        scores = cache_row.get("source_scores", [])
        if len(scores) != len(row.choices):
            errors["invalid_score_lengths"].append(content_id)
        if content_id in source_prediction_cache:
            expected = int(source_prediction_cache[content_id]["source_selected_index"])
            actual = int(cache_row["source_selected_index"])
            if expected != actual:
                errors["selected_index_mismatches"].append(
                    f"{content_id}:expected={expected}:actual={actual}"
                )
    if any(errors.values()):
        raise ValueError(f"{label} source score cache audit failed: {errors}")
    return {
        "label": label,
        "rows": len(score_cache),
        "forbidden_payload_keys_absent": True,
        "score_lengths_match_choices": True,
        "selected_indices_match_parent_cache": True,
        "source_families": sorted({str(row.get("source_family", "")) for row in score_cache.values()}),
    }


def _group_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["split"]), int(row["seed"]), str(row["content_id"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    output: dict[str, dict[int, list[dict[str, Any]]]] = {"validation": {}, "test": {}}
    for (split, seed, content_id), conditions in grouped.items():
        if ALT_CONDITION not in conditions or QWEN_CONDITION not in conditions:
            continue
        output.setdefault(split, {}).setdefault(seed, []).append(
            {
                "split": split,
                "seed": seed,
                "content_id": content_id,
                "alt": conditions[ALT_CONDITION],
                "qwen": conditions[QWEN_CONDITION],
            }
        )
    for split in output:
        for seed in output[split]:
            output[split][seed].sort(key=lambda row: row["content_id"])
    return output


def _source_index_lookup(pair: dict[str, Any]) -> tuple[int, int]:
    return (
        int(pair["alt"]["metadata"]["source_selected_index"]),
        int(pair["qwen"]["metadata"]["source_selected_index"]),
    )


def _source_score_feature(
    pair: dict[str, Any],
    metric: str,
    alt_scores: dict[str, dict[str, Any]],
    qwen_scores: dict[str, dict[str, Any]],
) -> float:
    content_id = str(pair["content_id"])
    alt = alt_scores[content_id]
    qwen = qwen_scores[content_id]
    if metric == "alt_margin_minus_qwen_margin":
        return float(alt["margin"]) - float(qwen["margin"])
    if metric == "alt_neg_entropy_minus_qwen_neg_entropy":
        return float(alt["neg_entropy"]) - float(qwen["neg_entropy"])
    if metric == "alt_best_minus_qwen_best":
        return float(alt["best_score"]) - float(qwen["best_score"])
    if metric == "alt_std_minus_qwen_std":
        return float(alt["score_std"]) - float(qwen["score_std"])
    if metric == "alt_margin":
        return float(alt["margin"])
    if metric == "negative_qwen_margin":
        return -float(qwen["margin"])
    if metric == "alt_neg_entropy":
        return float(alt["neg_entropy"])
    if metric == "negative_qwen_neg_entropy":
        return -float(qwen["neg_entropy"])
    raise ValueError(f"unknown metric: {metric}")


def _threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [0.0]
    mids = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    span = max(unique[-1] - unique[0], 1.0)
    return [unique[0] - span - 1.0, *unique, *mids, unique[-1] + span + 1.0]


def _accuracy(pairs: list[dict[str, Any]], selector: Callable[[dict[str, Any]], dict[str, Any]]) -> float:
    if not pairs:
        return 0.0
    return float(sum(bool(selector(pair)["correct"]) for pair in pairs) / len(pairs))


def _alt_rate(pairs: list[dict[str, Any]], selector: Callable[[dict[str, Any]], dict[str, Any]]) -> float:
    if not pairs:
        return 0.0
    return float(sum(selector(pair) is pair["alt"] for pair in pairs) / len(pairs))


def _fit_scalar_rules(
    validation_pairs: list[dict[str, Any]],
    alt_scores: dict[str, dict[str, Any]],
    qwen_scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for metric in SCALAR_METRICS:
        values = [_source_score_feature(pair, metric, alt_scores, qwen_scores) for pair in validation_pairs]
        best: dict[str, Any] | None = None
        for threshold in _threshold_candidates(values):
            selector = lambda pair, metric=metric, threshold=threshold: (
                pair["alt"] if _source_score_feature(pair, metric, alt_scores, qwen_scores) >= threshold else pair["qwen"]
            )
            candidate = {
                "metric": metric,
                "threshold": float(threshold),
                "validation_accuracy": _accuracy(validation_pairs, selector),
                "validation_alt_rate": _alt_rate(validation_pairs, selector),
                "kind": "scalar_threshold",
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
    return rows


def _fit_source_index_lookup(validation_pairs: list[dict[str, Any]], *, min_count: int) -> dict[str, Any]:
    grouped: dict[tuple[int, int], dict[str, int]] = {}
    for pair in validation_pairs:
        row = grouped.setdefault(_source_index_lookup(pair), {"n": 0, "alt_correct": 0, "qwen_correct": 0})
        row["n"] += 1
        row["alt_correct"] += int(bool(pair["alt"]["correct"]))
        row["qwen_correct"] += int(bool(pair["qwen"]["correct"]))
    alt_keys = sorted(
        key
        for key, row in grouped.items()
        if row["n"] >= min_count and row["alt_correct"] > row["qwen_correct"]
    )
    alt_key_set = set(alt_keys)
    selector = lambda pair: pair["alt"] if _source_index_lookup(pair) in alt_key_set else pair["qwen"]
    return {
        "metric": SOURCE_INDEX_LOOKUP,
        "threshold": None,
        "validation_accuracy": _accuracy(validation_pairs, selector),
        "validation_alt_rate": _alt_rate(validation_pairs, selector),
        "kind": "categorical_lookup",
        "min_count": min_count,
        "alt_source_index_pairs": [list(key) for key in alt_keys],
        "validation_groups": {f"{key[0]}->{key[1]}": row for key, row in sorted(grouped.items())},
    }


def _selector_from_rule(
    rule: dict[str, Any],
    alt_scores: dict[str, dict[str, Any]],
    qwen_scores: dict[str, dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if rule["kind"] == "categorical_lookup":
        alt_keys = {tuple(key) for key in rule["alt_source_index_pairs"]}
        return lambda pair: pair["alt"] if _source_index_lookup(pair) in alt_keys else pair["qwen"]
    metric = str(rule["metric"])
    threshold = float(rule["threshold"])
    return lambda pair: (
        pair["alt"] if _source_score_feature(pair, metric, alt_scores, qwen_scores) >= threshold else pair["qwen"]
    )


def _bootstrap_rows(
    pairs: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        rows.append({**selector(pair), "condition": ROUTER_CONDITION, "content_id": pair["content_id"]})
        rows.append({**pair["qwen"], "condition": QWEN_CONDITION, "content_id": pair["content_id"]})
        rows.append({**pair["alt"], "condition": ALT_CONDITION, "content_id": pair["content_id"]})
    return rows


def _paired_ci(
    pairs: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    baseline: str,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, float]:
    return arc_gate._paired_bootstrap(
        _bootstrap_rows(pairs, selector),
        condition=ROUTER_CONDITION,
        baseline=baseline,
        seed=seed,
        samples=bootstrap_samples,
    )


def _summarize_rule(
    *,
    rule: dict[str, Any],
    pairs_by_seed: dict[int, list[dict[str, Any]]],
    alt_scores: dict[str, dict[str, Any]],
    qwen_scores: dict[str, dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    selector = _selector_from_rule(rule, alt_scores, qwen_scores)
    per_seed = []
    for seed, pairs in sorted(pairs_by_seed.items()):
        alt_accuracy = _accuracy(pairs, lambda pair: pair["alt"])
        qwen_accuracy = _accuracy(pairs, lambda pair: pair["qwen"])
        router_accuracy = _accuracy(pairs, selector)
        oracle_accuracy = _accuracy(pairs, lambda pair: pair["alt"] if pair["alt"]["correct"] else pair["qwen"])
        ci_qwen = _paired_ci(
            pairs,
            selector,
            baseline=QWEN_CONDITION,
            seed=seed + 5003,
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
                "router_alt_rate": _alt_rate(pairs, selector),
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


def _score_cache_digest(paths: list[pathlib.Path]) -> dict[str, str]:
    return {_display_path(path): _sha256_file(path) for path in paths}


def _write_rule_csv(path: pathlib.Path, rule_rows: list[dict[str, Any]], summaries: dict[str, Any]) -> None:
    fields = [
        "metric",
        "kind",
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
            summary = summaries[row["metric"]]["aggregate"]
            writer.writerow(
                {
                    "metric": row["metric"],
                    "kind": row["kind"],
                    "threshold": row.get("threshold"),
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
    lines = [
        "# ARC Source-Score Router Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected metric: `{payload['selected_rule']['metric']}`",
        f"- selected kind: `{payload['selected_rule']['kind']}`",
        f"- test disagreement rows: `{selected['n']}`",
        f"- source-score sidecar bytes: `{payload['source_confidence_sidecar_bytes']}`",
        "",
        "| Row | Accuracy | Delta vs Qwen | CI95 low vs Qwen | Alt-rate |",
        "|---|---:|---:|---:|---:|",
        f"| TinyLlama packet | {selected['alt_accuracy_mean']:.3f} | "
        f"{selected['alt_accuracy_mean'] - selected['qwen_accuracy_mean']:.3f} | - | - |",
        f"| Qwen-substituted packet | {selected['qwen_accuracy_mean']:.3f} | 0.000 | - | - |",
        f"| source-score router | {selected['router_accuracy_mean']:.3f} | "
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
        "Lay description: we let the source models attach how confident they were in their chosen answer, then "
        "trained a tiny validation-only rule to decide which packet to trust. This tests whether the previous "
        "failure was just missing source confidence or whether the packet itself needs a learned connector.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_source_score_router_gate(
    *,
    parent_dir: pathlib.Path,
    output_dir: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    tiny_validation_score_cache: pathlib.Path,
    tiny_test_score_cache: pathlib.Path,
    qwen_validation_score_cache: pathlib.Path,
    qwen_test_score_cache: pathlib.Path,
    materialize_score_caches: bool,
    force_rematerialize: bool,
    tiny_source_model: pathlib.Path,
    qwen_source_model: pathlib.Path,
    source_lm_device: str,
    source_lm_dtype: str,
    tiny_source_lm_max_length: int,
    qwen_source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
    bootstrap_samples: int,
    min_lookup_count: int,
    min_gap_over_qwen: float,
    source_confidence_sidecar_bytes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = parent_dir / "qwen_disagreement_predictions.jsonl"
    agreement_path = parent_dir / "source_cache_agreement.csv"
    parent_path = parent_dir / "source_family_cache_falsification.json"
    tiny_validation_parent_cache = parent_dir / "tinyllama_validation" / "source_prediction_cache.jsonl"
    tiny_test_parent_cache = parent_dir / "tinyllama_test" / "source_prediction_cache.jsonl"
    parent_payload = _read_json(parent_path)
    qwen_validation_parent_cache = _resolve(parent_payload["source_cache_audit"]["qwen_validation_cache"])
    qwen_test_parent_cache = _resolve(parent_payload["source_cache_audit"]["qwen_test_cache"])
    disagreement_ids = _split_disagreement_ids(_read_agreement(agreement_path))
    validation_rows = _select_rows(validation_path, disagreement_ids["validation"])
    test_rows = _select_rows(test_path, disagreement_ids["test"])
    materialization = {"requested": materialize_score_caches, "materialized_this_run": False, "runs": []}
    score_jobs = [
        (validation_rows, tiny_validation_score_cache, "tinyllama_1.1b", tiny_source_model, tiny_source_lm_max_length),
        (test_rows, tiny_test_score_cache, "tinyllama_1.1b", tiny_source_model, tiny_source_lm_max_length),
        (validation_rows, qwen_validation_score_cache, "qwen2.5_0.5b", qwen_source_model, qwen_source_lm_max_length),
        (test_rows, qwen_test_score_cache, "qwen2.5_0.5b", qwen_source_model, qwen_source_lm_max_length),
    ]
    if materialize_score_caches:
        for rows, path, family, model, max_length in score_jobs:
            if force_rematerialize or not path.exists():
                materialization["runs"].append(
                    _materialize_score_cache(
                        rows=rows,
                        output_path=path,
                        source_family=family,
                        source_model=model,
                        source_lm_device=source_lm_device,
                        source_lm_dtype=source_lm_dtype,
                        source_lm_max_length=max_length,
                        source_lm_normalization=source_lm_normalization,
                        source_lm_prompt_mode=source_lm_prompt_mode,
                        local_files_only=local_files_only,
                    )
                )
                materialization["materialized_this_run"] = True
    paths = [
        tiny_validation_score_cache,
        tiny_test_score_cache,
        qwen_validation_score_cache,
        qwen_test_score_cache,
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing score caches: {missing}")
    tiny_validation_scores = _load_score_cache(tiny_validation_score_cache)
    tiny_test_scores = _load_score_cache(tiny_test_score_cache)
    qwen_validation_scores = _load_score_cache(qwen_validation_score_cache)
    qwen_test_scores = _load_score_cache(qwen_test_score_cache)
    tiny_validation_parent = _load_score_cache(tiny_validation_parent_cache)
    tiny_test_parent = _load_score_cache(tiny_test_parent_cache)
    qwen_validation_parent = _load_score_cache(qwen_validation_parent_cache)
    qwen_test_parent = _load_score_cache(qwen_test_parent_cache)
    audit = {
        "tiny_validation": _audit_score_cache(
            rows=validation_rows,
            score_cache=tiny_validation_scores,
            source_prediction_cache=tiny_validation_parent,
            label="tiny_validation",
        ),
        "tiny_test": _audit_score_cache(
            rows=test_rows,
            score_cache=tiny_test_scores,
            source_prediction_cache=tiny_test_parent,
            label="tiny_test",
        ),
        "qwen_validation": _audit_score_cache(
            rows=validation_rows,
            score_cache=qwen_validation_scores,
            source_prediction_cache=qwen_validation_parent,
            label="qwen_validation",
        ),
        "qwen_test": _audit_score_cache(
            rows=test_rows,
            score_cache=qwen_test_scores,
            source_prediction_cache=qwen_test_parent,
            label="qwen_test",
        ),
    }
    pairs_by_split_seed = _group_pairs(_read_jsonl(prediction_path))
    validation_pairs = [
        pair for pairs in pairs_by_split_seed["validation"].values() for pair in pairs
    ]
    scalar_rules = _fit_scalar_rules(validation_pairs, tiny_validation_scores, qwen_validation_scores)
    lookup_rule = _fit_source_index_lookup(validation_pairs, min_count=min_lookup_count)
    rule_rows = [*scalar_rules, lookup_rule]
    rule_rows.sort(key=lambda row: (row["validation_accuracy"], -row["validation_alt_rate"], row["metric"]), reverse=True)
    validation_summaries = {
        row["metric"]: _summarize_rule(
            rule=row,
            pairs_by_seed=pairs_by_split_seed["validation"],
            alt_scores=tiny_validation_scores,
            qwen_scores=qwen_validation_scores,
            bootstrap_samples=bootstrap_samples,
        )
        for row in rule_rows
    }
    test_summaries = {
        row["metric"]: _summarize_rule(
            rule=row,
            pairs_by_seed=pairs_by_split_seed["test"],
            alt_scores=tiny_test_scores,
            qwen_scores=qwen_test_scores,
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
    selector = _selector_from_rule(selected_rule, tiny_test_scores, qwen_test_scores)
    selected_rows = []
    for split, seed_map in pairs_by_split_seed.items():
        alt_scores = tiny_validation_scores if split == "validation" else tiny_test_scores
        qwen_scores = qwen_validation_scores if split == "validation" else qwen_test_scores
        split_selector = _selector_from_rule(selected_rule, alt_scores, qwen_scores)
        for seed, pairs in seed_map.items():
            for pair in pairs:
                chosen = split_selector(pair)
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
                        "alt_source_margin": tiny_validation_scores.get(pair["content_id"], tiny_test_scores.get(pair["content_id"], {})).get("margin"),
                        "qwen_source_margin": qwen_validation_scores.get(pair["content_id"], qwen_test_scores.get(pair["content_id"], {})).get("margin"),
                    }
                )
    del selector
    payload = {
        "gate": "source_private_arc_challenge_source_score_router_gate",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "parent_gate": _display_path(parent_path),
        "parent_pass_gate": parent_payload["pass_gate"],
        "input_artifacts": {
            "qwen_disagreement_predictions": _display_path(prediction_path),
            "qwen_disagreement_predictions_sha256": _sha256_file(prediction_path),
            "source_cache_agreement": _display_path(agreement_path),
            "source_cache_agreement_sha256": _sha256_file(agreement_path),
            "source_score_caches_sha256": _score_cache_digest(paths),
        },
        "source_score_cache_audit": audit,
        "score_cache_materialization": materialization,
        "source_confidence_sidecar_bytes": source_confidence_sidecar_bytes,
        "rule_candidates": rule_rows,
        "validation_rule_summaries": validation_summaries,
        "test_rule_summaries": test_summaries,
        "selected_rule": selected_rule,
        "selected_rule_validation_summary": selected_validation,
        "selected_rule_test_summary": selected_test,
        "pass_rule": (
            "Pass requires the validation-selected source-score router to beat Qwen-substituted packets "
            f"on every ARC test seed by at least {min_gap_over_qwen:.3f}, with positive paired CI95 low."
        ),
        "interpretation": (
            "This gate tests whether source-side confidence is enough to repair the strict TinyLlama-vs-Qwen "
            "ARC disagreement failure. A positive result would promote risk-gated packet routing as a source-family "
            "repair. A negative result means the next branch should be a learned connector or a stronger alternate "
            "source, not another scalar confidence rule."
        ),
    }
    (output_dir / "source_score_router_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "source_score_router_gate.md", payload)
    _write_rule_csv(output_dir / "source_score_router_rule_metrics.csv", rule_rows, test_summaries)
    _write_jsonl(output_dir / "selected_source_score_router_predictions.jsonl", selected_rows)
    manifest = {
        "artifacts": [
            "source_score_router_gate.json",
            "source_score_router_gate.md",
            "source_score_router_rule_metrics.csv",
            "selected_source_score_router_predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "source_score_router_gate.json",
                "source_score_router_gate.md",
                "source_score_router_rule_metrics.csv",
                "selected_source_score_router_predictions.jsonl",
            )
        },
        "pass_gate": pass_gate,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# ARC Source-Score Router Gate Manifest",
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
    parser.add_argument("--parent-dir", type=pathlib.Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--tiny-validation-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--tiny-test-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--qwen-validation-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--qwen-test-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--skip-score-materialization", action="store_true")
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--tiny-source-model", type=pathlib.Path, default=DEFAULT_TINY_MODEL)
    parser.add_argument("--qwen-source-model", type=pathlib.Path, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--tiny-source-lm-max-length", type=int, default=192)
    parser.add_argument("--qwen-source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", default="mean", choices=("mean", "sum"))
    parser.add_argument("--source-lm-prompt-mode", default="qa")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lookup-count", type=int, default=3)
    parser.add_argument("--min-gap-over-qwen", type=float, default=0.01)
    parser.add_argument("--source-confidence-sidecar-bytes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    cache_dir = output_dir / "source_score_caches"
    tiny_validation_score_cache = (
        _resolve(args.tiny_validation_score_cache)
        if args.tiny_validation_score_cache is not None
        else cache_dir / "tinyllama_validation_source_scores.jsonl"
    )
    tiny_test_score_cache = (
        _resolve(args.tiny_test_score_cache)
        if args.tiny_test_score_cache is not None
        else cache_dir / "tinyllama_test_source_scores.jsonl"
    )
    qwen_validation_score_cache = (
        _resolve(args.qwen_validation_score_cache)
        if args.qwen_validation_score_cache is not None
        else cache_dir / "qwen_validation_source_scores.jsonl"
    )
    qwen_test_score_cache = (
        _resolve(args.qwen_test_score_cache)
        if args.qwen_test_score_cache is not None
        else cache_dir / "qwen_test_source_scores.jsonl"
    )
    build_source_score_router_gate(
        parent_dir=_resolve(args.parent_dir),
        output_dir=output_dir,
        validation_path=_resolve(args.validation_path),
        test_path=_resolve(args.test_path),
        tiny_validation_score_cache=tiny_validation_score_cache,
        tiny_test_score_cache=tiny_test_score_cache,
        qwen_validation_score_cache=qwen_validation_score_cache,
        qwen_test_score_cache=qwen_test_score_cache,
        materialize_score_caches=not args.skip_score_materialization,
        force_rematerialize=args.force_rematerialize,
        tiny_source_model=_resolve(args.tiny_source_model),
        qwen_source_model=_resolve(args.qwen_source_model),
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        tiny_source_lm_max_length=args.tiny_source_lm_max_length,
        qwen_source_lm_max_length=args.qwen_source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=not args.allow_downloads,
        bootstrap_samples=args.bootstrap_samples,
        min_lookup_count=args.min_lookup_count,
        min_gap_over_qwen=args.min_gap_over_qwen,
        source_confidence_sidecar_bytes=args.source_confidence_sidecar_bytes,
    )


if __name__ == "__main__":
    main()
