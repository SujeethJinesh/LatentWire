"""Rerank stochastic route candidates with interpretable non-oracle signals."""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys
from decimal import Decimal, InvalidOperation
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.evaluate import (  # noqa: E402
    add_paired_prediction_summary,
    write_prediction_records,
    write_prediction_sidecar,
)

from scripts.aggregate_stochastic_routes import load_records  # noqa: E402


_BOXED_RE = re.compile(r"\\boxed\s*\{?[-+]?\d")
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:answer is|final answer|therefore,? the answer|so the answer|the result is)",
    re.IGNORECASE,
)
_NUMERIC_MENTION_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?(?![A-Za-z0-9])")
_DANGLING_END_RE = re.compile(
    r"(?:[+\-*/=,:;]|\b(?:and|or|because|so|then|thus|therefore)\b)\s*$",
    re.IGNORECASE,
)
STRICT_FORMAT_DELTA = 2.5


def _rows_by_index(records: list[dict[str, Any]], method: str) -> dict[int, dict[str, Any]]:
    return {
        int(record["index"]): record
        for record in records
        if str(record.get("method")) == method
    }


def _prediction_key(record: dict[str, Any]) -> str:
    value = record.get("normalized_prediction")
    if value is None or value == "":
        value = record.get("prediction", "")
    return str(value)


def _normalize_numeric_text(value: str) -> str:
    cleaned = value.strip().replace(",", "")
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    if cleaned.startswith("."):
        cleaned = f"0{cleaned}"
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    if not cleaned:
        return ""
    if "/" in cleaned:
        return cleaned
    try:
        normalized = format(Decimal(cleaned), "f")
    except (InvalidOperation, ValueError):
        return cleaned
    normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _numeric_mentions(text: str) -> list[str]:
    return [_normalize_numeric_text(match.group(0)) for match in _NUMERIC_MENTION_RE.finditer(text)]


def _tail_numeric_mention(text: str) -> str:
    mentions = _numeric_mentions(text)
    return mentions[-1] if mentions else ""


def _vote_entropy(counts: collections.Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(max(probability, 1e-12))
    return float(entropy)


def answer_format_score(record: dict[str, Any]) -> float:
    """Score whether a candidate presents its answer as a complete answer.

    This deliberately avoids the gold label. It favors explicit final-answer
    markers and penalizes outputs that likely hit the generation cap without a
    clear answer marker.
    """

    prediction = str(record.get("prediction", ""))
    normalized = str(record.get("normalized_prediction") or "")
    stripped = prediction.strip()
    lower = stripped.lower()
    score = 0.0

    if normalized:
        score += 0.5
    if _BOXED_RE.search(stripped):
        score += 4.0
    if _EXPLICIT_ANSWER_RE.search(stripped):
        score += 3.0
    elif "therefore" in lower:
        score += 1.0
    if normalized and stripped.endswith(normalized):
        score += 2.0
    if normalized and re.search(rf"(?:=|is|are)\s*\$?{re.escape(normalized)}\$?\s*[.!\n]*$", stripped):
        score += 1.0

    generated_tokens = record.get("generated_tokens")
    if isinstance(generated_tokens, (int, float)) and generated_tokens >= 64 and not (
        _BOXED_RE.search(stripped) or _EXPLICIT_ANSWER_RE.search(stripped)
    ):
        score -= 1.0
    if len(stripped) < 24:
        score -= 0.5
    return float(score)


def numeric_consistency_score(record: dict[str, Any]) -> float:
    """Score whether the candidate's numeric content is self-consistent."""

    prediction = str(record.get("prediction", "")).strip()
    normalized = _normalize_numeric_text(str(record.get("normalized_prediction") or ""))
    mentions = _numeric_mentions(prediction)
    unique_mentions = set(mentions)
    score = 0.0

    if mentions:
        score += 1.0
    if normalized and normalized in unique_mentions:
        score += 3.0
    if normalized and mentions and mentions[-1] == normalized:
        score += 4.0
    if len(unique_mentions) == 1 and mentions:
        score += 1.0
    if normalized and len(mentions) >= 2 and mentions[-1] != normalized:
        score -= 3.0
    if normalized and mentions and mentions[0] == normalized and mentions[-1] == normalized:
        score += 1.0
    return float(score)


def completion_score(record: dict[str, Any]) -> float:
    """Score whether the candidate looks like a complete finished answer."""

    prediction = str(record.get("prediction", "")).strip()
    normalized = _normalize_numeric_text(str(record.get("normalized_prediction") or ""))
    if not prediction:
        return 0.0

    score = 0.0
    if prediction.endswith((".", "!", "?", ")", "]", "}")):
        score += 1.0
    if not _DANGLING_END_RE.search(prediction):
        score += 1.0
    tail_numeric = _tail_numeric_mention(prediction)
    if tail_numeric:
        score += 0.5
    if normalized and tail_numeric == normalized and prediction.rstrip().endswith(tail_numeric):
        score += 2.0
    if prediction.endswith(("...", "…")):
        score -= 0.5
    if prediction.endswith((",", ":", ";")):
        score -= 1.0
    return float(score)


def _candidate_metadata(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter(_prediction_key(row) for row in candidates)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_count = ranked[0][1] if ranked else 0
    next_count = ranked[1][1] if len(ranked) > 1 else 0
    numeric_scores = [float(row.get("candidate_numeric_consistency_score", 0.0)) for row in candidates]
    completion_scores = [float(row.get("candidate_completion_score", 0.0)) for row in candidates]
    return {
        "candidate_count": len(candidates),
        "candidate_unique_predictions": len(counts),
        "candidate_vote_prediction": ranked[0][0] if ranked else "",
        "candidate_vote_count": int(top_count),
        "candidate_vote_margin": int(top_count - next_count),
        "candidate_vote_entropy": _vote_entropy(counts),
        "candidate_numeric_consistency_mean": sum(numeric_scores) / max(len(numeric_scores), 1),
        "candidate_completion_mean": sum(completion_scores) / max(len(completion_scores), 1),
        "candidate_numeric_consistency_max": max(numeric_scores) if numeric_scores else 0.0,
        "candidate_completion_max": max(completion_scores) if completion_scores else 0.0,
        "candidate_oracle_correct": bool(any(bool(row.get("correct")) for row in candidates)),
        "seed_correct_count": int(
            sum(bool(row.get("correct")) for row in candidates if row.get("candidate_source") != "target")
        ),
    }


def _annotate_candidates(baseline: dict[str, Any], seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target = dict(baseline)
    target["candidate_source"] = "target"
    candidates.append(target)
    for seed_idx, row in enumerate(seed_rows):
        candidate = dict(row)
        candidate["candidate_source"] = f"seed_{seed_idx}"
        candidate["source_input_index"] = seed_idx
        candidates.append(candidate)

    counts: collections.Counter[str] = collections.Counter(_prediction_key(row) for row in candidates)
    for candidate in candidates:
        candidate["candidate_answer_agreement"] = int(counts[_prediction_key(candidate)])
        candidate["candidate_format_score"] = answer_format_score(candidate)
        candidate["candidate_numeric_mentions"] = _numeric_mentions(str(candidate.get("prediction", "")))
        candidate["candidate_numeric_mention_count"] = len(candidate["candidate_numeric_mentions"])
        candidate["candidate_unique_numeric_mention_count"] = len(set(candidate["candidate_numeric_mentions"]))
        candidate["candidate_tail_numeric_mention"] = _tail_numeric_mention(str(candidate.get("prediction", "")))
        candidate["candidate_numeric_consistency_score"] = numeric_consistency_score(candidate)
        candidate["candidate_completion_score"] = completion_score(candidate)
        candidate["candidate_has_terminal_punctuation"] = int(
            str(candidate.get("prediction", "")).strip().endswith((".", "!", "?", ")", "]", "}"))
        )
    return candidates


def _choose(candidates: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    target = next(row for row in candidates if row["candidate_source"] == "target")
    seeds = [row for row in candidates if row["candidate_source"] != "target"]

    if policy == "agreement_then_format":
        return max(
            candidates,
            key=lambda row: (
                int(row["candidate_answer_agreement"]),
                float(row["candidate_format_score"]),
                row["candidate_source"] == "target",
            ),
        )
    if policy == "format_then_agreement":
        return max(
            candidates,
            key=lambda row: (
                float(row["candidate_format_score"]),
                int(row["candidate_answer_agreement"]),
                row["candidate_source"] == "target",
            ),
        )
    if policy == "seed_format_confidence":
        return max(
            seeds,
            key=lambda row: (
                float(row["candidate_format_score"]),
                int(row["candidate_answer_agreement"]),
            ),
        )
    if policy == "target_on_low_format":
        best_seed = max(
            seeds,
            key=lambda row: (
                float(row["candidate_format_score"]),
                int(row["candidate_answer_agreement"]),
            ),
        )
        if float(best_seed["candidate_format_score"]) >= float(target["candidate_format_score"]) + 1.0:
            return best_seed
        return target
    if policy == "target_on_strict_format":
        best_seed = max(
            seeds,
            key=lambda row: (
                float(row["candidate_format_score"]),
                int(row["candidate_answer_agreement"]),
            ),
        )
        if float(best_seed["candidate_format_score"]) >= float(target["candidate_format_score"]) + STRICT_FORMAT_DELTA:
            return best_seed
        return target
    if policy == "agreement_or_target":
        best = max(
            candidates,
            key=lambda row: (
                int(row["candidate_answer_agreement"]),
                float(row["candidate_format_score"]),
                row["candidate_source"] == "target",
            ),
        )
        return best if int(best["candidate_answer_agreement"]) > 1 else target
    if policy == "numeric_consistency_then_completion":
        return max(
            candidates,
            key=lambda row: (
                float(row["candidate_numeric_consistency_score"]),
                float(row["candidate_completion_score"]),
                row["candidate_source"] == "target",
            ),
        )
    if policy == "completion_then_numeric_consistency":
        return max(
            candidates,
            key=lambda row: (
                float(row["candidate_completion_score"]),
                float(row["candidate_numeric_consistency_score"]),
                row["candidate_source"] == "target",
            ),
        )
    if policy == "numeric_consistency_or_target":
        best = max(
            candidates,
            key=lambda row: (
                float(row["candidate_numeric_consistency_score"]),
                float(row["candidate_completion_score"]),
                row["candidate_source"] == "target",
            ),
        )
        return best if float(best["candidate_numeric_consistency_score"]) > 0.0 else target
    raise ValueError(f"Unknown reranking policy: {policy}")


def _reranked_record(
    *,
    method_name: str,
    policy: str,
    chosen: dict[str, Any],
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    record = dict(chosen)
    record["method"] = method_name
    record["reranking_policy"] = policy
    record["answer"] = baseline.get("answer", chosen.get("answer"))
    record["index"] = int(baseline["index"])
    if baseline.get("example_id") is not None:
        record["example_id"] = baseline.get("example_id")
    record.update(_candidate_metadata(candidates))
    record["target_correct"] = bool(baseline.get("correct"))
    record["selected_candidate_source"] = chosen.get("candidate_source")
    record["selected_candidate_format_score"] = float(chosen.get("candidate_format_score", 0.0))
    record["selected_candidate_answer_agreement"] = int(chosen.get("candidate_answer_agreement", 0))
    target = next(row for row in candidates if row.get("candidate_source") == "target")
    record["selected_candidate_format_delta_vs_target"] = float(chosen.get("candidate_format_score", 0.0)) - float(
        target.get("candidate_format_score", 0.0)
    )
    record["strict_format_delta_threshold"] = STRICT_FORMAT_DELTA if policy == "target_on_strict_format" else None
    record["selected_candidate_numeric_consistency_score"] = float(
        chosen.get("candidate_numeric_consistency_score", 0.0)
    )
    record["selected_candidate_completion_score"] = float(chosen.get("candidate_completion_score", 0.0))
    record["selected_candidate_numeric_mention_count"] = int(chosen.get("candidate_numeric_mention_count", 0))
    record["selected_candidate_tail_numeric_mention"] = str(chosen.get("candidate_tail_numeric_mention", ""))
    record["candidate_scores"] = [
        {
            "source": row.get("candidate_source"),
            "normalized_prediction": row.get("normalized_prediction"),
            "answer_agreement": int(row.get("candidate_answer_agreement", 0)),
            "format_score": float(row.get("candidate_format_score", 0.0)),
            "numeric_consistency_score": float(row.get("candidate_numeric_consistency_score", 0.0)),
            "completion_score": float(row.get("candidate_completion_score", 0.0)),
            "numeric_mention_count": int(row.get("candidate_numeric_mention_count", 0)),
            "unique_numeric_mention_count": int(row.get("candidate_unique_numeric_mention_count", 0)),
            "tail_numeric_mention": row.get("candidate_tail_numeric_mention", ""),
            "has_terminal_punctuation": int(row.get("candidate_has_terminal_punctuation", 0)),
            "correct": bool(row.get("correct")),
        }
        for row in candidates
    ]
    return record


def rerank_records(
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

    output: list[dict[str, Any]] = []
    for idx in indices:
        baseline = dict(baseline_rows[idx])
        baseline["method"] = baseline_method
        output.append(baseline)

        seed_rows = [rows[idx] for rows in seed_method_rows]
        candidates = _annotate_candidates(baseline, seed_rows)
        for policy in [
            "agreement_then_format",
            "format_then_agreement",
            "seed_format_confidence",
            "target_on_low_format",
            "target_on_strict_format",
            "agreement_or_target",
            "numeric_consistency_then_completion",
            "completion_then_numeric_consistency",
            "numeric_consistency_or_target",
        ]:
            chosen = _choose(candidates, policy)
            output.append(
                _reranked_record(
                    method_name=f"rerank_{policy}",
                    policy=policy,
                    chosen=chosen,
                    baseline=baseline,
                    candidates=candidates,
                )
            )
    return output


def summarize_results(records: list[dict[str, Any]]) -> dict[str, float]:
    results: dict[str, float] = {}
    for method in sorted({str(record["method"]) for record in records}):
        rows = [record for record in records if str(record["method"]) == method]
        results[method] = sum(bool(row.get("correct")) for row in rows) / max(len(rows), 1)
    add_paired_prediction_summary(results, records)
    return results


def write_markdown_summary(results: dict[str, float], output_md: str | pathlib.Path) -> None:
    methods = [
        "target_alone",
        "rerank_agreement_then_format",
        "rerank_format_then_agreement",
        "rerank_seed_format_confidence",
        "rerank_target_on_low_format",
        "rerank_target_on_strict_format",
        "rerank_agreement_or_target",
        "rerank_numeric_consistency_then_completion",
        "rerank_completion_then_numeric_consistency",
        "rerank_numeric_consistency_or_target",
    ]
    lines = [
        "# Stochastic Route Reranker Summary",
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
    if "rerank_format_then_agreement" in results:
        lines.extend(
            [
                "",
                "Interpretation:",
                "",
                "Format-first reranking is the first non-oracle selector to test whether stochastic "
                "route candidates can be used without label leakage. Compare it to target-alone and "
                "to the oracle aggregate to separate candidate quality from selection quality.",
            ]
        )
    if "rerank_target_on_strict_format" in results:
        lines.extend(
            [
                "",
                "Strict fallback ablation:",
                "",
                f"The strict target fallback uses a `{STRICT_FORMAT_DELTA:.1f}` format-score margin above "
                "target-alone before selecting a seed. Treat this as an explicit ablation threshold; it "
                "needs held-out validation before becoming a paper claim.",
            ]
        )
    if "rerank_numeric_consistency_then_completion" in results:
        lines.extend(
            [
                "",
                "Numeric ablation:",
                "",
                "Numeric-consistency-first reranking is a disjoint ablation that uses only the candidate's "
                "own numeric text and completion cues. It checks whether the reranker can prefer self-"
                "consistent numeric answers even when format markers are weak or misleading.",
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
    records = rerank_records(record_sets, method=args.method, baseline_method=args.baseline_method)
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
            "reranking_methods": sorted({record["method"] for record in records if record["method"] != args.baseline_method}),
        },
    )
    if args.output_md:
        write_markdown_summary(results, args.output_md)
    for key, value in sorted(results.items()):
        if not key.startswith("paired_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
