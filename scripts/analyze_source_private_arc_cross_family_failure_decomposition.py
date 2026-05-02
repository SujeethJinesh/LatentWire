from __future__ import annotations

"""Decompose ARC cross-family source-private packet failures.

The source-family falsification gates tell us whether an alternate source cache
passes. This diagnostic separates three failure modes that matter for choosing
the next method branch:

1. weak source endpoint,
2. packet/receiver decoding loss,
3. source-family mismatch against the Qwen cache baseline.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_cross_family_failure_decomposition_20260502")
DEFAULT_WRAPPERS = [
    pathlib.Path(
        "results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000"
    ),
    pathlib.Path("results/source_private_arc_challenge_source_family_cache_falsification_20260502_qwen15_cpu"),
    pathlib.Path("results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"),
]
MATCHED = arc_gate.MATCHED_CONDITION
QWEN_SUB = "qwen_substituted_packet"


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
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: pathlib.Path | str) -> list[dict[str, str]]:
    with _resolve(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _source_bucket(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        alt_correct = _bool(row["alt_source_correct"])
        qwen_correct = _bool(row["qwen_source_correct"])
        if alt_correct and qwen_correct:
            counts["both_correct"] += 1
        elif alt_correct:
            counts["alt_only_correct"] += 1
        elif qwen_correct:
            counts["qwen_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
    total = len(rows)
    return {
        "n": total,
        "agreement_rate": _safe_div(sum(1 for row in rows if _bool(row["agree"])), total),
        "disagreement_count": int(sum(1 for row in rows if not _bool(row["agree"]))),
        "alt_source_accuracy": _safe_div(sum(1 for row in rows if _bool(row["alt_source_correct"])), total),
        "qwen_source_accuracy": _safe_div(sum(1 for row in rows if _bool(row["qwen_source_correct"])), total),
        "alt_qwen_oracle_accuracy": _safe_div(
            sum(
                1
                for row in rows
                if _bool(row["alt_source_correct"]) or _bool(row["qwen_source_correct"])
            ),
            total,
        ),
        "bucket_counts": {name: int(counts[name]) for name in sorted(counts)},
        "bucket_rates": {
            name: _safe_div(float(counts[name]), total)
            for name in ("alt_only_correct", "qwen_only_correct", "both_correct", "both_wrong")
        },
    }


def _source_agreement_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        disagreement_rows = [row for row in split_rows if not _bool(row["agree"])]
        by_split[split] = {
            "full": _source_bucket(split_rows),
            "qwen_disagreement": _source_bucket(disagreement_rows),
        }
    return by_split


def _packet_seed_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_rows": 0,
            "accuracy": float("nan"),
            "source_selected_correct_rate": float("nan"),
            "prediction_matches_source_selected_rate": float("nan"),
            "packet_minus_source_selected_correct": float("nan"),
            "mean_payload_bytes": float("nan"),
        }
    correct = [float(row["correct"]) for row in rows]
    source_selected_correct = [
        float(int(row["metadata"]["source_selected_index"]) == int(row["answer_index"])) for row in rows
    ]
    follows_source = [
        float(int(row["prediction_index"]) == int(row["metadata"]["source_selected_index"])) for row in rows
    ]
    return {
        "n_rows": len(rows),
        "accuracy": _mean(correct),
        "source_selected_correct_rate": _mean(source_selected_correct),
        "prediction_matches_source_selected_rate": _mean(follows_source),
        "packet_minus_source_selected_correct": _mean(correct) - _mean(source_selected_correct),
        "mean_payload_bytes": _mean([float(row.get("payload_bytes", float("nan"))) for row in rows]),
    }


def _aggregate_seed_metrics(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_rows:
        return {
            "seed_count": 0,
            "n_rows_per_seed": float("nan"),
            "accuracy_mean": float("nan"),
            "accuracy_min": float("nan"),
            "source_selected_correct_mean": float("nan"),
            "prediction_matches_source_selected_mean": float("nan"),
            "prediction_matches_source_selected_min": float("nan"),
            "packet_minus_source_selected_correct_mean": float("nan"),
            "mean_payload_bytes": float("nan"),
        }
    return {
        "seed_count": len(seed_rows),
        "n_rows_per_seed": _mean([float(row["n_rows"]) for row in seed_rows]),
        "accuracy_mean": _mean([float(row["accuracy"]) for row in seed_rows]),
        "accuracy_min": min(float(row["accuracy"]) for row in seed_rows),
        "source_selected_correct_mean": _mean(
            [float(row["source_selected_correct_rate"]) for row in seed_rows]
        ),
        "prediction_matches_source_selected_mean": _mean(
            [float(row["prediction_matches_source_selected_rate"]) for row in seed_rows]
        ),
        "prediction_matches_source_selected_min": min(
            float(row["prediction_matches_source_selected_rate"]) for row in seed_rows
        ),
        "packet_minus_source_selected_correct_mean": _mean(
            [float(row["packet_minus_source_selected_correct"]) for row in seed_rows]
        ),
        "mean_payload_bytes": _mean([float(row["mean_payload_bytes"]) for row in seed_rows]),
    }


def _packet_summary(*, full_rows: list[dict[str, Any]], disagreement_rows: list[dict[str, Any]]) -> dict[str, Any]:
    surfaces = {
        "full_matched": full_rows,
        "qwen_disagreement": disagreement_rows,
    }
    summary: dict[str, Any] = {}
    for surface, rows in surfaces.items():
        surface_summary: dict[str, Any] = {}
        split_condition_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            split_condition_seed[(str(row["split"]), str(row["condition"]), int(row["seed"]))].append(row)
        split_conditions = sorted({(split, condition) for split, condition, _seed in split_condition_seed})
        for split, condition in split_conditions:
            seed_metrics = [
                _packet_seed_metrics(seed_rows)
                for (row_split, row_condition, _seed), seed_rows in split_condition_seed.items()
                if row_split == split and row_condition == condition
            ]
            surface_summary.setdefault(split, {})[condition] = _aggregate_seed_metrics(seed_metrics)
        summary[surface] = surface_summary
    return summary


def _payload_metric(payload: dict[str, Any], split: str, slice_name: str, key: str) -> float:
    try:
        return float(payload["splits"][split][slice_name]["aggregate"][key])
    except KeyError:
        return float("nan")


def _infer_blocker(
    *,
    payload: dict[str, Any],
    source_agreement: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    test_full = source_agreement.get("test", {}).get("full", {})
    test_disagreement = source_agreement.get("test", {}).get("qwen_disagreement", {})
    full_packet = packet.get("full_matched", {}).get("test", {}).get(MATCHED, {})
    disagreement_matched = packet.get("qwen_disagreement", {}).get("test", {}).get(MATCHED, {})
    disagreement_qwen = packet.get("qwen_disagreement", {}).get("test", {}).get(QWEN_SUB, {})
    target_accuracy = _payload_metric(payload, "test", "full_slice", "target_accuracy")
    qwen_sub_disagreement = float(disagreement_qwen.get("accuracy_mean", float("nan")))
    matched_disagreement = float(disagreement_matched.get("accuracy_mean", float("nan")))
    source_quality_gap = float(test_full.get("alt_source_accuracy", float("nan"))) - target_accuracy
    packet_decode_gap = float(full_packet.get("accuracy_mean", float("nan"))) - float(
        test_full.get("alt_source_accuracy", float("nan"))
    )
    packet_follows_source = float(
        full_packet.get("prediction_matches_source_selected_mean", float("nan"))
    )
    source_family_gap = matched_disagreement - qwen_sub_disagreement
    oracle_headroom = float(test_disagreement.get("alt_qwen_oracle_accuracy", float("nan"))) - max(
        float(test_disagreement.get("alt_source_accuracy", float("nan"))),
        float(test_disagreement.get("qwen_source_accuracy", float("nan"))),
    )
    if source_quality_gap < 0.0 and abs(packet_decode_gap) <= 0.02 and packet_follows_source >= 0.95:
        primary = "source_endpoint_quality"
        next_gate = "replace or improve the cross-family source endpoint before revising the 8B packet codec"
    elif packet_follows_source < 0.95 or packet_decode_gap < -0.02:
        primary = "packet_decoding_loss"
        next_gate = "train a stronger receiver/codec that preserves the source choice before changing source families"
    elif source_family_gap < 0.0:
        primary = "source_family_mismatch"
        next_gate = "learn a common-feature selector/router that decides when to trust the alternate source"
    else:
        primary = "mixed_or_unresolved"
        next_gate = "rerun with larger slices and a stronger alternate source to disambiguate"
    return {
        "primary_blocker": primary,
        "next_gate": next_gate,
        "target_accuracy": target_accuracy,
        "source_quality_gap_vs_target": source_quality_gap,
        "packet_decode_gap_vs_alt_source": packet_decode_gap,
        "packet_follows_source_rate": packet_follows_source,
        "matched_minus_qwen_substituted_on_disagreement": source_family_gap,
        "alt_qwen_disagreement_oracle_headroom": oracle_headroom,
    }


def _flat_rows_for_csv(wrapper_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wrapper in wrapper_summaries:
        family = wrapper["source_family"]
        for split, split_summary in wrapper["source_agreement"].items():
            for surface, source_metrics in split_summary.items():
                packet_surface = "full_matched" if surface == "full" else "qwen_disagreement"
                for condition in sorted(
                    wrapper["packet_summary"].get(packet_surface, {}).get(split, {})
                ):
                    packet = wrapper["packet_summary"][packet_surface][split][condition]
                    rows.append(
                        {
                            "source_family": family,
                            "split": split,
                            "surface": surface,
                            "condition": condition,
                            "source_rows": source_metrics["n"],
                            "alt_source_accuracy": source_metrics["alt_source_accuracy"],
                            "qwen_source_accuracy": source_metrics["qwen_source_accuracy"],
                            "alt_qwen_oracle_accuracy": source_metrics["alt_qwen_oracle_accuracy"],
                            "packet_accuracy_mean": packet["accuracy_mean"],
                            "source_selected_correct_mean": packet["source_selected_correct_mean"],
                            "packet_follows_source_mean": packet[
                                "prediction_matches_source_selected_mean"
                            ],
                            "packet_minus_source_selected_correct_mean": packet[
                                "packet_minus_source_selected_correct_mean"
                            ],
                        }
                    )
    return rows


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ARC Cross-Family Failure Decomposition",
        "",
        f"- wrappers analyzed: `{len(payload['wrappers'])}`",
        f"- selected next gate: `{payload['headline']['selected_next_gate']}`",
        "",
        "## Lay Explanation",
        "",
        (
            "This diagnostic asks whether cross-family transfer failed because the "
            "sender chose weak answers, because the tiny packet failed to carry the "
            "sender's choice, or because the receiver needs a better common feature "
            "space for deciding which model to trust."
        ),
        "",
        "## Per-Wrapper Decisions",
        "",
    ]
    for wrapper in payload["wrappers"]:
        d = wrapper["decision"]
        lines.extend(
            [
                f"### {wrapper['source_family']}",
                "",
                f"- pass gate: `{wrapper['pass_gate']}`",
                f"- primary blocker: `{d['primary_blocker']}`",
                f"- source quality gap vs target: `{d['source_quality_gap_vs_target']:.6f}`",
                f"- packet decode gap vs source: `{d['packet_decode_gap_vs_alt_source']:.6f}`",
                f"- packet follows source rate: `{d['packet_follows_source_rate']:.6f}`",
                (
                    "- matched minus Qwen-substituted on disagreement: "
                    f"`{d['matched_minus_qwen_substituted_on_disagreement']:.6f}`"
                ),
                f"- next gate: {d['next_gate']}",
                "",
            ]
        )
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_wrapper(wrapper_dir: pathlib.Path) -> dict[str, Any]:
    wrapper_dir = _resolve(wrapper_dir)
    payload_path = wrapper_dir / "source_family_cache_falsification.json"
    agreement_path = wrapper_dir / "source_cache_agreement.csv"
    full_predictions_path = wrapper_dir / "matched_predictions.jsonl"
    disagreement_predictions_path = wrapper_dir / "qwen_disagreement_predictions.jsonl"
    payload = _read_json(payload_path)
    source_agreement = _source_agreement_summary(_read_csv(agreement_path))
    packet = _packet_summary(
        full_rows=_read_jsonl(full_predictions_path),
        disagreement_rows=_read_jsonl(disagreement_predictions_path),
    )
    decision = _infer_blocker(payload=payload, source_agreement=source_agreement, packet=packet)
    source_family = str(payload.get("alternate_source_family", wrapper_dir.name))
    return {
        "source_family": source_family,
        "wrapper_dir": _display_path(wrapper_dir),
        "pass_gate": bool(payload.get("pass_gate", False)),
        "basis": payload.get("basis", {}),
        "headline": payload.get("headline", {}),
        "inputs": {
            "source_family_cache_falsification": _display_path(payload_path),
            "source_family_cache_falsification_sha256": _sha256_file(payload_path),
            "source_cache_agreement": _display_path(agreement_path),
            "source_cache_agreement_sha256": _sha256_file(agreement_path),
            "matched_predictions": _display_path(full_predictions_path),
            "matched_predictions_sha256": _sha256_file(full_predictions_path),
            "qwen_disagreement_predictions": _display_path(disagreement_predictions_path),
            "qwen_disagreement_predictions_sha256": _sha256_file(disagreement_predictions_path),
        },
        "source_agreement": source_agreement,
        "packet_summary": packet,
        "decision": decision,
    }


def build_decomposition(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    wrapper_dirs: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = wrapper_dirs if wrapper_dirs is not None else DEFAULT_WRAPPERS
    existing = [_resolve(path) for path in candidates if _resolve(path).exists()]
    if not existing:
        raise FileNotFoundError("no source-family wrapper directories exist")
    wrappers = [_summarize_wrapper(path) for path in existing]
    blocker_counts = Counter(wrapper["decision"]["primary_blocker"] for wrapper in wrappers)
    selected = "common_feature_connector"
    if blocker_counts.get("packet_decoding_loss", 0) >= blocker_counts.get("source_endpoint_quality", 0):
        selected = "packet_fidelity_receiver"
    if blocker_counts.get("source_endpoint_quality", 0) > blocker_counts.get("packet_decoding_loss", 0):
        selected = "stronger_cross_family_source_endpoint"
    if any(
        wrapper["decision"]["matched_minus_qwen_substituted_on_disagreement"] < -0.05
        and wrapper["decision"]["packet_follows_source_rate"] >= 0.95
        for wrapper in wrappers
    ):
        selected = "common_feature_connector_with_stronger_source"
    payload = {
        "gate": "source_private_arc_cross_family_failure_decomposition",
        "pass_gate": False,
        "headline": {
            "selected_next_gate": selected,
            "primary_blocker_counts": dict(sorted(blocker_counts.items())),
            "wrappers_analyzed": len(wrappers),
        },
        "wrappers": wrappers,
        "claim_policy": {
            "paper_positive_allowed": False,
            "diagnostic_only": True,
            "safe_claim": (
                "Cross-family failure is now decomposed into source quality, packet fidelity, "
                "and source-family mismatch terms; it is not evidence against the same-family "
                "public-coordinate result."
            ),
        },
    }
    json_path = output_dir / "arc_cross_family_failure_decomposition.json"
    md_path = output_dir / "arc_cross_family_failure_decomposition.md"
    csv_path = output_dir / "family_failure_summary.csv"
    _write_json(json_path, payload)
    _write_md(md_path, payload)
    flat_rows = _flat_rows_for_csv(wrappers)
    if flat_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat_rows)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "files": [
            {"path": _display_path(json_path), "sha256": _sha256_file(json_path)},
            {"path": _display_path(md_path), "sha256": _sha256_file(md_path)},
            {"path": _display_path(csv_path), "sha256": _sha256_file(csv_path)},
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--wrapper-dir",
        dest="wrapper_dirs",
        type=pathlib.Path,
        action="append",
        default=None,
        help="Source-family falsification artifact directory. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_decomposition(output_dir=args.output_dir, wrapper_dirs=args.wrapper_dirs)
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
