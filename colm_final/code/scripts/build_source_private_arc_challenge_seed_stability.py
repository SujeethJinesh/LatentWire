from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_seed_stability_20260501")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_ANCHOR = pathlib.Path(
    "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/predictions.jsonl"
)
ANCHOR_CONTROLS = ("none", "anchor_id_shuffle", "anchor_value_shuffle", "random_anchors_same_count")


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


def _read_anchor_source_choices(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    choices: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") != arc_gate.MATCHED_CONDITION:
                continue
            metadata = dict(row.get("metadata", {}))
            content_id = str(row["content_id"])
            choices[content_id] = {
                "row_id": str(row["row_id"]),
                "content_id": content_id,
                "source_selected_index": int(metadata["source_selected_index"]),
                "source_selected_label": str(metadata.get("source_selected_label", "")),
                "source_selected_choice_sha256": str(metadata.get("source_selected_choice_sha256", "")),
                "source_visible_fields": list(metadata.get("source_visible_fields", ["question", "choices"])),
                "forbidden_source_fields": list(metadata.get("forbidden_source_fields", arc_gate.FORBIDDEN_SOURCE_KEYS)),
            }
    if not choices:
        raise ValueError(f"{path} contained no matched source-private packet rows")
    return choices


def _source_predictions_from_anchor(
    eval_rows: list[arc_gate.ArcRow],
    anchor_choices: dict[str, dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    predictions: list[int] = []
    cache_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatch: list[str] = []
    for row in eval_rows:
        entry = anchor_choices.get(row.content_id)
        if entry is None:
            missing.append(row.content_id)
            continue
        selected_index = int(entry["source_selected_index"])
        if selected_index < 0 or selected_index >= len(row.choices):
            mismatch.append(row.content_id)
            continue
        selected_hash = arc_gate._sha256_text(row.choices[selected_index])
        if entry.get("source_selected_choice_sha256") and entry["source_selected_choice_sha256"] != selected_hash:
            mismatch.append(row.content_id)
            continue
        predictions.append(selected_index)
        cache_rows.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "source_selected_index": selected_index,
                "source_selected_choice_sha256": selected_hash,
                "source_visible_fields": ["question", "choices"],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            }
        )
    if missing or mismatch:
        raise ValueError(
            "anchor source choices did not match eval rows: "
            f"missing={len(missing)} mismatch={len(mismatch)}"
        )
    return predictions, cache_rows


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _anchor_digest(anchor_texts: list[str]) -> str:
    return hashlib.sha256("\n".join(anchor_texts).encode("utf-8")).hexdigest()


def _random_anchor_texts(*, count: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    anchors: list[str] = []
    for index in range(count):
        values = rng.integers(0, 2**31 - 1, size=8)
        anchors.append(" ".join(f"random_anchor_{int(value):08x}" for value in values) + f" slot_{index}")
    return anchors


def _feature_columns_permuted(features: np.ndarray, *, seed: int) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    permutation = [int(index) for index in rng.permutation(features.shape[1])]
    if len(permutation) > 1 and permutation == list(range(len(permutation))):
        permutation = permutation[1:] + permutation[:1]
    return features[:, permutation], permutation


def _pair_features_for_anchor_control(
    *,
    eval_rows: list[arc_gate.ArcRow],
    anchor_texts: list[str],
    feature_dim: int,
    feature_mode: str,
    feature_model: str,
    feature_device: str,
    feature_dtype: str,
    feature_max_length: int,
    local_files_only: bool,
    anchor_control: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    if anchor_control not in ANCHOR_CONTROLS:
        raise ValueError(f"unknown anchor control {anchor_control!r}")
    pair_texts = arc_gate._choice_pair_texts(eval_rows)
    if feature_mode != "anchor_relative_hashed":
        if anchor_control != "none":
            raise ValueError("anchor controls require feature_mode='anchor_relative_hashed'")
        features = arc_gate._features(
            pair_texts,
            dim=feature_dim,
            feature_mode=feature_mode,
            feature_model=feature_model,
            feature_device=feature_device,
            feature_dtype=feature_dtype,
            feature_max_length=feature_max_length,
            local_files_only=local_files_only,
            anchor_texts=anchor_texts,
        )
        return features, features, None

    source_anchors = arc_gate._select_anchor_texts(anchor_texts, dim=feature_dim)
    source_features = arc_gate._anchor_relative_hashed_features_from_anchors(
        pair_texts,
        anchors=source_anchors,
        dim=feature_dim,
    )
    receiver_anchors = source_anchors
    permutation: list[int] | None = None
    if anchor_control == "none":
        receiver_features = source_features
        interpretation = "source and receiver share the same deterministic train-anchor coordinate chart"
    elif anchor_control == "anchor_id_shuffle":
        receiver_features, permutation = _feature_columns_permuted(
            source_features,
            seed=_stable_seed("arc-anchor-id-shuffle"),
        )
        interpretation = "receiver uses the same anchor values with a shuffled coordinate identity map"
    elif anchor_control == "anchor_value_shuffle":
        shift = _stable_seed("arc-anchor-value-shuffle") % max(1, len(source_anchors))
        if len(source_anchors) > 1 and shift == 0:
            shift = 1
        receiver_anchors = source_anchors[shift:] + source_anchors[:shift]
        receiver_features = arc_gate._anchor_relative_hashed_features_from_anchors(
            pair_texts,
            anchors=receiver_anchors,
            dim=feature_dim,
        )
        interpretation = "receiver preserves coordinate slots but deranges the public anchor values"
    else:
        random_anchors = _random_anchor_texts(
            count=len(source_anchors),
            seed=_stable_seed("arc-random-anchors-same-count"),
        )
        source_anchors = random_anchors
        receiver_anchors = random_anchors
        source_features = arc_gate._anchor_relative_hashed_features_from_anchors(
            pair_texts,
            anchors=source_anchors,
            dim=feature_dim,
        )
        receiver_features = source_features
        interpretation = "source and receiver share synthetic random anchors with the same count as train anchors"

    metadata: dict[str, Any] = {
        "anchor_control": anchor_control,
        "source_anchor_count": len(source_anchors),
        "receiver_anchor_count": len(receiver_anchors),
        "source_anchor_sha256": _anchor_digest(source_anchors),
        "receiver_anchor_sha256": _anchor_digest(receiver_anchors),
        "interpretation": interpretation,
    }
    if permutation is not None:
        metadata["receiver_column_permutation_sha256"] = hashlib.sha256(
            ",".join(str(index) for index in permutation).encode("utf-8")
        ).hexdigest()
    return source_features, receiver_features, metadata


def _condition_accuracy(metrics: dict[str, dict[str, Any]], condition: str) -> float:
    return float(metrics[condition]["accuracy"])


def _summarize_seed(
    *,
    seed: int,
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    has_overlap: bool,
) -> dict[str, Any]:
    metrics = arc_gate._condition_metrics(rows)
    matched = _condition_accuracy(metrics, arc_gate.MATCHED_CONDITION)
    target = _condition_accuracy(metrics, "target_only")
    same_byte_text = _condition_accuracy(metrics, "same_byte_structured_text")
    best_control_name = max(arc_gate.STRICT_DESTRUCTIVE_CONTROLS, key=lambda condition: metrics[condition]["accuracy"])
    best_control = _condition_accuracy(metrics, best_control_name)
    target_ci = arc_gate._paired_bootstrap(
        rows,
        condition=arc_gate.MATCHED_CONDITION,
        baseline="target_only",
        seed=seed + 1001,
        samples=bootstrap_samples,
    )
    control_ci = arc_gate._paired_bootstrap(
        rows,
        condition=arc_gate.MATCHED_CONDITION,
        baseline=best_control_name,
        seed=seed + 1002,
        samples=bootstrap_samples,
    )
    pass_gate = (
        not has_overlap
        and matched >= target + min_lift_over_target
        and matched >= best_control + min_gap_over_control
        and matched >= same_byte_text + min_gap_over_text
        and target_ci["ci95_low"] > 0.0
        and _condition_accuracy(metrics, "candidate_derangement") <= target + 0.05
    )
    return {
        "seed": seed,
        "pass_gate": bool(pass_gate),
        "matched_accuracy": matched,
        "target_accuracy": target,
        "same_byte_structured_text_accuracy": same_byte_text,
        "best_destructive_control": best_control_name,
        "best_destructive_control_accuracy": best_control,
        "candidate_derangement_accuracy": _condition_accuracy(metrics, "candidate_derangement"),
        "shuffled_source_packet_accuracy": _condition_accuracy(metrics, "shuffled_source_packet"),
        "random_same_byte_packet_accuracy": _condition_accuracy(metrics, "random_same_byte_packet"),
        "target_derived_sidecar_accuracy": _condition_accuracy(metrics, "target_derived_sidecar"),
        "matched_minus_target": matched - target,
        "matched_minus_best_destructive": matched - best_control,
        "matched_minus_same_byte_text": matched - same_byte_text,
        "paired_ci95_vs_target": target_ci,
        "paired_ci95_vs_best_destructive": control_ci,
        "condition_metrics": metrics,
    }


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in per_seed if row["pass_gate"]]
    return {
        "seed_count": len(per_seed),
        "pass_count": len(passed),
        "all_seeds_pass": len(passed) == len(per_seed),
        "matched_accuracy_mean": float(statistics.fmean(row["matched_accuracy"] for row in per_seed)),
        "matched_accuracy_min": float(min(row["matched_accuracy"] for row in per_seed)),
        "matched_accuracy_max": float(max(row["matched_accuracy"] for row in per_seed)),
        "matched_minus_target_mean": float(statistics.fmean(row["matched_minus_target"] for row in per_seed)),
        "matched_minus_target_min": float(min(row["matched_minus_target"] for row in per_seed)),
        "matched_minus_best_destructive_min": float(min(row["matched_minus_best_destructive"] for row in per_seed)),
        "matched_minus_same_byte_text_min": float(min(row["matched_minus_same_byte_text"] for row in per_seed)),
        "paired_ci95_low_vs_target_min": float(
            min(row["paired_ci95_vs_target"]["ci95_low"] for row in per_seed)
        ),
        "candidate_derangement_accuracy_max": float(max(row["candidate_derangement_accuracy"] for row in per_seed)),
        "same_byte_structured_text_accuracy": float(per_seed[0]["same_byte_structured_text_accuracy"]),
        "target_accuracy": float(per_seed[0]["target_accuracy"]),
    }


def _write_csv(path: pathlib.Path, per_seed: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "pass_gate",
        "matched_accuracy",
        "target_accuracy",
        "same_byte_structured_text_accuracy",
        "best_destructive_control",
        "best_destructive_control_accuracy",
        "candidate_derangement_accuracy",
        "matched_minus_target",
        "matched_minus_best_destructive",
        "matched_minus_same_byte_text",
        "paired_ci95_low_vs_target",
        "paired_ci95_high_vs_target",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_seed:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "pass_gate": row["pass_gate"],
                    "matched_accuracy": row["matched_accuracy"],
                    "target_accuracy": row["target_accuracy"],
                    "same_byte_structured_text_accuracy": row["same_byte_structured_text_accuracy"],
                    "best_destructive_control": row["best_destructive_control"],
                    "best_destructive_control_accuracy": row["best_destructive_control_accuracy"],
                    "candidate_derangement_accuracy": row["candidate_derangement_accuracy"],
                    "matched_minus_target": row["matched_minus_target"],
                    "matched_minus_best_destructive": row["matched_minus_best_destructive"],
                    "matched_minus_same_byte_text": row["matched_minus_same_byte_text"],
                    "paired_ci95_low_vs_target": row["paired_ci95_vs_target"]["ci95_low"],
                    "paired_ci95_high_vs_target": row["paired_ci95_vs_target"]["ci95_high"],
                }
            )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private ARC-Challenge Seed-Stability Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- split: `{payload['split_name']}`",
        f"- eval rows: `{payload['eval_rows']}`",
        f"- packet budget: `{payload['budget_bytes']}B`",
        f"- seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        "",
        "| Seed | Pass | Matched | Target | Best control | Same-byte text | Derange | CI95 low vs target |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['pass_gate']} | {row['matched_accuracy']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['best_destructive_control_accuracy']:.3f} "
            f"({row['best_destructive_control']}) | {row['same_byte_structured_text_accuracy']:.3f} | "
            f"{row['candidate_derangement_accuracy']:.3f} | "
            f"{row['paired_ci95_vs_target']['ci95_low']:.3f} |"
        )
    aggregate = payload["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- pass count: `{aggregate['pass_count']} / {aggregate['seed_count']}`",
            f"- matched accuracy mean/min/max: `{aggregate['matched_accuracy_mean']:.3f}` / "
            f"`{aggregate['matched_accuracy_min']:.3f}` / `{aggregate['matched_accuracy_max']:.3f}`",
            f"- minimum matched-target lift: `{aggregate['matched_minus_target_min']:.3f}`",
            f"- minimum matched-best-control lift: `{aggregate['matched_minus_best_destructive_min']:.3f}`",
            f"- minimum matched-same-byte-text lift: `{aggregate['matched_minus_same_byte_text_min']:.3f}`",
            f"- minimum CI95 lower bound vs target: `{aggregate['paired_ci95_low_vs_target_min']:.3f}`",
            "",
            "This gate varies only the packet projection/random-control seed while reusing the "
            "answer-key-forbidden source-choice cache from the anchor ARC run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_seed_stability(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    anchor_predictions: pathlib.Path,
    split_name: str,
    seeds: list[int],
    budget_bytes: int,
    feature_dim: int,
    code_dim: int,
    feature_mode: str,
    feature_model: str,
    feature_device: str,
    feature_dtype: str,
    feature_max_length: int,
    local_files_only: bool,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    anchor_control: str = "none",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    anchor_texts = arc_gate._choice_pair_texts(train_rows)
    overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in eval_rows})

    anchor_choices = _read_anchor_source_choices(anchor_predictions)
    source_predictions, cache_rows = _source_predictions_from_anchor(eval_rows, anchor_choices)
    _write_jsonl(output_dir / "source_prediction_cache.jsonl", cache_rows)

    source_pair_features, receiver_pair_features, anchor_control_metadata = _pair_features_for_anchor_control(
        eval_rows=eval_rows,
        anchor_texts=anchor_texts,
        feature_dim=feature_dim,
        feature_mode=feature_mode,
        feature_model=feature_model,
        feature_device=feature_device,
        feature_dtype=feature_dtype,
        feature_max_length=feature_max_length,
        local_files_only=local_files_only,
        anchor_control=anchor_control,
    )
    source_residuals = arc_gate._candidate_residuals(eval_rows, source_pair_features)
    receiver_residuals = arc_gate._candidate_residuals(eval_rows, receiver_pair_features)
    priors = arc_gate._index_prior(train_rows)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        projection = arc_gate._projection_matrix(feature_dim, code_dim, seed=seed + 171)
        prediction_rows = arc_gate._rows_for_predictions(
            eval_rows=eval_rows,
            residuals=source_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=source_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=priors,
            seed=seed + 911,
        )
        per_seed.append(
            _summarize_seed(
                seed=seed,
                rows=prediction_rows,
                bootstrap_samples=bootstrap_samples,
                min_lift_over_target=min_lift_over_target,
                min_gap_over_control=min_gap_over_control,
                min_gap_over_text=min_gap_over_text,
                has_overlap=bool(overlap),
            )
        )

    aggregate = _aggregate(per_seed)
    payload = {
        "gate": "source_private_arc_challenge_seed_stability",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "split_name": split_name,
        "train_path": _display_path(train_path),
        "eval_path": _display_path(eval_path),
        "anchor_predictions": _display_path(anchor_predictions),
        "train_sha256": _sha256_file(train_path),
        "eval_sha256": _sha256_file(eval_path),
        "anchor_predictions_sha256": _sha256_file(anchor_predictions),
        "source_prediction_cache_sha256": _sha256_file(output_dir / "source_prediction_cache.jsonl"),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_eval_content_overlap_count": len(overlap),
        "train_eval_content_overlap_sha256": hashlib.sha256("\n".join(overlap).encode("utf-8")).hexdigest(),
        "budget_bytes": budget_bytes,
        "feature_dim": feature_dim,
        "code_dim": code_dim,
        "feature_mode": feature_mode,
        "feature_model": feature_model if feature_mode.startswith("hf_") else None,
        "feature_device": arc_gate.syn._resolve_torch_device(feature_device) if feature_mode.startswith("hf_") else None,
        "feature_dtype": feature_dtype if feature_mode.startswith("hf_") else None,
        "feature_max_length": feature_max_length if feature_mode.startswith("hf_") else None,
        "anchor_control": anchor_control,
        "anchor_control_metadata": anchor_control_metadata,
        "anchor_relative_basis": (
            {
                "anchor_source": "train split question/candidate texts",
                "anchor_count": min(feature_dim, len(set(anchor_texts))),
                "base_feature_mode": "hashed",
                "basis_claim": "public anchor-relative coordinate chart plus downstream random projection",
                "anchor_control": anchor_control,
            }
            if feature_mode == "anchor_relative_hashed"
            else None
        ),
        "seeds": seeds,
        "bootstrap_samples": bootstrap_samples,
        "source_cache_contract": {
            "source_cache_fields": [
                "row_id",
                "content_id",
                "source_selected_index",
                "source_selected_choice_sha256",
                "source_visible_fields",
                "forbidden_source_fields",
            ],
            "source_visible_fields": ["question", "choices"],
            "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "source_cache_origin": "anchor matched_source_private_packet metadata from an answer-key-forbidden run",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "pass_gate": bool(
            not overlap
            and aggregate["all_seeds_pass"]
            and aggregate["paired_ci95_low_vs_target_min"] > 0.0
        ),
        "public_benchmark_seed_stable": bool(
            not overlap
            and aggregate["all_seeds_pass"]
            and aggregate["paired_ci95_low_vs_target_min"] > 0.0
        ),
        "pass_rule": (
            "Pass requires no train/eval content overlap and every projection seed to pass the fixed-packet "
            "ARC gate: matched packet beats target-only, the best strict destructive control, and same-byte "
            "structured text; paired CI95 lower bound versus target is positive; derangement stays near target."
        ),
    }
    (output_dir / "arc_challenge_seed_stability.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "per_seed_metrics.csv", per_seed)
    _write_markdown(output_dir / "arc_challenge_seed_stability.md", payload)
    manifest = {
        "artifacts": [
            "arc_challenge_seed_stability.json",
            "arc_challenge_seed_stability.md",
            "per_seed_metrics.csv",
            "source_prediction_cache.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "arc_challenge_seed_stability.json",
                "arc_challenge_seed_stability.md",
                "per_seed_metrics.csv",
                "source_prediction_cache.jsonl",
            )
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private ARC-Challenge Seed-Stability Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- public benchmark seed stable: `{payload['public_benchmark_seed_stable']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--anchor-predictions", type=pathlib.Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--seeds", type=_parse_seeds, default="47,53,59")
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument(
        "--feature-mode",
        choices=("hashed", "anchor_relative_hashed", "hf_last_mean", "hf_mid_last_mean"),
        default="hashed",
    )
    parser.add_argument("--anchor-control", choices=ANCHOR_CONTROLS, default="none")
    parser.add_argument("--feature-model", default="BAAI/bge-small-en")
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--feature-dtype", default="float32")
    parser.add_argument("--feature-max-length", type=int, default=128)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lift-over-target", type=float, default=0.03)
    parser.add_argument("--min-gap-over-control", type=float, default=0.03)
    parser.add_argument("--min-gap-over-text", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_seed_stability(
        output_dir=args.output_dir,
        train_path=_resolve(args.train_path),
        eval_path=_resolve(args.eval_path),
        anchor_predictions=_resolve(args.anchor_predictions),
        split_name=args.split_name,
        seeds=args.seeds,
        budget_bytes=args.budget_bytes,
        feature_dim=args.feature_dim,
        code_dim=args.code_dim,
        feature_mode=args.feature_mode,
        feature_model=args.feature_model,
        feature_device=args.feature_device,
        feature_dtype=args.feature_dtype,
        feature_max_length=args.feature_max_length,
        local_files_only=not args.allow_downloads,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
        anchor_control=args.anchor_control,
    )


if __name__ == "__main__":
    main()
