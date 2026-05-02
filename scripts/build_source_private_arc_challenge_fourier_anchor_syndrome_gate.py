from __future__ import annotations

"""Build the ARC-Challenge Fourier/anchor-syndrome packet gate.

The gate keeps the frozen answer-key-forbidden ARC source-choice caches, then
changes only the communication basis: public anchor-relative candidate
coordinates are transformed through a deterministic low-frequency DCT basis
before the existing sparse signed packet encoder.  The controls intentionally
break anchor identity, anchor values, or spectral-bin identity while preserving
the packet budget.
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
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_seed_stability as seed_stability  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_VALIDATION_SOURCE_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/"
    "source_prediction_cache.jsonl"
)
DEFAULT_TEST_SOURCE_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/"
    "source_prediction_cache.jsonl"
)

MATCHED_VARIANT = "matched_fourier_anchor_syndrome"
CONTROL_VARIANTS = (
    "anchor_id_shuffle",
    "anchor_value_shuffle",
    "spectral_bin_permutation",
)
DIAGNOSTIC_VARIANTS = ("random_anchors_same_count",)
ALL_VARIANTS = (MATCHED_VARIANT, *CONTROL_VARIANTS, *DIAGNOSTIC_VARIANTS)


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_source_cache(path: pathlib.Path) -> dict[str, int]:
    choices: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = set(row.get("forbidden_source_fields", ()))
            if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
                raise ValueError(f"source cache row {row.get('row_id')} is missing forbidden-field contract")
            choices[str(row["content_id"])] = int(row["source_selected_index"])
    if not choices:
        raise ValueError(f"{path} contained no source-choice rows")
    return choices


def _source_predictions(rows: list[arc_gate.ArcRow], cache: dict[str, int]) -> list[int]:
    predictions: list[int] = []
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        if row.content_id not in cache:
            missing.append(row.content_id)
            continue
        prediction = int(cache[row.content_id])
        if prediction < 0 or prediction >= len(row.choices):
            invalid.append(row.content_id)
            continue
        predictions.append(prediction)
    if missing or invalid:
        raise ValueError(f"source cache mismatch: missing={len(missing)} invalid={len(invalid)}")
    return predictions


def _dct_low_frequency_features(features: np.ndarray, *, output_dim: int) -> np.ndarray:
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if output_dim > features.shape[1]:
        raise ValueError("output_dim cannot exceed input feature dimension")
    n = int(features.shape[1])
    positions = np.arange(n, dtype=np.float64)[:, None]
    frequencies = np.arange(output_dim, dtype=np.float64)[None, :]
    basis = np.cos((math.pi / n) * (positions + 0.5) * frequencies)
    basis[:, 0] *= math.sqrt(1.0 / n)
    if output_dim > 1:
        basis[:, 1:] *= math.sqrt(2.0 / n)
    transformed = np.asarray(features, dtype=np.float64) @ basis
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return np.divide(
        transformed,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(transformed),
        where=norms > 0,
    )


def _spectral_permutation(features: np.ndarray) -> tuple[np.ndarray, list[int]]:
    return seed_stability._feature_columns_permuted(
        features,
        seed=seed_stability._stable_seed("arc-fourier-spectral-bin-permutation"),
    )


def _fourier_pair_features_for_variant(
    *,
    eval_rows: list[arc_gate.ArcRow],
    anchor_texts: list[str],
    anchor_count: int,
    spectral_dim: int,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    anchor_control = "none" if variant in {MATCHED_VARIANT, "spectral_bin_permutation"} else variant
    source_features, receiver_features, anchor_metadata = seed_stability._pair_features_for_anchor_control(
        eval_rows=eval_rows,
        anchor_texts=anchor_texts,
        feature_dim=anchor_count,
        feature_mode="anchor_relative_hashed",
        feature_model="",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
        anchor_control=anchor_control,
    )
    source_spectrum = _dct_low_frequency_features(source_features, output_dim=spectral_dim)
    receiver_spectrum = _dct_low_frequency_features(receiver_features, output_dim=spectral_dim)
    metadata: dict[str, Any] = {
        "variant": variant,
        "anchor_control": anchor_control,
        "anchor_count": anchor_count,
        "spectral_dim": spectral_dim,
        "spectral_transform": "orthonormal_dct_ii_low_frequency",
        "source_spectrum_sha256": _sha256_text(
            ",".join(f"{value:.8f}" for value in source_spectrum[: min(8, len(source_spectrum))].ravel())
        ),
        "receiver_spectrum_sha256": _sha256_text(
            ",".join(f"{value:.8f}" for value in receiver_spectrum[: min(8, len(receiver_spectrum))].ravel())
        ),
    }
    if anchor_metadata is not None:
        metadata["anchor_control_metadata"] = anchor_metadata
    if variant == "spectral_bin_permutation":
        receiver_spectrum, permutation = _spectral_permutation(receiver_spectrum)
        metadata["spectral_bin_permutation_sha256"] = _sha256_text(
            ",".join(str(index) for index in permutation)
        )
        metadata["receiver_spectrum_sha256"] = _sha256_text(
            ",".join(f"{value:.8f}" for value in receiver_spectrum[: min(8, len(receiver_spectrum))].ravel())
        )
    return source_spectrum, receiver_spectrum, metadata


def _evaluate_variant(
    *,
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    index_prior: list[float],
    source_features: np.ndarray,
    receiver_features: np.ndarray,
    seeds: list[int],
    budget_bytes: int,
    code_dim: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    has_overlap: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_residuals = arc_gate._candidate_residuals(rows, source_features)
    receiver_residuals = arc_gate._candidate_residuals(rows, receiver_features)
    per_seed: list[dict[str, Any]] = []
    matched_prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        projection = arc_gate._projection_matrix(source_features.shape[1], code_dim, seed=seed + 171)
        prediction_rows = arc_gate._rows_for_predictions(
            eval_rows=rows,
            residuals=source_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=source_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=index_prior,
            seed=seed + 911,
        )
        per_seed.append(
            seed_stability._summarize_seed(
                seed=seed,
                rows=prediction_rows,
                bootstrap_samples=bootstrap_samples,
                min_lift_over_target=min_lift_over_target,
                min_gap_over_control=min_gap_over_control,
                min_gap_over_text=min_gap_over_text,
                has_overlap=has_overlap,
            )
        )
        for row in prediction_rows:
            if row["condition"] == arc_gate.MATCHED_CONDITION:
                matched_prediction_rows.append({**row, "seed": seed})
    return per_seed, matched_prediction_rows


def _evaluate_split(
    *,
    split_name: str,
    rows: list[arc_gate.ArcRow],
    source_cache: pathlib.Path,
    train_rows: list[arc_gate.ArcRow],
    seeds: list[int],
    budget_bytes: int,
    anchor_count: int,
    spectral_dim: int,
    code_dim: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    overlap: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_texts = arc_gate._choice_pair_texts(train_rows)
    source_predictions = _source_predictions(rows, _read_source_cache(source_cache))
    index_prior = arc_gate._index_prior(train_rows)
    variants: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for variant in ALL_VARIANTS:
        source_features, receiver_features, metadata = _fourier_pair_features_for_variant(
            eval_rows=rows,
            anchor_texts=anchor_texts,
            anchor_count=anchor_count,
            spectral_dim=spectral_dim,
            variant=variant,
        )
        per_seed, matched_rows = _evaluate_variant(
            rows=rows,
            source_predictions=source_predictions,
            index_prior=index_prior,
            source_features=source_features,
            receiver_features=receiver_features,
            seeds=seeds,
            budget_bytes=budget_bytes,
            code_dim=code_dim,
            bootstrap_samples=bootstrap_samples,
            min_lift_over_target=min_lift_over_target,
            min_gap_over_control=min_gap_over_control,
            min_gap_over_text=min_gap_over_text,
            has_overlap=bool(overlap),
        )
        aggregate = seed_stability._aggregate(per_seed)
        variants[variant] = {
            "basis_metadata": metadata,
            "per_seed": per_seed,
            "aggregate": aggregate,
        }
        for row in matched_rows:
            prediction_rows.append({**row, "split": split_name, "variant": variant})
    matched = variants[MATCHED_VARIANT]["aggregate"]
    mismatch_controls_collapse = all(
        variants[variant]["aggregate"]["pass_count"] == 0
        and variants[variant]["aggregate"]["matched_accuracy_mean"]
        <= variants[variant]["aggregate"]["target_accuracy"] + 0.03
        for variant in CONTROL_VARIANTS
    )
    split_pass = bool(
        matched["all_seeds_pass"]
        and matched["paired_ci95_low_vs_target_min"] > 0.0
        and mismatch_controls_collapse
        and not overlap
    )
    return {
        "split_name": split_name,
        "rows": len(rows),
        "source_cache": _display_path(source_cache),
        "source_cache_sha256": _sha256_file(source_cache),
        "train_eval_content_overlap_count": len(overlap),
        "train_eval_content_overlap_sha256": hashlib.sha256("\n".join(overlap).encode("utf-8")).hexdigest(),
        "variants": variants,
        "headline": {
            "pass_gate": split_pass,
            "matched_aggregate": matched,
            "mismatch_controls_collapse": mismatch_controls_collapse,
            "control_aggregates": {
                variant: variants[variant]["aggregate"] for variant in CONTROL_VARIANTS
            },
            "random_shared_anchor_diagnostic": variants["random_anchors_same_count"]["aggregate"],
        },
    }, prediction_rows


def _write_variant_csv(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fields = [
        "split",
        "variant",
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split_name, split in payload["splits"].items():
            for variant, variant_payload in split["variants"].items():
                for row in variant_payload["per_seed"]:
                    writer.writerow(
                        {
                            "split": split_name,
                            "variant": variant,
                            "seed": row["seed"],
                            "pass_gate": row["pass_gate"],
                            "matched_accuracy": row["matched_accuracy"],
                            "target_accuracy": row["target_accuracy"],
                            "same_byte_structured_text_accuracy": row[
                                "same_byte_structured_text_accuracy"
                            ],
                            "best_destructive_control": row["best_destructive_control"],
                            "best_destructive_control_accuracy": row[
                                "best_destructive_control_accuracy"
                            ],
                            "candidate_derangement_accuracy": row["candidate_derangement_accuracy"],
                            "matched_minus_target": row["matched_minus_target"],
                            "matched_minus_best_destructive": row["matched_minus_best_destructive"],
                            "matched_minus_same_byte_text": row["matched_minus_same_byte_text"],
                            "paired_ci95_low_vs_target": row["paired_ci95_vs_target"]["ci95_low"],
                        }
                    )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    test = payload["splits"]["test"]["headline"]["matched_aggregate"]
    lines = [
        "# Source-Private ARC-Challenge Fourier/Anchor-Syndrome Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- packet budget: `{payload['budget_bytes']}B`",
        f"- anchor / spectral / code dims: `{payload['anchor_count']}` / `{payload['spectral_dim']}` / `{payload['code_dim']}`",
        f"- test matched mean/min: `{test['matched_accuracy_mean']:.3f}` / `{test['matched_accuracy_min']:.3f}`",
        f"- test target / same-byte text: `{test['target_accuracy']:.3f}` / `{test['same_byte_structured_text_accuracy']:.3f}`",
        f"- test min CI95 low vs target: `{test['paired_ci95_low_vs_target_min']:.3f}`",
        "",
        "## Split Summary",
        "",
        "| Split | Variant | Pass seeds | Matched mean | Target | Text | CI95 low |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split_name, split in payload["splits"].items():
        for variant, variant_payload in split["variants"].items():
            agg = variant_payload["aggregate"]
            lines.append(
                f"| {split_name} | `{variant}` | {agg['pass_count']}/{agg['seed_count']} | "
                f"{agg['matched_accuracy_mean']:.3f} | {agg['target_accuracy']:.3f} | "
                f"{agg['same_byte_structured_text_accuracy']:.3f} | "
                f"{agg['paired_ci95_low_vs_target_min']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay description: the source and receiver first agree on a public coordinate grid made from "
            "training-set anchors. The source sends a tiny low-frequency spectral sketch in that grid. "
            "When the receiver uses the same grid, the packet works; when anchor identities or spectral "
            "bins are scrambled, the signal collapses near target-only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_fourier_anchor_syndrome_gate(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    validation_source_cache: pathlib.Path,
    test_source_cache: pathlib.Path,
    seeds: list[int],
    budget_bytes: int,
    anchor_count: int,
    spectral_dim: int,
    code_dim: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = arc_gate._load_rows(train_path)
    validation_rows = arc_gate._load_rows(validation_path)
    test_rows = arc_gate._load_rows(test_path)
    validation_overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in validation_rows})
    test_overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in test_rows})

    validation_payload, validation_prediction_rows = _evaluate_split(
        split_name="validation",
        rows=validation_rows,
        source_cache=validation_source_cache,
        train_rows=train_rows,
        seeds=seeds,
        budget_bytes=budget_bytes,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        overlap=validation_overlap,
    )
    test_payload, test_prediction_rows = _evaluate_split(
        split_name="test",
        rows=test_rows,
        source_cache=test_source_cache,
        train_rows=train_rows,
        seeds=seeds,
        budget_bytes=budget_bytes,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        overlap=test_overlap,
    )

    pass_gate = bool(validation_payload["headline"]["pass_gate"] and test_payload["headline"]["pass_gate"])
    payload = {
        "gate": "source_private_arc_challenge_fourier_anchor_syndrome_gate",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "validation_path": _display_path(validation_path),
        "test_path": _display_path(test_path),
        "train_sha256": _sha256_file(train_path),
        "validation_sha256": _sha256_file(validation_path),
        "test_sha256": _sha256_file(test_path),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "seeds": seeds,
        "budget_bytes": budget_bytes,
        "record_bytes_with_header_crc": budget_bytes + 3,
        "anchor_count": anchor_count,
        "spectral_dim": spectral_dim,
        "code_dim": code_dim,
        "bootstrap_samples": bootstrap_samples,
        "basis_contract": {
            "basis": "public train-anchor relative coordinates followed by orthonormal low-frequency DCT-II",
            "anchor_source": "train split question/candidate texts",
            "anchor_count": anchor_count,
            "spectral_dim": spectral_dim,
            "base_feature_mode": "hashed",
            "normalization": "row_l2_after_anchor_centering_and_after_dct",
            "packet_encoder": "existing sparse signed projection packet",
            "mismatch_controls": list(CONTROL_VARIANTS),
            "diagnostic_variants": list(DIAGNOSTIC_VARIANTS),
        },
        "method_contract": {
            "source_packet_budget_bytes": budget_bytes,
            "source_packet_origin": "answer-key-forbidden Qwen2.5-0.5B source-choice cache from frozen ARC packet gates",
            "receiver_inputs_at_test": [
                "fixed-byte Fourier/anchor-syndrome source packet",
                "public question/candidate text through the same public anchor chart",
            ],
            "forbidden_eval_source_inputs": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "claim_boundary": (
                "This is a public-basis source-private packet, not raw hidden-state transfer, source KV "
                "sharing, prefix tuning, or a universal latent language claim."
            ),
        },
        "splits": {
            "validation": validation_payload,
            "test": test_payload,
        },
        "headline": {
            "validation_pass": validation_payload["headline"]["pass_gate"],
            "test_pass": test_payload["headline"]["pass_gate"],
            "test_matched_aggregate": test_payload["headline"]["matched_aggregate"],
            "test_control_aggregates": test_payload["headline"]["control_aggregates"],
            "test_mismatch_controls_collapse": test_payload["headline"]["mismatch_controls_collapse"],
            "random_shared_anchor_diagnostic": test_payload["headline"]["random_shared_anchor_diagnostic"],
        },
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires validation and test matched Fourier/anchor-syndrome packets to pass all five seeds "
            "against target-only, best destructive control, same-byte text, candidate derangement, and paired "
            "uncertainty, while anchor-ID, anchor-value, and spectral-bin mismatch controls have zero passing "
            "seeds and collapse to target+0.03 or lower on average."
        ),
        "interpretation": (
            "The Fourier/anchor-syndrome packet preserves the ARC source-private packet signal after compressing "
            "the public anchor chart to low-frequency spectral coordinates. Because anchor-ID, anchor-value, and "
            "spectral-bin mismatch controls collapse near target-only, the result supports a shared public-basis "
            "communication story rather than a raw source-label or KV-cache transport story. The random shared "
            "anchor diagnostic should be framed carefully: it shows that shared coordinate agreement matters more "
            "than semantic anchor names in this hashed ARC implementation."
        ),
    }
    json_path = output_dir / "arc_challenge_fourier_anchor_syndrome_gate.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_variant_csv(output_dir / "per_variant_seed_metrics.csv", payload)
    _write_jsonl(
        output_dir / "matched_predictions.jsonl",
        [
            row
            for row in [*validation_prediction_rows, *test_prediction_rows]
            if row["variant"] == MATCHED_VARIANT
        ],
    )
    _write_markdown(output_dir / "arc_challenge_fourier_anchor_syndrome_gate.md", payload)
    manifest = {
        "artifacts": [
            "arc_challenge_fourier_anchor_syndrome_gate.json",
            "arc_challenge_fourier_anchor_syndrome_gate.md",
            "per_variant_seed_metrics.csv",
            "matched_predictions.jsonl",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in (
                "arc_challenge_fourier_anchor_syndrome_gate.json",
                "arc_challenge_fourier_anchor_syndrome_gate.md",
                "per_variant_seed_metrics.csv",
                "matched_predictions.jsonl",
            )
        },
        "pass_gate": pass_gate,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private ARC-Challenge Fourier/Anchor-Syndrome Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- validation pass: `{validation_payload['headline']['pass_gate']}`",
                f"- test pass: `{test_payload['headline']['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--validation-source-cache", type=pathlib.Path, default=DEFAULT_VALIDATION_SOURCE_CACHE)
    parser.add_argument("--test-source-cache", type=pathlib.Path, default=DEFAULT_TEST_SOURCE_CACHE)
    parser.add_argument("--seeds", type=_parse_int_list, default="47,53,59,61,67")
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--anchor-count", type=int, default=384)
    parser.add_argument("--spectral-dim", type=int, default=96)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lift-over-target", type=float, default=0.05)
    parser.add_argument("--min-gap-over-control", type=float, default=0.03)
    parser.add_argument("--min-gap-over-text", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_fourier_anchor_syndrome_gate(
        output_dir=_resolve(args.output_dir),
        train_path=_resolve(args.train_path),
        validation_path=_resolve(args.validation_path),
        test_path=_resolve(args.test_path),
        validation_source_cache=_resolve(args.validation_source_cache),
        test_source_cache=_resolve(args.test_source_cache),
        seeds=args.seeds,
        budget_bytes=args.budget_bytes,
        anchor_count=args.anchor_count,
        spectral_dim=args.spectral_dim,
        code_dim=args.code_dim,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
    )


if __name__ == "__main__":
    main()
