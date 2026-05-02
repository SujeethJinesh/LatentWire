from __future__ import annotations

"""Summarize HellaSwag hidden-innovation evidence across source families."""

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_source_family_stress_card_20260502")
DEFAULT_QWEN_GLOBAL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/"
    "hellaswag_hidden_innovation_global_stability.json"
)
DEFAULT_TINYLLAMA_SLICE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260502_tinyllama_train512_validation1024_2048/"
    "hellaswag_hidden_innovation_eval_slice_stress.json"
)

CSV_COLUMNS = (
    "row_id",
    "source_family",
    "source_model_path",
    "artifact_kind",
    "artifact_path",
    "artifact_sha256",
    "eval_rows",
    "eval_slice_start",
    "eval_slice_end_exclusive",
    "pass_gate",
    "primary_policy",
    "selected_accuracy",
    "best_label_copy_accuracy",
    "source_label_copy_accuracy",
    "score_only_accuracy",
    "zero_hidden_accuracy",
    "wrong_example_hidden_accuracy",
    "candidate_roll_hidden_accuracy",
    "selected_minus_best_label_copy",
    "selected_minus_score_only",
    "paired_ci95_low_vs_best_label_copy",
    "paired_ci95_low_vs_score_only",
    "jackknife_pass_count",
    "jackknife_row_count",
    "slice_pass_count",
    "slice_count",
    "raw_payload_bytes",
    "framed_record_bytes",
    "wall_seconds",
    "interpretation",
)


@dataclasses.dataclass(frozen=True)
class SourceFamilyArtifact:
    row_id: str
    source_family: str
    artifact_kind: str
    path: pathlib.Path
    primary_policy: str


DEFAULT_ARTIFACTS = (
    SourceFamilyArtifact(
        row_id="qwen25_full_validation",
        source_family="Qwen2.5",
        artifact_kind="same_family_full_validation_global",
        path=DEFAULT_QWEN_GLOBAL,
        primary_policy="mean_zscore",
    ),
    SourceFamilyArtifact(
        row_id="tinyllama_validation1024_2048",
        source_family="TinyLlama",
        artifact_kind="non_qwen_source_family_heldout_slice",
        path=DEFAULT_TINYLLAMA_SLICE,
        primary_policy="mean_zscore",
    ),
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _policy_row(payload: dict[str, Any], policy: str) -> dict[str, Any]:
    for row in payload.get("policy_rows", []):
        if row.get("policy") == policy:
            return row
    raise ValueError(f"missing policy row {policy!r}")


def _source_model_path(payload: dict[str, Any]) -> str | None:
    if payload.get("source_models"):
        return ";".join(str(value) for value in payload["source_models"])
    score_model = payload.get("eval_cache_metadata", {}).get("score_model", {})
    return score_model.get("model_path")


def _row_from_global_artifact(spec: SourceFamilyArtifact, payload: dict[str, Any]) -> dict[str, Any]:
    headline = payload["headline"]
    row = _policy_row(payload, spec.primary_policy)
    return {
        "row_id": spec.row_id,
        "source_family": spec.source_family,
        "source_model_path": _source_model_path(payload),
        "artifact_kind": spec.artifact_kind,
        "artifact_path": _display_path(spec.path),
        "artifact_sha256": _sha256_file(spec.path),
        "eval_rows": headline["eval_rows"],
        "eval_slice_start": 0,
        "eval_slice_end_exclusive": headline["eval_rows"],
        "pass_gate": bool(payload["pass_gate"]),
        "primary_policy": spec.primary_policy,
        "selected_accuracy": row["selected_eval_accuracy"],
        "best_label_copy_accuracy": row["best_label_copy_eval_accuracy"],
        "source_label_copy_accuracy": row["source_label_copy_eval_accuracy"],
        "score_only_accuracy": row["score_only_bagged_control_accuracy"],
        "zero_hidden_accuracy": row["zero_hidden_control_accuracy"],
        "wrong_example_hidden_accuracy": row["wrong_example_hidden_control_accuracy"],
        "candidate_roll_hidden_accuracy": row["candidate_roll_hidden_control_accuracy"],
        "selected_minus_best_label_copy": row["selected_minus_best_label_copy"],
        "selected_minus_score_only": row["selected_minus_score_only_bagged_control"],
        "paired_ci95_low_vs_best_label_copy": row["paired_ci95_low_vs_best_label_copy"],
        "paired_ci95_low_vs_score_only": row["paired_ci95_low_vs_score_only_bagged"],
        "jackknife_pass_count": headline.get("mean_zscore_subbag_pass_count"),
        "jackknife_row_count": headline.get("train_sample_seed_count"),
        "slice_pass_count": headline.get("mean_zscore_slice_pass_count"),
        "slice_count": headline.get("eval_slice_count"),
        "raw_payload_bytes": headline["source_private_packet_raw_bytes"],
        "framed_record_bytes": headline["source_private_packet_framed_bytes"],
        "wall_seconds": payload.get("timing", {}).get("total_seconds"),
        "interpretation": "same-family full-validation positive anchor row",
    }


def _row_from_slice_artifact(spec: SourceFamilyArtifact, payload: dict[str, Any]) -> dict[str, Any]:
    headline = payload["headline"]
    return {
        "row_id": spec.row_id,
        "source_family": spec.source_family,
        "source_model_path": _source_model_path(payload),
        "artifact_kind": spec.artifact_kind,
        "artifact_path": _display_path(spec.path),
        "artifact_sha256": _sha256_file(spec.path),
        "eval_rows": headline["eval_rows"],
        "eval_slice_start": headline["eval_slice_start"],
        "eval_slice_end_exclusive": headline["eval_slice_end_exclusive"],
        "pass_gate": bool(payload["pass_gate"]),
        "primary_policy": spec.primary_policy,
        "selected_accuracy": headline["selected_eval_accuracy"],
        "best_label_copy_accuracy": headline["best_label_copy_eval_accuracy"],
        "source_label_copy_accuracy": headline["source_label_copy_eval_accuracy"],
        "score_only_accuracy": headline["score_only_bagged_control_accuracy"],
        "zero_hidden_accuracy": headline["zero_hidden_control_accuracy"],
        "wrong_example_hidden_accuracy": headline["wrong_example_hidden_control_accuracy"],
        "candidate_roll_hidden_accuracy": headline["candidate_roll_hidden_control_accuracy"],
        "selected_minus_best_label_copy": headline["selected_minus_best_label_copy"],
        "selected_minus_score_only": headline["selected_minus_score_only_bagged_control"],
        "paired_ci95_low_vs_best_label_copy": headline["paired_ci95_low_vs_best_label_copy"],
        "paired_ci95_low_vs_score_only": headline["paired_ci95_low_vs_score_only_bagged"],
        "jackknife_pass_count": headline["jackknife_pass_count"],
        "jackknife_row_count": headline["jackknife_row_count"],
        "slice_pass_count": 1 if payload["pass_gate"] else 0,
        "slice_count": 1,
        "raw_payload_bytes": headline["raw_payload_bytes"],
        "framed_record_bytes": headline["framed_record_bytes"],
        "wall_seconds": payload.get("timing", {}).get("total_seconds"),
        "interpretation": (
            "non-Qwen source-family heldout slice positive row; promotes a full TinyLlama validation run "
            "but does not close strict cross-family ICLR evidence by itself"
        ),
    }


def _artifact_row(spec: SourceFamilyArtifact) -> dict[str, Any]:
    payload = _read_json(spec.path)
    if "policy_rows" in payload:
        return _row_from_global_artifact(spec, payload)
    return _row_from_slice_artifact(spec, payload)


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Source-Family Stress Card",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source families represented: `{h['source_family_count']}`",
        f"- Qwen full-validation pass: `{h['qwen_full_validation_pass']}`",
        f"- TinyLlama heldout-slice pass: `{h['tinyllama_heldout_slice_pass']}`",
        f"- TinyLlama delta vs best label-copy: `{h['tinyllama_delta_vs_best_label_copy']:.6f}`",
        f"- TinyLlama CI95 low vs best label-copy: `{h['tinyllama_ci95_low_vs_best_label_copy']:.6f}`",
        f"- ICLR ready: `{h['iclr_ready']}`",
        "",
        "## Rows",
        "",
        "| Source family | Scope | Rows | Accuracy | Best label-copy | Delta | CI95 low | Jackknife | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['source_family']} | {row['artifact_kind']} | {row['eval_rows']} | "
            f"{row['selected_accuracy']:.6f} | {row['best_label_copy_accuracy']:.6f} | "
            f"{row['selected_minus_best_label_copy']:.6f} | "
            f"{row['paired_ci95_low_vs_best_label_copy']:.6f} | "
            f"{row['jackknife_pass_count']}/{row['jackknife_row_count']} | "
            f"{row['pass_gate']} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_card(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    qwen_global: pathlib.Path = DEFAULT_QWEN_GLOBAL,
    tinyllama_slice: pathlib.Path = DEFAULT_TINYLLAMA_SLICE,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    artifacts = (
        dataclasses.replace(DEFAULT_ARTIFACTS[0], path=qwen_global),
        dataclasses.replace(DEFAULT_ARTIFACTS[1], path=tinyllama_slice),
    )
    rows = [_artifact_row(spec) for spec in artifacts]
    by_id = {row["row_id"]: row for row in rows}
    tiny = by_id["tinyllama_validation1024_2048"]
    qwen = by_id["qwen25_full_validation"]
    pass_gate = bool(
        qwen["pass_gate"]
        and tiny["pass_gate"]
        and tiny["eval_rows"] >= 1024
        and tiny["selected_minus_best_label_copy"] >= 0.02
        and tiny["paired_ci95_low_vs_best_label_copy"] > 0.0
        and tiny["jackknife_pass_count"] == tiny["jackknife_row_count"]
    )
    headline = {
        "source_family_count": len({row["source_family"] for row in rows}),
        "qwen_full_validation_pass": bool(qwen["pass_gate"]),
        "tinyllama_heldout_slice_pass": bool(tiny["pass_gate"]),
        "tinyllama_accuracy": tiny["selected_accuracy"],
        "tinyllama_best_label_copy_accuracy": tiny["best_label_copy_accuracy"],
        "tinyllama_score_only_accuracy": tiny["score_only_accuracy"],
        "tinyllama_delta_vs_best_label_copy": tiny["selected_minus_best_label_copy"],
        "tinyllama_delta_vs_score_only": tiny["selected_minus_score_only"],
        "tinyllama_ci95_low_vs_best_label_copy": tiny["paired_ci95_low_vs_best_label_copy"],
        "tinyllama_jackknife_pass_count": tiny["jackknife_pass_count"],
        "tinyllama_jackknife_row_count": tiny["jackknife_row_count"],
        "raw_payload_bytes": tiny["raw_payload_bytes"],
        "framed_record_bytes": tiny["framed_record_bytes"],
        "iclr_ready": False,
        "remaining_iclr_gaps": [
            "run TinyLlama or another non-Qwen source over full frozen HellaSwag validation",
            "test a true source-family-to-receiver-family transfer surface when target-model receiver artifacts exist",
            "add native NVIDIA systems rows before throughput/HBM claims",
        ],
    }
    payload = {
        "gate": "source_private_hellaswag_source_family_stress_card",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if the Qwen full-validation anchor remains positive and a non-Qwen source-family "
            "heldout slice with at least 1024 rows passes the same hidden-innovation gate by >=0.02 "
            "over best label-copy with CI95 low > 0 and all train-sample jackknife subbags passing. "
            "This is a source-family stress card, not a complete cross-family ICLR claim."
        ),
        "headline": headline,
        "rows": rows,
        "interpretation": (
            "The TinyLlama heldout-slice pass weakens the concern that the HellaSwag hidden-innovation "
            "packet only works because of Qwen-specific hidden coordinates. It promotes a full non-Qwen "
            "validation run and a true receiver-family transfer gate, while preserving the no-overclaim "
            "boundary that ICLR still needs broader cross-family and native systems evidence."
        ),
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hellaswag_source_family_stress_card.json"
    csv_path = output_dir / "hellaswag_source_family_stress_card.csv"
    md_path = output_dir / "hellaswag_source_family_stress_card.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, csv_path, md_path)
        ],
        "inputs": [
            {"path": _display_path(spec.path), "sha256": _sha256_file(spec.path)}
            for spec in artifacts
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-global", type=pathlib.Path, default=DEFAULT_QWEN_GLOBAL)
    parser.add_argument("--tinyllama-slice", type=pathlib.Path, default=DEFAULT_TINYLLAMA_SLICE)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_card(
        output_dir=args.output_dir,
        qwen_global=args.qwen_global,
        tinyllama_slice=args.tinyllama_slice,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
