from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_learned_residual_basis_multi_slice_stress_20260501_qwen05_validation0_5120"
)
DEFAULT_SLICE_ARTIFACTS = (
    pathlib.Path(
        "results/source_private_hellaswag_learned_residual_basis_gate_20260501_qwen05_train512_validation0_1024/hellaswag_learned_residual_basis_scout.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_learned_residual_basis_gate_20260501_qwen05_train512_validation1024_2048/hellaswag_learned_residual_basis_scout.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_learned_residual_basis_gate_20260501_qwen05_train512_validation2048_3072/hellaswag_learned_residual_basis_scout.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_learned_residual_basis_gate_20260501_qwen05_train512_validation3072_4096/hellaswag_learned_residual_basis_scout.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_learned_residual_basis_gate_20260501_qwen05_train512_validation4096_5120/hellaswag_learned_residual_basis_scout.json"
    ),
)
STRICT_DELTA = 0.02


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _bounds_from_path(path: pathlib.Path, index: int, rows: int) -> tuple[int, int]:
    name = str(path)
    for start in (0, 1024, 2048, 3072, 4096):
        token = f"validation{start}_{start + 1024}"
        if token in name:
            return start, start + 1024
    return index * rows, (index + 1) * rows


def _slice_row(path: pathlib.Path, payload: dict[str, Any], index: int) -> dict[str, Any]:
    headline = payload["headline"]
    packet_contract = payload["packet_contract"]
    variant = payload["variant_rows"][0] if len(payload["variant_rows"]) == 1 else max(
        payload["variant_rows"],
        key=lambda row: (
            row["scout_pass_rule"],
            row["selected_minus_best_label_copy"],
            row["paired_ci95_low_vs_best_label_copy"],
            row["selected_minus_score_only_bagged_control"],
        ),
    )
    rows = int(headline["eval_rows"])
    start, end = _bounds_from_path(_resolve(path), index, rows)
    row = {
        "artifact_path": _display_path(_resolve(path)),
        "gate": payload["gate"],
        "variant": variant["variant"],
        "eval_slice_start": start,
        "eval_slice_end_exclusive": end,
        "eval_rows": rows,
        "scout_pass": bool(payload["scout_pass"]),
        "slice_pass_rule": bool(variant["scout_pass_rule"]),
        "selected_eval_accuracy": float(variant["selected_eval_accuracy"]),
        "source_label_copy_eval_accuracy": float(variant["source_label_copy_eval_accuracy"]),
        "trained_choice_bias_label_copy_eval_accuracy": float(
            variant["trained_choice_bias_label_copy_eval_accuracy"]
        ),
        "best_label_copy_eval_accuracy": float(variant["best_label_copy_eval_accuracy"]),
        "selected_minus_best_label_copy": float(variant["selected_minus_best_label_copy"]),
        "paired_ci95_low_vs_best_label_copy": float(variant["paired_ci95_low_vs_best_label_copy"]),
        "paired_ci95_high_vs_best_label_copy": float(variant["paired_ci95_high_vs_best_label_copy"]),
        "score_only_bagged_control_accuracy": float(variant["score_only_bagged_control_accuracy"]),
        "selected_minus_score_only_bagged_control": float(
            variant["selected_minus_score_only_bagged_control"]
        ),
        "paired_ci95_low_vs_score_only_bagged": float(variant["paired_ci95_low_vs_score_only_bagged"]),
        "zero_hidden_control_accuracy": float(variant["zero_hidden_control_accuracy"]),
        "selected_minus_zero_hidden_control": float(variant["selected_minus_zero_hidden_control"]),
        "wrong_example_hidden_control_accuracy": float(variant["wrong_example_hidden_control_accuracy"]),
        "candidate_roll_hidden_control_accuracy": float(variant["candidate_roll_hidden_control_accuracy"]),
        "basis_dim_roll_control_accuracy": float(variant["basis_dim_roll_control_accuracy"]),
        "basis_sign_flip_control_accuracy": float(variant["basis_sign_flip_control_accuracy"]),
        "random_basis_same_dim_control_accuracy": float(variant["random_basis_same_dim_control_accuracy"]),
        "raw_payload_bytes": int(packet_contract["raw_payload_bytes"]),
        "framed_record_bytes": int(packet_contract["framed_record_bytes"]),
        "source_text_exposed": bool(packet_contract["source_text_exposed"]),
        "source_kv_exposed": bool(packet_contract["source_kv_exposed"]),
        "raw_hidden_vector_transmitted": bool(packet_contract["raw_hidden_vector_transmitted"]),
        "raw_scores_transmitted": bool(packet_contract["raw_scores_transmitted"]),
        "basis_coefficients_transmitted": bool(packet_contract["basis_coefficients_transmitted"]),
    }
    row["strict_slice_pass_rule"] = bool(
        row["slice_pass_rule"]
        and row["selected_minus_best_label_copy"] >= STRICT_DELTA
        and row["paired_ci95_low_vs_best_label_copy"] > 0.0
        and row["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
        and row["paired_ci95_low_vs_score_only_bagged"] > 0.0
        and row["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and row["wrong_example_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["candidate_roll_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["basis_dim_roll_control_accuracy"] <= row["best_label_copy_eval_accuracy"] + 0.005
        and row["basis_sign_flip_control_accuracy"] <= row["best_label_copy_eval_accuracy"] + 0.005
        and row["random_basis_same_dim_control_accuracy"] <= row["best_label_copy_eval_accuracy"] + 0.005
        and row["raw_payload_bytes"] == 2
        and row["framed_record_bytes"] == 5
        and not row["source_text_exposed"]
        and not row["source_kv_exposed"]
        and not row["raw_hidden_vector_transmitted"]
        and not row["raw_scores_transmitted"]
        and not row["basis_coefficients_transmitted"]
    )
    return row


def _contiguous(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    sorted_rows = sorted(rows, key=lambda row: row["eval_slice_start"])
    if sorted_rows[0]["eval_slice_start"] != 0:
        return False
    return all(
        left["eval_slice_end_exclusive"] == right["eval_slice_start"]
        for left, right in zip(sorted_rows, sorted_rows[1:], strict=False)
    )


def _weighted_accuracy(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["eval_rows"]) for row in rows)
    if total <= 0:
        return 0.0
    return sum(float(row[key]) * int(row["eval_rows"]) for row in rows) / total


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Learned Residual Basis Multi-Slice Stress",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- slice count: `{h['slice_count']}`",
        f"- strict pass slices: `{h['pass_slice_count']}/{h['slice_count']}`",
        f"- total eval rows: `{h['total_eval_rows']}`",
        f"- contiguous validation prefix: `{h['contiguous_validation_prefix']}`",
        f"- weighted selected accuracy: `{h['weighted_selected_eval_accuracy']:.6f}`",
        f"- weighted best label-copy accuracy: `{h['weighted_best_label_copy_eval_accuracy']:.6f}`",
        f"- weighted score-only accuracy: `{h['weighted_score_only_bagged_control_accuracy']:.6f}`",
        f"- weighted delta vs best label-copy: `{h['weighted_delta_vs_best_label_copy']:.6f}`",
        f"- min delta vs best label-copy: `{h['min_delta_vs_best_label_copy']:.6f}`",
        f"- min CI95 low vs best label-copy: `{h['min_ci95_low_vs_best_label_copy']:.6f}`",
        f"- min delta vs score-only: `{h['min_delta_vs_score_only_bagged']:.6f}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    slice_artifacts: tuple[pathlib.Path, ...] = DEFAULT_SLICE_ARTIFACTS,
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_rows = [
        _slice_row(path, _read_json(path), index)
        for index, path in enumerate(slice_artifacts)
    ]
    slice_rows = sorted(slice_rows, key=lambda row: row["eval_slice_start"])
    total_rows = sum(int(row["eval_rows"]) for row in slice_rows)
    contiguous = _contiguous(slice_rows)
    packet_bytes = sorted({(row["raw_payload_bytes"], row["framed_record_bytes"]) for row in slice_rows})
    source_private_packet = bool(
        packet_bytes == [(2, 5)]
        and all(not row["source_text_exposed"] for row in slice_rows)
        and all(not row["source_kv_exposed"] for row in slice_rows)
        and all(not row["raw_hidden_vector_transmitted"] for row in slice_rows)
        and all(not row["raw_scores_transmitted"] for row in slice_rows)
        and all(not row["basis_coefficients_transmitted"] for row in slice_rows)
    )
    controls_below_label = all(
        row["wrong_example_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["candidate_roll_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["basis_dim_roll_control_accuracy"] <= row["best_label_copy_eval_accuracy"] + 0.005
        and row["basis_sign_flip_control_accuracy"] <= row["best_label_copy_eval_accuracy"] + 0.005
        and row["random_basis_same_dim_control_accuracy"] <= row["best_label_copy_eval_accuracy"] + 0.005
        for row in slice_rows
    )
    headline = {
        "slice_count": len(slice_rows),
        "pass_slice_count": sum(1 for row in slice_rows if row["strict_slice_pass_rule"]),
        "total_eval_rows": total_rows,
        "contiguous_validation_prefix": contiguous,
        "variant": slice_rows[0]["variant"] if slice_rows else "",
        "weighted_selected_eval_accuracy": _weighted_accuracy(slice_rows, "selected_eval_accuracy"),
        "weighted_source_label_copy_eval_accuracy": _weighted_accuracy(slice_rows, "source_label_copy_eval_accuracy"),
        "weighted_trained_choice_bias_label_copy_eval_accuracy": _weighted_accuracy(
            slice_rows,
            "trained_choice_bias_label_copy_eval_accuracy",
        ),
        "weighted_best_label_copy_eval_accuracy": _weighted_accuracy(
            slice_rows,
            "best_label_copy_eval_accuracy",
        ),
        "weighted_score_only_bagged_control_accuracy": _weighted_accuracy(
            slice_rows,
            "score_only_bagged_control_accuracy",
        ),
        "weighted_zero_hidden_control_accuracy": _weighted_accuracy(slice_rows, "zero_hidden_control_accuracy"),
        "weighted_delta_vs_best_label_copy": _weighted_accuracy(slice_rows, "selected_eval_accuracy")
        - _weighted_accuracy(slice_rows, "best_label_copy_eval_accuracy"),
        "weighted_delta_vs_score_only_bagged": _weighted_accuracy(slice_rows, "selected_eval_accuracy")
        - _weighted_accuracy(slice_rows, "score_only_bagged_control_accuracy"),
        "weighted_delta_vs_zero_hidden": _weighted_accuracy(slice_rows, "selected_eval_accuracy")
        - _weighted_accuracy(slice_rows, "zero_hidden_control_accuracy"),
        "min_delta_vs_best_label_copy": min(row["selected_minus_best_label_copy"] for row in slice_rows),
        "min_ci95_low_vs_best_label_copy": min(
            row["paired_ci95_low_vs_best_label_copy"] for row in slice_rows
        ),
        "min_delta_vs_score_only_bagged": min(
            row["selected_minus_score_only_bagged_control"] for row in slice_rows
        ),
        "min_ci95_low_vs_score_only_bagged": min(
            row["paired_ci95_low_vs_score_only_bagged"] for row in slice_rows
        ),
        "min_delta_vs_zero_hidden": min(row["selected_minus_zero_hidden_control"] for row in slice_rows),
        "controls_below_label_copy": controls_below_label,
        "source_private_packet": source_private_packet,
        "raw_payload_bytes": 2 if packet_bytes == [(2, 5)] else None,
        "framed_record_bytes": 5 if packet_bytes == [(2, 5)] else None,
    }
    pass_gate = bool(
        len(slice_rows) >= 5
        and headline["pass_slice_count"] == headline["slice_count"]
        and total_rows >= 5120
        and contiguous
        and headline["weighted_delta_vs_best_label_copy"] >= STRICT_DELTA
        and headline["min_ci95_low_vs_best_label_copy"] > 0.0
        and headline["weighted_delta_vs_score_only_bagged"] >= STRICT_DELTA
        and headline["min_ci95_low_vs_score_only_bagged"] > 0.0
        and headline["weighted_delta_vs_zero_hidden"] >= STRICT_DELTA
        and controls_below_label
        and source_private_packet
    )
    payload = {
        "gate": "source_private_hellaswag_learned_residual_basis_multi_slice_stress",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Predeclared pca256 learned residual basis must pass all five contiguous 1024-row slices, beat "
            "best label-copy and score-only by at least 0.02 with positive slice-level CI lows, beat zero-hidden, "
            "keep all source-destroying and basis controls below label-copy, and preserve the 2B raw / 5B framed "
            "source-private packet contract."
        ),
        "headline": headline,
        "slice_rows": slice_rows,
        "interpretation": (
            "The learned residual basis is a promising bridge on the strongest scout slice, but it does not "
            "survive the all-slice promotion gate. It should be treated as alive only as evidence that learned "
            "basis dimension matters; PCA is not enough for the ICLR common-basis claim. The next branch should "
            "use a richer sparse/crosscoder objective or focus the paper on dense hidden innovation plus systems."
        ),
    }
    json_path = output_dir / "hellaswag_learned_residual_basis_multi_slice_stress.json"
    csv_path = output_dir / "slice_rows.csv"
    md_path = output_dir / "hellaswag_learned_residual_basis_multi_slice_stress.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, slice_rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "pass_gate": pass_gate,
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, csv_path, md_path)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slice-artifact", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()
    artifacts = tuple(args.slice_artifact) if args.slice_artifact else DEFAULT_SLICE_ARTIFACTS
    payload = build_gate(output_dir=args.output_dir, slice_artifacts=artifacts, run_date=args.run_date)
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
