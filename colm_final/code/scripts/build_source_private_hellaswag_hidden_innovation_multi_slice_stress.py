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
    "results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_5120"
)
DEFAULT_SLICE_ARTIFACTS = (
    pathlib.Path(
        "results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_bagged_gate.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/hellaswag_hidden_innovation_eval_slice_stress.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/hellaswag_hidden_innovation_eval_slice_stress.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/hellaswag_hidden_innovation_eval_slice_stress.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/hellaswag_hidden_innovation_eval_slice_stress.json"
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


def _rows_from_bagged(payload: dict[str, Any]) -> int:
    for key in (
        "source_label_copy",
        "trained_choice_bias_label_copy",
        "score_only_bagged_control",
        "zero_hidden_control",
    ):
        readout = payload.get("control_readouts", {}).get(key)
        if isinstance(readout, dict) and int(readout.get("rows", 0)) > 0:
            return int(readout["rows"])
    raise ValueError("bagged artifact did not expose an evaluable row count")


def _slice_row(path: pathlib.Path, payload: dict[str, Any], index: int) -> dict[str, Any]:
    headline = payload["headline"]
    packet_contract = payload["packet_contract"]
    is_eval_slice = payload.get("gate") == "source_private_hellaswag_hidden_innovation_eval_slice_stress"
    rows = int(headline["eval_rows"]) if is_eval_slice else _rows_from_bagged(payload)
    start = int(headline["eval_slice_start"]) if is_eval_slice else index * rows
    end = int(headline["eval_slice_end_exclusive"]) if is_eval_slice else start + rows
    if is_eval_slice:
        jackknife_pass_count = int(headline["jackknife_pass_count"])
        jackknife_row_count = int(headline["jackknife_row_count"])
        jackknife_min_delta = float(headline["jackknife_min_delta_vs_best_label_copy"])
        jackknife_min_ci_low = float(headline["jackknife_min_ci95_low_vs_best_label_copy"])
    else:
        jackknife = payload["jackknife_summary"]
        jackknife_pass_count = int(jackknife["pass_count"])
        jackknife_row_count = int(jackknife["row_count"])
        jackknife_min_delta = float(jackknife["selected_minus_best_label_copy_min"])
        jackknife_min_ci_low = float(jackknife["paired_ci95_low_vs_best_label_copy_min"])
    row = {
        "artifact_path": _display_path(_resolve(path)),
        "gate": str(payload.get("gate", "source_private_hellaswag_hidden_innovation_bagged_gate")),
        "eval_slice_start": start,
        "eval_slice_end_exclusive": end,
        "eval_rows": rows,
        "pass_gate": bool(payload["pass_gate"]),
        "selected_eval_accuracy": float(headline["selected_eval_accuracy"]),
        "source_label_copy_eval_accuracy": float(headline["source_label_copy_eval_accuracy"]),
        "trained_choice_bias_label_copy_eval_accuracy": float(
            headline["trained_choice_bias_label_copy_eval_accuracy"]
        ),
        "best_label_copy_eval_accuracy": float(headline["best_label_copy_eval_accuracy"]),
        "selected_minus_best_label_copy": float(headline["selected_minus_best_label_copy"]),
        "paired_ci95_low_vs_best_label_copy": float(headline["paired_ci95_low_vs_best_label_copy"]),
        "paired_ci95_high_vs_best_label_copy": float(headline["paired_ci95_high_vs_best_label_copy"]),
        "score_only_bagged_control_accuracy": float(headline["score_only_bagged_control_accuracy"]),
        "selected_minus_score_only_bagged_control": float(
            headline["selected_minus_score_only_bagged_control"]
        ),
        "paired_ci95_low_vs_score_only_bagged": float(headline["paired_ci95_low_vs_score_only_bagged"]),
        "zero_hidden_control_accuracy": float(headline["zero_hidden_control_accuracy"]),
        "selected_minus_zero_hidden_control": float(headline["selected_minus_zero_hidden_control"]),
        "wrong_example_hidden_control_accuracy": float(headline["wrong_example_hidden_control_accuracy"]),
        "candidate_roll_hidden_control_accuracy": float(headline["candidate_roll_hidden_control_accuracy"]),
        "jackknife_pass_count": jackknife_pass_count,
        "jackknife_row_count": jackknife_row_count,
        "jackknife_min_delta_vs_best_label_copy": jackknife_min_delta,
        "jackknife_min_ci95_low_vs_best_label_copy": jackknife_min_ci_low,
        "raw_payload_bytes": int(packet_contract["raw_payload_bytes"]),
        "framed_record_bytes": int(packet_contract["framed_record_bytes"]),
        "source_text_exposed": bool(packet_contract["source_text_exposed"]),
        "source_kv_exposed": bool(packet_contract["source_kv_exposed"]),
        "raw_hidden_vector_transmitted": bool(packet_contract["raw_hidden_vector_transmitted"]),
        "raw_scores_transmitted": bool(packet_contract["raw_scores_transmitted"]),
    }
    row["slice_pass_rule"] = bool(
        row["pass_gate"]
        and row["selected_minus_best_label_copy"] >= STRICT_DELTA
        and row["paired_ci95_low_vs_best_label_copy"] > 0.0
        and row["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
        and row["paired_ci95_low_vs_score_only_bagged"] > 0.0
        and row["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and row["wrong_example_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["candidate_roll_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["jackknife_pass_count"] == row["jackknife_row_count"]
        and row["jackknife_min_delta_vs_best_label_copy"] >= STRICT_DELTA
        and row["jackknife_min_ci95_low_vs_best_label_copy"] > 0.0
        and row["raw_payload_bytes"] == 2
        and row["framed_record_bytes"] == 5
        and not row["source_text_exposed"]
        and not row["source_kv_exposed"]
        and not row["raw_hidden_vector_transmitted"]
        and not row["raw_scores_transmitted"]
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
        "# HellaSwag Hidden-Innovation Multi-Slice Stress",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- slice count: `{h['slice_count']}`",
        f"- total eval rows: `{h['total_eval_rows']}`",
        f"- contiguous validation prefix: `{h['contiguous_validation_prefix']}`",
        f"- weighted selected accuracy: `{h['weighted_selected_eval_accuracy']:.6f}`",
        f"- weighted best label-copy accuracy: `{h['weighted_best_label_copy_eval_accuracy']:.6f}`",
        f"- min delta vs best label-copy: `{h['min_delta_vs_best_label_copy']:.6f}`",
        f"- min CI95 low vs best label-copy: `{h['min_ci95_low_vs_best_label_copy']:.6f}`",
        f"- min delta vs score-only bagged: `{h['min_delta_vs_score_only_bagged']:.6f}`",
        f"- min score-only CI95 low: `{h['min_ci95_low_vs_score_only_bagged']:.6f}`",
        f"- min delta vs zero-hidden: `{h['min_delta_vs_zero_hidden']:.6f}`",
        f"- all corrupted-hidden controls below label-copy: `{h['corrupted_hidden_controls_below_label_copy']}`",
        f"- jackknife slices passing: `{h['jackknife_slice_pass_count']}/{h['slice_count']}`",
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
    )
    corrupted_controls_below_label = all(
        row["wrong_example_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        and row["candidate_roll_hidden_control_accuracy"] <= row["best_label_copy_eval_accuracy"]
        for row in slice_rows
    )
    headline = {
        "slice_count": len(slice_rows),
        "pass_slice_count": sum(1 for row in slice_rows if row["slice_pass_rule"]),
        "total_eval_rows": total_rows,
        "contiguous_validation_prefix": contiguous,
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
        "max_wrong_example_hidden_control_accuracy": max(
            row["wrong_example_hidden_control_accuracy"] for row in slice_rows
        ),
        "max_candidate_roll_hidden_control_accuracy": max(
            row["candidate_roll_hidden_control_accuracy"] for row in slice_rows
        ),
        "corrupted_hidden_controls_below_label_copy": corrupted_controls_below_label,
        "jackknife_slice_pass_count": sum(
            1 for row in slice_rows if row["jackknife_pass_count"] == row["jackknife_row_count"]
        ),
        "source_private_packet": source_private_packet,
        "raw_payload_bytes": packet_bytes[0][0] if packet_bytes else None,
        "framed_record_bytes": packet_bytes[0][1] if packet_bytes else None,
        "strict_delta_required": STRICT_DELTA,
    }
    pass_gate = bool(
        headline["slice_count"] >= 3
        and headline["total_eval_rows"] >= 3072
        and headline["pass_slice_count"] == headline["slice_count"]
        and headline["contiguous_validation_prefix"]
        and headline["min_delta_vs_best_label_copy"] >= STRICT_DELTA
        and headline["min_ci95_low_vs_best_label_copy"] > 0.0
        and headline["min_delta_vs_score_only_bagged"] >= STRICT_DELTA
        and headline["min_ci95_low_vs_score_only_bagged"] > 0.0
        and headline["min_delta_vs_zero_hidden"] >= STRICT_DELTA
        and headline["corrupted_hidden_controls_below_label_copy"]
        and headline["jackknife_slice_pass_count"] == headline["slice_count"]
        and headline["source_private_packet"]
    )
    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_multi_slice_stress",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if at least three contiguous frozen HellaSwag validation slices totaling at least "
            "3072 rows each pass the same hidden-innovation gate: >=0.02 over best label-copy, "
            "score-only bag, and zero-hidden controls; paired CI95 lows > 0; corrupted-hidden "
            "controls below best label-copy; all jackknife subbags pass; and the packet remains "
            "2B raw / 5B framed with no source text/KV/raw hidden/raw score exposure."
        ),
        "slice_artifacts": [_display_path(_resolve(path)) for path in slice_artifacts],
        "headline": headline,
        "slice_rows": slice_rows,
        "interpretation": (
            "This aggregate gate is the Mac-feasible bridge between a single heldout HellaSwag slice "
            "and full-validation evidence. It turns the current headline-candidate into a stricter "
            "multi-slice claim only if every contiguous validation block preserves the source-private "
            "advantage and all leakage controls still collapse."
        ),
    }
    json_path = output_dir / "hellaswag_hidden_innovation_multi_slice_stress.json"
    md_path = output_dir / "hellaswag_hidden_innovation_multi_slice_stress.md"
    csv_path = output_dir / "slice_rows.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(csv_path, slice_rows)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path, csv_path)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_paths(value: str) -> tuple[pathlib.Path, ...]:
    paths = tuple(pathlib.Path(part.strip()) for part in value.split(",") if part.strip())
    if not paths:
        raise argparse.ArgumentTypeError("at least one artifact path is required")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slice-artifacts", type=_parse_paths, default=DEFAULT_SLICE_ARTIFACTS)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        slice_artifacts=args.slice_artifacts,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
