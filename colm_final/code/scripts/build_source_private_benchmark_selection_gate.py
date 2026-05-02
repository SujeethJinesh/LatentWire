from __future__ import annotations

"""Select the next source-private benchmark/receiver gate from frozen artifacts."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_benchmark_selection_gate_20260502")
DEFAULT_BENCHMARKS = (
    {
        "row_id": "openbookqa_test_3b",
        "dataset": "OpenBookQA",
        "split": "test",
        "priority": 0,
        "seed_artifact": pathlib.Path(
            "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/"
            "arc_challenge_seed_stability.json"
        ),
        "oracle_predictions_jsonl": pathlib.Path(
            "results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/"
            "predictions.jsonl"
        ),
        "oracle_budget_note": "oracle computed from 4B fixed-gate predictions; 3B seed row is the promoted packet budget",
    },
    {
        "row_id": "arc_challenge_test_12b",
        "dataset": "ARC-Challenge",
        "split": "test",
        "priority": 1,
        "seed_artifact": pathlib.Path(
            "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/"
            "arc_challenge_seed_stability.json"
        ),
        "oracle_predictions_jsonl": pathlib.Path(
            "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/"
            "predictions.jsonl"
        ),
        "oracle_budget_note": "oracle computed from the same ARC fixed-gate prediction artifact",
    },
    {
        "row_id": "commonsenseqa_validation_2b",
        "dataset": "CommonsenseQA",
        "split": "validation",
        "priority": 2,
        "seed_artifact": pathlib.Path(
            "results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b/"
            "arc_challenge_seed_stability.json"
        ),
        "oracle_predictions_jsonl": pathlib.Path(
            "results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b/"
            "predictions.jsonl"
        ),
        "oracle_budget_note": "oracle computed from 12B fixed-gate predictions; 2B seed row is the strict text-margin packet budget",
    },
)
CSV_COLUMNS = (
    "row_id",
    "dataset",
    "split",
    "budget_bytes",
    "eval_rows",
    "seed_count",
    "pass_count",
    "seed_stability_pass",
    "packet_target_pass",
    "packet_text_margin_pass",
    "destructive_control_pass",
    "receiver_headroom_pass",
    "receiver_gate_status",
    "selection_role",
    "matched_accuracy_mean",
    "matched_accuracy_min",
    "target_accuracy",
    "same_byte_text_accuracy",
    "matched_minus_target_min",
    "matched_minus_same_byte_text_min",
    "matched_minus_best_destructive_min",
    "paired_ci95_low_vs_target_min",
    "oracle_accuracy",
    "oracle_headroom_vs_packet",
    "target_correct_packet_wrong",
    "packet_correct_target_wrong",
    "target_and_packet_correct",
    "recommended_next_gate",
)
STRICT_TARGET_DELTA = 0.05
STRICT_TEXT_DELTA = 0.02
STRICT_DESTRUCTIVE_DELTA = 0.05
STRICT_ORACLE_HEADROOM = 0.03


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve(path)
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _oracle_headroom(path: pathlib.Path | str) -> dict[str, Any]:
    resolved = _resolve(path)
    grouped: dict[str, dict[str, Any]] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(str(row["row_id"]), {})[str(row["condition"])] = row
    rows = []
    for row_id, conditions in grouped.items():
        target = conditions.get("target_only")
        packet = conditions.get("matched_source_private_packet")
        if target is None or packet is None:
            continue
        rows.append((row_id, bool(target["correct"]), bool(packet["correct"])))
    if not rows:
        return {
            "available": False,
            "row_count": 0,
            "target_accuracy": None,
            "packet_accuracy": None,
            "oracle_accuracy": None,
            "oracle_headroom_vs_packet": None,
            "target_correct_packet_wrong": None,
            "packet_correct_target_wrong": None,
            "target_and_packet_correct": None,
        }
    row_count = len(rows)
    target_correct = sum(int(target) for _, target, _ in rows)
    packet_correct = sum(int(packet) for _, _, packet in rows)
    oracle_correct = sum(int(target or packet) for _, target, packet in rows)
    target_only = sum(int(target and not packet) for _, target, packet in rows)
    packet_only = sum(int(packet and not target) for _, target, packet in rows)
    both = sum(int(target and packet) for _, target, packet in rows)
    return {
        "available": True,
        "row_count": row_count,
        "target_accuracy": target_correct / row_count,
        "packet_accuracy": packet_correct / row_count,
        "oracle_accuracy": oracle_correct / row_count,
        "oracle_headroom_vs_packet": (oracle_correct - packet_correct) / row_count,
        "target_correct_packet_wrong": target_only,
        "packet_correct_target_wrong": packet_only,
        "target_and_packet_correct": both,
    }


def _build_row(config: dict[str, Any]) -> dict[str, Any]:
    seed_artifact = _resolve(config["seed_artifact"])
    payload = _read_json(seed_artifact)
    aggregate = payload["aggregate"]
    oracle = _oracle_headroom(config["oracle_predictions_jsonl"])
    seed_count = int(aggregate.get("seed_count", 1))
    pass_count = int(aggregate.get("pass_count", int(bool(payload.get("pass_gate")))))
    matched_minus_target_min = float(aggregate["matched_minus_target_min"])
    matched_minus_text_min = float(aggregate["matched_minus_same_byte_text_min"])
    matched_minus_destructive_min = float(aggregate["matched_minus_best_destructive_min"])
    ci_low = float(aggregate["paired_ci95_low_vs_target_min"])
    seed_stability_pass = bool(aggregate.get("all_seeds_pass", payload.get("pass_gate"))) and pass_count == seed_count
    packet_target_pass = matched_minus_target_min >= STRICT_TARGET_DELTA and ci_low > 0.0
    packet_text_margin_pass = matched_minus_text_min >= STRICT_TEXT_DELTA
    destructive_control_pass = matched_minus_destructive_min >= STRICT_DESTRUCTIVE_DELTA
    receiver_headroom = oracle["oracle_headroom_vs_packet"]
    receiver_headroom_pass = bool(receiver_headroom is not None and receiver_headroom >= STRICT_ORACLE_HEADROOM)
    receiver_gate_status = "missing_train_only_receiver"
    if not receiver_headroom_pass:
        receiver_gate_status = "insufficient_oracle_headroom"
    if seed_stability_pass and packet_target_pass and packet_text_margin_pass and destructive_control_pass and receiver_headroom_pass:
        selection_role = "receiver_candidate"
    elif packet_target_pass and not packet_text_margin_pass:
        selection_role = "diagnostic_text_saturated"
    elif seed_stability_pass and packet_target_pass:
        selection_role = "packet_positive_but_receiver_incomplete"
    else:
        selection_role = "not_selected"
    return {
        "row_id": config["row_id"],
        "dataset": config["dataset"],
        "split": config["split"],
        "priority": int(config.get("priority", 999)),
        "seed_artifact": _display_path(seed_artifact),
        "seed_artifact_sha256": _sha256_file(seed_artifact),
        "oracle_predictions_jsonl": _display_path(config["oracle_predictions_jsonl"]),
        "oracle_predictions_sha256": _sha256_file(config["oracle_predictions_jsonl"]),
        "oracle_budget_note": config.get("oracle_budget_note", ""),
        "budget_bytes": int(payload.get("budget_bytes", aggregate.get("budget_bytes", 0))),
        "eval_rows": int(payload.get("eval_rows", aggregate.get("eval_rows", oracle["row_count"] or 0))),
        "seed_count": seed_count,
        "pass_count": pass_count,
        "seed_stability_pass": seed_stability_pass,
        "packet_target_pass": packet_target_pass,
        "packet_text_margin_pass": packet_text_margin_pass,
        "destructive_control_pass": destructive_control_pass,
        "receiver_headroom_pass": receiver_headroom_pass,
        "receiver_gate_status": receiver_gate_status,
        "selection_role": selection_role,
        "matched_accuracy_mean": float(aggregate["matched_accuracy_mean"]),
        "matched_accuracy_min": float(aggregate["matched_accuracy_min"]),
        "target_accuracy": float(aggregate["target_accuracy"]),
        "same_byte_text_accuracy": float(aggregate["same_byte_structured_text_accuracy"]),
        "matched_minus_target_min": matched_minus_target_min,
        "matched_minus_same_byte_text_min": matched_minus_text_min,
        "matched_minus_best_destructive_min": matched_minus_destructive_min,
        "paired_ci95_low_vs_target_min": ci_low,
        "oracle_available": oracle["available"],
        "oracle_accuracy": oracle["oracle_accuracy"],
        "oracle_headroom_vs_packet": receiver_headroom,
        "target_correct_packet_wrong": oracle["target_correct_packet_wrong"],
        "packet_correct_target_wrong": oracle["packet_correct_target_wrong"],
        "target_and_packet_correct": oracle["target_and_packet_correct"],
        "recommended_next_gate": "",
    }


def _select_next(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["selection_role"] == "receiver_candidate"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (row["priority"], row["budget_bytes"], -row["oracle_headroom_vs_packet"]))[0]


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Benchmark Selection Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- recommended next gate: `{h['recommended_next_gate']}`",
        f"- selected benchmark: `{h['selected_benchmark']}`",
        f"- receiver candidates: `{h['receiver_candidate_count']}`",
        f"- text-saturated diagnostics: `{h['diagnostic_text_saturated_count']}`",
        "",
        "## Rows",
        "",
        "| Benchmark | Budget | Seeds | Matched | Target | Text | Lift vs target | Lift vs text | Oracle headroom | Role |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset']} `{row['split']}` | {row['budget_bytes']}B | "
            f"{row['pass_count']}/{row['seed_count']} | {row['matched_accuracy_mean']:.3f} | "
            f"{row['target_accuracy']:.3f} | {row['same_byte_text_accuracy']:.3f} | "
            f"{row['matched_minus_target_min']:.3f} | {row['matched_minus_same_byte_text_min']:.3f} | "
            f"{(row['oracle_headroom_vs_packet'] or 0.0):.3f} | `{row['selection_role']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Next Gate",
            "",
            payload["next_exact_gate"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    benchmarks: tuple[dict[str, Any], ...] = DEFAULT_BENCHMARKS,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_build_row(config) for config in benchmarks]
    selected = _select_next(rows)
    if selected is not None:
        selected["recommended_next_gate"] = (
            f"train-only receiver/headroom gate on {selected['dataset']} {selected['split']} "
            f"using the promoted {selected['budget_bytes']}B packet, target-cache-only, "
            "candidate-only, target-derived, row-shuffle, random same-rate, label-permutation, "
            "candidate-derangement, same-byte text, and source-label-copy controls"
        )
    receiver_candidates = [row for row in rows if row["selection_role"] == "receiver_candidate"]
    diagnostics = [row for row in rows if row["selection_role"] == "diagnostic_text_saturated"]
    pass_gate = bool(selected is not None)
    selected_name = f"{selected['dataset']} {selected['split']}" if selected is not None else "none"
    next_exact_gate = (
        selected["recommended_next_gate"]
        if selected is not None
        else "No benchmark currently clears packet/text/control/headroom selection; move to a true learned connector."
    )
    headline = {
        "recommended_next_gate": next_exact_gate,
        "selected_benchmark": selected_name,
        "receiver_candidate_count": len(receiver_candidates),
        "diagnostic_text_saturated_count": len(diagnostics),
        "strict_target_delta": STRICT_TARGET_DELTA,
        "strict_text_delta": STRICT_TEXT_DELTA,
        "strict_destructive_delta": STRICT_DESTRUCTIVE_DELTA,
        "strict_oracle_headroom": STRICT_ORACLE_HEADROOM,
        "iclr_ready": False,
        "iclr_blocker": "selected benchmark still needs a train-only receiver that beats packet-only with paired uncertainty",
        "colm_ready": True,
    }
    interpretation = (
        "OpenBookQA and ARC both retain large target-or-packet oracle headroom after fixed-packet "
        "seed stability, while CommonsenseQA remains useful but text-saturated. This makes "
        "OpenBookQA the highest-value next ICLR method gate: it is already a second public benchmark, "
        "has a strict text-margin packet row, and has enough target/packet complementarity to test a "
        "real receiver rather than another HellaSwag hidden-code variant."
    )
    payload = {
        "gate": "source_private_benchmark_selection_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if at least one frozen benchmark has 5/5 seed stability, packet lift >=0.05 vs "
            "target with positive paired CI, packet lift >=0.02 vs same-byte text, destructive-control "
            "separation >=0.05, and target-or-packet oracle headroom >=0.03. Passing this gate selects "
            "the next receiver experiment; it is not itself an ICLR positive learned-receiver result."
        ),
        "headline": headline,
        "rows": rows,
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
    }
    json_path = output_dir / "benchmark_selection_gate.json"
    csv_path = output_dir / "benchmark_selection_gate.csv"
    md_path = output_dir / "benchmark_selection_gate.md"
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
            {
                "row_id": config["row_id"],
                "seed_artifact": _display_path(config["seed_artifact"]),
                "seed_artifact_sha256": _sha256_file(config["seed_artifact"]),
                "oracle_predictions_jsonl": _display_path(config["oracle_predictions_jsonl"]),
                "oracle_predictions_sha256": _sha256_file(config["oracle_predictions_jsonl"]),
            }
            for config in benchmarks
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(output_dir=args.output_dir, run_date=args.run_date)
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
