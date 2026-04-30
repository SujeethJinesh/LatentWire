from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIRS = (
    pathlib.Path("results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed29_30_full"),
    pathlib.Path("results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed31_32_full"),
    pathlib.Path("results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed29_30_semantic"),
    pathlib.Path("results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed29_30_slot_remap901"),
    pathlib.Path("results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed31_32_slot_remap907"),
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _run_label(payload: dict[str, Any], run_dir: pathlib.Path) -> str:
    view = payload.get("candidate_view", "unknown")
    remap = payload.get("remap_slot_seed")
    seed_pair = f"{payload.get('train_seed', '?')}_{payload.get('eval_seed', '?')}"
    suffix = f"_remap{remap}" if remap is not None else ""
    return f"{view}{suffix}_seed{seed_pair}"


def _row(run_dir: pathlib.Path) -> dict[str, Any]:
    resolved = _resolve(run_dir)
    payload = _read_json(resolved / "run_summary.json")
    summary = payload["summary"]
    learned = summary["learned_metrics"]
    target = float(summary["target_only_accuracy"])
    controls = [
        condition
        for condition in summary["conditions"]
        if condition
        not in {
            "target_only",
            "matched_consistency_packet",
            "masked_matched_packet",
            "full_diag_oracle",
        }
    ]
    control_deltas = {condition: float(learned[condition]["accuracy"]) - target for condition in controls}
    hamming = summary["hamming_metrics"]["matched_consistency_packet"]["accuracy"]
    learned_matched = float(summary["learned_matched_accuracy"])
    n = int(summary["n"])
    candidate_view = payload.get("candidate_view", "unknown")
    remap_slot_seed = payload.get("remap_slot_seed")
    is_full_reference = candidate_view == "full" and remap_slot_seed is None
    is_opaque_slot = candidate_view == "slot" and remap_slot_seed is not None
    max_control_delta = max(control_deltas.values()) if control_deltas else 0.0
    return {
        "run_dir": str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved),
        "label": _run_label(payload, run_dir),
        "n": n,
        "candidate_view": candidate_view,
        "remap_slot_seed": remap_slot_seed,
        "pass_gate": bool(summary["pass_gate"]),
        "source_packet_pass": bool(summary["source_packet_pass"]),
        "exact_id_parity": bool(summary["exact_id_parity"]),
        "target_only_accuracy": target,
        "learned_matched_accuracy": learned_matched,
        "hamming_matched_accuracy": float(hamming),
        "best_control_condition": summary["best_control_condition"],
        "best_control_accuracy": float(summary["best_control_accuracy"]),
        "learned_minus_target": float(summary["learned_minus_target"]),
        "learned_minus_best_control": float(summary["learned_minus_best_control"]),
        "hamming_minus_target": float(hamming) - target,
        "max_control_delta": max_control_delta,
        "controls_ok_at_005": max_control_delta <= 0.05,
        "ci95_low_vs_target": summary["paired_bootstrap"]["learned_matched_vs_target"]["ci95_low"],
        "ci95_high_vs_target": summary["paired_bootstrap"]["learned_matched_vs_target"]["ci95_high"],
        "ci95_low_vs_best_control": summary["paired_bootstrap"]["learned_matched_vs_best_control"]["ci95_low"],
        "condition_id_parity": bool(summary.get("condition_id_parity", summary.get("exact_id_parity", False))),
        "train_eval_id_intersection_count": int(payload.get("train_eval_id_intersection_count", 0)),
        "is_full_reference": is_full_reference,
        "is_opaque_slot": is_opaque_slot,
        "opaque_learned_collapse": bool(is_opaque_slot and learned_matched <= target + 0.05),
        "opaque_hamming_collapse": bool(is_opaque_slot and float(hamming) <= target + 0.05),
    }


def summarize_runs(*, run_dirs: list[pathlib.Path], output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(path) for path in run_dirs if _resolve(path).exists()]
    if not rows:
        raise ValueError("no run directories found")
    reference_rows = [row for row in rows if row["is_full_reference"] and row["n"] >= 256]
    opaque_slot_rows = [row for row in rows if row["is_opaque_slot"] and row["n"] >= 256]
    blinded_rows = [
        row
        for row in rows
        if row["candidate_view"] in {"no_diag", "semantic"} and row["n"] >= 256 and row["remap_slot_seed"] is None
    ]
    reference_full_pass = bool(reference_rows) and all(row["pass_gate"] for row in reference_rows)
    all_disjoint_train_eval = all(row["train_eval_id_intersection_count"] == 0 for row in rows)
    opaque_slot_collapse = bool(opaque_slot_rows) and all(
        row["exact_id_parity"]
        and row["condition_id_parity"]
        and row["train_eval_id_intersection_count"] == 0
        and row["opaque_learned_collapse"]
        and row["opaque_hamming_collapse"]
        and row["ci95_high_vs_target"] <= 0.10
        and row["controls_ok_at_005"]
        for row in opaque_slot_rows
    )
    blinded_controls_clean = all(row["exact_id_parity"] and row["controls_ok_at_005"] for row in blinded_rows)
    headline = {
        "pass_gate": reference_full_pass and opaque_slot_collapse and blinded_controls_clean and all_disjoint_train_eval,
        "runs": len(rows),
        "reference_full_n256_runs": len(reference_rows),
        "reference_full_n256_pass": reference_full_pass,
        "opaque_slot_n256_runs": len(opaque_slot_rows),
        "opaque_slot_collapse": opaque_slot_collapse,
        "blinded_n256_runs": len(blinded_rows),
        "blinded_controls_clean": blinded_controls_clean,
        "all_disjoint_train_eval": all_disjoint_train_eval,
        "min_reference_full_lift_vs_target": min((row["learned_minus_target"] for row in reference_rows), default=None),
        "max_opaque_slot_lift_vs_target": max((row["learned_minus_target"] for row in opaque_slot_rows), default=None),
        "max_opaque_slot_hamming_lift_vs_target": max((row["hamming_minus_target"] for row in opaque_slot_rows), default=None),
        "max_blinded_lift_vs_target": max((row["learned_minus_target"] for row in blinded_rows), default=None),
        "min_blinded_lift_vs_target": min((row["learned_minus_target"] for row in blinded_rows), default=None),
        "max_control_delta": max(row["max_control_delta"] for row in rows),
        "all_exact_id_parity": all(row["exact_id_parity"] for row in rows),
    }
    payload = {
        "gate": "source_private_masked_consistency_receiver_label_blind_stress",
        "headline": headline,
        "rows": rows,
        "pass_rule": (
            "Pass requires existing full-view n256 receiver runs to pass; every opaque slot-view n256 run with "
            "per-example remapped candidate order to collapse to target-only within +0.05 for learned and "
            "deterministic Hamming decoders with paired CI95 high <= +0.10; all decisive rows to have zero train/eval "
            "ID overlap; all destructive controls to stay within target+0.05; and exact-ID parity for all rows. "
            "no_diag/semantic rows are diagnostic: if they remain high, the receiver can use "
            "public semantic candidate side information; if they collapse, the normal result is diagnostic-key dependent."
        ),
        "interpretation": (
            "This gate distinguishes source-private packet communication with decoder side information from opaque "
            "candidate-index lookup. The headline claim is strengthened when normal full-view packets pass while "
            "per-example remapped slot-only candidate views collapse."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Source-Private Masked Consistency Receiver Label-Blind Stress",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- reference full n256 pass: `{headline['reference_full_n256_pass']}`",
        f"- opaque slot collapse: `{headline['opaque_slot_collapse']}`",
        f"- blinded controls clean: `{headline['blinded_controls_clean']}`",
        f"- all disjoint train/eval: `{headline['all_disjoint_train_eval']}`",
        f"- min reference full lift vs target: `{headline['min_reference_full_lift_vs_target']}`",
        f"- max opaque slot lift vs target: `{headline['max_opaque_slot_lift_vs_target']}`",
        f"- max opaque slot Hamming lift vs target: `{headline['max_opaque_slot_hamming_lift_vs_target']}`",
        f"- max blinded lift vs target: `{headline['max_blinded_lift_vs_target']}`",
        "",
        "| Run | n | view | remap | train/eval overlap | pass | learned | Hamming | target | lift | CI high | best control | max control delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        remap = "" if row["remap_slot_seed"] is None else str(row["remap_slot_seed"])
        lines.append(
            f"| `{row['label']}` | {row['n']} | `{row['candidate_view']}` | {remap} | "
            f"{row['train_eval_id_intersection_count']} | `{row['pass_gate']}` | "
            f"{row['learned_matched_accuracy']:.3f} | {row['hamming_matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['learned_minus_target']:.3f} | "
            f"{row['ci95_high_vs_target']:.3f} | "
            f"`{row['best_control_condition']}` {row['best_control_accuracy']:.3f} | {row['max_control_delta']:.3f} |"
        )
    lines.extend(["", payload["pass_rule"], "", payload["interpretation"], ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "headline": headline,
        "pass_gate": headline["pass_gate"],
        "artifacts": ["summary.json", "summary.md", "manifest.json", "manifest.md"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Masked Consistency Receiver Label-Blind Stress Manifest",
                "",
                f"- pass gate: `{headline['pass_gate']}`",
                f"- reference full n256 runs: `{headline['reference_full_n256_runs']}`",
                f"- opaque slot n256 runs: `{headline['opaque_slot_n256_runs']}`",
                f"- blinded n256 runs: `{headline['blinded_n256_runs']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_masked_consistency_receiver_label_blind_20260430/summary"),
    )
    args = parser.parse_args()
    payload = summarize_runs(run_dirs=args.run_dir or list(DEFAULT_RUN_DIRS), output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
