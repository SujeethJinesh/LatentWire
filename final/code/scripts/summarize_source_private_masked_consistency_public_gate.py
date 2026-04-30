from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    return str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved)


def _public_explained_fraction(packet_lift: float, public_lift: float) -> float:
    if packet_lift <= 0.0:
        return 1.0 if public_lift > 0.0 else 0.0
    return max(0.0, public_lift) / packet_lift


def _row(packet_dir: pathlib.Path, public_dir: pathlib.Path, *, public_fraction_limit: float) -> dict[str, Any]:
    packet = _read_json(_resolve(packet_dir) / "run_summary.json")
    public = _read_json(_resolve(public_dir) / "run_summary.json")
    packet_summary = packet["summary"]
    public_summary = public["summary"]

    target = float(packet_summary["target_only_accuracy"])
    learned = float(packet_summary["learned_matched_accuracy"])
    best_control = float(packet_summary["best_control_accuracy"])
    public_target = float(public_summary["target_only_accuracy"])
    public_acc = float(public_summary["public_only_accuracy"])
    packet_lift = learned - target
    public_lift = public_acc - public_target
    fraction = _public_explained_fraction(packet_lift, public_lift)
    same_eval_hash = packet_summary["exact_id_sha256"] == public_summary["exact_id_sha256"]
    same_candidate_view = packet.get("candidate_view") == public.get("candidate_view")
    same_diag_mode = packet.get("diagnostic_table_mode") == public.get("diagnostic_table_mode")
    ci_vs_target = packet_summary["paired_bootstrap"]["learned_matched_vs_target"]
    ci_vs_best_control = packet_summary["paired_bootstrap"]["learned_matched_vs_best_control"]
    public_ci = public_summary["paired_bootstrap_public_vs_target"]
    public_near_target = public_acc <= public_target + float(public_summary["max_allowed_lift"])

    pass_gate = (
        bool(packet_summary["exact_id_parity"])
        and same_eval_hash
        and same_candidate_view
        and same_diag_mode
        and int(packet.get("train_eval_id_intersection_count", -1)) == 0
        and int(public_summary["train_eval_id_intersection_count"]) == 0
        and learned >= target + 0.15
        and learned >= best_control + 0.15
        and bool(packet_summary["learned_controls_ok"])
        and float(ci_vs_target["ci95_low"]) > 0.10
        and float(ci_vs_best_control["ci95_low"]) > 0.10
        and public_near_target
        and fraction < public_fraction_limit
    )
    return {
        "packet_run_dir": _rel(packet_dir),
        "public_run_dir": _rel(public_dir),
        "candidate_view": packet.get("candidate_view"),
        "diagnostic_table_mode": packet.get("diagnostic_table_mode"),
        "n": int(packet_summary["n"]),
        "same_eval_hash": same_eval_hash,
        "same_candidate_view": same_candidate_view,
        "same_diagnostic_table_mode": same_diag_mode,
        "packet_exact_id_parity": bool(packet_summary["exact_id_parity"]),
        "packet_train_eval_id_intersection_count": int(packet.get("train_eval_id_intersection_count", -1)),
        "public_train_eval_id_intersection_count": int(public_summary["train_eval_id_intersection_count"]),
        "target_only_accuracy": target,
        "learned_matched_accuracy": learned,
        "hamming_matched_accuracy": float(packet_summary["hamming_matched_accuracy"]),
        "best_control_condition": packet_summary["best_control_condition"],
        "best_control_accuracy": best_control,
        "learned_minus_target": packet_lift,
        "learned_minus_best_control": learned - best_control,
        "packet_controls_ok": bool(packet_summary["learned_controls_ok"]),
        "packet_pass_gate": bool(packet_summary["pass_gate"]),
        "packet_ci95_low_vs_target": float(ci_vs_target["ci95_low"]),
        "packet_ci95_low_vs_best_control": float(ci_vs_best_control["ci95_low"]),
        "public_only_accuracy": public_acc,
        "public_minus_target": public_lift,
        "public_ci95_high_vs_target": float(public_ci["ci95_high"]),
        "public_near_target": public_near_target,
        "public_explained_fraction": fraction,
        "public_fraction_limit": public_fraction_limit,
        "pass_gate": pass_gate,
    }


def summarize_runs(
    *,
    packet_dirs: list[pathlib.Path],
    public_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    public_fraction_limit: float,
) -> dict[str, Any]:
    if len(packet_dirs) != len(public_dirs):
        raise ValueError("--packet-run-dir and --public-run-dir counts must match")
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _row(packet_dir, public_dir, public_fraction_limit=public_fraction_limit)
        for packet_dir, public_dir in zip(packet_dirs, public_dirs, strict=True)
    ]
    headline = {
        "pass_gate": bool(rows) and all(row["pass_gate"] for row in rows),
        "rows": len(rows),
        "passed_rows": sum(1 for row in rows if row["pass_gate"]),
        "min_learned_minus_target": min((row["learned_minus_target"] for row in rows), default=None),
        "min_learned_minus_best_control": min((row["learned_minus_best_control"] for row in rows), default=None),
        "max_public_minus_target": max((row["public_minus_target"] for row in rows), default=None),
        "max_public_explained_fraction": max((row["public_explained_fraction"] for row in rows), default=None),
        "all_same_eval_hash": all(row["same_eval_hash"] for row in rows),
        "all_public_near_target": all(row["public_near_target"] for row in rows),
        "all_packet_controls_ok": all(row["packet_controls_ok"] for row in rows),
        "all_train_eval_disjoint": all(
            row["packet_train_eval_id_intersection_count"] == 0
            and row["public_train_eval_id_intersection_count"] == 0
            for row in rows
        ),
    }
    payload = {
        "gate": "source_private_masked_consistency_public_separation",
        "headline": headline,
        "rows": rows,
        "pass_rule": (
            "Pass requires exact ID parity, same eval IDs between packet and public-only rows, disjoint train/eval IDs, "
            "matched learned packet accuracy >= target+0.15 and >= best destructive control+0.15, paired CI95 lower "
            "bounds > +0.10 vs target and best control, destructive controls within target+0.05, public-only accuracy "
            "within target+0.05, and public-only lift explaining less than the configured fraction of packet lift."
        ),
        "interpretation": (
            "This aggregate decides whether a learned masked-consistency receiver has source-packet lift that is not "
            "explained by a separately trained public-only receiver on the same eval IDs."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Source-Private Masked Consistency Public-Separation Gate",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- rows: `{headline['rows']}`",
        f"- passed rows: `{headline['passed_rows']}`",
        f"- min learned minus target: `{headline['min_learned_minus_target']}`",
        f"- min learned minus best control: `{headline['min_learned_minus_best_control']}`",
        f"- max public minus target: `{headline['max_public_minus_target']}`",
        f"- max public explained fraction: `{headline['max_public_explained_fraction']}`",
        f"- all same eval hash: `{headline['all_same_eval_hash']}`",
        f"- all public near target: `{headline['all_public_near_target']}`",
        "",
        "| Packet run | Public run | n | view | learned | target | best control | packet lift | public | public lift | public frac | pass |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['packet_run_dir']}` | `{row['public_run_dir']}` | {row['n']} | "
            f"`{row['candidate_view']}` | {row['learned_matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | "
            f"{row['learned_minus_target']:.3f} | {row['public_only_accuracy']:.3f} | "
            f"{row['public_minus_target']:.3f} | {row['public_explained_fraction']:.3f} | "
            f"`{row['pass_gate']}` |"
        )
    lines.extend(["", payload["pass_rule"], "", payload["interpretation"], ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "gate": payload["gate"],
        "pass_gate": headline["pass_gate"],
        "artifacts": ["summary.json", "summary.md", "manifest.json", "manifest.md"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Masked Consistency Public-Separation Manifest",
                "",
                f"- pass gate: `{headline['pass_gate']}`",
                f"- rows: `{headline['rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-run-dir", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--public-run-dir", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--public-fraction-limit", type=float, default=0.25)
    args = parser.parse_args()
    payload = summarize_runs(
        packet_dirs=args.packet_run_dir,
        public_dirs=args.public_run_dir,
        output_dir=args.output_dir,
        public_fraction_limit=args.public_fraction_limit,
    )
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
