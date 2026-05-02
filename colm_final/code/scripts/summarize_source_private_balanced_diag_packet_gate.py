from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_PAIRS = (
    (
        pathlib.Path("results/source_private_diag_only_public_ablation_20260430/direct_diag_n500_seed29"),
        pathlib.Path("results/source_private_diag_only_public_ablation_20260430/n500_seed29_diag_only_public_same_eval"),
    ),
    (
        pathlib.Path("results/source_private_diag_only_public_ablation_20260430/direct_diag_n500_seed31"),
        pathlib.Path("results/source_private_diag_only_public_ablation_20260430/n500_seed31_diag_only_public_same_eval"),
    ),
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_optional_json(path: pathlib.Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _paired_bootstrap(diffs: list[float]) -> dict[str, float]:
    arr = np.asarray(diffs, dtype=np.float32)
    point = float(np.mean(arr)) if len(arr) else 0.0
    if len(arr) <= 1:
        return {"point": point, "ci95_low": point, "ci95_high": point}
    rng = np.random.default_rng(20260430 + len(arr))
    samples = np.empty(2000, dtype=np.float32)
    for index in range(len(samples)):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples[index] = float(np.mean(arr[idx]))
    return {
        "point": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _pair_row(*, direct_dir: pathlib.Path, public_dir: pathlib.Path, budget_bytes: int) -> dict[str, Any]:
    direct_dir = _resolve(direct_dir)
    public_dir = _resolve(public_dir)
    sweep = _read_json(direct_dir / "sweep_summary.json")
    direct_manifest = _read_optional_json(direct_dir / "manifest.json")
    direct_args = direct_manifest.get("args", {})
    direct_summary = next(row for row in sweep["budget_summaries"] if int(row["budget_bytes"]) == budget_bytes)
    direct_rows = _read_jsonl(direct_dir / f"predictions_budget{budget_bytes}.jsonl")
    public_payload = _read_json(public_dir / "run_summary.json")
    public_rows = _read_jsonl(public_dir / "predictions.jsonl")
    direct_ids = [row["example_id"] for row in direct_rows]
    public_ids = [row["example_id"] for row in public_rows]
    public_by_id = {row["example_id"]: row for row in public_rows}
    same_eval_ids = direct_ids == public_ids
    content_parity = same_eval_ids and all(
        row.get("family_name") == public_by_id[row["example_id"]].get("family_name")
        and row.get("answer_label") == public_by_id[row["example_id"]].get("answer_label")
        for row in direct_rows
    )
    public_eval_family_set = public_payload.get("eval_family_set")
    direct_family_set = direct_args.get("family_set")
    direct_eval_config_matches_public = (
        direct_family_set == public_eval_family_set
        and int(direct_args.get("examples", -1)) == int(public_payload.get("eval_examples", -2))
        and int(direct_args.get("seed", -1)) == int(public_payload.get("eval_seed", -2))
    )
    public_train_eval_disjoint = public_payload.get("summary", {}).get("train_eval_id_intersection_count") == 0
    balanced_diag_config = (
        direct_args.get("diagnostic_table_mode") == "plausible_decoys"
        and public_payload.get("diagnostic_table_mode") == "plausible_decoys"
        and public_payload.get("candidate_view") == "diag_only"
    )
    packet_correct = [bool(row["conditions"]["matched_repair_packet"]["correct"]) for row in direct_rows]
    public_correct = [bool(public_by_id[row["example_id"]]["public_correct"]) for row in direct_rows]
    target_correct = [bool(row["conditions"]["target_only"]["correct"]) for row in direct_rows]
    packet_minus_public = _paired_bootstrap([float(packet) - float(public) for packet, public in zip(packet_correct, public_correct, strict=True)])
    public_minus_target = _paired_bootstrap([float(public) - float(target) for public, target in zip(public_correct, target_correct, strict=True)])
    return {
        "direct_dir": str(direct_dir.relative_to(ROOT) if direct_dir.is_relative_to(ROOT) else direct_dir),
        "public_dir": str(public_dir.relative_to(ROOT) if public_dir.is_relative_to(ROOT) else public_dir),
        "budget_bytes": budget_bytes,
        "n": len(direct_rows),
        "same_eval_ids": same_eval_ids,
        "content_parity": content_parity,
        "direct_eval_config_matches_public": direct_eval_config_matches_public,
        "public_train_eval_disjoint": public_train_eval_disjoint,
        "balanced_diag_config": balanced_diag_config,
        "direct_family_set": direct_family_set,
        "public_train_family_set": public_payload.get("train_family_set"),
        "public_eval_family_set": public_eval_family_set,
        "exact_id_sha256": _sha256_ids(direct_ids),
        "direct_pass_gate": bool(direct_summary["pass_gate"]),
        "public_pass_gate": bool(public_payload["pass_gate"]),
        "packet_accuracy": float(direct_summary["matched_selector_accuracy"]),
        "target_only_accuracy": float(direct_summary["best_no_source_accuracy"]),
        "best_control_accuracy": float(direct_summary["best_source_destroying_control_accuracy"]),
        "best_reviewer_negative_accuracy": float(direct_summary["best_reviewer_negative_control_accuracy"]),
        "public_only_accuracy": float(public_payload["summary"]["public_only_accuracy"]),
        "public_minus_target": public_minus_target,
        "packet_minus_public": packet_minus_public,
        "packet_payload_bytes": direct_summary["metrics"]["matched_repair_packet"]["mean_payload_bytes"],
        "packet_p50_latency_ms": direct_summary["metrics"]["matched_repair_packet"]["p50_latency_ms"],
        "public_p50_latency_ms": public_payload["summary"]["p50_latency_ms"],
    }


def summarize_pairs(
    *,
    pairs: list[tuple[pathlib.Path, pathlib.Path]],
    output_dir: pathlib.Path,
    budget_bytes: int,
    min_packet_minus_public_ci_low: float,
    max_public_lift: float,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_pair_row(direct_dir=direct, public_dir=public, budget_bytes=budget_bytes) for direct, public in pairs]
    headline = {
        "pass_gate": all(
            row["same_eval_ids"]
            and row["content_parity"]
            and row["direct_eval_config_matches_public"]
            and row["public_train_eval_disjoint"]
            and row["balanced_diag_config"]
            and row["direct_pass_gate"]
            and row["public_pass_gate"]
            and row["packet_minus_public"]["ci95_low"] >= min_packet_minus_public_ci_low
            and row["public_minus_target"]["ci95_high"] <= max_public_lift
            for row in rows
        ),
        "runs": len(rows),
        "budget_bytes": budget_bytes,
        "min_packet_accuracy": min(row["packet_accuracy"] for row in rows),
        "max_public_only_accuracy": max(row["public_only_accuracy"] for row in rows),
        "min_packet_minus_public_ci95_low": min(row["packet_minus_public"]["ci95_low"] for row in rows),
        "max_public_minus_target_ci95_high": max(row["public_minus_target"]["ci95_high"] for row in rows),
        "all_same_eval_ids": all(row["same_eval_ids"] for row in rows),
        "all_content_parity": all(row["content_parity"] for row in rows),
        "all_direct_eval_config_matches_public": all(row["direct_eval_config_matches_public"] for row in rows),
        "all_public_train_eval_disjoint": all(row["public_train_eval_disjoint"] for row in rows),
        "all_balanced_diag_config": all(row["balanced_diag_config"] for row in rows),
        "all_direct_pass": all(row["direct_pass_gate"] for row in rows),
        "all_public_no_leak": all(row["public_pass_gate"] for row in rows),
    }
    payload = {
        "gate": "source_private_balanced_diag_packet_gate",
        "headline": headline,
        "rows": rows,
        "pass_rule": (
            f"Budget-{budget_bytes} direct diagnostic packet must pass strict controls; public-only diag receiver "
            f"must have CI95 high <= target+{max_public_lift:.2f}; packet-public CI95 low must be >= "
            f"{min_packet_minus_public_ci_low:.2f}; eval IDs/families/answers must match exactly; public train/eval "
            "IDs must be disjoint; and both runs must use plausible-decoy diag_only config."
        ),
        "interpretation": (
            "Balanced plausible-decoy diagnostic tables remove obvious X-code distractors and public semantic shortcuts. "
            "A direct 2-byte private diagnostic packet remains sufficient, while a trained public-only diagnostic receiver "
            "does not solve the task."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Source-Private Balanced Diagnostic Packet Gate",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- runs: `{headline['runs']}`",
        f"- budget bytes: `{budget_bytes}`",
        f"- min packet accuracy: `{headline['min_packet_accuracy']:.3f}`",
        f"- max public-only accuracy: `{headline['max_public_only_accuracy']:.3f}`",
        f"- min packet-public CI95 low: `{headline['min_packet_minus_public_ci95_low']:.3f}`",
        f"- max public-target CI95 high: `{headline['max_public_minus_target_ci95_high']:.3f}`",
        "",
        "| Direct run | Public-only run | families | n | packet | public | target | best control | packet-public CI | public-target CI | parity |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['direct_dir']}` | `{row['public_dir']}` | "
            f"{row['public_train_family_set']}->{row['public_eval_family_set']} | {row['n']} | "
            f"{row['packet_accuracy']:.3f} | {row['public_only_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | "
            f"[{row['packet_minus_public']['ci95_low']:.3f}, {row['packet_minus_public']['ci95_high']:.3f}] | "
            f"[{row['public_minus_target']['ci95_low']:.3f}, {row['public_minus_target']['ci95_high']:.3f}] | "
            f"`{row['same_eval_ids'] and row['content_parity']}` |"
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
                "# Source-Private Balanced Diagnostic Packet Gate Manifest",
                "",
                f"- pass gate: `{headline['pass_gate']}`",
                f"- runs: `{headline['runs']}`",
                f"- min packet accuracy: `{headline['min_packet_accuracy']:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", type=pathlib.Path, metavar=("DIRECT_DIR", "PUBLIC_DIR"))
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_diag_only_public_ablation_20260430/summary"),
    )
    parser.add_argument("--budget-bytes", type=int, default=2)
    parser.add_argument("--min-packet-minus-public-ci-low", type=float, default=0.10)
    parser.add_argument("--max-public-lift", type=float, default=0.05)
    args = parser.parse_args()
    pairs = [(direct, public) for direct, public in args.pair] if args.pair else list(DEFAULT_PAIRS)
    payload = summarize_pairs(
        pairs=pairs,
        output_dir=args.output_dir,
        budget_bytes=args.budget_bytes,
        min_packet_minus_public_ci_low=args.min_packet_minus_public_ci_low,
        max_public_lift=args.max_public_lift,
    )
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
