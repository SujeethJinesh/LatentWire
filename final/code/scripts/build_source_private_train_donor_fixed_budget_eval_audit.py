from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_train_donor_antishuffle_fixed12b_eval_audit_20260501")
DEFAULT_EVAL_RUNS = (
    pathlib.Path("results/source_private_train_donor_antishuffle_seed47_n512_budget12_cross"),
    pathlib.Path("results/source_private_train_donor_antishuffle_seed53_n512_budget12_14"),
    pathlib.Path("results/source_private_train_donor_antishuffle_seed59_n512_budget12_14"),
)
DEFAULT_DIRECTIONS = ("core_to_holdout", "holdout_to_core")

CSV_COLUMNS = (
    "seed",
    "direction",
    "budget_bytes",
    "n",
    "candidate_accuracy",
    "base_accuracy",
    "target_accuracy",
    "best_control_name",
    "best_control_accuracy",
    "candidate_minus_base",
    "paired_ci95_low_vs_base",
    "controls_ok",
    "pass_gate",
    "source_path",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_path(path: pathlib.Path) -> pathlib.Path:
    resolved = _resolve(path)
    if resolved.is_dir():
        return resolved / "candidate_conditioned_packet_builder_smoke.json"
    return resolved


def _read_run(path: pathlib.Path) -> dict[str, Any]:
    summary = _summary_path(path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["_summary_path"] = summary
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _compact_row(seed: int, row: dict[str, Any], source_path: pathlib.Path) -> dict[str, Any]:
    return {
        "seed": seed,
        "direction": row["direction"],
        "budget_bytes": row["budget_bytes"],
        "n": row["n"],
        "candidate_accuracy": row["candidate_conditioned_packet_accuracy"],
        "base_accuracy": row["base_matched_accuracy"],
        "target_accuracy": row["target_accuracy"],
        "best_control_name": row["best_control_name"],
        "best_control_accuracy": row["best_control_accuracy"],
        "candidate_minus_base": row["candidate_minus_base"],
        "paired_ci95_low_vs_base": row["paired_ci95_low_vs_base"],
        "controls_ok": row["controls_ok"],
        "pass_gate": row["pass_gate"],
        "source_path": _rel(source_path),
    }


def _headline(rows: list[dict[str, Any]], *, budget: int, expected_rows: int) -> dict[str, Any]:
    pass_rows = sum(1 for row in rows if row["pass_gate"])
    return {
        "budget_bytes": budget,
        "pass_gate": len(rows) == expected_rows and pass_rows == len(rows),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "pass_rows": pass_rows,
        "seeds": sorted({row["seed"] for row in rows}),
        "directions": sorted({row["direction"] for row in rows}),
        "min_candidate_accuracy": min((float(row["candidate_accuracy"]) for row in rows), default=None),
        "max_best_control_accuracy": max((float(row["best_control_accuracy"]) for row in rows), default=None),
        "min_paired_ci95_low_vs_base": min((float(row["paired_ci95_low_vs_base"]) for row in rows), default=None),
        "min_candidate_minus_base": min((float(row["candidate_minus_base"]) for row in rows), default=None),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Train-Donor Anti-Shuffle Fixed-Budget Eval Audit",
        "",
        f"- budget: `{headline['budget_bytes']}B`",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- passing rows: `{headline['pass_rows']}/{headline['rows']}`",
        f"- min CI95 low vs base: `{_fmt(headline['min_paired_ci95_low_vs_base'])}`",
        "",
        "## Rows",
        "",
        "| Seed | Direction | Budget | Candidate | Base | Target | Best control | CI95 low vs base | Pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["direction"]),
                    str(row["budget_bytes"]),
                    _fmt(row["candidate_accuracy"]),
                    _fmt(row["base_accuracy"]),
                    _fmt(row["target_accuracy"]),
                    _fmt(row["best_control_accuracy"]),
                    _fmt(row["paired_ci95_low_vs_base"]),
                    _fmt(row["pass_gate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_fixed_budget_eval_audit(
    *,
    eval_runs: list[pathlib.Path],
    budget: int,
    directions: tuple[str, ...],
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for path in eval_runs:
        run = _read_run(path)
        seed = int(run["seed"])
        summary_path = run["_summary_path"]
        source_hashes[_rel(summary_path)] = _sha256_file(summary_path)
        for row in run.get("rows", []):
            if int(row.get("budget_bytes", -1)) == budget and row.get("direction") in directions:
                rows.append(_compact_row(seed, row, summary_path))
    rows.sort(key=lambda row: (row["seed"], row["direction"]))
    expected_rows = len(eval_runs) * len(directions)
    payload = {
        "gate": "source_private_train_donor_fixed_budget_eval_audit",
        "pass_gate": False,
        "budget_bytes": budget,
        "directions": list(directions),
        "eval_runs": [_rel(path) for path in eval_runs],
        "eval_run_sha256": source_hashes,
        "rows": rows,
        "headline": _headline(rows, budget=budget, expected_rows=expected_rows),
        "interpretation": (
            "This is an eval-only fixed-budget audit. It shows whether a predeclared byte rate "
            "passes all n512 cross-family seed rows, but it does not by itself justify how that "
            "rate was selected before final eval."
        ),
    }
    payload["pass_gate"] = payload["headline"]["pass_gate"]

    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "fixed_budget_eval_audit.json"
    csv_path = output / "fixed_budget_eval_audit.csv"
    md_path = output / "fixed_budget_eval_audit.md"
    manifest_path = output / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": payload["headline"],
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name, "manifest.md"],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "manifest.md").write_text(
        "\n".join(
            [
                "# Fixed-Budget Eval Audit Manifest",
                "",
                f"- budget: `{budget}B`",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- rows: `{payload['headline']['pass_rows']}/{payload['headline']['rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", action="append", type=pathlib.Path, default=None)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_fixed_budget_eval_audit(
        eval_runs=args.eval_run or list(DEFAULT_EVAL_RUNS),
        budget=args.budget,
        directions=tuple(args.directions),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(_resolve(args.output_dir)),
                "pass_gate": payload["pass_gate"],
                "headline": payload["headline"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
