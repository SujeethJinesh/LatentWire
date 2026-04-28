from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_hidden_repair_packet_smoke as repair_gate


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _codebook_hash(examples: list[repair_gate.Example]) -> str:
    lines: list[str] = []
    for example in examples:
        candidate_map = ",".join(
            f"{candidate.label}:{candidate.handles_diagnostic}" for candidate in example.candidates
        )
        lines.append(f"{example.example_id}:{example.diagnostic_code}:{candidate_map}")
    return _sha256_lines(lines)


def _public_surface_hash(examples: list[repair_gate.Example]) -> str:
    lines = [
        f"{example.example_id}:{example.family_name}:{example.answer_label}:"
        + ",".join(candidate.label for candidate in example.candidates)
        for example in examples
    ]
    return _sha256_lines(lines)


def _diagnostic_values(examples: list[repair_gate.Example]) -> list[str]:
    return [example.diagnostic_code for example in examples]


def _compact_budget_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    return {
        "budget_bytes": summary["budget_bytes"],
        "pass_gate": summary["pass_gate"],
        "matched": metrics["matched_repair_packet"]["accuracy"],
        "best_no_source": summary["best_no_source_accuracy"],
        "best_source_destroying_control": summary["best_source_destroying_control_accuracy"],
        "best_reviewer_negative_control": summary["best_reviewer_negative_control_accuracy"],
        "min_reviewer_positive_oracle": summary["min_reviewer_positive_oracle_accuracy"],
        "structured_json": metrics["structured_json_matched"]["accuracy"],
        "structured_free_text": metrics["structured_free_text_matched"]["accuracy"],
        "diag_masked_full_log": metrics["diag_masked_full_log"]["accuracy"],
        "full_diag_text": metrics["full_diag_text"]["accuracy"],
    }


def run_gate(
    *,
    examples: int,
    candidates: int,
    family_set: str,
    seeds: list[int],
    budgets: list[int],
) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    public_hashes: list[str] = []
    codebook_hashes: list[str] = []
    for seed in seeds:
        benchmark = repair_gate.make_benchmark(
            examples=examples,
            candidates=candidates,
            seed=seed,
            family_set=family_set,
        )
        public_hashes.append(_public_surface_hash(benchmark))
        codebook_hashes.append(_codebook_hash(benchmark))
        budget_rows: list[dict[str, Any]] = []
        for budget in budgets:
            _, summary = repair_gate.run_budget(examples=benchmark, seed=seed, budget_bytes=budget)
            budget_rows.append(_compact_budget_summary(summary))
        seed_rows.append(
            {
                "seed": seed,
                "n": len(benchmark),
                "exact_id_sha256": _sha256_lines([example.example_id for example in benchmark]),
                "public_surface_sha256": public_hashes[-1],
                "codebook_sha256": codebook_hashes[-1],
                "diagnostic_preview": _diagnostic_values(benchmark)[:12],
                "budget_summaries": budget_rows,
                "pass_gate": all(row["pass_gate"] for row in budget_rows),
            }
        )
    exact_id_hashes = {row["exact_id_sha256"] for row in seed_rows}
    public_surface_hashes = set(public_hashes)
    unique_codebooks = len(set(codebook_hashes))
    return {
        "examples": examples,
        "family_set": family_set,
        "seeds": seeds,
        "budgets": budgets,
        "seed_rows": seed_rows,
        "exact_id_parity_across_seeds": len(exact_id_hashes) == 1,
        "public_surface_parity_across_seeds": len(public_surface_hashes) == 1,
        "unique_codebook_count": unique_codebooks,
        "codebook_remapped": unique_codebooks == len(seeds),
        "pass_gate": (
            all(row["pass_gate"] for row in seed_rows)
            and len(exact_id_hashes) == 1
            and len(public_surface_hashes) == 1
            and unique_codebooks == len(seeds)
        ),
        "pass_rule": (
            "Every seed/budget must pass the hidden-repair packet gate; exact IDs and public "
            "candidate labels must remain identical across seeds; diagnostic codebooks must differ."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Codebook-Remap Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- examples: `{payload['examples']}`",
        f"- family set: `{payload['family_set']}`",
        f"- seeds: `{payload['seeds']}`",
        f"- budgets: `{payload['budgets']}`",
        f"- exact ID parity across seeds: `{payload['exact_id_parity_across_seeds']}`",
        f"- public surface parity across seeds: `{payload['public_surface_parity_across_seeds']}`",
        f"- unique codebooks: `{payload['unique_codebook_count']}`",
        "",
        "| Seed | Budget | Pass | Matched | No-source | Source controls | Reviewer negatives | Oracles | JSON | Free text | Diag masked |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_row in payload["seed_rows"]:
        for row in seed_row["budget_summaries"]:
            lines.append(
                "| "
                f"{seed_row['seed']} | {row['budget_bytes']} | `{row['pass_gate']}` | "
                f"{row['matched']:.3f} | {row['best_no_source']:.3f} | "
                f"{row['best_source_destroying_control']:.3f} | "
                f"{row['best_reviewer_negative_control']:.3f} | "
                f"{row['min_reviewer_positive_oracle']:.3f} | "
                f"{row['structured_json']:.3f} | {row['structured_free_text']:.3f} | "
                f"{row['diag_masked_full_log']:.3f} |"
            )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--seeds", default="29,31,37")
    parser.add_argument("--budgets", default="2,4,8,16")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    budgets = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_gate(
        examples=args.examples,
        candidates=args.candidates,
        family_set=args.family_set,
        seeds=seeds,
        budgets=budgets,
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
    artifacts = ["summary.json", "summary.md", "manifest.json", "manifest.md"]
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_codebook_remap_gate.py",
                f"--examples {args.examples}",
                f"--candidates {args.candidates}",
                f"--family-set {args.family_set}",
                f"--seeds {args.seeds}",
                f"--budgets {args.budgets}",
                f"--output-dir {args.output_dir}",
            ]
        ),
        "args": vars(args) | {"output_dir": str(args.output_dir), "seeds": seeds, "budgets": budgets},
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "python": sys.version,
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": {
            "pass_gate": payload["pass_gate"],
            "examples": payload["examples"],
            "seeds": payload["seeds"],
            "budgets": payload["budgets"],
            "unique_codebook_count": payload["unique_codebook_count"],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Source-Private Codebook-Remap Gate Manifest",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Outcome",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- examples: `{payload['examples']}`",
        f"- unique codebooks: `{payload['unique_codebook_count']}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts)
    (output_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
