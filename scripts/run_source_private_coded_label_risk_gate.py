from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys
from dataclasses import replace
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_hidden_repair_packet_smoke as repair_gate


TRANSFORMS = (
    "baseline",
    "label_rename",
    "diagnostic_code_remap",
    "candidate_pool_permutation",
    "label_code_order_composed",
)

CONDITIONS = (
    "target_only",
    "matched_repair_packet",
    "zero_source",
    "shuffled_source",
    "random_same_byte",
    "answer_only",
    "answer_masked",
    "target_derived_sidecar",
    "structured_json_matched",
    "structured_free_text_matched",
    "diag_masked_full_log",
    "full_hidden_log",
    "full_diag_text",
)

SOURCE_DESTROYING_CONTROLS = (
    "zero_source",
    "shuffled_source",
    "random_same_byte",
    "answer_only",
    "answer_masked",
    "target_derived_sidecar",
)

REVIEWER_NEGATIVE_CONTROLS = (
    "structured_json_matched",
    "structured_free_text_matched",
    "diag_masked_full_log",
)

POSITIVE_ORACLES = (
    "full_hidden_log",
    "full_diag_text",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _candidate_block(candidates: tuple[repair_gate.Candidate, ...]) -> str:
    return "\n".join(
        "- "
        f"{candidate.label}: patch={candidate.patch_name}; intent={candidate.patch_intent}; "
        f"handles_repair_diag={candidate.handles_diagnostic}"
        for candidate in candidates
    )


def _rebuild_target_prompt(example: repair_gate.Example, candidates: tuple[repair_gate.Candidate, ...]) -> str:
    prefix, sep, _ = example.target_prompt.partition("Candidates:\n")
    if not sep:
        raise ValueError(f"cannot rebuild target prompt for {example.example_id}")
    return prefix + "Candidates:\n" + _candidate_block(candidates)


def _rebuild_source_prompt(example: repair_gate.Example, private_log: str) -> str:
    trace_line = next(line for line in private_log.splitlines() if "REPAIR_DIAG=" in line)
    prompt = re.sub(
        r"Private hidden-test log:\n.*?\nPrivate REPAIR_DIAG line copied from the log:",
        f"Private hidden-test log:\n{private_log}\nPrivate REPAIR_DIAG line copied from the log:",
        example.source_prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"Private REPAIR_DIAG line copied from the log: .*\nPacket:",
        f"Private REPAIR_DIAG line copied from the log: {trace_line}\nPacket:",
        prompt,
        flags=re.DOTALL,
    )
    return prompt


def _opaque_label(*, transform: str, seed: int, example_index: int, candidate_index: int) -> str:
    digest = hashlib.sha256(f"{transform}:{seed}:{example_index}:{candidate_index}".encode("utf-8")).hexdigest()[:10]
    return f"opaque_choice_{example_index:04d}_{candidate_index}_{digest}"


def _remapped_diag(*, seed: int, example_index: int) -> str:
    alphabet = repair_gate.DIAG_LETTERS
    return f"{alphabet[(example_index * 7 + seed * 3 + 5) % len(alphabet)]}{(example_index * 3 + seed) % 10}"


def _transform_example(
    example: repair_gate.Example,
    *,
    transform: str,
    seed: int,
    example_index: int,
) -> repair_gate.Example:
    candidates = list(example.candidates)
    answer_original_label = example.answer_label
    diagnostic_code = example.diagnostic_code
    private_log = example.private_test_log

    if transform == "baseline":
        return example

    if transform in {"diagnostic_code_remap", "label_code_order_composed"}:
        diagnostic_code = _remapped_diag(seed=seed, example_index=example_index)
        private_log = re.sub(r"REPAIR_DIAG=[A-Z][0-9]", f"REPAIR_DIAG={diagnostic_code}", private_log)
        remapped: list[repair_gate.Candidate] = []
        for candidate_index, candidate in enumerate(candidates):
            handle = diagnostic_code if candidate.label == answer_original_label else f"wrong_diag_{candidate_index}"
            remapped.append(replace(candidate, handles_diagnostic=handle))
        candidates = remapped

    if transform in {"label_rename", "candidate_pool_permutation", "label_code_order_composed"}:
        label_map: dict[str, str] = {}
        renamed: list[repair_gate.Candidate] = []
        for candidate_index, candidate in enumerate(candidates):
            label = _opaque_label(
                transform=transform,
                seed=seed,
                example_index=example_index,
                candidate_index=candidate_index,
            )
            label_map[candidate.label] = label
            renamed.append(replace(candidate, label=label))
        candidates = renamed
        answer_original_label = label_map[answer_original_label]

    if transform in {"candidate_pool_permutation", "label_code_order_composed"}:
        rng = random.Random(seed * 1000003 + example_index)
        order = list(range(len(candidates)))
        rng.shuffle(order)
        if order == list(range(len(candidates))):
            order = order[1:] + order[:1]
        candidates = [candidates[index] for index in order]

    candidate_tuple = tuple(candidates)
    return replace(
        example,
        private_test_log=private_log,
        source_prompt=_rebuild_source_prompt(example, private_log),
        candidates=candidate_tuple,
        answer_label=answer_original_label,
        diagnostic_code=diagnostic_code,
        target_prompt=_rebuild_target_prompt(example, candidate_tuple),
    )


def transform_examples(
    examples: list[repair_gate.Example],
    *,
    transform: str,
    seed: int,
) -> list[repair_gate.Example]:
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}")
    return [
        _transform_example(example, transform=transform, seed=seed, example_index=example_index)
        for example_index, example in enumerate(examples)
    ]


def _hashes(examples: list[repair_gate.Example]) -> dict[str, str]:
    return {
        "exact_id_sha256": _sha256_lines([example.example_id for example in examples]),
        "label_sha256": _sha256_lines(
            [
                f"{example.example_id}:"
                + ",".join(candidate.label for candidate in example.candidates)
                + f":answer={example.answer_label}"
                for example in examples
            ]
        ),
        "candidate_order_sha256": _sha256_lines(
            [
                f"{example.example_id}:"
                + ",".join(candidate.patch_name for candidate in example.candidates)
                for example in examples
            ]
        ),
        "codebook_sha256": _sha256_lines(
            [
                f"{example.example_id}:{example.diagnostic_code}:"
                + ",".join(f"{candidate.patch_name}:{candidate.handles_diagnostic}" for candidate in example.candidates)
                for example in examples
            ]
        ),
    }


def _condition_prediction(
    *,
    condition: str,
    example: repair_gate.Example,
    examples: list[repair_gate.Example],
    index: int,
    budget_bytes: int,
    rng: random.Random,
) -> dict[str, Any]:
    start = repair_gate.time.perf_counter()
    payload, payload_bytes, payload_tokens, metadata = repair_gate._condition_payload(
        condition=condition,
        example=example,
        examples=examples,
        index=index,
        budget_bytes=budget_bytes,
        rng=rng,
    )
    if condition == "answer_only":
        prediction = payload if payload == example.answer_label else repair_gate._prior_prediction(example)
        decode_metadata = {"answer_exact_match": payload == example.answer_label}
    elif condition in {
        "matched_repair_packet",
        "shuffled_source",
        "random_same_byte",
        "target_derived_sidecar",
        "structured_json_matched",
        "structured_free_text_matched",
        "diag_masked_full_log",
        "full_hidden_log",
        "full_diag_text",
    }:
        prediction, decode_metadata = repair_gate._decode_packet(example, payload)
    else:
        prediction = repair_gate._prior_prediction(example)
        decode_metadata = {}
    return {
        "prediction": prediction,
        "correct": prediction == example.answer_label,
        "payload": payload,
        "payload_bytes": payload_bytes,
        "payload_tokens": payload_tokens,
        "latency_ms": (repair_gate.time.perf_counter() - start) * 1000.0,
        **metadata,
        **decode_metadata,
    }


def _run_condition_table(
    *,
    examples: list[repair_gate.Example],
    seed: int,
    budget_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed + budget_bytes * 1009)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        conditions = {
            condition: _condition_prediction(
                condition=condition,
                example=example,
                examples=examples,
                index=index,
                budget_bytes=budget_bytes,
                rng=rng,
            )
            for condition in CONDITIONS
        }
        rows.append(
            {
                "example_id": example.example_id,
                "family_name": example.family_name,
                "answer_label": example.answer_label,
                "diagnostic_code": example.diagnostic_code,
                "candidate_labels": [candidate.label for candidate in example.candidates],
                "candidate_order": [candidate.patch_name for candidate in example.candidates],
                "candidate_diags": [candidate.handles_diagnostic for candidate in example.candidates],
                "conditions": conditions,
            }
        )
    return rows, _summarize(rows=rows, budget_bytes=budget_bytes)


def _summarize(*, rows: list[dict[str, Any]], budget_bytes: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row["conditions"][condition] for row in rows]
        correct_ids = [row["example_id"] for row in rows if row["conditions"][condition]["correct"]]
        metrics[condition] = {
            "correct": len(correct_ids),
            "accuracy": len(correct_ids) / len(rows),
            "correct_ids": correct_ids,
            "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in condition_rows),
            "max_payload_bytes": max(row["payload_bytes"] for row in condition_rows),
            "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in condition_rows),
            "p50_latency_ms": statistics.median(row["latency_ms"] for row in condition_rows),
        }
    target = metrics["target_only"]["accuracy"]
    matched = metrics["matched_repair_packet"]["accuracy"]
    best_source_destroying = max(metrics[name]["accuracy"] for name in SOURCE_DESTROYING_CONTROLS)
    best_reviewer_negative = max(metrics[name]["accuracy"] for name in REVIEWER_NEGATIVE_CONTROLS)
    min_positive_oracle = min(metrics[name]["accuracy"] for name in POSITIVE_ORACLES)
    pass_gate = (
        matched >= 0.95
        and matched - target >= 0.15
        and best_source_destroying <= target + 0.03
        and best_reviewer_negative <= target + 0.03
        and min_positive_oracle >= matched
    )
    return {
        "budget_bytes": budget_bytes,
        "n": len(rows),
        "target_accuracy": target,
        "matched_accuracy": matched,
        "matched_minus_target": matched - target,
        "best_source_destroying_control_accuracy": best_source_destroying,
        "best_reviewer_negative_control_accuracy": best_reviewer_negative,
        "min_positive_oracle_accuracy": min_positive_oracle,
        "pass_gate": pass_gate,
        "metrics": metrics,
        "pass_rule": (
            "matched packet >= 0.95, matched-target >= 0.15, source-destroying "
            "and reviewer-negative controls <= target+0.03, and positive oracles >= matched"
        ),
    }


def run_gate(
    *,
    examples: int,
    candidates: int,
    family_set: str,
    seeds: list[int],
    budget_bytes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_hashes: dict[int, dict[str, str]] = {}
    for seed in seeds:
        base = repair_gate.make_benchmark(
            examples=examples,
            candidates=candidates,
            seed=seed,
            family_set=family_set,
        )
        for transform in TRANSFORMS:
            transformed = transform_examples(base, transform=transform, seed=seed)
            hashes = _hashes(transformed)
            if transform == "baseline":
                baseline_hashes[seed] = hashes
            predictions, summary = _run_condition_table(
                examples=transformed,
                seed=seed,
                budget_bytes=budget_bytes,
            )
            expected_hash_changes = {
                "exact_id_same_as_baseline": hashes["exact_id_sha256"] == baseline_hashes[seed]["exact_id_sha256"],
                "label_changed_vs_baseline": hashes["label_sha256"] != baseline_hashes[seed]["label_sha256"],
                "candidate_order_changed_vs_baseline": (
                    hashes["candidate_order_sha256"] != baseline_hashes[seed]["candidate_order_sha256"]
                ),
                "codebook_changed_vs_baseline": hashes["codebook_sha256"] != baseline_hashes[seed]["codebook_sha256"],
            }
            if transform == "baseline":
                expected_transform_pass = expected_hash_changes["exact_id_same_as_baseline"]
            elif transform == "label_rename":
                expected_transform_pass = (
                    expected_hash_changes["exact_id_same_as_baseline"]
                    and expected_hash_changes["label_changed_vs_baseline"]
                    and not expected_hash_changes["candidate_order_changed_vs_baseline"]
                )
            elif transform == "diagnostic_code_remap":
                expected_transform_pass = (
                    expected_hash_changes["exact_id_same_as_baseline"]
                    and not expected_hash_changes["label_changed_vs_baseline"]
                    and expected_hash_changes["codebook_changed_vs_baseline"]
                )
            elif transform == "candidate_pool_permutation":
                expected_transform_pass = (
                    expected_hash_changes["exact_id_same_as_baseline"]
                    and expected_hash_changes["label_changed_vs_baseline"]
                    and expected_hash_changes["candidate_order_changed_vs_baseline"]
                )
            else:
                expected_transform_pass = (
                    expected_hash_changes["exact_id_same_as_baseline"]
                    and expected_hash_changes["label_changed_vs_baseline"]
                    and expected_hash_changes["candidate_order_changed_vs_baseline"]
                    and expected_hash_changes["codebook_changed_vs_baseline"]
                )
            rows.append(
                {
                    "seed": seed,
                    "transform": transform,
                    "hashes": hashes,
                    "hash_change_checks": expected_hash_changes,
                    "hash_gate": expected_transform_pass,
                    "summary": summary,
                    "pass_gate": summary["pass_gate"] and expected_transform_pass,
                }
            )
            for prediction in predictions:
                prediction_rows.append(
                    {
                        "seed": seed,
                        "transform": transform,
                        **prediction,
                    }
                )
    by_transform: dict[str, dict[str, Any]] = {}
    for transform in TRANSFORMS:
        transform_rows = [row for row in rows if row["transform"] == transform]
        by_transform[transform] = {
            "pass_gate": all(row["pass_gate"] for row in transform_rows),
            "min_matched_accuracy": min(row["summary"]["matched_accuracy"] for row in transform_rows),
            "max_target_accuracy": max(row["summary"]["target_accuracy"] for row in transform_rows),
            "max_best_source_destroying_control": max(
                row["summary"]["best_source_destroying_control_accuracy"] for row in transform_rows
            ),
            "max_best_reviewer_negative_control": max(
                row["summary"]["best_reviewer_negative_control_accuracy"] for row in transform_rows
            ),
        }
    payload = {
        "gate": "source_private_coded_label_risk_gate",
        "examples": examples,
        "candidates": candidates,
        "family_set": family_set,
        "seeds": seeds,
        "budget_bytes": budget_bytes,
        "transforms": list(TRANSFORMS),
        "rows": rows,
        "by_transform": by_transform,
        "pass_gate": all(row["pass_gate"] for row in rows),
        "pass_rule": (
            "For every seed and transform, exact IDs must stay fixed; the intended "
            "surface hash must change; matched 2-byte packet accuracy must be >=0.95; "
            "source-destroying and reviewer-negative controls must remain within +0.03 "
            "of target-only; and positive oracles must not trail the packet."
        ),
    }
    return payload | {"prediction_rows": prediction_rows}


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Coded-Label Risk Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- examples per seed: `{payload['examples']}`",
        f"- seeds: `{payload['seeds']}`",
        f"- budget bytes: `{payload['budget_bytes']}`",
        "",
        "| Seed | Transform | Pass | Hash gate | Matched | Target | Source controls | Reviewer negatives | Oracles |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        summary = row["summary"]
        lines.append(
            "| "
            f"{row['seed']} | `{row['transform']}` | `{row['pass_gate']}` | `{row['hash_gate']}` | "
            f"{summary['matched_accuracy']:.3f} | {summary['target_accuracy']:.3f} | "
            f"{summary['best_source_destroying_control_accuracy']:.3f} | "
            f"{summary['best_reviewer_negative_control_accuracy']:.3f} | "
            f"{summary['min_positive_oracle_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Transform Summary",
            "",
            "| Transform | Pass | Min matched | Max target | Max source control | Max reviewer negative |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for transform, summary in payload["by_transform"].items():
        lines.append(
            "| "
            f"`{transform}` | `{summary['pass_gate']}` | {summary['min_matched_accuracy']:.3f} | "
            f"{summary['max_target_accuracy']:.3f} | {summary['max_best_source_destroying_control']:.3f} | "
            f"{summary['max_best_reviewer_negative_control']:.3f} |"
        )
    lines.extend(["", f"Pass rule: {payload['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=160)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--seeds", default="29,31,37")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_with_predictions = run_gate(
        examples=args.examples,
        candidates=args.candidates,
        family_set=args.family_set,
        seeds=seeds,
        budget_bytes=args.budget,
    )
    prediction_rows = payload_with_predictions.pop("prediction_rows")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload_with_predictions, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload_with_predictions)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)

    artifacts = ["summary.json", "summary.md", "predictions.jsonl", "manifest.json", "manifest.md"]
    command = " ".join(
        [
            "./venv_arm64/bin/python",
            "scripts/run_source_private_coded_label_risk_gate.py",
            f"--examples {args.examples}",
            f"--candidates {args.candidates}",
            f"--family-set {args.family_set}",
            f"--seeds {args.seeds}",
            f"--budget {args.budget}",
            f"--output-dir {args.output_dir}",
        ]
    )
    manifest = {
        "command": command,
        "args": vars(args) | {"output_dir": str(args.output_dir), "seeds": seeds},
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
            "pass_gate": payload_with_predictions["pass_gate"],
            "examples": payload_with_predictions["examples"],
            "seeds": payload_with_predictions["seeds"],
            "transforms": payload_with_predictions["transforms"],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Coded-Label Risk Gate Manifest",
                "",
                "## Command",
                "",
                "```bash",
                command,
                "```",
                "",
                "## Outcome",
                "",
                f"- pass gate: `{payload_with_predictions['pass_gate']}`",
                f"- examples per seed: `{payload_with_predictions['examples']}`",
                f"- transforms: `{payload_with_predictions['transforms']}`",
                "",
                "## Artifacts",
                "",
                *[f"- `{artifact}`" for artifact in artifacts],
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not payload_with_predictions["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
