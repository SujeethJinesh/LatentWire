#!/usr/bin/env python3
"""Build candidate pools for condition-specific receiver likelihood controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


CONDITIONS = ("matched", "zero_source", "shuffled_source", "label_shuffle", "target_only", "slots_only")


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _answer_list(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [str(answer)]


def _is_correct_for_current_answer(prediction: str, current_row: dict[str, Any]) -> bool:
    return harness._generation_match(str(prediction), _answer_list(current_row))


def _currentized_candidate(
    *,
    current: dict[str, Any],
    donor: dict[str, Any],
    method: str,
    condition: str,
    donor_example_id: str | None = None,
) -> dict[str, Any]:
    prediction = str(donor.get("prediction", ""))
    normalized_prediction = donor.get("normalized_prediction")
    if normalized_prediction is None:
        normalized_prediction = harness._extract_prediction_numeric_answer(prediction)
    row = dict(current)
    row.update(
        {
            "method": method,
            "prediction": prediction,
            "normalized_prediction": "" if normalized_prediction is None else str(normalized_prediction),
            "correct": _is_correct_for_current_answer(prediction, current),
            "control_condition": condition,
        }
    )
    if donor_example_id is not None:
        row["control_donor_example_id"] = donor_example_id
    for key in ("generated_tokens", "raw_target_token_count", "target_prompt_token_count"):
        if key in donor:
            row[key] = donor[key]
    return row


def _blank_candidate(*, current: dict[str, Any], method: str, condition: str) -> dict[str, Any]:
    row = dict(current)
    row.update(
        {
            "method": method,
            "prediction": "",
            "normalized_prediction": "",
            "correct": False,
            "control_condition": condition,
        }
    )
    return row


def _ordered(rows_by_id: dict[str, dict[str, Any]], reference_ids: Sequence[str]) -> list[dict[str, Any]]:
    missing = [example_id for example_id in reference_ids if example_id not in rows_by_id]
    if missing:
        raise ValueError(f"Missing IDs: {missing[:5]}")
    return [dict(rows_by_id[example_id]) for example_id in reference_ids]


def build(args: argparse.Namespace) -> dict[str, Any]:
    target_path = _resolve(args.target_jsonl)
    text_path = _resolve(args.text_jsonl)
    source_path = _resolve(args.source_jsonl)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = _read_jsonl(target_path)
    text_rows = _read_jsonl(text_path)
    source_rows = _read_jsonl(source_path)
    target_by_id = _by_id(target_rows)
    text_by_id = _by_id(text_rows)
    source_by_id = _by_id(source_rows)
    reference_ids = [str(row["example_id"]) for row in target_rows]
    text_ordered = _ordered(text_by_id, reference_ids)
    source_ordered = _ordered(source_by_id, reference_ids)

    condition_outputs: dict[str, dict[str, str]] = {}
    condition_output_sha256: dict[str, dict[str, str]] = {}

    def write_condition(condition: str, label_rows: dict[str, list[dict[str, Any]]]) -> None:
        condition_outputs[condition] = {}
        condition_output_sha256[condition] = {}
        condition_dir = output_dir / condition
        for label, rows in label_rows.items():
            path = condition_dir / f"{label}.jsonl"
            _write_jsonl(path, rows)
            condition_outputs[condition][label] = _display_path(path)
            condition_output_sha256[condition][label] = _sha256_file(path)

    write_condition("matched", {"target": target_rows, "text": text_ordered, "source": source_ordered})

    write_condition(
        "target_only",
        {
            "target": [dict(row, control_condition="target_only") for row in target_rows],
            "text": [
                _currentized_candidate(
                    current=target_by_id[example_id],
                    donor=target_by_id[example_id],
                    method="target_only_text_candidate",
                    condition="target_only",
                    donor_example_id=example_id,
                )
                for example_id in reference_ids
            ],
            "source": [
                _currentized_candidate(
                    current=target_by_id[example_id],
                    donor=target_by_id[example_id],
                    method="target_only_source_candidate",
                    condition="target_only",
                    donor_example_id=example_id,
                )
                for example_id in reference_ids
            ],
        },
    )

    write_condition(
        "zero_source",
        {
            "target": [dict(row, control_condition="zero_source") for row in target_rows],
            "text": [dict(row, control_condition="zero_source") for row in text_ordered],
            "source": [
                _blank_candidate(current=target_by_id[example_id], method="zero_source_candidate", condition="zero_source")
                for example_id in reference_ids
            ],
        },
    )

    write_condition(
        "slots_only",
        {
            "target": [dict(row, control_condition="slots_only") for row in target_rows],
            "text": [
                _blank_candidate(current=target_by_id[example_id], method="slots_only_text_candidate", condition="slots_only")
                for example_id in reference_ids
            ],
            "source": [
                _blank_candidate(current=target_by_id[example_id], method="slots_only_source_candidate", condition="slots_only")
                for example_id in reference_ids
            ],
        },
    )

    rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(reference_ids):
        donor = source_ordered[(index + int(args.shuffle_offset)) % len(source_ordered)]
        rows.append(
            _currentized_candidate(
                current=target_by_id[example_id],
                donor=donor,
                method="shuffled_source_candidate",
                condition="shuffled_source",
                donor_example_id=str(donor["example_id"]),
            )
        )
    write_condition(
        "shuffled_source",
        {
            "target": [dict(row, control_condition="shuffled_source") for row in target_rows],
            "text": [dict(row, control_condition="shuffled_source") for row in text_ordered],
            "source": rows,
        },
    )

    label_target_rows: list[dict[str, Any]] = []
    label_text_rows: list[dict[str, Any]] = []
    label_source_rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(reference_ids):
        del index
        current = target_by_id[example_id]
        # Permute slot labels before receiver scoring. Source content occupies
        # the target-labeled slot, target content occupies the source-labeled
        # slot, and text remains text.
        label_target_rows.append(
            _currentized_candidate(
                current=current,
                donor=source_by_id[example_id],
                method="label_shuffle_target_candidate",
                condition="label_shuffle",
                donor_example_id=example_id,
            )
        )
        label_text_rows.append(dict(text_by_id[example_id], control_condition="label_shuffle"))
        label_source_rows.append(
            _currentized_candidate(
                current=current,
                donor=target_by_id[example_id],
                method="label_shuffle_source_candidate",
                condition="label_shuffle",
                donor_example_id=example_id,
            )
        )
    write_condition(
        "label_shuffle",
        {
            "target": label_target_rows,
            "text": label_text_rows,
            "source": label_source_rows,
        },
    )

    manifest_path = output_dir / "manifest.json"
    ordered_ids_text = "\n".join(reference_ids) + "\n"
    payload = {
        "date": str(args.date),
        "status": "condition_candidate_pools_built",
        "git_commit": _git_commit(),
        "target_jsonl": _display_path(target_path),
        "target_jsonl_sha256": _sha256_file(target_path),
        "text_jsonl": _display_path(text_path),
        "text_jsonl_sha256": _sha256_file(text_path),
        "source_jsonl": _display_path(source_path),
        "source_jsonl_sha256": _sha256_file(source_path),
        "output_dir": _display_path(output_dir),
        "reference_n": len(reference_ids),
        "reference_ids": reference_ids,
        "reference_ids_sha256": hashlib.sha256(ordered_ids_text.encode("utf-8")).hexdigest(),
        "shuffle_offset": int(args.shuffle_offset),
        "label_shuffle_offset": int(args.label_shuffle_offset),
        "condition_outputs": condition_outputs,
        "condition_output_sha256": condition_output_sha256,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = output_dir / "manifest.md"
    _write_markdown(md_path, payload)
    print(json.dumps({"status": payload["status"], "manifest": _display_path(manifest_path)}, indent=2))
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Condition Likelihood Candidate Pools",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload['git_commit'] or 'unknown'}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- reference IDs sha256: `{payload['reference_ids_sha256']}`",
        f"- shuffle offset: `{payload['shuffle_offset']}`",
        f"- label-shuffle offset: `{payload['label_shuffle_offset']}`",
        "",
        "## Conditions",
        "",
    ]
    for condition, labels in payload["condition_outputs"].items():
        lines.append(f"### {condition}")
        lines.append("")
        for label, output in labels.items():
            digest = payload["condition_output_sha256"][condition][label]
            lines.append(f"- `{label}`: `{output}` sha256 `{digest}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-jsonl", required=True)
    parser.add_argument("--text-jsonl", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--label-shuffle-offset", type=int, default=17)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return build(parse_args(argv))


if __name__ == "__main__":
    main()
