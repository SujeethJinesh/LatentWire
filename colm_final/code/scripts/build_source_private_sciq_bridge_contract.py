from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import random
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_sciq_bridge_contract_20260501")
DEFAULT_HF_CACHE = pathlib.Path(".debug/hf_datasets")
DEFAULT_HF_DATASET = "sciq"
DEFAULT_OFFICIAL_SPLITS = ("train", "validation", "test")
EXPECTED_OFFICIAL_COUNTS = {"train": 11679, "validation": 1000, "test": 1000}
FORBIDDEN_SOURCE_FIELDS = (
    "answer",
    "answerKey",
    "answer_index",
    "answer_label",
    "correct_answer",
    "gold",
    "support",
)
REQUIRED_PUBLIC_CONTROLS = (
    "target_only",
    "zero_source",
    "matched_source_private_packet",
    "shuffled_source_packet",
    "random_same_byte_packet",
    "target_derived_sidecar",
    "answer_only_text_forbidden_oracle",
    "same_byte_structured_text",
    "label_permutation",
    "candidate_derangement",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_id(question: str, choices: list[str]) -> str:
    payload = json.dumps({"question": question, "choices": choices}, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _choice_shuffle_seed(raw: dict[str, Any]) -> int:
    parts = [
        str(raw.get("question", "")).strip(),
        str(raw.get("correct_answer", "")).strip(),
        str(raw.get("distractor1", "")).strip(),
        str(raw.get("distractor2", "")).strip(),
        str(raw.get("distractor3", "")).strip(),
    ]
    return int(hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def canonical_sciq_row(raw: dict[str, Any], *, source_name: str, row_index: int) -> dict[str, Any]:
    question = str(raw["question"]).strip()
    correct = str(raw["correct_answer"]).strip()
    choices = [
        correct,
        str(raw["distractor1"]).strip(),
        str(raw["distractor2"]).strip(),
        str(raw["distractor3"]).strip(),
    ]
    rng = random.Random(_choice_shuffle_seed(raw))
    shuffled = list(choices)
    rng.shuffle(shuffled)
    answer_index = shuffled.index(correct)
    labels = [chr(ord("A") + index) for index in range(len(shuffled))]
    cid = _content_id(question, shuffled)
    return {
        "id": str(raw.get("id") or f"sciq_{cid[:16]}"),
        "content_id": cid,
        "source_name": source_name,
        "row_index": int(row_index),
        "question": question,
        "choices": shuffled,
        "choice_labels": labels,
        "answer_index": int(answer_index),
        "answer_label": labels[answer_index],
    }


def _write_canonical_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_official_split(*, split: str, hf_dataset: str, cache_dir: pathlib.Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(hf_dataset, split=split, cache_dir=str(cache_dir))
    return [
        canonical_sciq_row(dict(row), source_name=f"{hf_dataset}/{split}", row_index=index + 1)
        for index, row in enumerate(dataset)
    ]


def _slice_summary(name: str, rows: list[dict[str, Any]], *, path: pathlib.Path | None = None) -> dict[str, Any]:
    content_ids = [row["content_id"] for row in rows]
    duplicate_counts = {cid: count for cid, count in collections.Counter(content_ids).items() if count > 1}
    duplicate_choice_rows = [
        str(row["id"]) for row in rows if len(set(str(choice) for choice in row["choices"])) != len(row["choices"])
    ]
    choice_count_counts = collections.Counter(str(len(row["choices"])) for row in rows)
    answer_index_counts = collections.Counter(str(row["answer_index"]) for row in rows)
    answer_label_counts = collections.Counter(str(row["answer_label"]) for row in rows)
    schema_valid = (
        len(rows) > 0
        and all(len(row["choices"]) == 4 for row in rows)
        and all(0 <= int(row["answer_index"]) < len(row["choices"]) for row in rows)
    )
    summary = {
        "name": name,
        "path": _display_path(path) if path is not None else None,
        "sha256": _sha256_file(path) if path is not None and path.exists() else None,
        "n": len(rows),
        "valid": schema_valid,
        "unique_content_ids": len(set(content_ids)),
        "choice_count_counts": dict(sorted(choice_count_counts.items())),
        "answer_index_counts": dict(sorted(answer_index_counts.items())),
        "answer_label_counts": dict(sorted(answer_label_counts.items())),
        "duplicate_content_ids": sorted(duplicate_counts),
        "duplicate_choice_row_count": len(duplicate_choice_rows),
        "duplicate_choice_row_ids_sample": duplicate_choice_rows[:10],
        "content_digest": hashlib.sha256("\n".join(content_ids).encode("utf-8")).hexdigest(),
        "first_ids": [row["id"] for row in rows[:3]],
        "last_ids": [row["id"] for row in rows[-3:]],
    }
    if path is not None and path.exists():
        summary["bytes"] = path.stat().st_size
    return summary


def _overlap_matrix(slices: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    sets = {name: {row["content_id"] for row in rows} for name, rows in slices.items()}
    return {
        left: {right: len(sets[left] & sets[right]) for right in sorted(sets)}
        for left in sorted(sets)
    }


def _method_contract() -> dict[str, Any]:
    return {
        "fixed_packet_budget_bytes": 12,
        "source_visible_fields": ["question", "choices"],
        "source_private_allowed_fields": [
            "source model trace generated from question+choices only",
            "rate-capped source-private packet bytes",
        ],
        "forbidden_source_fields": list(FORBIDDEN_SOURCE_FIELDS),
        "required_controls": list(REQUIRED_PUBLIC_CONTROLS),
        "claim_boundary": (
            "This contract freezes the SciQ bridge and controls. It is not a positive second-benchmark "
            "result until the fixed-12B packet beats target-only and destructive controls on validation/test."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private SciQ Bridge Contract",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- public result ready: `{payload['public_benchmark_result_ready']}`",
        f"- fixed packet budget: `{payload['method_contract']['fixed_packet_budget_bytes']}B`",
        "",
        "## Official Splits",
        "",
        "| Split | Rows | Expected | Valid | Duplicate choice rows | Materialized Path | Answer Index Counts |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, summary in payload["official_summaries"].items():
        expected = EXPECTED_OFFICIAL_COUNTS.get(name)
        lines.append(
            f"| `{name}` | {summary['n']} | {expected} | `{summary['valid']}` | "
            f"{summary['duplicate_choice_row_count']} | `{summary['path']}` | "
            f"`{summary['answer_index_counts']}` |"
        )
    lines.extend(["", "## Required Controls", ""])
    for control in payload["method_contract"]["required_controls"]:
        lines.append(f"- `{control}`")
    lines.extend(["", "## Next Gate", "", payload["next_exact_gate"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_contract(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    official_output_dir: pathlib.Path | None = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_cache_dir: pathlib.Path = DEFAULT_HF_CACHE,
    run_date: str = str(dt.date.today()),
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    official_root = _resolve(official_output_dir) if official_output_dir else output_dir / "official_splits"
    cache_dir = _resolve(hf_cache_dir)

    official_slices: dict[str, list[dict[str, Any]]] = {}
    official_paths: dict[str, pathlib.Path] = {}
    for split in DEFAULT_OFFICIAL_SPLITS:
        rows = _load_official_split(split=split, hf_dataset=hf_dataset, cache_dir=cache_dir)
        path = official_root / f"sciq_{split}.jsonl"
        _write_canonical_jsonl(path, rows)
        official_slices[split] = rows
        official_paths[split] = path

    official_summaries = {
        split: _slice_summary(split, rows, path=official_paths.get(split))
        for split, rows in official_slices.items()
    }
    official_overlap = _overlap_matrix(official_slices)
    official_counts_match = all(
        official_summaries[split]["n"] == expected
        for split, expected in EXPECTED_OFFICIAL_COUNTS.items()
        if split in official_summaries
    )
    official_disjoint = all(
        official_overlap[left][right] == 0
        for left in official_overlap
        for right in official_overlap[left]
        if left != right
    )
    schema_valid = all(summary["valid"] for summary in official_summaries.values())
    no_duplicate_content = all(not summary["duplicate_content_ids"] for summary in official_summaries.values())
    duplicate_choice_count = int(
        sum(summary["duplicate_choice_row_count"] for summary in official_summaries.values())
    )
    pass_gate = (
        len(official_summaries) == len(DEFAULT_OFFICIAL_SPLITS)
        and official_counts_match
        and official_disjoint
        and schema_valid
        and no_duplicate_content
    )
    payload = {
        "gate": "source_private_sciq_bridge_contract",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": bool(pass_gate),
        "public_benchmark_result_ready": False,
        "readiness": "SciQ bridge contract is ready" if pass_gate else "SciQ bridge contract has blockers",
        "official_summaries": official_summaries,
        "official_overlap_matrix": official_overlap,
        "checks": {
            "official_splits_materialized": True,
            "official_counts_match_expected": official_counts_match,
            "official_splits_disjoint": official_disjoint,
            "official_schema_valid": schema_valid,
            "official_no_duplicate_content": no_duplicate_content,
            "duplicate_choice_row_count": duplicate_choice_count,
            "source_answer_fields_forbidden": True,
            "support_context_forbidden_for_this_gate": True,
        },
        "method_contract": _method_contract(),
        "next_exact_gate": (
            "Run the fixed-12B source-private packet on SciQ validation/test with the same "
            "target-only, source-shuffle, random, target-derived, same-byte text, label-permutation, "
            "and candidate-derangement controls used on ARC. Do not consume correct_answer or support "
            "when building source packets."
        ),
        "sources": {
            "sciq_paper": "https://aclanthology.org/W17-4413/",
            "sciq_arxiv": "https://arxiv.org/abs/1707.06209",
            "hf_dataset": f"https://huggingface.co/datasets/{hf_dataset}",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_json = output_dir / "sciq_bridge_contract.json"
    contract_md = output_dir / "sciq_bridge_contract.md"
    manifest_json = output_dir / "manifest.json"
    contract_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(contract_md, payload)
    manifest = {
        "created_utc": payload["created_utc"],
        "files": {
            _display_path(contract_json): _sha256_file(contract_json),
            _display_path(contract_md): _sha256_file(contract_md),
        },
    }
    for path in official_paths.values():
        manifest["files"][_display_path(path)] = _sha256_file(path)
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest"] = _display_path(manifest_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the source-private SciQ bridge contract.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official-output-dir", type=pathlib.Path)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-cache-dir", type=pathlib.Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--run-date", default=str(dt.date.today()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_contract(
        output_dir=args.output_dir,
        official_output_dir=args.official_output_dir,
        hf_dataset=args.hf_dataset,
        hf_cache_dir=args.hf_cache_dir,
        run_date=args.run_date,
    )


if __name__ == "__main__":
    main()
