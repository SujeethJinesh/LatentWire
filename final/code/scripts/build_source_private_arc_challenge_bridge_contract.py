from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_bridge_contract_20260501")
DEFAULT_VALIDATION_SMOKE = pathlib.Path("data/arc_challenge_gate_15.jsonl")
DEFAULT_EVAL_SMOKE = pathlib.Path("data/arc_challenge_eval_35.jsonl")
DEFAULT_COMBINED_SMOKE = pathlib.Path("data/arc_challenge_50.jsonl")
DEFAULT_HF_CACHE = pathlib.Path(".debug/hf_datasets")
DEFAULT_HF_DATASET = "allenai/ai2_arc"
DEFAULT_HF_CONFIG = "ARC-Challenge"
DEFAULT_OFFICIAL_SPLITS = ("train", "validation", "test")
EXPECTED_OFFICIAL_COUNTS = {"train": 1119, "validation": 299, "test": 1172}
FORBIDDEN_SOURCE_FIELDS = (
    "answer",
    "answerKey",
    "answer_index",
    "answer_label",
    "gold_answer",
    "gold_rationale",
    "correct_choice",
    "correct_option",
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


def _answer_to_index(answer: Any, labels: list[str], choice_count: int) -> int:
    if isinstance(answer, bool):
        raise ValueError("boolean answer is not a valid ARC choice index")
    if isinstance(answer, int):
        if 0 <= answer < choice_count:
            return answer
        if 1 <= answer <= choice_count:
            return answer - 1
    answer_text = str(answer).strip()
    if answer_text in labels:
        return labels.index(answer_text)
    upper = answer_text.upper()
    if upper in labels:
        return labels.index(upper)
    if answer_text.isdigit():
        value = int(answer_text)
        if str(value) in labels:
            return labels.index(str(value))
        if 0 <= value < choice_count:
            return value
        if 1 <= value <= choice_count:
            return value - 1
    raise ValueError(f"could not map ARC answer {answer!r} onto labels {labels!r}")


def canonical_arc_row(raw: dict[str, Any], *, source_name: str, row_index: int) -> dict[str, Any]:
    question = str(raw["question"]).strip()
    raw_choices = raw["choices"]
    if isinstance(raw_choices, dict):
        choices = [str(value).strip() for value in raw_choices["text"]]
        labels = [str(value).strip() for value in raw_choices.get("label", [])]
    else:
        choices = [str(value).strip() for value in raw_choices]
        labels = []
    if not labels:
        labels = [chr(ord("A") + index) for index in range(len(choices))]
    if len(labels) != len(choices):
        raise ValueError(f"choice labels/text length mismatch in {source_name} row {row_index}")
    answer = raw.get("answerKey", raw.get("answer"))
    answer_index = _answer_to_index(answer, labels, len(choices))
    cid = _content_id(question, choices)
    return {
        "id": str(raw.get("id") or f"arc_{cid[:16]}"),
        "content_id": cid,
        "source_name": source_name,
        "row_index": int(row_index),
        "question": question,
        "choices": choices,
        "choice_labels": labels,
        "answer_index": int(answer_index),
        "answer_label": labels[answer_index],
    }


def _read_jsonl(path: pathlib.Path, *, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(canonical_arc_row(json.loads(line), source_name=source_name, row_index=row_index))
    if not rows:
        raise ValueError(f"{path} did not contain any ARC rows")
    return rows


def _write_canonical_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_official_split(
    *,
    split: str,
    hf_dataset: str,
    hf_config: str,
    cache_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(hf_dataset, hf_config, split=split, cache_dir=str(cache_dir))
    return [
        canonical_arc_row(dict(row), source_name=f"{hf_dataset}/{hf_config}/{split}", row_index=index + 1)
        for index, row in enumerate(dataset)
    ]


def _slice_summary(name: str, rows: list[dict[str, Any]], *, path: pathlib.Path | None = None) -> dict[str, Any]:
    content_ids = [row["content_id"] for row in rows]
    duplicate_counts = {
        cid: count for cid, count in collections.Counter(content_ids).items() if count > 1
    }
    choice_count_counts = collections.Counter(str(len(row["choices"])) for row in rows)
    answer_index_counts = collections.Counter(str(row["answer_index"]) for row in rows)
    answer_label_counts = collections.Counter(str(row["answer_label"]) for row in rows)
    schema_valid = (
        len(rows) > 0
        and all(len(row["choices"]) >= 2 for row in rows)
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


def _covers_union(combined: list[dict[str, Any]], parts: list[list[dict[str, Any]]]) -> bool:
    combined_ids = {row["content_id"] for row in combined}
    part_ids: set[str] = set()
    for rows in parts:
        part_ids.update(row["content_id"] for row in rows)
    return part_ids <= combined_ids and len(combined_ids) == len(part_ids)


def _method_contract() -> dict[str, Any]:
    return {
        "fixed_packet_budget_bytes": 12,
        "selector_source": "results/source_private_train_donor_antishuffle_stable_gap_seed47_53_59_20260501/train_donor_locked_rate_frontier.json",
        "selector_rule": "global stable_interior source_private_gap selector",
        "source_visible_fields": ["question", "choices"],
        "source_private_allowed_fields": [
            "source model latent/trace generated from question+choices only",
            "answer-masked source trace",
            "rate-capped source-private packet bytes",
        ],
        "forbidden_source_fields": list(FORBIDDEN_SOURCE_FIELDS),
        "required_controls": list(REQUIRED_PUBLIC_CONTROLS),
        "claim_boundary": (
            "This contract freezes the ARC bridge and controls. It is not itself a positive public "
            "benchmark result until the fixed-12B packet beats target-only and every destructive "
            "control on the frozen ARC evaluation split."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private ARC-Challenge Bridge Contract",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- public result ready: `{payload['public_benchmark_result_ready']}`",
        f"- fixed packet budget: `{payload['method_contract']['fixed_packet_budget_bytes']}B`",
        "",
        "## Local Smoke Slices",
        "",
        "| Slice | Rows | Valid | SHA256 | Answer Index Counts |",
        "|---|---:|---:|---|---|",
    ]
    for name, summary in payload["local_summaries"].items():
        lines.append(
            f"| `{name}` | {summary['n']} | `{summary['valid']}` | "
            f"`{summary['sha256']}` | `{summary['answer_index_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Official Splits",
            "",
        ]
    )
    if payload["official_summaries"]:
        lines.extend(["| Split | Rows | Expected | Valid | Materialized Path |", "|---|---:|---:|---:|---|"])
        for name, summary in payload["official_summaries"].items():
            expected = EXPECTED_OFFICIAL_COUNTS.get(name)
            lines.append(
                f"| `{name}` | {summary['n']} | {expected} | `{summary['valid']}` | "
                f"`{summary['path']}` |"
            )
    else:
        lines.append("- not materialized in this run")
    lines.extend(
        [
            "",
            "## Required Controls",
            "",
        ]
    )
    for control in payload["method_contract"]["required_controls"]:
        lines.append(f"- `{control}`")
    lines.extend(
        [
            "",
            "## Blocker",
            "",
            payload["next_exact_gate"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contract(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    validation_smoke_jsonl: pathlib.Path = DEFAULT_VALIDATION_SMOKE,
    eval_smoke_jsonl: pathlib.Path = DEFAULT_EVAL_SMOKE,
    combined_smoke_jsonl: pathlib.Path = DEFAULT_COMBINED_SMOKE,
    materialize_official: bool = False,
    official_output_dir: pathlib.Path | None = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_config: str = DEFAULT_HF_CONFIG,
    hf_cache_dir: pathlib.Path = DEFAULT_HF_CACHE,
    run_date: str = str(dt.date.today()),
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    validation_path = _resolve(validation_smoke_jsonl)
    eval_path = _resolve(eval_smoke_jsonl)
    combined_path = _resolve(combined_smoke_jsonl)
    local_slices = {
        "validation_smoke": _read_jsonl(validation_path, source_name="validation_smoke"),
        "evaluation_smoke": _read_jsonl(eval_path, source_name="evaluation_smoke"),
        "combined_smoke": _read_jsonl(combined_path, source_name="combined_smoke"),
    }
    official_slices: dict[str, list[dict[str, Any]]] = {}
    official_paths: dict[str, pathlib.Path] = {}
    if materialize_official:
        official_root = _resolve(official_output_dir) if official_output_dir else output_dir / "official_splits"
        cache_dir = _resolve(hf_cache_dir)
        for split in DEFAULT_OFFICIAL_SPLITS:
            rows = _load_official_split(
                split=split,
                hf_dataset=hf_dataset,
                hf_config=hf_config,
                cache_dir=cache_dir,
            )
            path = official_root / f"arc_challenge_{split}.jsonl"
            _write_canonical_jsonl(path, rows)
            official_slices[split] = rows
            official_paths[split] = path

    local_summaries = {
        "validation_smoke": _slice_summary("validation_smoke", local_slices["validation_smoke"], path=validation_path),
        "evaluation_smoke": _slice_summary("evaluation_smoke", local_slices["evaluation_smoke"], path=eval_path),
        "combined_smoke": _slice_summary("combined_smoke", local_slices["combined_smoke"], path=combined_path),
    }
    official_summaries = {
        split: _slice_summary(split, rows, path=official_paths.get(split))
        for split, rows in official_slices.items()
    }
    local_overlap = _overlap_matrix(local_slices)
    official_overlap = _overlap_matrix(official_slices) if official_slices else {}
    local_eval_disjoint = local_overlap["validation_smoke"]["evaluation_smoke"] == 0
    combined_covers_local = _covers_union(
        local_slices["combined_smoke"],
        [local_slices["validation_smoke"], local_slices["evaluation_smoke"]],
    )
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
    official_ready = (not materialize_official) or (
        len(official_summaries) == len(DEFAULT_OFFICIAL_SPLITS)
        and official_counts_match
        and official_disjoint
        and all(summary["valid"] for summary in official_summaries.values())
    )
    pass_gate = (
        local_eval_disjoint
        and combined_covers_local
        and all(summary["valid"] for summary in local_summaries.values())
        and official_ready
    )
    payload = {
        "gate": "source_private_arc_challenge_bridge_contract",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "public_benchmark_result_ready": False,
        "readiness": (
            "ARC bridge contract is ready" if pass_gate else "ARC bridge contract has slice/schema blockers"
        ),
        "local_summaries": local_summaries,
        "official_summaries": official_summaries,
        "local_overlap_matrix": local_overlap,
        "official_overlap_matrix": official_overlap,
        "checks": {
            "local_validation_eval_disjoint": local_eval_disjoint,
            "combined_smoke_covers_validation_and_eval": combined_covers_local,
            "official_splits_materialized": bool(official_summaries),
            "official_counts_match_expected": official_counts_match if official_summaries else None,
            "official_splits_disjoint": official_disjoint if official_summaries else None,
            "source_answer_fields_forbidden": True,
        },
        "method_contract": _method_contract(),
        "next_exact_gate": (
            "Implement/run the fixed-12B source-private packet on official ARC-Challenge validation/test "
            "with label permutation, shuffled-source, same-byte text, target-derived, random, and "
            "candidate-derangement controls. Do not consume answerKey when building the source packet."
        ),
        "sources": {
            "arc_paper": "https://arxiv.org/abs/1803.05457",
            "tfds_arc_card": "https://www.tensorflow.org/datasets/catalog/ai2_arc",
            "hf_dataset": f"https://huggingface.co/datasets/{hf_dataset}",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_json = output_dir / "arc_challenge_bridge_contract.json"
    contract_md = output_dir / "arc_challenge_bridge_contract.md"
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
    parser = argparse.ArgumentParser(description="Build the source-private ARC-Challenge bridge contract.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-smoke-jsonl", type=pathlib.Path, default=DEFAULT_VALIDATION_SMOKE)
    parser.add_argument("--eval-smoke-jsonl", type=pathlib.Path, default=DEFAULT_EVAL_SMOKE)
    parser.add_argument("--combined-smoke-jsonl", type=pathlib.Path, default=DEFAULT_COMBINED_SMOKE)
    parser.add_argument("--materialize-official", action="store_true")
    parser.add_argument("--official-output-dir", type=pathlib.Path)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-cache-dir", type=pathlib.Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--run-date", default=str(dt.date.today()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_contract(
        output_dir=args.output_dir,
        validation_smoke_jsonl=args.validation_smoke_jsonl,
        eval_smoke_jsonl=args.eval_smoke_jsonl,
        combined_smoke_jsonl=args.combined_smoke_jsonl,
        materialize_official=args.materialize_official,
        official_output_dir=args.official_output_dir,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_cache_dir=args.hf_cache_dir,
        run_date=args.run_date,
    )


if __name__ == "__main__":
    main()
