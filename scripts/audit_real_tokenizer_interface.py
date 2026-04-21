#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/gsm8k_gate_search_30.jsonl"
DEFAULT_OUTPUT_JSON = ROOT / ".debug/real_tokenizer_interface_audit.json"
DEFAULT_OUTPUT_MD = ROOT / ".debug/real_tokenizer_interface_audit.md"
DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"

_DIGIT_RE = re.compile(r"^[0-9]+(?:[.,:/][0-9]+)*$")
_OPERATOR_CHARS = set("+-*/=()[]{}<>%^&|~,:;.!?")
_SPECIAL_MARKERS = ("▁", "Ġ", "Ċ")


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt:
        return str(prompt)
    fallback = row.get("source_question")
    if fallback:
        return str(fallback)
    raise KeyError("Row is missing prompt/source_question text")


def _encoded_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise TypeError("Tokenizer output did not include input_ids")
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    if not text:
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            try:
                converted = convert(int(token_id))
                if isinstance(converted, list):
                    converted = converted[0] if converted else ""
                text = str(converted)
            except Exception:
                text = ""
    return str(text)


def _canonical_piece(piece: str) -> str:
    text = piece
    for marker in _SPECIAL_MARKERS:
        text = text.replace(marker, " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text


def _token_surface(tokenizer: Any, token_id: int) -> str:
    return _canonical_piece(_decode_piece(tokenizer, token_id))


def _family(surface: str) -> str:
    if not surface:
        return "empty"
    if _DIGIT_RE.fullmatch(surface) or (any(ch.isdigit() for ch in surface) and not any(ch.isalpha() for ch in surface)):
        return "digit"
    if all((not ch.isalnum()) and (not ch.isspace()) and (ch in _OPERATOR_CHARS) for ch in surface):
        return "operator"
    return "other"


def _token_stats(tokenizer: Any, text: str) -> dict[str, Any]:
    token_ids = _encoded_ids(tokenizer, text)
    surfaces = [_token_surface(tokenizer, token_id) for token_id in token_ids]
    kept_surfaces = [surface for surface in surfaces if surface]
    unique_surfaces = set(kept_surfaces)
    family_counts = {"digit": 0, "operator": 0, "other": 0}
    for surface in kept_surfaces:
        family_counts[_family(surface)] += 1
    total = max(len(kept_surfaces), 1)
    return {
        "token_ids": token_ids,
        "surfaces": kept_surfaces,
        "unique_surfaces": unique_surfaces,
        "token_count": len(kept_surfaces),
        "unique_token_count": len(unique_surfaces),
        "digit_count": family_counts["digit"],
        "operator_count": family_counts["operator"],
        "other_count": family_counts["other"],
        "digit_share": family_counts["digit"] / total,
        "operator_share": family_counts["operator"] / total,
        "other_share": family_counts["other"] / total,
    }


def _shared_rate(lhs: set[str], rhs: set[str]) -> tuple[float, int, int]:
    union = lhs | rhs
    if not union:
        return 1.0, 0, 0
    shared = len(lhs & rhs)
    return shared / len(union), shared, len(union)


def _summarize_row(index: int, prompt: str, source: Any, target: Any) -> dict[str, Any]:
    byte_count = len(prompt.encode("utf-8"))
    source_stats = _token_stats(source, prompt)
    target_stats = _token_stats(target, prompt)
    shared_rate, shared_count, union_count = _shared_rate(source_stats["unique_surfaces"], target_stats["unique_surfaces"])

    source_digit = {surface for surface in source_stats["unique_surfaces"] if _family(surface) == "digit"}
    target_digit = {surface for surface in target_stats["unique_surfaces"] if _family(surface) == "digit"}
    source_operator = {surface for surface in source_stats["unique_surfaces"] if _family(surface) == "operator"}
    target_operator = {surface for surface in target_stats["unique_surfaces"] if _family(surface) == "operator"}
    digit_shared_rate, digit_shared_count, digit_union_count = _shared_rate(source_digit, target_digit)
    operator_shared_rate, operator_shared_count, operator_union_count = _shared_rate(source_operator, target_operator)

    return {
        "index": index,
        "prompt": prompt,
        "bytes_per_example": byte_count,
        "source_token_count": source_stats["token_count"],
        "target_token_count": target_stats["token_count"],
        "source_fragmentation": source_stats["token_count"] / max(byte_count, 1),
        "target_fragmentation": target_stats["token_count"] / max(byte_count, 1),
        "shared_decoded_token_rate": shared_rate,
        "shared_decoded_token_count": shared_count,
        "shared_decoded_token_union": union_count,
        "source_unique_decoded_token_count": source_stats["unique_token_count"],
        "target_unique_decoded_token_count": target_stats["unique_token_count"],
        "source_digit_token_count": source_stats["digit_count"],
        "target_digit_token_count": target_stats["digit_count"],
        "source_operator_token_count": source_stats["operator_count"],
        "target_operator_token_count": target_stats["operator_count"],
        "source_digit_share": source_stats["digit_share"],
        "target_digit_share": target_stats["digit_share"],
        "source_operator_share": source_stats["operator_share"],
        "target_operator_share": target_stats["operator_share"],
        "shared_digit_token_rate": digit_shared_rate,
        "shared_digit_token_count": digit_shared_count,
        "shared_digit_token_union": digit_union_count,
        "shared_operator_token_rate": operator_shared_rate,
        "shared_operator_token_count": operator_shared_count,
        "shared_operator_token_union": operator_union_count,
    }


def _summary(rows: list[dict[str, Any]], *, source_model: str, target_model: str, input_path: Path) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "input_path": str(input_path),
        "source_model": source_model,
        "target_model": target_model,
        "total_bytes": sum(row["bytes_per_example"] for row in rows),
        "mean_bytes_per_example": mean(row["bytes_per_example"] for row in rows),
        "mean_source_token_count": mean(row["source_token_count"] for row in rows),
        "mean_target_token_count": mean(row["target_token_count"] for row in rows),
        "mean_source_fragmentation": mean(row["source_fragmentation"] for row in rows),
        "mean_target_fragmentation": mean(row["target_fragmentation"] for row in rows),
        "mean_shared_decoded_token_rate": mean(row["shared_decoded_token_rate"] for row in rows),
        "mean_shared_digit_token_rate": mean(row["shared_digit_token_rate"] for row in rows),
        "mean_shared_operator_token_rate": mean(row["shared_operator_token_rate"] for row in rows),
        "mean_source_digit_share": mean(row["source_digit_share"] for row in rows),
        "mean_target_digit_share": mean(row["target_digit_share"] for row in rows),
        "mean_source_operator_share": mean(row["source_operator_share"] for row in rows),
        "mean_target_operator_share": mean(row["target_operator_share"] for row in rows),
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Real Tokenizer Interface Audit",
        "",
        f"- Source model: `{summary['source_model']}`",
        f"- Target model: `{summary['target_model']}`",
        f"- Input: `{summary['input_path']}`",
        f"- Examples: `{summary['examples']}`",
        f"- Mean bytes/example: `{summary['mean_bytes_per_example']:.2f}`",
        f"- Mean source fragmentation: `{summary['mean_source_fragmentation']:.4f}`",
        f"- Mean target fragmentation: `{summary['mean_target_fragmentation']:.4f}`",
        f"- Mean shared decoded-token rate: `{summary['mean_shared_decoded_token_rate']:.4f}`",
        f"- Mean shared digit-token rate: `{summary['mean_shared_digit_token_rate']:.4f}`",
        f"- Mean shared operator-token rate: `{summary['mean_shared_operator_token_rate']:.4f}`",
        "",
        "| Example | Bytes | Src toks | Tgt toks | Src frag | Tgt frag | Shared rate | Digit share (src/tgt) | Operator share (src/tgt) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {index} | {bytes_per_example} | {source_token_count} | {target_token_count} | "
            "{source_fragmentation:.4f} | {target_fragmentation:.4f} | {shared_decoded_token_rate:.4f} | "
            "{source_digit_share:.4f}/{target_digit_share:.4f} | {source_operator_share:.4f}/{target_operator_share:.4f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit real tokenizer fragmentation and byte interface statistics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    rows_raw = _load_jsonl(args.input, limit=args.limit)
    source = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    target = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    rows = [
        _summarize_row(
            index=index,
            prompt=_prompt_text(row),
            source=source,
            target=target,
        )
        for index, row in enumerate(rows_raw)
    ]
    payload = {
        "config": {
            "input": str(args.input),
            "source_model": args.source_model,
            "target_model": args.target_model,
            "limit": args.limit,
        },
        "summary": _summary(rows, source_model=args.source_model, target_model=args.target_model, input_path=args.input),
        "rows": rows,
    }
    _write_json(payload, args.output_json)
    _write_markdown(payload, args.output_md)
    print(
        "Real tokenizer audit: "
        f"examples={payload['summary']['examples']}, "
        f"mean_bytes={payload['summary']['mean_bytes_per_example']:.2f}, "
        f"mean_shared_rate={payload['summary']['mean_shared_decoded_token_rate']:.4f}, "
        f"mean_source_frag={payload['summary']['mean_source_fragmentation']:.4f}, "
        f"mean_target_frag={payload['summary']['mean_target_fragmentation']:.4f}"
    )
    return payload


if __name__ == "__main__":
    main()
