#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/gsm8k_gate_search_30.jsonl"
DEFAULT_OUTPUT_JSON = ROOT / "results/query_pool_toy_20260421/real_tokenizer_interface_pair_sweep_20260421.json"
DEFAULT_OUTPUT_MD = ROOT / "results/query_pool_toy_20260421/real_tokenizer_interface_pair_sweep_20260421.md"
DEFAULT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("qwen25_to_qwen3", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"),
    ("qwen25_to_mistral", "Qwen/Qwen2.5-0.5B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("qwen25_to_phi3", "Qwen/Qwen2.5-0.5B-Instruct", "microsoft/Phi-3-mini-4k-instruct"),
)
_SPECIAL_MARKERS = ("▁", "Ġ", "Ċ")


@dataclass(frozen=True)
class PairSpec:
    label: str
    source_model: str
    target_model: str


@dataclass(frozen=True)
class RealTokenizerInterfacePairSweepConfig:
    input_path: str
    limit: int | None = None
    calibration_examples: int = 15
    remap_capacity: int = 24
    pairs: tuple[PairSpec, ...] = tuple(PairSpec(*item) for item in DEFAULT_PAIRS)


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


def _canonical_piece(piece: str) -> str:
    text = piece
    for marker in _SPECIAL_MARKERS:
        text = text.replace(marker, " ")
    text = " ".join(text.split())
    return text.strip()


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
            converted = convert(int(token_id))
            if isinstance(converted, list):
                converted = converted[0] if converted else ""
            text = str(converted)
    return str(text)


def _byte_offsets(text: str) -> list[int]:
    offsets = [0]
    cursor = 0
    for char in text:
        cursor += len(char.encode("utf-8"))
        offsets.append(cursor)
    return offsets


def _encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    input_ids = getattr(encoded, "input_ids", None)
    offset_mapping = getattr(encoded, "offset_mapping", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
        offset_mapping = encoded.get("offset_mapping")
    if input_ids is None or offset_mapping is None:
        raise TypeError("Tokenizer output did not include input_ids/offset_mapping")
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(offset_mapping, "tolist"):
        offset_mapping = offset_mapping.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offset_mapping and isinstance(offset_mapping[0], list) and offset_mapping and isinstance(offset_mapping[0][0], list):
        offset_mapping = offset_mapping[0]
    token_ids = [int(token_id) for token_id in input_ids]
    char_to_byte = _byte_offsets(text)
    byte_spans: list[tuple[int, int]] = []
    substrings: list[str] = []
    for start, end in offset_mapping:
        start = int(start)
        end = int(end)
        substrings.append(text[start:end])
        byte_spans.append((char_to_byte[start], char_to_byte[end]))
    surfaces = [_canonical_piece(_decode_piece(tokenizer, token_id)) for token_id in token_ids]
    return token_ids, byte_spans, surfaces if len(surfaces) == len(byte_spans) else [""] * len(byte_spans)


def _shared_rate(lhs: set[str], rhs: set[str]) -> tuple[float, int, int]:
    union = lhs | rhs
    if not union:
        return 1.0, 0, 0
    shared = len(lhs & rhs)
    return shared / len(union), shared, len(union)


def _boundary_positions(spans: Sequence[tuple[int, int]]) -> set[int]:
    return {end for _, end in spans[:-1]}


def _boundary_f1(lhs: set[int], rhs: set[int]) -> float:
    if not lhs and not rhs:
        return 1.0
    return float(2.0 * len(lhs & rhs) / max(len(lhs) + len(rhs), 1))


def _fragmentation(token_count: int, byte_count: int) -> float:
    return float(token_count / max(byte_count, 1))


def _build_remap_table(
    prompts: Sequence[str],
    *,
    source_tokenizer: Any,
    target_tokenizer: Any,
    capacity: int,
) -> dict[str, tuple[str, ...]]:
    counts: Counter[str] = Counter()
    for prompt in prompts:
        _, spans, _ = _encode_with_offsets(source_tokenizer, prompt)
        char_to_byte = _byte_offsets(prompt)
        byte_to_char = {byte: index for index, byte in enumerate(char_to_byte)}
        for start, end in spans:
            token_text = prompt[byte_to_char[start] : byte_to_char[end]]
            counts[token_text] += 1

    candidates: list[tuple[float, str, tuple[str, ...]]] = []
    for token_text, count in counts.items():
        if not token_text:
            continue
        _, _, target_surfaces = _encode_with_offsets(target_tokenizer, token_text)
        target_pieces = tuple(piece for piece in target_surfaces if piece)
        if len(token_text) <= 1 or len(target_pieces) <= 1:
            continue
        byte_len = len(token_text.encode("utf-8"))
        multibyte_bonus = 0.75 if byte_len != len(token_text) else 0.0
        score = count * (len(target_pieces) - 1) + multibyte_bonus + 0.02 * byte_len
        candidates.append((float(score), token_text, target_pieces))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return {token: pieces for _, token, pieces in candidates[: max(0, capacity)]}


def _summarize_prompt(prompt: str, *, source_tokenizer: Any, target_tokenizer: Any, remap: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    _, source_spans, source_surfaces = _encode_with_offsets(source_tokenizer, prompt)
    _, target_spans, target_surfaces = _encode_with_offsets(target_tokenizer, prompt)
    byte_count = len(prompt.encode("utf-8"))
    source_boundaries = _boundary_positions(source_spans)
    target_boundaries = _boundary_positions(target_spans)
    source_unique = {piece for piece in source_surfaces if piece}
    target_unique = {piece for piece in target_surfaces if piece}
    shared_rate, shared_count, union_count = _shared_rate(source_unique, target_unique)
    char_to_byte = _byte_offsets(prompt)
    byte_to_char = {byte: index for index, byte in enumerate(char_to_byte)}
    covered_bytes = 0
    for start, end in source_spans:
        token_text = prompt[byte_to_char[start] : byte_to_char[end]]
        if len(token_text.encode("utf-8")) <= 1 or token_text in remap:
            covered_bytes += end - start
    return {
        "bytes_per_example": byte_count,
        "source_token_count": len(source_spans),
        "target_token_count": len(target_spans),
        "source_fragmentation": _fragmentation(len(source_spans), byte_count),
        "target_fragmentation": _fragmentation(len(target_spans), byte_count),
        "shared_decoded_token_rate": shared_rate,
        "shared_decoded_token_count": shared_count,
        "shared_decoded_token_union": union_count,
        "boundary_f1": _boundary_f1(source_boundaries, target_boundaries),
        "boundary_jaccard": len(source_boundaries & target_boundaries) / max(len(source_boundaries | target_boundaries), 1),
        "byte_span_remap_coverage": covered_bytes / max(byte_count, 1),
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(mean(float(row[key]) for row in rows)) if rows else 0.0


def _evaluate_pair(pair: PairSpec, prompts: Sequence[str], *, calibration_examples: int, remap_capacity: int) -> dict[str, Any]:
    source_tokenizer = AutoTokenizer.from_pretrained(pair.source_model, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(pair.target_model, trust_remote_code=True)
    calibration = list(prompts[: max(0, min(calibration_examples, len(prompts)))])
    evaluation = list(prompts[max(0, min(calibration_examples, len(prompts))) :])
    if not evaluation:
        evaluation = list(prompts)
    remap = _build_remap_table(
        calibration if calibration else prompts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        capacity=remap_capacity,
    )
    rows = [_summarize_prompt(prompt, source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer, remap=remap) for prompt in evaluation]
    return {
        "label": pair.label,
        "source_model": pair.source_model,
        "target_model": pair.target_model,
        "examples": len(rows),
        "calibration_examples": len(calibration),
        "remap_table_size": len(remap),
        "mean_bytes_per_example": _mean(rows, "bytes_per_example"),
        "mean_source_token_count": _mean(rows, "source_token_count"),
        "mean_target_token_count": _mean(rows, "target_token_count"),
        "mean_source_fragmentation": _mean(rows, "source_fragmentation"),
        "mean_target_fragmentation": _mean(rows, "target_fragmentation"),
        "mean_fragmentation_delta": _mean(rows, "target_fragmentation") - _mean(rows, "source_fragmentation"),
        "mean_shared_decoded_token_rate": _mean(rows, "shared_decoded_token_rate"),
        "mean_boundary_f1": _mean(rows, "boundary_f1"),
        "mean_boundary_jaccard": _mean(rows, "boundary_jaccard"),
        "mean_byte_span_remap_coverage": _mean(rows, "byte_span_remap_coverage"),
    }


def run_sweep(config: RealTokenizerInterfacePairSweepConfig) -> dict[str, Any]:
    input_path = Path(config.input_path)
    prompts = [_prompt_text(row) for row in _load_jsonl(input_path, limit=config.limit)]
    rows = [
        _evaluate_pair(
            pair,
            prompts,
            calibration_examples=config.calibration_examples,
            remap_capacity=config.remap_capacity,
        )
        for pair in config.pairs
    ]
    return {
        "config": {
            **asdict(config),
            "pairs": [asdict(pair) for pair in config.pairs],
        },
        "rows": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Real Tokenizer Interface Pair Sweep",
        "",
        f"- Input: `{config['input_path']}`",
        f"- Calibration examples / pair: `{config['calibration_examples']}`",
        f"- Remap capacity: `{config['remap_capacity']}`",
        f"- Pair count: `{len(payload['rows'])}`",
        "",
        "| Pair | Src frag | Tgt frag | Frag delta | Shared decoded | Boundary F1 | Remap coverage | Src toks | Tgt toks | Remap table |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {label} | {mean_source_fragmentation:.4f} | {mean_target_fragmentation:.4f} | {mean_fragmentation_delta:.4f} | {mean_shared_decoded_token_rate:.4f} | {mean_boundary_f1:.4f} | {mean_byte_span_remap_coverage:.4f} | {mean_source_token_count:.2f} | {mean_target_token_count:.2f} | {remap_table_size} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- `mean_shared_decoded_token_rate` near `1.0` means the two tokenizers expose almost the same surface pieces on this slice.",
            "- `mean_boundary_f1` and `mean_fragmentation_delta` capture whether token boundaries diverge even when decoded surfaces overlap.",
            "- `mean_byte_span_remap_coverage` measures how much of the byte stream is covered by a bounded source-token to target-piece remap table learned on a calibration subset.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real tokenizer interface pair sweep over a JSONL prompt slice.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calibration-examples", type=int, default=15)
    parser.add_argument("--remap-capacity", type=int, default=24)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "SOURCE_MODEL", "TARGET_MODEL"),
        help="Add a tokenizer pair as: LABEL SOURCE_MODEL TARGET_MODEL. May be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    pairs = tuple(PairSpec(*item) for item in args.pair) if args.pair else tuple(PairSpec(*item) for item in DEFAULT_PAIRS)
    config = RealTokenizerInterfacePairSweepConfig(
        input_path=str(args.input),
        limit=args.limit,
        calibration_examples=args.calibration_examples,
        remap_capacity=args.remap_capacity,
        pairs=pairs,
    )
    payload = run_sweep(config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload))
    return payload


if __name__ == "__main__":
    main()
