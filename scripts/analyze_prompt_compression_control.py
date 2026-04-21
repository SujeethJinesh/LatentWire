#!/usr/bin/env python3
"""Deterministic LLMLingua-style prompt-compression control.

This is a lightweight competitor control for prompt compression when the
original LLMLingua / LongLLMLingua inference stack is unavailable. Instead of
loading a compressor model, it performs a lexical token-budget analysis over the
existing GSM8K / SVAMP prompt data and records the same style of telemetry we
would want from a learned compressor.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TOKEN_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_INPUTS = (
    pathlib.Path("data/gsm8k_eval_70.jsonl"),
    pathlib.Path("data/svamp_eval_70.jsonl"),
)
DEFAULT_OUTPUT_DIR = pathlib.Path("results/competitor_gap_20260421")
DEFAULT_JSON = DEFAULT_OUTPUT_DIR / "prompt_compression_control_20260421.json"
DEFAULT_MD = DEFAULT_OUTPUT_DIR / "prompt_compression_control_20260421.md"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "let",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
}

ANSWER_MARKERS = {
    "answer",
    "final",
    "question",
    "solve",
    "step",
    "steps",
    "think",
    "let",
    "lets",
    "let's",
}

PRESERVE_MARKERS = {"question", "answer", "step", "think", "final", "solve"}


@dataclass(frozen=True)
class ExampleTelemetry:
    index: int
    original_chars: int
    original_bytes: int
    original_tokens_proxy: int
    compressed_budget_tokens: int
    compressed_chars: int
    compressed_bytes: int
    compressed_tokens_proxy: int
    number_total: int
    number_kept: int
    number_preservation_rate: float | None
    answer_known: bool
    answer_span_present: bool
    answer_span_preserved: bool | None
    bytes_saved: int
    compression_ratio: float | None
    compressed_prompt_preview: str


@dataclass(frozen=True)
class SourceTelemetry:
    source: str
    n_examples: int
    original_chars_mean: float
    original_bytes_mean: float
    original_tokens_proxy_mean: float
    compressed_budget_tokens_mean: float
    compressed_chars_mean: float
    compressed_bytes_mean: float
    compressed_tokens_proxy_mean: float
    bytes_saved_mean: float
    compression_ratio_mean: float | None
    number_preservation_rate: float | None
    answer_span_coverage_rate: float
    answer_span_preservation_rate: float | None
    examples: list[ExampleTelemetry]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _normalized_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def _normalize_number(value: str) -> str:
    return value.replace(",", "").strip()


def _tokens_to_text(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    pieces: list[str] = [tokens[0]]
    attach_left = {",", ".", ":", ";", "?", "!", "%", ")", "]", "}", "'"}
    attach_right = {"(", "[", "{", "$", "#"}
    for token in tokens[1:]:
        if token in attach_left:
            pieces[-1] = pieces[-1] + token
        elif pieces[-1] in attach_right:
            pieces[-1] = pieces[-1] + token
        else:
            pieces.append(token)
    return " ".join(pieces)


def _prompt_text(record: dict[str, Any]) -> str:
    if isinstance(record.get("prompt"), str):
        return record["prompt"]
    if isinstance(record.get("question"), str):
        return f"{record['question']}\nAnswer:"
    if isinstance(record.get("source_question"), str):
        return f"{record['source_question']}\nAnswer:"
    raise ValueError(f"Could not find prompt/question text in record keys: {sorted(record)}")


def _answer_strings(record: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for key in ("answer_text", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            answers.append(value.strip())
    aliases = record.get("aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                answers.append(alias.strip())
    # Normalize common GSM/SVAMP answer wrappers.
    expanded: list[str] = []
    for answer in answers:
        expanded.append(answer)
        stripped = answer.replace("####", "").strip()
        if stripped and stripped != answer:
            expanded.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for answer in expanded:
        norm = _normalized_text(answer)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(answer)
    return deduped


def _number_strings(text: str) -> list[str]:
    seen: set[str] = set()
    numbers: list[str] = []
    for match in NUMBER_RE.findall(text):
        normalized = _normalize_number(match)
        if normalized not in seen:
            seen.add(normalized)
            numbers.append(normalized)
    return numbers


def _find_span_positions(tokens: Sequence[str], pattern_tokens: Sequence[str]) -> list[tuple[int, int]]:
    if not pattern_tokens or len(pattern_tokens) > len(tokens):
        return []
    matches: list[tuple[int, int]] = []
    for start in range(len(tokens) - len(pattern_tokens) + 1):
        window = tokens[start : start + len(pattern_tokens)]
        if [tok.lower() for tok in window] == [tok.lower() for tok in pattern_tokens]:
            matches.append((start, start + len(pattern_tokens)))
    return matches


def _token_bonus(token: str, index: int, total: int) -> float:
    lower = token.lower()
    score = 0.0
    if NUMBER_RE.fullmatch(token):
        score += 8.0 + min(len(token), 4)
    elif lower in PRESERVE_MARKERS:
        score += 5.0
    elif lower in ANSWER_MARKERS:
        score += 3.0
    elif lower in STOPWORDS:
        score += 0.2
    elif token.isalpha():
        score += 1.0 + min(len(token) / 8.0, 1.5)
    else:
        score += 0.1
    if token[:1].isupper() and token.isalpha():
        score += 0.3
    if index < 6 or index >= max(0, total - 6):
        score += 1.0
    return score


def _compress_prompt(
    prompt: str,
    *,
    answer_strings: Sequence[str],
    budget_ratio: float,
    min_budget: int,
) -> tuple[str, int, int, int, float | None, float | None, bool, bool]:
    tokens = _tokenize(prompt)
    original_token_count = len(tokens)
    if original_token_count == 0:
        return "", 0, 0, 0, None, None, False, False

    compressed_budget = max(min_budget, int(math.ceil(original_token_count * budget_ratio)))
    compressed_budget = min(compressed_budget, original_token_count)
    scores = [_token_bonus(token, idx, original_token_count) for idx, token in enumerate(tokens)]

    selected: set[int] = set()
    protected: set[int] = set()

    # Always preserve the opening framing and terminal answer marker if present.
    for idx in range(min(4, original_token_count)):
        protected.add(idx)
        scores[idx] += 2.0
    for idx in range(max(0, original_token_count - 4), original_token_count):
        protected.add(idx)
        scores[idx] += 2.0

    # Preserve answer spans when the answer string actually appears in the prompt.
    answer_span_present = False
    for answer in answer_strings:
        answer_tokens = _tokenize(answer)
        for start, end in _find_span_positions(tokens, answer_tokens):
            answer_span_present = True
            for idx in range(start, end):
                protected.add(idx)
                scores[idx] += 10.0
            for idx in range(max(0, start - 1), min(original_token_count, end + 1)):
                scores[idx] += 1.0

    # Preserve numeric spans and their immediate lexical neighborhood.
    for idx, token in enumerate(tokens):
        if NUMBER_RE.fullmatch(token):
            protected.add(idx)
            scores[idx] += 4.0
            for neighbor in (idx - 1, idx + 1):
                if 0 <= neighbor < original_token_count:
                    scores[neighbor] += 1.25

    selected.update(protected)
    if len(selected) > compressed_budget:
        compressed_budget = len(selected)

    ranking = sorted(range(original_token_count), key=lambda idx: (-scores[idx], idx))
    for idx in ranking:
        if len(selected) >= compressed_budget:
            break
        selected.add(idx)

    selected_indices = sorted(selected)
    compressed_tokens = [tokens[idx] for idx in selected_indices]
    compressed_prompt = _tokens_to_text(compressed_tokens)

    original_numbers = _number_strings(prompt)
    compressed_numbers = _number_strings(compressed_prompt)
    original_number_set = set(original_numbers)
    compressed_number_set = set(compressed_numbers)
    number_total = len(original_number_set)
    number_kept = len(original_number_set & compressed_number_set)
    number_rate = (number_kept / number_total) if number_total else None

    answer_span_preserved = None
    if answer_span_present:
        answer_span_preserved = any(
            _normalized_text(answer) in _normalized_text(compressed_prompt)
            for answer in answer_strings
            if answer.strip()
        )

    compressed_token_count = len(_tokenize(compressed_prompt))
    compression_ratio = (
        (original_token_count / compressed_token_count) if compressed_token_count else None
    )
    return (
        compressed_prompt,
        compressed_budget,
        original_token_count,
        compressed_token_count,
        number_rate,
        compression_ratio,
        answer_span_present,
        bool(answer_span_preserved) if answer_span_preserved is not None else False,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def analyze_source(
    path: pathlib.Path,
    *,
    budget_ratio: float,
    min_budget: int,
) -> SourceTelemetry:
    records = load_jsonl(path)
    examples: list[ExampleTelemetry] = []
    for index, record in enumerate(records):
        prompt = _prompt_text(record)
        answers = _answer_strings(record)
        compressed_prompt, compressed_budget, original_tokens, compressed_tokens, number_rate, compression_ratio, answer_span_present, answer_span_preserved = _compress_prompt(
            prompt,
            answer_strings=answers,
            budget_ratio=budget_ratio,
            min_budget=min_budget,
        )
        original_chars = len(prompt)
        original_bytes = len(prompt.encode("utf-8"))
        compressed_chars = len(compressed_prompt)
        compressed_bytes = len(compressed_prompt.encode("utf-8"))
        original_numbers = _number_strings(prompt)
        compressed_numbers = _number_strings(compressed_prompt)
        number_total = len(set(original_numbers))
        number_kept = len(set(original_numbers) & set(compressed_numbers))
        examples.append(
            ExampleTelemetry(
                index=index,
                original_chars=original_chars,
                original_bytes=original_bytes,
                original_tokens_proxy=original_tokens,
                compressed_budget_tokens=compressed_budget,
                compressed_chars=compressed_chars,
                compressed_bytes=compressed_bytes,
                compressed_tokens_proxy=compressed_tokens,
                number_total=number_total,
                number_kept=number_kept,
                number_preservation_rate=number_rate,
                answer_known=bool(answers),
                answer_span_present=answer_span_present,
                answer_span_preserved=answer_span_preserved if answer_span_present else None,
                bytes_saved=original_bytes - compressed_bytes,
                compression_ratio=compression_ratio,
                compressed_prompt_preview=compressed_prompt[:180],
            )
        )
    answer_present_examples = [example for example in examples if example.answer_span_present]
    answer_preserved_examples = [
        example for example in answer_present_examples if example.answer_span_preserved
    ]
    return SourceTelemetry(
        source=path.name,
        n_examples=len(examples),
        original_chars_mean=_mean(example.original_chars for example in examples),
        original_bytes_mean=_mean(example.original_bytes for example in examples),
        original_tokens_proxy_mean=_mean(example.original_tokens_proxy for example in examples),
        compressed_budget_tokens_mean=_mean(example.compressed_budget_tokens for example in examples),
        compressed_chars_mean=_mean(example.compressed_chars for example in examples),
        compressed_bytes_mean=_mean(example.compressed_bytes for example in examples),
        compressed_tokens_proxy_mean=_mean(example.compressed_tokens_proxy for example in examples),
        bytes_saved_mean=_mean(example.bytes_saved for example in examples),
        compression_ratio_mean=_mean(
            example.compression_ratio for example in examples if example.compression_ratio is not None
        )
        if any(example.compression_ratio is not None for example in examples)
        else None,
        number_preservation_rate=_mean(
            example.number_preservation_rate
            for example in examples
            if example.number_preservation_rate is not None
        )
        if any(example.number_preservation_rate is not None for example in examples)
        else None,
        answer_span_coverage_rate=(
            len(answer_present_examples) / len(examples) if examples else 0.0
        ),
        answer_span_preservation_rate=(
            len(answer_preserved_examples) / len(answer_present_examples)
            if answer_present_examples
            else None
        ),
        examples=examples,
    )


def _telemetry_to_dict(source: SourceTelemetry) -> dict[str, Any]:
    return {
        "source": source.source,
        "n_examples": source.n_examples,
        "original_chars_mean": source.original_chars_mean,
        "original_bytes_mean": source.original_bytes_mean,
        "original_tokens_proxy_mean": source.original_tokens_proxy_mean,
        "compressed_budget_tokens_mean": source.compressed_budget_tokens_mean,
        "compressed_chars_mean": source.compressed_chars_mean,
        "compressed_bytes_mean": source.compressed_bytes_mean,
        "compressed_tokens_proxy_mean": source.compressed_tokens_proxy_mean,
        "bytes_saved_mean": source.bytes_saved_mean,
        "compression_ratio_mean": source.compression_ratio_mean,
        "number_preservation_rate": source.number_preservation_rate,
        "answer_span_coverage_rate": source.answer_span_coverage_rate,
        "answer_span_preservation_rate": source.answer_span_preservation_rate,
        "examples": [example.__dict__ for example in source.examples],
    }


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_markdown(
    sources: Sequence[SourceTelemetry],
    *,
    budget_ratio: float,
    min_budget: int,
) -> str:
    lines = [
        "# LLMLingua-Style Prompt Compression Control",
        "",
        "This control is deterministic and does not load the LLMLingua model stack.",
        "It uses a lexical token-budget proxy over the existing GSM8K / SVAMP prompt data",
        "to separate prompt-budget effects from learned compressor effects.",
        "",
        f"Budget ratio: `{budget_ratio:.2f}`",
        f"Minimum budget: `{min_budget}` token-proxy units",
        "",
        "| Source | N | Original chars | Original tokens proxy | Compressed budget | Compressed tokens proxy | Number preservation | Answer-span coverage | Answer-span preservation | Est. bytes saved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in sources:
        lines.append(
            "| {source} | {n} | {orig_chars:.1f} | {orig_tokens:.1f} | {budget:.1f} | {comp_tokens:.1f} | {num_rate} | {coverage:.1%} | {answer_rate} | {bytes_saved:.1f} |".format(
                source=source.source,
                n=source.n_examples,
                orig_chars=source.original_chars_mean,
                orig_tokens=source.original_tokens_proxy_mean,
                budget=source.compressed_budget_tokens_mean,
                comp_tokens=source.compressed_tokens_proxy_mean,
                num_rate=_format_float(source.number_preservation_rate, 2),
                coverage=source.answer_span_coverage_rate,
                answer_rate=_format_float(source.answer_span_preservation_rate, 2),
                bytes_saved=source.bytes_saved_mean,
            )
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Tokens are approximated with a regex word / number / punctuation proxy, not a model tokenizer.",
            "- Compression keeps high-salience lexical items, all numeric tokens, and any answer span that is actually present in the prompt.",
            "- The goal is a fair no-download control for LLMLingua-style prompt compression, not a replacement for learned prompt compression.",
            "",
            "## Claim Risks",
            "",
            "- This is a lexical baseline, so it cannot support claims about LLMLingua's learned ranking quality.",
            "- Answer-span preservation is only meaningful on examples where the answer string appears in the source prompt.",
            "- Token-budget values are proxy counts, so they are descriptive rather than tokenizer-exact.",
            "- Accuracy is not measured here; this control only measures compression and preservation telemetry.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=pathlib.Path,
        default=list(DEFAULT_INPUTS),
        help="JSONL prompt files to analyze.",
    )
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=0.5,
        help="Fraction of the original token-proxy budget to retain.",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=16,
        help="Lower bound on the compressed token-proxy budget.",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=DEFAULT_JSON,
        help="Where to write the telemetry JSON artifact.",
    )
    parser.add_argument(
        "--output-md",
        type=pathlib.Path,
        default=DEFAULT_MD,
        help="Where to write the markdown readout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    sources = [
        analyze_source(path, budget_ratio=args.budget_ratio, min_budget=args.min_budget)
        for path in args.inputs
    ]
    payload = {
        "analysis": "llmlingua_style_prompt_compression_control",
        "date": "2026-04-21",
        "budget_ratio": args.budget_ratio,
        "min_budget": args.min_budget,
        "token_proxy": "regex word-number-punctuation",
        "sources": [_telemetry_to_dict(source) for source in sources],
        "claim_risks": [
            "Lexical proxy only; no learned LLMLingua inference was run.",
            "No downstream accuracy or latency claims are supported by this artifact.",
            "Answer-span preservation is only measurable where the answer string appears in the original prompt.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(
        build_markdown(sources, budget_ratio=args.budget_ratio, min_budget=args.min_budget),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    main()
