#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = ROOT / "results/query_pool_toy_20260421/tokenizer_stress_split_20260421.json"
DEFAULT_OUTPUT_MD = ROOT / "results/query_pool_toy_20260421/tokenizer_stress_split_20260421.md"


@dataclass(frozen=True)
class TokenizerStressSplitConfig:
    seed: int = 0
    remap_capacity: int = 12


@dataclass(frozen=True)
class StressExample:
    example_id: str
    text: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class ToyTokenizer:
    name: str
    merge_tokens: tuple[str, ...]
    vocab_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: index for index, token in enumerate(self.vocab_tokens)})
        object.__setattr__(self, "id_to_token", self.vocab_tokens)
        object.__setattr__(self, "_merge_sorted", tuple(sorted(self.merge_tokens, key=lambda token: (-len(token), token))))

    def segment(self, text: str) -> list[str]:
        tokens: list[str] = []
        cursor = 0
        while cursor < len(text):
            match = None
            for token in self._merge_sorted:
                if text.startswith(token, cursor):
                    match = token
                    break
            if match is None:
                tokens.append(text[cursor])
                cursor += 1
            else:
                tokens.append(match)
                cursor += len(match)
        return tokens

    def encode_ids(self, text: str) -> list[int]:
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unk) for token in self.segment(text)]

    def decode_ids(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[token_id] if 0 <= token_id < len(self.id_to_token) else "<unk>" for token_id in ids)


def _build_tokenizers() -> tuple[ToyTokenizer, ToyTokenizer]:
    common = (
        "<unk>",
        "<pad>",
        " ",
        "\n",
        "+",
        "-",
        "*",
        "/",
        "=",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ",",
        ".",
        ":",
        ";",
        "?",
        "!",
        "_",
        "^",
        "|",
        "<",
        ">",
        "'",
        '"',
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "µ",
        "μ",
        "Δ",
        "λ",
        "π",
        "σ",
        "Ω",
        "°",
        "±",
        "≈",
        "≤",
        "≥",
        "→",
        "∑",
        "∂",
        "東京",
        "东",
        "京",
        "漢",
        "字",
        "🙂",
        "🚀",
    )
    source_merges = (
        "µs",
        "ms",
        "kg",
        "m/s",
        "°C",
        "Δx",
        "λ_i",
        "var_",
        "foo_bar",
        "alpha_beta",
        "10.50",
        "3.1415",
        "0.001",
        ">= ",
        "<= ",
        "->",
        "東京",
        "漢字",
        "🙂🚀",
        "cache_key",
        "KV[",
        "token-boundary",
    )
    target_merges = (
        "micro",
        "seconds",
        "meter",
        "per",
        "sec",
        "deg",
        "C",
        "Delta",
        "lambda",
        "_i",
        "foo",
        "_bar",
        "alpha",
        "_beta",
        "10",
        ".50",
        "3",
        ".1415",
        "0",
        ".001",
        ">=",
        "<=",
        "→",
        "東",
        "京",
        "漢",
        "字",
        "🙂",
        "🚀",
        "cache",
        "_key",
        "KV",
        "token",
        "boundary",
    )
    source_vocab = tuple(dict.fromkeys((*common, *source_merges)))
    target_vocab = tuple(dict.fromkeys((*common, *target_merges)))
    return (
        ToyTokenizer(name="source_stress", merge_tokens=source_merges, vocab_tokens=source_vocab),
        ToyTokenizer(name="target_stress", merge_tokens=target_merges, vocab_tokens=target_vocab),
    )


def _default_examples() -> list[StressExample]:
    rows = [
        ("unicode_0", "東京 cache_key maps 漢字 -> value🙂🚀.", ("unicode", "multi_byte_span", "punctuation", "variables")),
        ("unicode_1", "Normalize café naïve résumé; 東京 stays intact.", ("unicode", "multi_byte_span", "punctuation")),
        ("units_0", "Latency is 7 µs + 20 ms, not 0.001 s.", ("math_units", "decimals", "punctuation")),
        ("units_1", "Speed v=3.50 m/s at 22°C with ±0.25 error.", ("math_units", "decimals", "variables", "punctuation")),
        ("decimal_0", "Compute 10.50 / 2.5 + 3.1415.", ("decimals", "punctuation")),
        ("decimal_1", "Threshold p<=0.001 and q>=0.990?", ("decimals", "variables", "punctuation")),
        ("variable_0", "Update Δx = λ_i * foo_bar + alpha_beta.", ("variables", "unicode", "punctuation")),
        ("variable_1", "Set cache_key=KV[token-boundary] before reroute.", ("variables", "punctuation")),
        ("punct_0", "Why: (a+b), [c-d], {e/f}; answer -> yes!", ("punctuation", "variables")),
        ("punct_1", "Quote 'x_y' then compare A>=B and C<=D.", ("punctuation", "variables")),
        ("multi_0", "Emoji span 🙂🚀 crosses byte boundaries near λ_i.", ("multi_byte_span", "unicode", "variables")),
        ("multi_1", "漢字東京 mix with µs, Ω, σ, and π≈3.1415.", ("multi_byte_span", "unicode", "math_units", "decimals")),
    ]
    return [StressExample(example_id=example_id, text=text, categories=tuple(categories)) for example_id, text, categories in rows]


def _load_examples(path: pathlib.Path | None) -> list[StressExample]:
    if path is None:
        return _default_examples()
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("examples", raw.get("rows", []))
    examples: list[StressExample] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError("Input examples must be objects with text and categories")
        categories = row.get("categories", ())
        if isinstance(categories, str):
            categories = (categories,)
        examples.append(
            StressExample(
                example_id=str(row.get("example_id", row.get("id", f"example_{index}"))),
                text=str(row["text"]),
                categories=tuple(str(category) for category in categories),
            )
        )
    return examples


def _byte_offsets(text: str) -> list[int]:
    offsets = [0]
    cursor = 0
    for char in text:
        cursor += len(char.encode("utf-8"))
        offsets.append(cursor)
    return offsets


def _token_spans(text: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    char_to_byte = _byte_offsets(text)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        cursor += len(token)
        spans.append((char_to_byte[start], char_to_byte[cursor]))
    if cursor != len(text):
        raise ValueError("Tokenizer segmentation did not consume the full string")
    return spans


def _boundary_positions(spans: Sequence[tuple[int, int]]) -> set[int]:
    return {end for _, end in spans[:-1]}


def _boundary_f1(lhs: set[int], rhs: set[int]) -> float:
    if not lhs and not rhs:
        return 1.0
    return float(2.0 * len(lhs & rhs) / max(len(lhs) + len(rhs), 1))


def _fragmentation(token_count: int, byte_count: int) -> float:
    return float(token_count / max(byte_count, 1))


def _has_multibyte_span(text: str, span: tuple[int, int]) -> bool:
    encoded = text.encode("utf-8")
    try:
        encoded[span[0] : span[1]].decode("ascii")
    except UnicodeDecodeError:
        return True
    return False


def _build_byte_span_remap(
    examples: Sequence[StressExample],
    source: ToyTokenizer,
    target: ToyTokenizer,
    *,
    capacity: int,
) -> dict[str, tuple[str, ...]]:
    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(source.segment(example.text))
    candidates: list[tuple[float, str, tuple[str, ...]]] = []
    for token, count in counts.items():
        target_pieces = tuple(target.segment(token))
        if not token or len(target_pieces) <= 1:
            continue
        byte_len = len(token.encode("utf-8"))
        multibyte_bonus = 0.75 if byte_len != len(token) else 0.0
        score = count * (len(target_pieces) - 1) + multibyte_bonus + 0.02 * byte_len
        candidates.append((float(score), token, target_pieces))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return {token: pieces for _, token, pieces in candidates[: max(0, capacity)]}


def _covered_token_mask(tokens: Sequence[str], remap: dict[str, tuple[str, ...]]) -> list[bool]:
    return [token in remap or len(token) == 1 for token in tokens]


def _stress_flags(text: str) -> dict[str, bool]:
    return {
        "has_unicode": any(ord(char) > 127 for char in text),
        "has_decimal": bool(re.search(r"\d+\.\d+", text)),
        "has_variable": bool(re.search(r"[A-Za-z_]+(?:_[A-Za-z0-9]+)?|[ΔλπσΩ][A-Za-z0-9_]*", text)),
        "has_punctuation": bool(re.search(r"[\[\]{}(),:;?!'\"<>+=*/^|.-]", text)),
        "has_math_unit": any(unit in text for unit in ("µs", "μs", "ms", "m/s", "°C", "kg", "Ω")),
        "has_multibyte": any(ord(char) > 127 and len(char.encode("utf-8")) > 1 for char in text),
    }


def _analyze_example(
    example: StressExample,
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
    remap: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    source_tokens = source.segment(example.text)
    target_tokens = target.segment(example.text)
    source_spans = _token_spans(example.text, source_tokens)
    target_spans = _token_spans(example.text, target_tokens)
    source_boundaries = _boundary_positions(source_spans)
    target_boundaries = _boundary_positions(target_spans)
    byte_count = len(example.text.encode("utf-8"))
    token_id_decoded = target.decode_ids(source.encode_ids(example.text))
    covered_mask = _covered_token_mask(source_tokens, remap)
    covered_bytes = sum(end - start for (start, end), covered in zip(source_spans, covered_mask) if covered)
    multibyte_spans = sum(1 for span in source_spans if _has_multibyte_span(example.text, span))
    return {
        "example_id": example.example_id,
        "text": example.text,
        "categories": list(example.categories),
        "byte_count": byte_count,
        "char_count": len(example.text),
        "source_token_count": len(source_tokens),
        "target_token_count": len(target_tokens),
        "source_fragmentation": _fragmentation(len(source_tokens), byte_count),
        "target_fragmentation": _fragmentation(len(target_tokens), byte_count),
        "fragmentation_delta": _fragmentation(len(target_tokens), byte_count) - _fragmentation(len(source_tokens), byte_count),
        "source_target_boundary_f1": _boundary_f1(source_boundaries, target_boundaries),
        "boundary_jaccard": len(source_boundaries & target_boundaries) / max(len(source_boundaries | target_boundaries), 1),
        "byte_span_remap_coverage": covered_bytes / max(byte_count, 1),
        "remapped_source_token_rate": sum(covered_mask) / max(len(covered_mask), 1),
        "token_id_exact_reconstruction": float(token_id_decoded == example.text),
        "byte_span_exact_reconstruction": 1.0,
        "target_regroup_exact_reconstruction": float("".join(target_tokens) == example.text),
        "multi_byte_source_span_rate": multibyte_spans / max(len(source_spans), 1),
        "stress_flags": _stress_flags(example.text),
    }


def _aggregate_rows(example_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["overall"] = list(example_rows)
    for row in example_rows:
        for category in row["categories"]:
            grouped[category].append(row)

    metrics = (
        "byte_count",
        "source_token_count",
        "target_token_count",
        "source_fragmentation",
        "target_fragmentation",
        "fragmentation_delta",
        "source_target_boundary_f1",
        "boundary_jaccard",
        "byte_span_remap_coverage",
        "remapped_source_token_rate",
        "token_id_exact_reconstruction",
        "byte_span_exact_reconstruction",
        "target_regroup_exact_reconstruction",
        "multi_byte_source_span_rate",
    )
    rows: list[dict[str, Any]] = []
    for category in sorted(grouped, key=lambda name: (name != "overall", name)):
        values = grouped[category]
        rows.append(
            {
                "category": category,
                "example_count": len(values),
                **{metric: float(mean(float(row[metric]) for row in values)) for metric in metrics},
            }
        )
    return rows


def run_analysis(config: TokenizerStressSplitConfig, examples: Sequence[StressExample]) -> dict[str, Any]:
    source, target = _build_tokenizers()
    remap = _build_byte_span_remap(examples, source, target, capacity=config.remap_capacity)
    example_rows = [_analyze_example(example, source=source, target=target, remap=remap) for example in examples]
    category_rows = _aggregate_rows(example_rows)
    return {
        "config": asdict(config),
        "source_tokenizer": source.name,
        "target_tokenizer": target.name,
        "remap_table": {token: list(pieces) for token, pieces in remap.items()},
        "summary": next(row for row in category_rows if row["category"] == "overall"),
        "rows": category_rows,
        "examples": example_rows,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tokenizer Stress Split",
        "",
        f"- Source tokenizer: `{payload['source_tokenizer']}`",
        f"- Target tokenizer: `{payload['target_tokenizer']}`",
        f"- Examples: `{payload['summary']['example_count']}`",
        f"- Remap capacity: `{payload['config']['remap_capacity']}`",
        f"- Overall boundary F1: `{payload['summary']['source_target_boundary_f1']:.4f}`",
        f"- Overall byte-span remap coverage: `{payload['summary']['byte_span_remap_coverage']:.4f}`",
        f"- Token-ID exact reconstruction proxy: `{payload['summary']['token_id_exact_reconstruction']:.4f}`",
        "",
        "## Category Metrics",
        "",
        "| category | n | boundary_f1 | frag_delta | remap_coverage | token_id_exact | multibyte_span_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {category} | {example_count} | {source_target_boundary_f1} | {fragmentation_delta} | "
            "{byte_span_remap_coverage} | {token_id_exact_reconstruction} | {multi_byte_source_span_rate} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretability Notes",
            "",
            "- `source_target_boundary_f1` measures byte-boundary agreement between source and target tokenizations.",
            "- `fragmentation_delta` is target tokens per byte minus source tokens per byte; positive values mean target-side fragmentation is worse.",
            "- `byte_span_remap_coverage` is the byte share covered by a bounded source-token-to-target-piece remap table, with single-character spans treated as trivially coverable.",
            "- `token_id_exact_reconstruction` is the brittle token-ID transfer proxy: source token IDs decoded under the target vocabulary.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic tokenizer mismatch stress split diagnostic.")
    parser.add_argument("--input-json", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=pathlib.Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remap-capacity", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = TokenizerStressSplitConfig(seed=args.seed, remap_capacity=args.remap_capacity)
    examples = _load_examples(args.input_json)
    payload = run_analysis(config, examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(payload))
    print(
        "Tokenizer stress split: "
        f"examples={payload['summary']['example_count']}, "
        f"boundary_f1={payload['summary']['source_target_boundary_f1']:.4f}, "
        f"remap_coverage={payload['summary']['byte_span_remap_coverage']:.4f}, "
        f"token_id_exact={payload['summary']['token_id_exact_reconstruction']:.4f}"
    )
    return payload


if __name__ == "__main__":
    main()
