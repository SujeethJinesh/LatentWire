#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


_DIGITS = "0123456789"
_OPS = "+-*/()"


@dataclass(frozen=True)
class ToyTokenizerBridgeConfig:
    seed: int = 0
    examples: int = 192
    min_terms: int = 3
    max_terms: int = 6
    byte_noise_rate: float = 0.10
    span_noise_rate: float = 0.10


@dataclass(frozen=True)
class ToyTokenizer:
    name: str
    merge_tokens: tuple[str, ...]
    vocab_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {tok: i for i, tok in enumerate(self.vocab_tokens)})
        object.__setattr__(self, "id_to_token", self.vocab_tokens)
        object.__setattr__(self, "_merge_sorted", tuple(sorted(self.merge_tokens, key=lambda tok: (-len(tok), tok))))

    def segment(self, text: str) -> list[str]:
        tokens: list[str] = []
        i = 0
        while i < len(text):
            matched = None
            for token in self._merge_sorted:
                if text.startswith(token, i):
                    matched = token
                    break
            if matched is None:
                tokens.append(text[i])
                i += 1
            else:
                tokens.append(matched)
                i += len(matched)
        return tokens

    def encode_ids(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in self.segment(text)]

    def decode_tokens(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)

    def decode_ids(self, ids: Iterable[int]) -> str:
        tokens = [self.id_to_token[i] if 0 <= i < len(self.id_to_token) else "<unk>" for i in ids]
        return self.decode_tokens(tokens)


def _make_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _build_tokenizers() -> tuple[ToyTokenizer, ToyTokenizer]:
    source_merge_tokens = (
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "42",
        "64",
        "99",
        "100",
        "101",
        "123",
        "256",
        "314",
        "512",
        "777",
        "999",
    )
    target_merge_tokens = (
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "11",
        "12",
        "21",
        "22",
        "34",
        "56",
        "78",
        "88",
        "90",
        "101",
        "111",
        "144",
        "256",
        "314",
        "606",
        "808",
        "909",
    )

    source_vocab = (
        "<unk>",
        "<pad>",
        *tuple(_DIGITS),
        *tuple(_OPS),
        *source_merge_tokens,
    )
    target_vocab = (
        "<unk>",
        "<pad>",
        "(", ")",
        "*",
        "/",
        "-",
        "+",
        *tuple(reversed(_DIGITS)),
        *target_merge_tokens,
    )
    return (
        ToyTokenizer(name="source_a", merge_tokens=source_merge_tokens, vocab_tokens=source_vocab),
        ToyTokenizer(name="target_b", merge_tokens=target_merge_tokens, vocab_tokens=target_vocab),
    )


def _sample_number(rng: random.Random, *, index: int) -> str:
    common_numbers = (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "34",
        "42",
        "56",
        "64",
        "78",
        "88",
        "90",
        "99",
        "100",
        "101",
        "111",
        "123",
        "144",
        "256",
        "314",
        "512",
        "606",
        "777",
        "808",
        "909",
        "999",
    )
    if rng.random() < 0.55:
        return common_numbers[(index * 7 + rng.randrange(len(common_numbers))) % len(common_numbers)]
    value = rng.randint(0, 999)
    if rng.random() < 0.25:
        return f"{value:03d}"
    return str(value)


def _generate_expression(seed: int, index: int, config: ToyTokenizerBridgeConfig) -> str:
    rng = _make_rng(seed * 10_000 + index * 97 + 13)
    terms = config.min_terms + rng.randrange(config.max_terms - config.min_terms + 1)
    numbers = [_sample_number(rng, index=index + i) for i in range(terms)]
    ops = [rng.choice(_OPS[:4]) for _ in range(terms - 1)]
    parts: list[str] = [numbers[0]]
    for op, number in zip(ops, numbers[1:]):
        parts.extend([op, number])
    text = "".join(parts)
    if terms >= 4 and rng.random() < 0.5:
        op_positions = [i for i, ch in enumerate(text) if ch in "+-*/"]
        if op_positions:
            center = op_positions[len(op_positions) // 2]
            left = text.rfind("(", 0, center)
            right = text.find(")", center)
            if left == -1 and right == -1:
                insert_left = max(0, center - 2)
                insert_right = min(len(text), center + 3)
                text = text[:insert_left] + "(" + text[insert_left:insert_right] + ")" + text[insert_right:]
    return text


def _perturb_char(ch: str) -> str:
    if ch in _DIGITS:
        return _DIGITS[(int(ch) + 1) % len(_DIGITS)]
    if ch in _OPS:
        return _OPS[(_OPS.index(ch) + 1) % len(_OPS)]
    return ch


def _should_corrupt(rate: float, *, seed: int, index: int, salt: int) -> bool:
    if rate <= 0.0:
        return False
    period = max(1, round(1.0 / rate))
    return (seed + index * 17 + salt) % period == 0


def _byte_noise(text: str, *, seed: int, index: int) -> str:
    if not text:
        return text
    pos = (seed * 31 + index * 17) % len(text)
    chars = list(text)
    chars[pos] = _perturb_char(chars[pos])
    return "".join(chars)


def _span_noise(text: str, tokenizer: ToyTokenizer, *, seed: int, index: int) -> str:
    tokens = tokenizer.segment(text)
    if not tokens:
        return text
    pos = (seed * 37 + index * 19) % len(tokens)
    chars = list(text)
    cursor = 0
    for token_index, token in enumerate(tokens):
        start = cursor
        end = cursor + len(token)
        if token_index == pos:
            if len(token) > 1:
                chars[start + len(token) - 1] = _perturb_char(chars[start + len(token) - 1])
            else:
                chars[start] = _perturb_char(chars[start])
            break
        cursor = end
    return "".join(chars)


def _token_spans(text: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        cursor += len(token)
        spans.append((start, cursor))
    if cursor != len(text):
        raise ValueError("Tokenizer segmentation did not consume the full string")
    return spans


def _class_accuracy(original: str, reconstructed: str, chars: set[str]) -> float:
    total = 0
    correct = 0
    for i, ch in enumerate(original):
        if ch not in chars:
            continue
        total += 1
        if i < len(reconstructed) and reconstructed[i] == ch:
            correct += 1
    return float(correct / total) if total else 1.0


def _exact_reconstruction(original: str, reconstructed: str) -> float:
    return 1.0 if original == reconstructed else 0.0


def _fragmentation_rate(reconstructed: str, tokenizer: ToyTokenizer) -> float:
    tokens = tokenizer.segment(reconstructed)
    return float(len(tokens) / max(len(reconstructed), 1))


def _estimated_bytes(
    method: str,
    text: str,
    source_tokens: Sequence[str],
    target_vocab: set[str],
) -> float:
    if method == "token_id":
        return float(1 + 2 * len(source_tokens))
    if method == "vocab_remap":
        shared = sum(1 for token in source_tokens if token in target_vocab)
        return float(1 + shared + sum(2 + len(token.encode("utf-8")) for token in source_tokens if token not in target_vocab))
    if method == "byte_span_canonical":
        return float(1 + len(text.encode("utf-8")) + 4 * len(source_tokens))
    if method == "byte_span_noisy_bytes":
        return float(2 + len(text.encode("utf-8")) + 4 * len(source_tokens))
    if method == "byte_span_noisy_spans":
        return float(2 + len(text.encode("utf-8")) + 4 * len(source_tokens))
    raise ValueError(f"Unknown method: {method}")


def _evaluate_methods(
    config: ToyTokenizerBridgeConfig,
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = ("token_id", "vocab_remap", "byte_span_canonical", "byte_span_noisy_bytes", "byte_span_noisy_spans")

    for method in methods:
        total_exact = 0.0
        total_digit = 0.0
        total_operator = 0.0
        total_fragmentation = 0.0
        total_bytes = 0.0
        total_chars = 0
        total_source_tokens = 0
        total_target_tokens = 0

        for index in range(config.examples):
            text = _generate_expression(config.seed, index, config)
            source_tokens = source.segment(text)
            source_ids = source.encode_ids(text)

            if method == "token_id":
                reconstructed = target.decode_ids(source_ids)
            elif method == "vocab_remap":
                remapped = [token if token in target.token_to_id else "<unk>" for token in source_tokens]
                reconstructed = target.decode_tokens(remapped)
            elif method == "byte_span_canonical":
                reconstructed = text
            elif method == "byte_span_noisy_bytes":
                reconstructed = _byte_noise(text, seed=config.seed, index=index) if _should_corrupt(
                    config.byte_noise_rate, seed=config.seed, index=index, salt=11
                ) else text
            elif method == "byte_span_noisy_spans":
                reconstructed = _span_noise(text, source, seed=config.seed, index=index) if _should_corrupt(
                    config.span_noise_rate, seed=config.seed, index=index, salt=29
                ) else text
            else:  # pragma: no cover - guarded above.
                raise ValueError(method)

            target_tokens = target.segment(reconstructed)
            total_exact += _exact_reconstruction(text, reconstructed)
            total_digit += _class_accuracy(text, reconstructed, set(_DIGITS))
            total_operator += _class_accuracy(text, reconstructed, set(_OPS))
            total_fragmentation += len(target_tokens) / max(len(text), 1)
            total_bytes += _estimated_bytes(method, text, source_tokens, set(target.token_to_id))
            total_chars += len(text)
            total_source_tokens += len(source_tokens)
            total_target_tokens += len(target_tokens)

        count = float(config.examples)
        rows.append(
            {
                "method": method,
                "exact_reconstruction": total_exact / count,
                "digit_accuracy": total_digit / count,
                "operator_accuracy": total_operator / count,
                "fragmentation_rate": total_fragmentation / count,
                "bytes_per_example": total_bytes / count,
                "avg_source_tokens": total_source_tokens / count,
                "avg_target_tokens": total_target_tokens / count,
                "avg_chars": total_chars / count,
            }
        )
    return rows


def write_json_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Tokenizer Bridge",
        "",
        f"- Seed: `{payload['config']['seed']}`",
        f"- Examples: `{payload['config']['examples']}`",
        f"- Source tokenizer: `{payload['source_tokenizer']}`",
        f"- Target tokenizer: `{payload['target_tokenizer']}`",
        "",
        "| Method | Exact recon | Digit acc | Operator acc | Fragmentation | Bytes/example |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {exact_reconstruction:.4f} | {digit_accuracy:.4f} | {operator_accuracy:.4f} | {fragmentation_rate:.4f} | {bytes_per_example:.2f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy tokenizer/vocab bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=192)
    parser.add_argument("--min-terms", type=int, default=3)
    parser.add_argument("--max-terms", type=int, default=6)
    parser.add_argument("--byte-noise-rate", type=float, default=0.10)
    parser.add_argument("--span-noise-rate", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyTokenizerBridgeConfig(
        seed=args.seed,
        examples=args.examples,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
        byte_noise_rate=args.byte_noise_rate,
        span_noise_rate=args.span_noise_rate,
    )
    source, target = _build_tokenizers()
    rows = _evaluate_methods(config, source=source, target=target)
    payload = {
        "config": asdict(config),
        "source_tokenizer": source.name,
        "target_tokenizer": target.name,
        "rows": rows,
    }
    write_json_summary(payload, pathlib.Path(args.output))
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    return payload


if __name__ == "__main__":
    main()
