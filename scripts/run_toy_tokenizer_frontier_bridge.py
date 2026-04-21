#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


METHODS: tuple[str, ...] = (
    "token_id",
    "frontier_regroup",
    "learned_remap",
)

_JOINERS = ("-", "_", "/")
_NUMBERS = ("0", "1", "2", "7", "10", "11", "42", "64", "99", "128", "256")

_SOURCE_WORDS = (
    "prefix",
    "frontier",
    "token",
    "bridge",
    "latent",
    "compress",
    "routing",
    "cache",
    "signal",
    "update",
    "vector",
    "query",
    "context",
    "search",
    "buffer",
    "memory",
)

_TARGET_MERGES = (
    "pref",
    "ix",
    "fro",
    "ntier",
    "to",
    "ken",
    "brid",
    "ge",
    "lat",
    "ent",
    "comp",
    "ress",
    "rou",
    "ting",
    "ca",
    "che",
    "sig",
    "nal",
    "up",
    "date",
    "vec",
    "tor",
    "que",
    "ry",
    "con",
    "text",
    "sea",
    "rch",
    "buf",
    "fer",
    "mem",
    "ory",
)


@dataclass(frozen=True)
class ToyTokenizerFrontierBridgeConfig:
    seed: int = 0
    train_examples: int = 96
    test_examples: int = 96
    min_segments: int = 3
    max_segments: int = 5
    remap_capacity: int = 10


@dataclass(frozen=True)
class ToyTokenizer:
    name: str
    merge_tokens: tuple[str, ...]
    vocab_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.vocab_tokens)})
        object.__setattr__(self, "id_to_token", self.vocab_tokens)
        object.__setattr__(self, "_merge_sorted", tuple(sorted(self.merge_tokens, key=lambda token: (-len(token), token))))

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
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unk) for token in self.segment(text)]

    def decode_tokens(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)

    def decode_ids(self, ids: Iterable[int]) -> str:
        tokens = [self.id_to_token[i] if 0 <= i < len(self.id_to_token) else "<unk>" for i in ids]
        return self.decode_tokens(tokens)


def _make_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _build_tokenizers() -> tuple[ToyTokenizer, ToyTokenizer]:
    source_vocab = ("<unk>", "<pad>", *_JOINERS, *_NUMBERS, *_SOURCE_WORDS)
    target_vocab = ("<unk>", "<pad>", *_JOINERS, *_NUMBERS, *_TARGET_MERGES)
    return (
        ToyTokenizer(name="source_frontier", merge_tokens=_SOURCE_WORDS, vocab_tokens=source_vocab),
        ToyTokenizer(name="target_frontier", merge_tokens=_TARGET_MERGES, vocab_tokens=target_vocab),
    )


def _sample_word(rng: random.Random, index: int) -> str:
    weighted_pool = (
        "prefix",
        "bridge",
        "token",
        "cache",
        "query",
        "latent",
        "prefix",
        "bridge",
        "token",
        "cache",
        "query",
        "latent",
        "frontier",
        "compress",
        "routing",
        "vector",
        "search",
        "context",
        "buffer",
        "memory",
        "update",
        "signal",
    )
    return weighted_pool[(index * 7 + rng.randrange(len(weighted_pool))) % len(weighted_pool)]


def _generate_example(config: ToyTokenizerFrontierBridgeConfig, index: int, *, split: str) -> str:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    offset = 0 if split == "train" else 1_000_000
    rng = _make_rng(config.seed * 10_000 + offset + index * 97 + 13)
    segments = config.min_segments + rng.randrange(config.max_segments - config.min_segments + 1)
    parts: list[str] = []
    for segment_index in range(segments):
        word = _sample_word(rng, index + segment_index)
        parts.append(word)
        if segment_index < segments - 1:
            parts.append(_JOINERS[(config.seed + index + segment_index) % len(_JOINERS)])
    parts.append(_NUMBERS[(config.seed * 3 + index * 5 + segments) % len(_NUMBERS)])
    return "".join(parts)


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


def _boundary_positions(text: str, tokens: Sequence[str]) -> set[int]:
    spans = _token_spans(text, tokens)
    return {end for _, end in spans[:-1]}


def _boundary_f1(lhs: set[int], rhs: set[int]) -> float:
    if not lhs and not rhs:
        return 1.0
    intersection = len(lhs & rhs)
    return float(2.0 * intersection / max(len(lhs) + len(rhs), 1))


def _token_accuracy(reference: Sequence[str], candidate: Sequence[str]) -> float:
    if not reference:
        return 1.0
    matches = sum(1 for ref, cand in zip(reference, candidate) if ref == cand)
    return float(matches / len(reference))


def _build_remap_table(
    config: ToyTokenizerFrontierBridgeConfig,
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
) -> tuple[dict[str, tuple[str, ...]], int, int]:
    counts: Counter[str] = Counter()
    for index in range(config.train_examples):
        text = _generate_example(config, index, split="train")
        counts.update(source.segment(text))

    scored: list[tuple[float, str, tuple[str, ...]]] = []
    for token, count in counts.items():
        target_tokens = tuple(target.segment(token))
        if len(target_tokens) <= 1:
            continue
        savings = float(len(target_tokens) - 1)
        score = float(count) * savings - 0.25 * len(token)
        if score > 0.0:
            scored.append((score, token, target_tokens))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: max(0, int(config.remap_capacity))]

    table: dict[str, tuple[str, ...]] = {}
    table_bytes = 0
    covered_instances = 0
    for _, token, target_tokens in selected:
        table[token] = target_tokens
        covered_instances += counts[token]
        table_bytes += 4 + len(token.encode("utf-8")) + sum(len(piece.encode("utf-8")) for piece in target_tokens)
    return table, table_bytes, covered_instances


def _decode_method(
    method: str,
    text: str,
    source: ToyTokenizer,
    target: ToyTokenizer,
    remap_table: dict[str, tuple[str, ...]] | None,
) -> tuple[str, float, int, int]:
    source_tokens = source.segment(text)
    if method == "token_id":
        return target.decode_ids(source.encode_ids(text)), 0.0, 0, 0
    if method == "frontier_regroup":
        target_tokens = target.segment(text)
        return target.decode_tokens(target_tokens), 1.0, 0, 0
    if method == "learned_remap":
        assert remap_table is not None
        decoded_tokens: list[str] = []
        remapped_instances = 0
        fallback_instances = 0
        for token in source_tokens:
            if token in remap_table:
                decoded_tokens.extend(remap_table[token])
                remapped_instances += 1
            else:
                decoded_tokens.extend(target.segment(token))
                fallback_instances += 1
        return target.decode_tokens(decoded_tokens), 1.0, remapped_instances, fallback_instances
    raise ValueError(f"Unknown method: {method}")


def _estimate_bytes(
    method: str,
    *,
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    remapped_instances: int,
    fallback_instances: int,
    table_bytes_amortized: float,
    target: ToyTokenizer,
    remap_table: dict[str, tuple[str, ...]] | None,
) -> float:
    if method == "token_id":
        return float(1 + 2 * len(source_tokens))
    if method == "frontier_regroup":
        return float(1 + len(target_tokens))
    if method == "learned_remap":
        assert remap_table is not None
        fallback_tokens = sum(len(target.segment(token)) for token in source_tokens if token not in remap_table)
        return float(1 + remapped_instances + fallback_tokens + table_bytes_amortized)
    raise ValueError(f"Unknown method: {method}")


def run_experiment(
    config: ToyTokenizerFrontierBridgeConfig,
    methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    source, target = _build_tokenizers()
    remap_table, table_bytes, covered_instances = _build_remap_table(config, source=source, target=target)
    method_list = tuple(methods) if methods is not None else METHODS

    rows: list[dict[str, Any]] = []
    for method in method_list:
        total_exact = 0.0
        total_boundary_f1 = 0.0
        total_source_target_boundary_f1 = 0.0
        total_bytes = 0.0
        total_source_tokens = 0
        total_target_tokens = 0
        total_remapped_instances = 0
        total_token_accuracy = 0.0
        total_examples = float(config.test_examples)

        for index in range(config.test_examples):
            text = _generate_example(config, index, split="test")
            source_tokens = source.segment(text)
            target_tokens = target.segment(text)
            decoded, token_accuracy, remapped_instances, fallback_instances = _decode_method(
                method,
                text,
                source,
                target,
                remap_table if method == "learned_remap" else None,
            )
            decoded_tokens = target.segment(decoded)
            source_boundary = _boundary_positions(text, source_tokens)
            target_boundary = _boundary_positions(text, target_tokens)
            decoded_boundary = _boundary_positions(decoded, decoded_tokens)

            total_exact += 1.0 if decoded == text else 0.0
            total_boundary_f1 += _boundary_f1(target_boundary, decoded_boundary)
            total_source_target_boundary_f1 += _boundary_f1(source_boundary, target_boundary)
            total_bytes += _estimate_bytes(
                method,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                remapped_instances=remapped_instances,
                fallback_instances=fallback_instances,
                table_bytes_amortized=table_bytes / (total_examples * 4.0) if method == "learned_remap" else 0.0,
                target=target,
                remap_table=remap_table if method == "learned_remap" else None,
            )
            total_source_tokens += len(source_tokens)
            total_target_tokens += len(target_tokens)
            total_remapped_instances += remapped_instances
            total_token_accuracy += token_accuracy

        rows.append(
            {
                "method": method,
                "exact_reconstruction": total_exact / total_examples,
                "decoded_boundary_f1": total_boundary_f1 / total_examples,
                "source_target_boundary_f1": total_source_target_boundary_f1 / total_examples,
                "bytes_per_example": total_bytes / total_examples,
                "avg_source_tokens": total_source_tokens / total_examples,
                "avg_target_tokens": total_target_tokens / total_examples,
                "learned_remap_coverage": (
                    total_remapped_instances / total_source_tokens if method == "learned_remap" and total_source_tokens else 0.0
                ),
                "token_accuracy": total_token_accuracy / total_examples,
                "remap_table_size": len(remap_table) if method == "learned_remap" else 0,
                "remap_table_bytes": table_bytes if method == "learned_remap" else 0,
                "remap_instances": total_remapped_instances if method == "learned_remap" else 0,
                "remap_covered_train_instances": covered_instances if method == "learned_remap" else 0,
            }
        )
    return rows


def write_json_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Tokenizer Frontier Bridge",
        "",
        f"- Seed: `{payload['config']['seed']}`",
        f"- Train examples: `{payload['config']['train_examples']}`",
        f"- Test examples: `{payload['config']['test_examples']}`",
        f"- Source tokenizer: `{payload['source_tokenizer']}`",
        f"- Target tokenizer: `{payload['target_tokenizer']}`",
        "",
        "| Method | Exact recon | Decoded boundary F1 | Source-target boundary F1 | Bytes/example | Learned remap coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {exact_reconstruction:.4f} | {decoded_boundary_f1:.4f} | {source_target_boundary_f1:.4f} | {bytes_per_example:.2f} | {learned_remap_coverage:.4f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy tokenizer frontier bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--test-examples", type=int, default=96)
    parser.add_argument("--min-segments", type=int, default=3)
    parser.add_argument("--max-segments", type=int, default=5)
    parser.add_argument("--remap-capacity", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyTokenizerFrontierBridgeConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        remap_capacity=args.remap_capacity,
    )
    source, target = _build_tokenizers()
    rows = run_experiment(config, METHODS)
    payload = {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "source_tokenizer": source.name,
        "target_tokenizer": target.name,
    }
    write_json_summary(payload, pathlib.Path(args.output))
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    return payload


if __name__ == "__main__":
    main()
