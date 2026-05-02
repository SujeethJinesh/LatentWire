#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


METHODS: tuple[str, ...] = (
    "token_id",
    "regroup_baseline",
    "shared_byte_span_remap_route_atoms",
    "oracle_shared_byte_span_route_atoms",
)

_JOINERS = ("-", "_", "/")
_DIGITS = tuple("0123456789")
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
class ToySharedByteSpanRouteAtomRemapConfig:
    seed: int = 0
    calibration_examples: int = 160
    test_examples: int = 128
    min_atoms: int = 4
    max_atoms: int = 6
    protected_atoms: int = 4
    remap_capacity: int = 10
    low_bits: int = 3
    high_bits: int = 8
    signal_scale: float = 3.2
    distractor_scale: float = 7.8
    activation_noise: float = 0.04
    calibration_noise: float = 0.03
    label_noise: float = 0.16


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


def _make_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _build_tokenizers() -> tuple[ToyTokenizer, ToyTokenizer]:
    source_vocab = ("<unk>", "<pad>", *_JOINERS, *_DIGITS, *_NUMBERS, *_SOURCE_WORDS)
    target_vocab = ("<unk>", "<pad>", *_JOINERS, *_DIGITS, *_NUMBERS, *_TARGET_MERGES)
    return (
        ToyTokenizer(name="source", merge_tokens=_SOURCE_WORDS, vocab_tokens=source_vocab),
        ToyTokenizer(name="target", merge_tokens=_TARGET_MERGES, vocab_tokens=target_vocab),
    )


def _sample_atom_word(rng: random.Random, index: int) -> str:
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


def _generate_example(config: ToySharedByteSpanRouteAtomRemapConfig, index: int, *, split: str) -> str:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    offset = 0 if split == "train" else 1_000_000
    rng = _make_rng(config.seed * 10_000 + offset + index * 97 + 13)
    atoms = config.min_atoms + rng.randrange(config.max_atoms - config.min_atoms + 1)
    parts: list[str] = []
    for atom_index in range(atoms):
        parts.append(_sample_atom_word(rng, index + atom_index))
        if atom_index < atoms - 1:
            parts.append(_JOINERS[(config.seed + index + atom_index) % len(_JOINERS)])
    parts.append(_NUMBERS[(config.seed * 3 + index * 5 + atoms) % len(_NUMBERS)])
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


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _select_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    k = max(0, min(int(k), int(values.numel())))
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    return torch.topk(values, k=k, largest=True).indices.sort().values


def _make_mask(indices: torch.Tensor, atoms: int) -> torch.Tensor:
    mask = torch.zeros(atoms, dtype=torch.bool, device=indices.device)
    if indices.numel() > 0:
        mask[indices] = True
    return mask


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _build_problem(
    config: ToySharedByteSpanRouteAtomRemapConfig,
    source: ToyTokenizer,
    target: ToyTokenizer,
) -> dict[str, Any]:
    gen = _make_generator(config.seed)
    token_pool = tuple(dict.fromkeys([*_SOURCE_WORDS, *_JOINERS, *_DIGITS, *_NUMBERS]))
    token_vectors: dict[str, torch.Tensor] = {}
    token_utility: dict[str, float] = {}
    token_class: dict[str, int] = {}

    class_count = 4
    prototypes = torch.randn(class_count, 16, generator=gen, dtype=torch.float32)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    classifier = prototypes.T.contiguous()

    for idx, token in enumerate(token_pool):
        signal_rank = idx % class_count
        base = torch.randn(16, generator=gen, dtype=torch.float32)
        if token in _SOURCE_WORDS[:6]:
            token_vectors[token] = config.signal_scale * (prototypes[signal_rank] + 0.18 * base)
            token_utility[token] = 1.4 + 0.18 * (6 - (idx % 6))
            token_class[token] = signal_rank
        elif token in _SOURCE_WORDS[6:12]:
            token_vectors[token] = 0.55 * base
            token_utility[token] = 0.2
            token_class[token] = -1
        elif token in _SOURCE_WORDS[12:]:
            token_vectors[token] = config.distractor_scale * (base - base.mean())
            token_utility[token] = -0.55
            token_class[token] = -1
        elif token in _JOINERS:
            token_vectors[token] = 0.18 * base
            token_utility[token] = -0.10
            token_class[token] = -1
        elif token in _DIGITS:
            token_vectors[token] = 0.12 * base
            token_utility[token] = -0.02
            token_class[token] = -1
        else:
            token_vectors[token] = 0.14 * base
            token_utility[token] = -0.05
            token_class[token] = -1

    def _sample_split(count: int, *, seed_offset: int) -> list[dict[str, Any]]:
        rng = _make_rng(config.seed + seed_offset)
        examples: list[dict[str, Any]] = []
        for index in range(count):
            text = _generate_example(config, index, split="train" if seed_offset < 20_000 else "test")
            source_tokens = source.segment(text)
            target_tokens = target.segment(text)
            activations = []
            values = []
            utilities = []
            for pos, token in enumerate(source_tokens):
                token_vec = token_vectors[token]
                token_util = token_utility[token]
                token_rng = _make_generator(config.seed + seed_offset + index * 31 + pos)
                token_noise = torch.randn(token_vec.shape, generator=token_rng)
                activation = 0.14 + float(config.activation_noise) * torch.rand(1, generator=token_rng).item() + max(token_util, 0.0)
                activation += 0.22 if token in _SOURCE_WORDS[:6] else 0.0
                activation = max(0.0, activation)
                activations.append(float(activation))
                values.append((token_vec + 0.03 * token_noise).float())
                utilities.append(float(token_util))
            activation_tensor = torch.tensor(activations, dtype=torch.float32)
            calibration_rng = _make_generator(config.seed + seed_offset + index * 131 + 7)
            activation_tensor = activation_tensor + float(config.calibration_noise) * torch.randn(
                activation_tensor.shape, generator=calibration_rng, dtype=activation_tensor.dtype
            )
            activation_tensor = activation_tensor.clamp_min(0.0)
            activation_tensor = activation_tensor / activation_tensor.sum().clamp_min(1e-8)
            value_tensor = torch.stack(values, dim=0)
            summary = torch.einsum("a,ad->d", activation_tensor, value_tensor) / activation_tensor.sum().clamp_min(1e-8)
            label = torch.argmax(summary @ classifier, dim=-1)
            examples.append(
                {
                    "text": text,
                    "source_tokens": source_tokens,
                    "target_tokens": target_tokens,
                    "activations": activation_tensor.float(),
                    "values": value_tensor.float(),
                    "utilities": torch.tensor(utilities, dtype=torch.float32),
                    "summary": summary.float(),
                    "label": label.long(),
                }
            )
        return examples

    return {
        "token_vectors": token_vectors,
        "token_utility": token_utility,
        "token_class": token_class,
        "classifier": classifier,
        "train": _sample_split(config.calibration_examples, seed_offset=11_000),
        "test": _sample_split(config.test_examples, seed_offset=21_000),
    }


def _build_remap_table(
    config: ToySharedByteSpanRouteAtomRemapConfig,
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
    train_examples: Sequence[dict[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], int, dict[str, float]]:
    counts: Counter[str] = Counter()
    energy: defaultdict[str, float] = defaultdict(float)
    target_piece_count: dict[str, int] = {}
    for example in train_examples:
        for token, activation in zip(example["source_tokens"], example["activations"].tolist()):
            counts[token] += 1
            energy[token] += float(activation)
            if token not in target_piece_count:
                target_piece_count[token] = len(target.segment(token))

    scored: list[tuple[float, str, tuple[str, ...]]] = []
    for token, count in counts.items():
        target_tokens = tuple(target.segment(token))
        if len(target_tokens) <= 1:
            continue
        savings = float(len(target_tokens) - 1)
        score = float(count) * savings - 0.15 * len(token)
        if score > 0.0:
            scored.append((score, token, target_tokens))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: max(0, int(config.remap_capacity))]

    table: dict[str, tuple[str, ...]] = {}
    table_bytes = 0
    for _, token, target_tokens in selected:
        table[token] = target_tokens
        table_bytes += 4 + len(token.encode("utf-8")) + sum(len(piece.encode("utf-8")) for piece in target_tokens)

    avg_energy = {token: energy[token] / max(counts[token], 1) for token in counts}
    return table, table_bytes, avg_energy


def _select_methods(
    *,
    config: ToySharedByteSpanRouteAtomRemapConfig,
    example: dict[str, Any],
    source: ToyTokenizer,
    target: ToyTokenizer,
    remap_table: dict[str, tuple[str, ...]],
    avg_energy: dict[str, float],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    source_tokens = example["source_tokens"]
    scores: dict[str, torch.Tensor] = {}
    for method in METHODS:
        if method == "token_id":
            score_values = torch.tensor(
                [1.0 / (1.0 + source.token_to_id.get(token, 0)) for token in source_tokens], dtype=torch.float32
            )
        elif method == "regroup_baseline":
            score_values = torch.tensor(
                [1.0 / max(len(target.segment(token)), 1) + 0.08 * avg_energy.get(token, 0.0) for token in source_tokens],
                dtype=torch.float32,
            )
        elif method == "shared_byte_span_remap_route_atoms":
            score_values = torch.tensor(
                [
                    (1.0 + (1.15 if token in remap_table else 0.0))
                    * (0.55 + avg_energy.get(token, 0.0))
                    / max(len(target.segment(token)), 1)
                    for token in source_tokens
                ],
                dtype=torch.float32,
            )
        else:
            score_values = torch.full((len(source_tokens),), float("-inf"), dtype=torch.float32)
            for atom_index in range(len(source_tokens)):
                mask = torch.zeros(len(source_tokens), dtype=torch.bool)
                mask[atom_index] = True
                score_values[atom_index] = _oracle_gain(
                    example,
                    source=source,
                    target=target,
                    remap_table=remap_table,
                    protected_mask=mask,
                    config=config,
                )
        scores[method] = score_values

    masks: dict[str, torch.Tensor] = {}
    for method in METHODS:
        selected = _select_topk(scores[method], config.protected_atoms)
        masks[method] = _make_mask(selected, len(source_tokens))
    return masks, scores


def _oracle_gain(
    example: dict[str, Any],
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
    remap_table: dict[str, tuple[str, ...]],
    protected_mask: torch.Tensor,
    config: ToySharedByteSpanRouteAtomRemapConfig,
) -> float:
    atom_values = example["values"]
    recon = _predict_summary(
        example,
        atom_values=atom_values,
        protected_mask=protected_mask,
        remap_table=remap_table,
        source=source,
        target=target,
        config=config,
        method="oracle_shared_byte_span_route_atoms",
    )
    return float(-F.mse_loss(recon, example["summary"]).item())


def _protected_mask_for_oracle(
    example: dict[str, Any],
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
    remap_table: dict[str, tuple[str, ...]],
    config: ToySharedByteSpanRouteAtomRemapConfig,
) -> torch.Tensor:
    scores = torch.full((len(example["source_tokens"]),), float("-inf"), dtype=torch.float32)
    for atom_index in range(len(example["source_tokens"])):
        mask = torch.zeros(len(example["source_tokens"]), dtype=torch.bool)
        mask[atom_index] = True
        scores[atom_index] = _oracle_gain(
            example,
            source=source,
            target=target,
            remap_table=remap_table,
            protected_mask=mask,
            config=config,
        )
    return _make_mask(_select_topk(scores, config.protected_atoms), len(scores))


def _predict_summary(
    example: dict[str, Any],
    *,
    atom_values: torch.Tensor,
    protected_mask: torch.Tensor,
    remap_table: dict[str, tuple[str, ...]],
    source: ToyTokenizer,
    target: ToyTokenizer,
    config: ToySharedByteSpanRouteAtomRemapConfig,
    method: str,
) -> torch.Tensor:
    recon_values = atom_values.clone()
    low_mask = ~protected_mask
    if low_mask.any():
        recon_values[low_mask] = _symmetric_quantize(atom_values[low_mask], config.low_bits)
    if protected_mask.any():
        if method == "shared_byte_span_remap_route_atoms":
            bonus = torch.tensor(
                [1.0 if token in remap_table else 0.0 for token in example["source_tokens"]],
                dtype=torch.float32,
            ).unsqueeze(-1)
            recon_values[protected_mask] = _symmetric_quantize(atom_values[protected_mask] * (1.0 + 0.05 * bonus[protected_mask]), config.high_bits)
        elif method == "regroup_baseline":
            recon_values[protected_mask] = _symmetric_quantize(atom_values[protected_mask], config.high_bits)
        elif method == "token_id":
            recon_values[protected_mask] = _symmetric_quantize(atom_values[protected_mask] * 0.98, config.high_bits)
        else:
            recon_values[protected_mask] = _symmetric_quantize(atom_values[protected_mask], config.high_bits)
    kept_activations = example["activations"] * torch.ones_like(example["activations"])
    if protected_mask.any():
        kept_activations = example["activations"]
    return torch.einsum("a,ad->d", kept_activations, recon_values) / kept_activations.sum().clamp_min(1e-8)


def _atom_recovery(selected: torch.Tensor, oracle: torch.Tensor) -> float:
    sel = set(int(i) for i in torch.where(selected)[0].tolist())
    ref = set(int(i) for i in torch.where(oracle)[0].tolist())
    if not ref:
        return 1.0 if not sel else 0.0
    return len(sel & ref) / float(len(ref))


def _boundary_metric(
    text: str,
    *,
    source: ToyTokenizer,
    target: ToyTokenizer,
    selected_mask: torch.Tensor,
    remap_table: dict[str, tuple[str, ...]],
    method: str,
) -> float:
    source_tokens = source.segment(text)
    selected = [token for token, keep in zip(source_tokens, selected_mask.tolist()) if keep]
    if not selected:
        return 1.0

    source_boundary_sets = []
    target_boundary_sets = []
    method_boundary_sets = []
    for token in selected:
        source_boundary_sets.append(_boundary_positions(token, source.segment(token)))
        target_boundary_sets.append(_boundary_positions(token, target.segment(token)))
        if method == "token_id":
            method_boundary_sets.append(_boundary_positions(token, source.segment(token)))
        elif method == "regroup_baseline":
            method_boundary_sets.append(_boundary_positions(token, target.segment(token)))
        else:
            interface_tokens = target.segment(token) if token in remap_table else source.segment(token)
            method_boundary_sets.append(_boundary_positions(token, interface_tokens))

    source_boundaries = set().union(*source_boundary_sets)
    target_boundaries = set().union(*target_boundary_sets)
    method_boundaries = set().union(*method_boundary_sets)
    del source_boundaries
    return _boundary_f1(method_boundaries, target_boundaries)


def _bytes_proxy(
    *,
    example: dict[str, Any],
    selected_mask: torch.Tensor,
    remap_table: dict[str, tuple[str, ...]],
    target: ToyTokenizer,
    method: str,
    config: ToySharedByteSpanRouteAtomRemapConfig,
    amortized_table_bytes: float,
) -> float:
    source_tokens = example["source_tokens"]
    base_bytes = 0.0
    for token in source_tokens:
        base_bytes += float(math.ceil(len(token.encode("utf-8")) * config.low_bits / 8.0))
    extra = 0.0
    for i, token in enumerate(source_tokens):
        if not selected_mask[i]:
            continue
        if method == "token_id":
            extra += 4.0 + len(token.encode("utf-8"))
        elif method == "regroup_baseline":
            extra += 4.0 + sum(len(piece.encode("utf-8")) for piece in target.segment(token))
        elif method == "shared_byte_span_remap_route_atoms":
            if token in remap_table:
                extra += 2.0
            else:
                extra += 4.0 + sum(len(piece.encode("utf-8")) for piece in target.segment(token))
        else:
            extra += 2.0
    return float(base_bytes + extra + amortized_table_bytes)


def run_experiment(config: ToySharedByteSpanRouteAtomRemapConfig) -> dict[str, Any]:
    source, target = _build_tokenizers()
    problem = _build_problem(config, source, target)
    remap_table, remap_table_bytes, avg_energy = _build_remap_table(
        config,
        source=source,
        target=target,
        train_examples=problem["train"],
    )
    amortized_table_bytes = float(remap_table_bytes / max(len(problem["test"]), 1))

    token_freq: Counter[str] = Counter()
    for example in problem["train"]:
        token_freq.update(example["source_tokens"])

    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for example in problem["test"]:
        selected_masks, _ = _select_methods(
            config=config,
            example=example,
            source=source,
            target=target,
            remap_table=remap_table,
            avg_energy=avg_energy,
        )
        oracle_mask = _protected_mask_for_oracle(
            example,
            source=source,
            target=target,
            remap_table=remap_table,
            config=config,
        )
        token_id_mask = selected_masks["token_id"]
        regroup_mask = selected_masks["regroup_baseline"]
        token_id_recon = _predict_summary(
            example,
            atom_values=example["values"],
            protected_mask=token_id_mask,
            remap_table=remap_table,
            source=source,
            target=target,
            config=config,
            method="token_id",
        )
        regroup_recon = _predict_summary(
            example,
            atom_values=example["values"],
            protected_mask=regroup_mask,
            remap_table=remap_table,
            source=source,
            target=target,
            config=config,
            method="regroup_baseline",
        )
        token_id_correct = bool(torch.argmax(token_id_recon @ problem["classifier"], dim=-1).item() == int(example["label"].item()))
        regroup_correct = bool(torch.argmax(regroup_recon @ problem["classifier"], dim=-1).item() == int(example["label"].item()))

        for method in METHODS:
            selected_mask = selected_masks[method]
            recon = _predict_summary(
                example,
                atom_values=example["values"],
                protected_mask=selected_mask,
                remap_table=remap_table,
                source=source,
                target=target,
                config=config,
                method=method,
            )
            pred = torch.argmax(recon @ problem["classifier"], dim=-1)
            correct = bool(pred.item() == int(example["label"].item()))
            boundary_f1 = _boundary_metric(
                example["text"],
                source=source,
                target=target,
                selected_mask=selected_mask,
                remap_table=remap_table,
                method=method,
            )
            remap_coverage = 0.0
            selected_indices = torch.where(selected_mask)[0].tolist()
            if selected_indices:
                remap_coverage = sum(1 for idx in selected_indices if example["source_tokens"][idx] in remap_table) / float(
                    len(selected_indices)
                )
            atom_recovery = _atom_recovery(selected_mask, oracle_mask)
            bytes_proxy = _bytes_proxy(
                example=example,
                selected_mask=selected_mask,
                remap_table=remap_table,
                target=target,
                method=method,
                config=config,
                amortized_table_bytes=amortized_table_bytes if method == "shared_byte_span_remap_route_atoms" else 0.0,
            )
            row = {
                "method": method,
                "seed": int(config.seed),
                "task_accuracy": float(correct),
                "mse": float(F.mse_loss(recon, example["summary"]).item()),
                "source_target_boundary_f1": float(boundary_f1),
                "remap_coverage": float(remap_coverage),
                "atom_recovery": float(atom_recovery),
                "bytes_proxy": float(bytes_proxy),
                "protected_atoms": int(selected_mask.sum().item()),
                "help_vs_token_id": float(correct and not token_id_correct),
                "harm_vs_token_id": float((not correct) and token_id_correct),
                "help_vs_regroup_baseline": float(correct and not regroup_correct),
                "harm_vs_regroup_baseline": float((not correct) and regroup_correct),
            }
            per_method[method].append(row)

    summary_rows: list[dict[str, Any]] = []
    baseline_lookup = {}
    regroup_lookup = {}
    for row in per_method["token_id"]:
        baseline_lookup.setdefault("task_accuracy", 0.0)
        baseline_lookup["task_accuracy"] += float(row["task_accuracy"])
        baseline_lookup.setdefault("mse", 0.0)
        baseline_lookup["mse"] += float(row["mse"])
    for row in per_method["regroup_baseline"]:
        regroup_lookup.setdefault("task_accuracy", 0.0)
        regroup_lookup["task_accuracy"] += float(row["task_accuracy"])
        regroup_lookup.setdefault("mse", 0.0)
        regroup_lookup["mse"] += float(row["mse"])
    token_id_acc = baseline_lookup["task_accuracy"] / max(len(per_method["token_id"]), 1)
    token_id_mse = baseline_lookup["mse"] / max(len(per_method["token_id"]), 1)
    regroup_acc = regroup_lookup["task_accuracy"] / max(len(per_method["regroup_baseline"]), 1)
    regroup_mse = regroup_lookup["mse"] / max(len(per_method["regroup_baseline"]), 1)

    for method in METHODS:
        rows = per_method[method]
        summary = {
            "method": method,
            "seed": int(config.seed),
            "task_accuracy": float(sum(float(row["task_accuracy"]) for row in rows) / len(rows)),
            "mse": float(sum(float(row["mse"]) for row in rows) / len(rows)),
            "source_target_boundary_f1": float(sum(float(row["source_target_boundary_f1"]) for row in rows) / len(rows)),
            "remap_coverage": float(sum(float(row["remap_coverage"]) for row in rows) / len(rows)),
            "atom_recovery": float(sum(float(row["atom_recovery"]) for row in rows) / len(rows)),
            "bytes_proxy": float(sum(float(row["bytes_proxy"]) for row in rows) / len(rows)),
            "protected_atoms": float(sum(float(row["protected_atoms"]) for row in rows) / len(rows)),
            "help_vs_token_id": float(sum(float(row["help_vs_token_id"]) for row in rows) / len(rows)),
            "harm_vs_token_id": float(sum(float(row["harm_vs_token_id"]) for row in rows) / len(rows)),
            "help_vs_regroup_baseline": float(sum(float(row["help_vs_regroup_baseline"]) for row in rows) / len(rows)),
            "harm_vs_regroup_baseline": float(sum(float(row["harm_vs_regroup_baseline"]) for row in rows) / len(rows)),
        }
        summary["task_accuracy_delta_vs_token_id"] = float(summary["task_accuracy"] - token_id_acc)
        summary["task_accuracy_delta_vs_regroup_baseline"] = float(summary["task_accuracy"] - regroup_acc)
        summary["mse_delta_vs_token_id"] = float(summary["mse"] - token_id_mse)
        summary["mse_delta_vs_regroup_baseline"] = float(summary["mse"] - regroup_mse)
        summary_rows.append(summary)

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "remap_table_size": len(remap_table),
        "remap_table_bytes": remap_table_bytes,
        "rows": summary_rows,
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]

    def fmt(value: Any) -> str:
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Shared Byte/Span Route Atom Remap",
        "",
        "- Shared remap is learned from calibration examples and used before route-atom selection.",
        "- Boundary F1 is measured against target-token boundaries on the same text; atom recovery is overlap with the oracle protected set.",
        "",
        "| Method | Task acc | Acc delta vs token_id | Acc delta vs regroup | MSE | Boundary F1 | Remap coverage | Atom recovery | Bytes proxy | Help vs token_id | Harm vs token_id | Help vs regroup | Harm vs regroup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {task_accuracy} | {task_accuracy_delta_vs_token_id} | {task_accuracy_delta_vs_regroup_baseline} | {mse} | {source_target_boundary_f1} | {remap_coverage} | {atom_recovery} | {bytes_proxy} | {help_vs_token_id} | {harm_vs_token_id} | {help_vs_regroup_baseline} | {harm_vs_regroup_baseline} |".format(
                method=row["method"],
                task_accuracy=fmt(row["task_accuracy"]),
                task_accuracy_delta_vs_token_id=fmt(row["task_accuracy_delta_vs_token_id"]),
                task_accuracy_delta_vs_regroup_baseline=fmt(row["task_accuracy_delta_vs_regroup_baseline"]),
                mse=fmt(row["mse"]),
                source_target_boundary_f1=fmt(row["source_target_boundary_f1"]),
                remap_coverage=fmt(row["remap_coverage"]),
                atom_recovery=fmt(row["atom_recovery"]),
                bytes_proxy=fmt(row["bytes_proxy"]),
                help_vs_token_id=fmt(row["help_vs_token_id"]),
                harm_vs_token_id=fmt(row["harm_vs_token_id"]),
                help_vs_regroup_baseline=fmt(row["help_vs_regroup_baseline"]),
                harm_vs_regroup_baseline=fmt(row["harm_vs_regroup_baseline"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy shared byte/span route-atom remap ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-examples", type=int, default=160)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--min-atoms", type=int, default=4)
    parser.add_argument("--max-atoms", type=int, default=6)
    parser.add_argument("--protected-atoms", type=int, default=4)
    parser.add_argument("--remap-capacity", type=int, default=10)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--signal-scale", type=float, default=3.2)
    parser.add_argument("--distractor-scale", type=float, default=7.8)
    parser.add_argument("--activation-noise", type=float, default=0.04)
    parser.add_argument("--calibration-noise", type=float, default=0.03)
    parser.add_argument("--label-noise", type=float, default=0.16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToySharedByteSpanRouteAtomRemapConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        protected_atoms=args.protected_atoms,
        remap_capacity=args.remap_capacity,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        signal_scale=args.signal_scale,
        distractor_scale=args.distractor_scale,
        activation_noise=args.activation_noise,
        calibration_noise=args.calibration_noise,
        label_noise=args.label_noise,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    return payload


if __name__ == "__main__":
    main()
