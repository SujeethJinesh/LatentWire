#!/usr/bin/env python3
"""Real-model sparse-anchor sidecar smoke on frozen SVAMP32 rows.

This probe is intentionally small. It tests whether rate-capped source-derived
features plus tokenizer-boundary alignment features can predict the C2C numeric
residue used by the cleared syndrome-sidecar bound.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from latent_bridge.evaluate import _generation_example_id, load_generation
from scripts import analyze_svamp32_source_latent_syndrome_probe as latent_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _torch_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def _boundary_positions(tokenizer: Any, text: str) -> set[int]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") or []
        return {int(end) for _start, end in offsets if int(end) > 0}
    except Exception:
        # Slow-tokenizer fallback: approximate boundaries by decoded token text
        # lengths. This is only used for the alignment sidecar, not scoring.
        tokens = tokenizer.tokenize(text)
        boundaries: set[int] = set()
        cursor = 0
        for token in tokens:
            clean = str(token).replace("##", "").replace("Ġ", " ").replace("▁", " ")
            cursor = min(len(text), cursor + max(1, len(clean)))
            boundaries.add(cursor)
        return boundaries


def _boundary_f1(source: set[int], target: set[int]) -> float:
    if not source and not target:
        return 1.0
    if not source or not target:
        return 0.0
    overlap = len(source & target)
    precision = overlap / max(len(source), 1)
    recall = overlap / max(len(target), 1)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _stable_hash(data: bytes) -> int:
    # FNV-1a 64-bit: deterministic across Python processes.
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= int(byte)
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def _sequence_alignment_sidecar(
    texts: Sequence[str],
    *,
    source_tokenizer: Any,
    target_tokenizer: Any,
    dim: int,
    token_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    features = torch.zeros((len(texts), dim), dtype=torch.float32)
    profiles = torch.zeros((len(texts), 6), dtype=torch.float32)
    metadata: list[dict[str, Any]] = []
    for row, text in enumerate(texts):
        source_bounds = _boundary_positions(source_tokenizer, text)
        target_bounds = _boundary_positions(target_tokenizer, text)
        union_bounds = sorted((source_bounds | target_bounds) | {len(text)})

        start = 0
        for end in union_bounds:
            span = text[start:end]
            if span:
                weight = 1.0
                weight += 0.25 * float(end in source_bounds)
                weight += 0.25 * float(end in target_bounds)
                bucket = _stable_hash(b"span:" + span.encode("utf-8")) % dim
                features[row, bucket] += weight
            start = end

        for end in source_bounds:
            bucket = _stable_hash(f"src:{end}".encode("utf-8")) % dim
            features[row, bucket] += float(token_scale)
        for end in target_bounds:
            bucket = _stable_hash(f"tgt:{end}".encode("utf-8")) % dim
            features[row, bucket] += float(token_scale)

        boundary_f1 = _boundary_f1(source_bounds, target_bounds)
        source_len = float(len(source_bounds))
        target_len = float(len(target_bounds))
        text_len = max(len(text), 1)
        fragmentation_gap = abs(source_len - target_len) / max(source_len + target_len, 1.0)
        profiles[row] = torch.tensor(
            [
                boundary_f1,
                1.0 - boundary_f1,
                source_len / text_len,
                target_len / text_len,
                fragmentation_gap,
                len(union_bounds) / text_len,
            ],
            dtype=torch.float32,
        )
        features[row] = features[row] / features[row].norm().clamp_min(1e-8)
        metadata.append(
            {
                "boundary_f1": float(boundary_f1),
                "source_boundary_count": int(len(source_bounds)),
                "target_boundary_count": int(len(target_bounds)),
                "union_boundary_count": int(len(union_bounds)),
            }
        )
    profiles = profiles / profiles.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return features, profiles, metadata


def _sparse_anchor_code(
    features: torch.Tensor,
    *,
    code_dim: int,
    topk: int,
    seed: int,
) -> torch.Tensor:
    if code_dim <= 0:
        raise ValueError("code_dim must be positive")
    if topk <= 0 or topk > code_dim:
        raise ValueError("topk must be in 1..code_dim")
    generator = torch.Generator().manual_seed(int(seed))
    projection = torch.randn(
        (int(features.shape[1]), int(code_dim)),
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(max(int(features.shape[1]), 1))
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    projected = centered @ projection
    values, indices = torch.topk(projected.abs(), k=int(topk), dim=1)
    sparse = torch.zeros_like(projected)
    sparse.scatter_(1, indices, torch.gather(projected, 1, indices))
    sparse = sparse / sparse.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return sparse


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    run = payload["run"]
    rows = [
        run["condition_summaries"][condition]
        for condition in (
            "matched",
            "zero_source",
            "shuffled_source",
            "label_shuffled",
            "target_only",
            "slots_only",
        )
    ]
    lines = [
        "# SVAMP32 Sparse-Anchor Sidecar Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- source model: `{payload['artifacts']['source_model']}`",
        f"- target tokenizer model: `{payload['artifacts']['target_tokenizer_model']}`",
        f"- moduli: `{','.join(str(value) for value in payload['config']['moduli'])}`",
        f"- anchor code dim: `{payload['config']['anchor_code_dim']}`",
        f"- anchor top-k: `{payload['config']['anchor_topk']}`",
        f"- sequence sidecar dim: `{payload['config']['sequence_sidecar_dim']}`",
        f"- estimated sidecar bytes: `{payload['config']['estimated_sidecar_bytes']}`",
        f"- teacher numeric coverage: `{payload['provenance']['teacher_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
        "",
        "## Summary",
        "",
        "| Condition | Correct | Clean Correct | Target-Self Correct |",
        "|---|---:|---:|---:|",
    ]
    for summary in rows:
        lines.append(
            "| {condition} | {correct} | {clean} | {target_self} |".format(
                condition=summary["condition"],
                correct=summary["correct_count"],
                clean=summary["clean_correct_count"],
                target_self=summary["target_self_correct_count"],
            )
        )
    lines.extend(
        [
            "",
            f"- clean source-necessary IDs: `{len(run['source_necessary_clean_ids'])}`",
            f"- source-necessary IDs: {', '.join(f'`{value}`' for value in run['source_necessary_clean_ids']) or 'none'}",
            f"- control clean union IDs: {', '.join(f'`{value}`' for value in run['control_clean_union_ids']) or 'none'}",
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-tokenizer-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--teacher", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--moduli", default="2,3,5,7")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=10)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=26)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--feature-layers", default="mid,last")
    parser.add_argument("--anchor-code-dim", type=int, default=64)
    parser.add_argument("--anchor-topk", type=int, default=4)
    parser.add_argument("--anchor-scale", type=float, default=1.0)
    parser.add_argument("--sequence-sidecar-dim", type=int, default=32)
    parser.add_argument("--sequence-scale", type=float, default=0.35)
    parser.add_argument("--profile-scale", type=float, default=0.20)
    parser.add_argument("--token-scale", type=float, default=0.75)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    examples = load_generation(str(_resolve(args.eval_file)))
    moduli = tuple(int(value) for value in str(args.moduli).split(",") if value.strip())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading source model: {args.source_model}", flush=True)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if source_tokenizer.pad_token_id is None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer_model, trust_remote_code=True)
    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source_model,
            torch_dtype=_torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        .to(args.device)
        .eval()
    )

    hidden_features, hidden_metadata = latent_probe._extract_source_features(
        model=model,
        tokenizer=source_tokenizer,
        examples=examples,
        device=str(args.device),
        source_reasoning_mode=str(args.source_reasoning_mode),
        use_chat_template=bool(args.source_use_chat_template),
        enable_thinking=_bool_arg(args.source_enable_thinking),
        feature_layers=str(args.feature_layers),
    )
    anchor_code = _sparse_anchor_code(
        hidden_features,
        code_dim=int(args.anchor_code_dim),
        topk=int(args.anchor_topk),
        seed=int(args.projection_seed),
    )
    prompts = [str(example.prompt) for example in examples]
    sequence_sidecar, sequence_profile, sidecar_metadata = _sequence_alignment_sidecar(
        prompts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        dim=int(args.sequence_sidecar_dim),
        token_scale=float(args.token_scale),
    )
    features = torch.cat(
        [
            float(args.anchor_scale) * anchor_code,
            float(args.sequence_scale) * sequence_sidecar,
            float(args.profile_scale) * sequence_profile,
        ],
        dim=1,
    )
    feature_metadata: list[dict[str, Any]] = []
    for hidden_row, sidecar_row in zip(hidden_metadata, sidecar_metadata, strict=True):
        feature_metadata.append(
            {
                **hidden_row,
                **sidecar_row,
                "feature_dim": int(features.shape[1]),
                "hidden_feature_dim": int(hidden_features.shape[1]),
                "anchor_code_dim": int(args.anchor_code_dim),
                "anchor_topk": int(args.anchor_topk),
                "sequence_sidecar_dim": int(args.sequence_sidecar_dim),
            }
        )

    config = latent_probe.ProbeConfig(
        moduli=moduli,
        probe_model="ridge",
        ridge_lambda=float(args.ridge_lambda),
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
    )
    payload = latent_probe.analyze_with_features(
        features=features,
        feature_metadata=feature_metadata,
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=args.candidate,
        target_set_path=_resolve(args.target_set_json),
        fallback_label=str(args.fallback_label),
        config=config,
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
    )
    estimated_sidecar_bytes = int(args.anchor_topk) * 6 + int(args.sequence_sidecar_dim) // 8 + 6
    payload["status"] = (
        "sparse_anchor_sidecar_clears_gate"
        if payload["status"] == "source_latent_syndrome_probe_clears_gate"
        else "sparse_anchor_sidecar_fails_gate"
    )
    payload["interpretation"] = (
        "This probe uses a rate-capped sparse anchor projection of source hidden "
        "summaries plus tokenizer-boundary alignment sidecar features to predict "
        "C2C numeric residue classes. It is a real-model smoke for the "
        "sequence-aligned sparse/anchor sidecar branch, not a paper claim."
    )
    payload["artifacts"]["source_model"] = str(args.source_model)
    payload["artifacts"]["target_tokenizer_model"] = str(args.target_tokenizer_model)
    payload["config"].update(
        {
            "anchor_code_dim": int(args.anchor_code_dim),
            "anchor_topk": int(args.anchor_topk),
            "anchor_scale": float(args.anchor_scale),
            "sequence_sidecar_dim": int(args.sequence_sidecar_dim),
            "sequence_scale": float(args.sequence_scale),
            "profile_scale": float(args.profile_scale),
            "token_scale": float(args.token_scale),
            "projection_seed": int(args.projection_seed),
            "estimated_sidecar_bytes": estimated_sidecar_bytes,
            "feature_layers": str(args.feature_layers),
        }
    )

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_md, payload)
    print(
        json.dumps(
            {"status": payload["status"], "output_json": _display_path(output_json)},
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
