from __future__ import annotations

"""Strict ARC-Challenge source-private soft-prefix resonance gate.

This wraps the Mac-local ARC/OpenBookQA soft-prefix preflight in a frozen
validation-to-test disagreement protocol.  The source encoder is trained only on
validation disagreement rows, evaluated once on held-out test disagreement rows,
and compared against target-only, source-destroying, source-index/rank/score,
same-byte text, and source-family substitution controls.
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
from typing import Any, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_soft_prefix_resonance_gate_20260504_"
    "tinyllama_to_qwen3_disagreement"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/"
    "arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/"
    "arc_challenge_test.jsonl"
)
DEFAULT_SOURCE_FAMILY_GATE_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"
)
DEFAULT_SCORE_CACHE_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_score_router_gate_20260502/source_score_caches"
)
DEFAULT_TINY_VALIDATION_SCORE_CACHE = DEFAULT_SCORE_CACHE_DIR / "tinyllama_validation_source_scores.jsonl"
DEFAULT_TINY_TEST_SCORE_CACHE = DEFAULT_SCORE_CACHE_DIR / "tinyllama_test_source_scores.jsonl"
DEFAULT_QWEN_VALIDATION_SCORE_CACHE = DEFAULT_SCORE_CACHE_DIR / "qwen_validation_source_scores.jsonl"
DEFAULT_QWEN_TEST_SCORE_CACHE = DEFAULT_SCORE_CACHE_DIR / "qwen_test_source_scores.jsonl"

STRICT_REQUIRED_CONTROLS = (
    "target_only",
    "slots_only_prefix",
    "zero_source",
    "source_row_shuffle",
    "candidate_roll",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "target_derived_prefix",
    "packet_only_source_index",
    "source_rank_control",
    "source_score_control",
    "same_byte_visible_text",
    "qwen_substituted_packet",
    "candidate_derangement",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_disagreement_content_ids(
    *,
    agreement_path: pathlib.Path,
    split: str,
    limit: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    with _resolve(agreement_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            if str(row.get("agree", "")).lower() == "true":
                continue
            content_id = str(row["content_id"])
            if content_id in seen:
                continue
            seen.add(content_id)
            selected.append(content_id)
            if len(selected) >= int(limit):
                break
    if len(selected) < int(limit):
        raise ValueError(f"only found {len(selected)} {split} disagreement rows, requested {limit}")
    return selected


def _filter_rows_by_content_ids(rows: Sequence[arc_gate.ArcRow], content_ids: Sequence[str]) -> list[arc_gate.ArcRow]:
    by_content = {row.content_id: row for row in rows}
    missing = [content_id for content_id in content_ids if content_id not in by_content]
    if missing:
        raise ValueError(f"missing {len(missing)} requested ARC rows")
    return [by_content[content_id] for content_id in content_ids]


def _arc_row_payload(row: arc_gate.ArcRow) -> dict[str, Any]:
    return {
        "id": row.row_id,
        "content_id": row.content_id,
        "question": row.question,
        "choices": list(row.choices),
        "choice_labels": list(row.choice_labels),
        "answer_index": int(row.answer_index),
        "answer_label": row.answer_label,
    }


def _select_cache_rows(
    *,
    cache_path: pathlib.Path,
    rows: Sequence[arc_gate.ArcRow],
) -> list[dict[str, Any]]:
    cache = {str(row["content_id"]): row for row in _read_jsonl(cache_path)}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        cached = cache.get(row.content_id)
        if cached is None:
            missing.append(row.content_id)
            continue
        selected.append(cached)
    if missing:
        raise ValueError(f"{cache_path} missing {len(missing)} selected rows")
    return selected


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    strict = payload["strict_headline"]
    lines = [
        "# ARC-Challenge Soft-Prefix Resonance Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/test disagreement rows: `{payload['train_rows']}` / `{payload['test_rows']}`",
        f"- matched accuracy: `{strict['matched_accuracy']:.6f}`",
        f"- best required control: `{strict['best_required_control']}`",
        f"- best required control accuracy: `{strict['best_required_control_accuracy']:.6f}`",
        f"- worst required paired CI95 low: `{strict['worst_required_ci95_low']:.6f}`",
        "",
        "## Strict Controls",
        "",
        "| Control | Accuracy | Delta | CI95 low |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["strict_control_metrics"].items():
        lines.append(
            f"| `{name}` | {row['control_accuracy']:.6f} | {row['delta_accuracy']:.6f} | "
            f"{row['ci95_low']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay explanation: this trains a small translator on validation questions where the source and target "
            "disagree, then tests once on new disagreement questions. The target gets hidden soft-prefix tokens, "
            "not the source text. The controls ask whether the same result comes from target-only memory, a row "
            "shuffle, candidate shuffle, raw source rank/score shortcuts, or visible same-byte text.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    source_family_gate_dir: pathlib.Path,
    tiny_validation_score_cache: pathlib.Path,
    tiny_test_score_cache: pathlib.Path,
    qwen_validation_score_cache: pathlib.Path,
    qwen_test_score_cache: pathlib.Path,
    train_disagreement_limit: int,
    test_disagreement_limit: int,
    source_feature_mode: str,
    source_feature_dim: int,
    target_feature_dim: int,
    source_model: str,
    target_model: str,
    source_device: str,
    target_device: str,
    train_device: str | None,
    target_attn_implementation: str | None,
    dtype: str,
    source_max_length: int,
    target_max_length: int,
    source_hidden_layer: int,
    source_token_pool_size: int,
    innovation_ridge: float,
    sparse_packet_rank: int,
    sparse_packet_top_k: int,
    sparse_packet_bits: int,
    local_files_only: bool,
    prefix_len: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    bootstrap_samples: int,
    continuation_mode: str,
    matched_use_target: bool,
    length_normalize: bool,
    contrastive_weight: float,
    contrastive_margin: float,
    contrastive_loss_cap: float,
    contrastive_controls: Sequence[str],
    same_byte_budget: int,
    min_accuracy_gap: float,
    min_ci_low: float,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "strict_inputs"
    agreement_path = _resolve(source_family_gate_dir) / "source_cache_agreement.csv"

    validation_rows_all = arc_gate._load_rows(_resolve(validation_path))
    test_rows_all = arc_gate._load_rows(_resolve(test_path))
    train_ids = _read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="validation",
        limit=train_disagreement_limit,
    )
    test_ids = _read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="test",
        limit=test_disagreement_limit,
    )
    train_rows = _filter_rows_by_content_ids(validation_rows_all, train_ids)
    test_rows = _filter_rows_by_content_ids(test_rows_all, test_ids)
    overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in test_rows})
    if overlap:
        raise ValueError(f"train/test content overlap: {overlap[:3]}")

    combined_rows = [*train_rows, *test_rows]
    combined_row_path = input_dir / "arc_challenge_validation_train_plus_test_disagreement.jsonl"
    combined_source_cache_path = input_dir / "tinyllama_source_score_cache.jsonl"
    combined_qwen_cache_path = input_dir / "qwen_source_score_cache.jsonl"
    _write_jsonl(combined_row_path, [_arc_row_payload(row) for row in combined_rows])
    _write_jsonl(
        combined_source_cache_path,
        [
            *_select_cache_rows(cache_path=tiny_validation_score_cache, rows=train_rows),
            *_select_cache_rows(cache_path=tiny_test_score_cache, rows=test_rows),
        ],
    )
    _write_jsonl(
        combined_qwen_cache_path,
        [
            *_select_cache_rows(cache_path=qwen_validation_score_cache, rows=train_rows),
            *_select_cache_rows(cache_path=qwen_test_score_cache, rows=test_rows),
        ],
    )

    preflight_payload = preflight.run_preflight(
        output_dir=output_dir / "preflight",
        eval_path=combined_row_path,
        source_cache_path=combined_source_cache_path,
        qwen_source_cache_path=combined_qwen_cache_path,
        source_score_cache_path=combined_source_cache_path,
        benchmark="ARC-Challenge-disagreement",
        row_limit=len(combined_rows),
        fit_fraction=0.5,
        fixed_fit_rows=len(train_rows),
        source_feature_mode=source_feature_mode,
        source_feature_dim=source_feature_dim,
        target_feature_dim=target_feature_dim,
        source_model=source_model,
        target_model_path=target_model,
        source_device=source_device,
        target_device=target_device,
        train_device=train_device,
        target_attn_implementation=target_attn_implementation,
        dtype=dtype,
        source_max_length=source_max_length,
        target_max_length=target_max_length,
        source_hidden_layer=source_hidden_layer,
        source_token_pool_size=source_token_pool_size,
        innovation_ridge=innovation_ridge,
        sparse_packet_rank=sparse_packet_rank,
        sparse_packet_top_k=sparse_packet_top_k,
        sparse_packet_bits=sparse_packet_bits,
        local_files_only=local_files_only,
        prefix_len=prefix_len,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        continuation_mode=continuation_mode,
        matched_use_target=matched_use_target,
        length_normalize=length_normalize,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        contrastive_loss_cap=contrastive_loss_cap,
        contrastive_controls=contrastive_controls,
        same_byte_budget=same_byte_budget,
        min_accuracy_gap=min_accuracy_gap,
        min_margin_gap=0.0,
    )

    metrics = preflight_payload["condition_metrics"]
    matched = metrics[preflight.MATCHED_CONDITION]
    source_feature_metadata = preflight_payload.get("feature_metadata", {}).get("source", {})
    sparse_packet_metadata = source_feature_metadata.get("sparse_packet") or {}
    packet_bytes_per_candidate = float(sparse_packet_metadata.get("packet_bytes_per_candidate", 0.0) or 0.0)
    estimated_packet_bytes_per_row = float(packet_bytes_per_candidate * int(source_token_pool_size))
    strict_control_metrics: dict[str, dict[str, float]] = {}
    for control in STRICT_REQUIRED_CONTROLS:
        paired = matched[f"paired_accuracy_vs_{control}"]
        strict_control_metrics[control] = {
            "control_accuracy": float(metrics[control]["accuracy"]),
            "delta_accuracy": float(matched["accuracy"] - metrics[control]["accuracy"]),
            "ci95_low": float(paired["ci95_low"]),
            "ci95_high": float(paired["ci95_high"]),
        }
    best_required_control = max(STRICT_REQUIRED_CONTROLS, key=lambda name: metrics[name]["accuracy"])
    worst_ci_low = min(row["ci95_low"] for row in strict_control_metrics.values())
    strict_pass = all(
        row["delta_accuracy"] >= float(min_accuracy_gap) and row["ci95_low"] > float(min_ci_low)
        for row in strict_control_metrics.values()
    )
    payload = {
        **preflight_payload,
        "gate": "source_private_arc_challenge_soft_prefix_resonance_gate",
        "pass_gate": bool(strict_pass),
        "preflight_pass_gate": bool(preflight_payload["pass_gate"]),
        "implementation_gate_only": False,
        "strict_required_controls": list(STRICT_REQUIRED_CONTROLS),
        "strict_control_metrics": strict_control_metrics,
        "strict_headline": {
            "matched_accuracy": float(matched["accuracy"]),
            "best_required_control": best_required_control,
            "best_required_control_accuracy": float(metrics[best_required_control]["accuracy"]),
            "worst_required_ci95_low": float(worst_ci_low),
        },
        "systems_packet_sideband": {
            "source_private": True,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "native_serving_throughput_measured": False,
            "packet_bytes_per_candidate": packet_bytes_per_candidate,
            "estimated_packet_bytes_per_row": estimated_packet_bytes_per_row,
            "sparse_packet_metadata": sparse_packet_metadata,
            "receiver_soft_prefix_tokens": int(prefix_len),
            "note": (
                "Byte counts cover the sparse packet sideband only. They are not native GPU throughput, "
                "HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "inputs": {
            **preflight_payload["inputs"],
            "validation_path": _display(validation_path),
            "test_path": _display(test_path),
            "source_family_gate_dir": _display(source_family_gate_dir),
            "agreement_path": _display(agreement_path),
            "combined_row_path": _display(combined_row_path),
            "combined_source_cache_path": _display(combined_source_cache_path),
            "combined_qwen_cache_path": _display(combined_qwen_cache_path),
        },
        "interpretation": (
            "This strict gate passes only if a validation-trained source-conditioned soft prefix beats every "
            "required target-only, source-destroying, same-byte, source-index/rank/score, and Qwen-substitution "
            "control on frozen test disagreement rows with positive paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_soft_prefix_resonance_gate.json"
    md_path = output_dir / "arc_challenge_soft_prefix_resonance_gate.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {"path": _display(json_path), "sha256": preflight._sha256_file(json_path), "bytes": json_path.stat().st_size},
            {"path": _display(md_path), "sha256": preflight._sha256_file(md_path), "bytes": md_path.stat().st_size},
            {
                "path": _display(output_dir / "preflight" / "prediction_audit.jsonl"),
                "sha256": preflight._sha256_file(output_dir / "preflight" / "prediction_audit.jsonl"),
                "bytes": (output_dir / "preflight" / "prediction_audit.jsonl").stat().st_size,
            },
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--source-family-gate-dir", type=pathlib.Path, default=DEFAULT_SOURCE_FAMILY_GATE_DIR)
    parser.add_argument("--tiny-validation-score-cache", type=pathlib.Path, default=DEFAULT_TINY_VALIDATION_SCORE_CACHE)
    parser.add_argument("--tiny-test-score-cache", type=pathlib.Path, default=DEFAULT_TINY_TEST_SCORE_CACHE)
    parser.add_argument("--qwen-validation-score-cache", type=pathlib.Path, default=DEFAULT_QWEN_VALIDATION_SCORE_CACHE)
    parser.add_argument("--qwen-test-score-cache", type=pathlib.Path, default=DEFAULT_QWEN_TEST_SCORE_CACHE)
    parser.add_argument("--train-disagreement-limit", type=int, default=16)
    parser.add_argument("--test-disagreement-limit", type=int, default=16)
    parser.add_argument("--source-feature-mode", default="cached_choice_score_pool_residual")
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--target-feature-dim", type=int, default=64)
    parser.add_argument("--source-model", default=preflight.DEFAULT_QWEN_SOURCE)
    parser.add_argument("--target-model", default=preflight.DEFAULT_QWEN_TARGET)
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--target-attn-implementation", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=192)
    parser.add_argument("--target-max-length", type=int, default=256)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--source-token-pool-size", type=int, default=4)
    parser.add_argument("--innovation-ridge", type=float, default=10.0)
    parser.add_argument("--sparse-packet-rank", type=int, default=16)
    parser.add_argument("--sparse-packet-top-k", type=int, default=4)
    parser.add_argument("--sparse-packet-bits", type=int, default=4)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--continuation-mode", choices=("label", "label_and_choice", "choice"), default="label")
    parser.add_argument("--matched-use-target", choices=("true", "false"), default="false")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.05)
    parser.add_argument("--contrastive-loss-cap", type=float, default=0.5)
    parser.add_argument("--contrastive-controls", default="")
    parser.add_argument("--same-byte-budget", type=int, default=4096)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_gate(
        output_dir=args.output_dir,
        validation_path=args.validation_path,
        test_path=args.test_path,
        source_family_gate_dir=args.source_family_gate_dir,
        tiny_validation_score_cache=args.tiny_validation_score_cache,
        tiny_test_score_cache=args.tiny_test_score_cache,
        qwen_validation_score_cache=args.qwen_validation_score_cache,
        qwen_test_score_cache=args.qwen_test_score_cache,
        train_disagreement_limit=int(args.train_disagreement_limit),
        test_disagreement_limit=int(args.test_disagreement_limit),
        source_feature_mode=str(args.source_feature_mode),
        source_feature_dim=int(args.source_feature_dim),
        target_feature_dim=int(args.target_feature_dim),
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        source_device=str(args.source_device),
        target_device=str(args.target_device),
        train_device=None if args.train_device is None else str(args.train_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        source_max_length=int(args.source_max_length),
        target_max_length=int(args.target_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        source_token_pool_size=int(args.source_token_pool_size),
        innovation_ridge=float(args.innovation_ridge),
        sparse_packet_rank=int(args.sparse_packet_rank),
        sparse_packet_top_k=int(args.sparse_packet_top_k),
        sparse_packet_bits=int(args.sparse_packet_bits),
        local_files_only=str(args.local_files_only).lower() == "true",
        prefix_len=int(args.prefix_len),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
        continuation_mode=str(args.continuation_mode),
        matched_use_target=str(args.matched_use_target).lower() == "true",
        length_normalize=str(args.length_normalize).lower() == "true",
        contrastive_weight=float(args.contrastive_weight),
        contrastive_margin=float(args.contrastive_margin),
        contrastive_loss_cap=float(args.contrastive_loss_cap),
        contrastive_controls=preflight._parse_contrastive_controls(str(args.contrastive_controls)),
        same_byte_budget=int(args.same_byte_budget),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "headline": payload["strict_headline"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
