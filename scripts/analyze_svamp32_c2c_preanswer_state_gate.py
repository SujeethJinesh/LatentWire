#!/usr/bin/env python3
"""Probe pre-answer C2C generation state on SVAMP32.

This is a C2C-distillation diagnostic, not a deployable source-private method.
It locates the final numeric answer in the repaired C2C generation, summarizes
C2C projector state plus generation-logit history before that answer appears,
and compares it with a post-answer leakage window using the existing strict
SVAMP32 syndrome evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.c2c_eval import (
    _history_aggregates,
    _logit_step_stats,
    build_c2c_kv_cache_index,
    build_c2c_messages,
    install_c2c_projector_trace_hooks,
    load_c2c_model,
    reset_c2c_projector_trace_history,
    stop_c2c_projector_trace_history,
    summarize_c2c_projector_generation_history,
)
from latent_bridge.evaluate import (
    _extract_prediction_numeric_answer,
    _generation_example_id,
    load_generation,
)
from scripts import analyze_svamp32_c2c_mechanism_syndrome_probe as mechanism_probe
from scripts import analyze_svamp32_source_latent_syndrome_probe as source_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
SCORE_KEYS = (
    "mean",
    "std",
    "min",
    "max",
    "top1_logit",
    "top2_logit",
    "top1_prob",
    "top2_prob",
    "top_margin",
    "entropy",
    "generated_logit",
    "generated_rank_frac",
)
AGGREGATE_NAMES = ("mean", "std", "min", "max", "first", "last", "delta")


@dataclass(frozen=True)
class AnswerSpan:
    value: str | None
    char_start: int | None
    char_end: int | None
    pre_answer_token_count: int
    answer_end_token_count: int


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _limit_spec_to_ids(
    spec: syndrome.RowSpec,
    *,
    reference_ids: Sequence[str],
    output_dir: pathlib.Path,
) -> syndrome.RowSpec:
    rows = syndrome._subset_reference_order(
        syndrome._records_for_method(spec),
        [str(value) for value in reference_ids],
    )
    path = output_dir / f"{spec.label}.jsonl"
    _write_jsonl(path, rows)
    return syndrome.RowSpec(label=spec.label, path=path, method=spec.method)


def _limit_target_set_to_ids(
    path: pathlib.Path,
    *,
    reference_ids: Sequence[str],
    output_dir: pathlib.Path,
) -> pathlib.Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keep = {str(value) for value in reference_ids}
    limited = dict(payload)
    limited["reference_ids"] = [str(value) for value in reference_ids]
    limited["reference_n"] = len(reference_ids)
    limited["ids"] = {
        key: [str(value) for value in values if str(value) in keep]
        for key, values in payload.get("ids", {}).items()
    }
    out = output_dir / "target_set.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(limited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    match = NUMBER_RE.search(str(value).replace(",", ""))
    if not match:
        return None
    try:
        dec = Decimal(match.group(0))
    except InvalidOperation:
        return None
    if dec == dec.to_integral_value():
        return str(int(dec))
    normalized = format(dec.normalize(), "f")
    return normalized.rstrip("0").rstrip(".")


def _final_numeric_span(text: str) -> tuple[str | None, int | None, int | None]:
    final_value = _canonical_number(_extract_prediction_numeric_answer(text))
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None, None, None
    if final_value is not None:
        matching = [
            match for match in matches
            if _canonical_number(match.group(0)) == final_value
        ]
        if matching:
            match = matching[-1]
            return final_value, int(match.start()), int(match.end())
    match = matches[-1]
    return _canonical_number(match.group(0)), int(match.start()), int(match.end())


def _token_count_before_char(tokenizer: Any, generated_tokens: torch.Tensor, char_start: int | None) -> int:
    if char_start is None:
        return int(generated_tokens.numel())
    best = 0
    flat = generated_tokens.detach().cpu().reshape(-1)
    for length in range(int(flat.numel()) + 1):
        decoded = tokenizer.decode(flat[:length].tolist(), skip_special_tokens=True)
        if len(decoded) <= int(char_start):
            best = length
        else:
            break
    return int(best)


def _token_count_through_char(tokenizer: Any, generated_tokens: torch.Tensor, char_end: int | None) -> int:
    if char_end is None:
        return int(generated_tokens.numel())
    flat = generated_tokens.detach().cpu().reshape(-1)
    for length in range(int(flat.numel()) + 1):
        decoded = tokenizer.decode(flat[:length].tolist(), skip_special_tokens=True)
        if len(decoded) >= int(char_end):
            return int(length)
    return int(flat.numel())


def locate_answer_span(tokenizer: Any, generated_tokens: torch.Tensor, decoded_text: str) -> AnswerSpan:
    value, start, end = _final_numeric_span(decoded_text)
    return AnswerSpan(
        value=value,
        char_start=start,
        char_end=end,
        pre_answer_token_count=_token_count_before_char(tokenizer, generated_tokens, start),
        answer_end_token_count=_token_count_through_char(tokenizer, generated_tokens, end),
    )


def summarize_score_window(
    scores: Sequence[torch.Tensor],
    generated_tokens: torch.Tensor,
    *,
    prefix: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    step_rows: list[dict[str, float]] = []
    flat_tokens = generated_tokens.detach().cpu().reshape(-1)
    for step_idx, logits in enumerate(scores):
        token = None
        if flat_tokens.numel() > step_idx:
            token = int(flat_tokens[step_idx].item())
        step_rows.append(_logit_step_stats(logits[0], token))
    feature_names: list[str] = []
    features: list[float] = []
    for key in SCORE_KEYS:
        values = [float(row[key]) for row in step_rows] if step_rows else []
        aggregates = (
            _history_aggregates(values)
            if values
            else {name: 0.0 for name in AGGREGATE_NAMES}
        )
        for aggregate_name in AGGREGATE_NAMES:
            feature_names.append(f"{prefix}.{key}.{aggregate_name}")
            features.append(float(aggregates[aggregate_name]))
    tensor = torch.tensor(features, dtype=torch.float32)
    return tensor, {
        "feature_family": prefix,
        "feature_dim": int(tensor.numel()),
        "feature_names": feature_names,
        "step_count": int(len(step_rows)),
        "aggregate_names": list(AGGREGATE_NAMES),
    }


@torch.no_grad()
def extract_preanswer_feature_sets(
    *,
    source_model: str,
    target_model: str,
    eval_file: pathlib.Path,
    device: str,
    max_new_tokens: int,
    residual_projection_dim: int,
    start_index: int = 0,
    limit: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    examples = load_generation(str(eval_file))
    if start_index:
        examples = examples[int(start_index) :]
    if limit is not None:
        examples = examples[: int(limit)]
    model, tokenizer, artifact = load_c2c_model(
        source_model=source_model,
        target_model=target_model,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    feature_rows: dict[str, list[torch.Tensor]] = {
        "pre_answer_exclusive": [],
        "post_answer_inclusive": [],
    }
    metadata: list[dict[str, Any]] = []
    for example in examples:
        text = tokenizer.apply_chat_template(
            build_c2c_messages(example.prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_len = int(inputs["input_ids"].shape[1])
        install_c2c_projector_trace_hooks(
            model,
            residual_projection_dim=int(residual_projection_dim),
        )
        reset_c2c_projector_trace_history(model, enabled=True)
        outputs = model.generate(
            **inputs,
            kv_cache_index=build_c2c_kv_cache_index(prompt_len, device=device),
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            return_dict_in_generate=True,
            output_scores=True,
        )
        stop_c2c_projector_trace_history(model)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs["sequences"]
        score_list = list(outputs.scores or []) if hasattr(outputs, "scores") else list(outputs.get("scores") or [])
        generated = sequences[0, prompt_len:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer_span = locate_answer_span(tokenizer, generated, decoded)

        projector_features, projector_meta = summarize_c2c_projector_generation_history(model)
        pre_end = min(int(answer_span.pre_answer_token_count), len(score_list))
        post_end = min(max(int(answer_span.answer_end_token_count), pre_end), len(score_list))
        pre_scores, pre_meta = summarize_score_window(
            score_list[:pre_end],
            generated[:pre_end],
            prefix="pre_answer_logits",
        )
        post_scores, post_meta = summarize_score_window(
            score_list[:post_end],
            generated[:post_end],
            prefix="post_answer_logits",
        )
        feature_rows["pre_answer_exclusive"].append(torch.cat([projector_features, pre_scores], dim=0))
        feature_rows["post_answer_inclusive"].append(torch.cat([projector_features, post_scores], dim=0))
        metadata.append(
            {
                "example_id": _generation_example_id(example),
                "formatted_prompt_tokens": prompt_len,
                "generated_tokens": int(generated.shape[0]),
                "decoded_prediction": decoded,
                "answer_value": answer_span.value,
                "answer_char_start": answer_span.char_start,
                "answer_char_end": answer_span.char_end,
                "pre_answer_token_count": int(pre_end),
                "post_answer_token_count": int(post_end),
                "residual_projection_dim": int(residual_projection_dim),
                "feature_dim": int(feature_rows["pre_answer_exclusive"][-1].numel()),
                "components": {
                    "projector": projector_meta,
                    "pre_answer_logits": pre_meta,
                    "post_answer_logits": post_meta,
                },
                "feature_names": [
                    *(f"projector::{name}" for name in projector_meta["feature_names"]),
                    *(f"pre::{name}" for name in pre_meta["feature_names"]),
                ],
            }
        )

    run_config = {
        "source_model": source_model,
        "target_model": target_model,
        "eval_file": _display_path(eval_file),
        "device": device,
        "max_new_tokens": int(max_new_tokens),
        "start_index": int(start_index),
        "limit": None if limit is None else int(limit),
        "residual_projection_dim": int(residual_projection_dim),
        "feature_family": "c2c_pre_answer_generation_state",
        "published_repo_id": artifact.repo_id,
        "published_subdir": artifact.subdir,
        "published_config_path": artifact.config_path,
        "published_checkpoint_dir": artifact.checkpoint_dir,
        "local_root": artifact.local_root,
    }
    return (
        {key: torch.stack(rows, dim=0) for key, rows in feature_rows.items()},
        metadata,
        run_config,
    )


def _feature_hash(features: torch.Tensor) -> dict[str, Any]:
    raw = features.detach().cpu().contiguous().numpy().tobytes()
    return {
        "shape": [int(dim) for dim in features.shape],
        "dtype": str(features.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "storage": "not_embedded_rerun_command_recorded",
    }


def _window_summary(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload["run"]
    matched = run["condition_summaries"]["matched"]
    target_only = run["condition_summaries"]["target_only"]
    zero = run["condition_summaries"]["zero_source"]
    shuffled = run["condition_summaries"]["shuffled_source"]
    label = run["condition_summaries"]["label_shuffled"]
    return {
        "status": payload["status"],
        "matched_correct": matched["correct_count"],
        "matched_clean_correct": matched["clean_correct_count"],
        "target_only_correct": target_only["correct_count"],
        "zero_source_correct": zero["correct_count"],
        "shuffled_source_correct": shuffled["correct_count"],
        "label_shuffled_correct": label["correct_count"],
        "clean_source_necessary": len(run["source_necessary_clean_ids"]),
        "source_necessary_clean_ids": run["source_necessary_clean_ids"],
        "control_clean_union_ids": run["control_clean_union_ids"],
    }


def _overall_status(pre: dict[str, Any], post: dict[str, Any]) -> str:
    pre_summary = _window_summary(pre)
    post_summary = _window_summary(post)
    if (
        pre_summary["matched_correct"] >= pre["config"]["min_correct"]
        and pre_summary["clean_source_necessary"] >= pre["config"]["min_clean_source_necessary"]
    ):
        return "pre_answer_c2c_state_proxy_clears_gate_not_deployable"
    if (
        post_summary["clean_source_necessary"] > pre_summary["clean_source_necessary"]
        and post_summary["matched_correct"] > pre_summary["matched_correct"]
    ):
        return "pre_answer_c2c_state_fails_post_answer_leakage_stronger"
    return "pre_answer_c2c_state_fails_controls"


def _write_manifest(output_dir: pathlib.Path, payload: dict[str, Any], artifacts: Sequence[pathlib.Path]) -> None:
    manifest = {
        "date": payload["date"],
        "status": payload["status"],
        "artifacts": {
            _display_path(path): {"sha256": _sha256(path)}
            for path in artifacts
            if path.exists()
        },
    }
    manifest_json = output_dir / "manifest.json"
    manifest_md = output_dir / "manifest.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SVAMP32 C2C Pre-Answer State Gate Manifest",
        "",
        f"- date: `{manifest['date']}`",
        f"- status: `{manifest['status']}`",
        "",
        "## Artifacts",
        "",
        "| Path | SHA256 |",
        "|---|---|",
    ]
    for path, meta in manifest["artifacts"].items():
        lines.append(f"| `{path}` | `{meta['sha256']}` |")
    manifest_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SVAMP32 C2C Pre-Answer State Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- output JSON: `{payload['artifacts']['output_json']}`",
        "",
        "## Summary",
        "",
        "| Window | Status | Matched | Target-only | Zero | Row-shuffle | Label-shuffle | Clean source-necessary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in payload["window_summaries"].items():
        lines.append(
            "| {name} | `{status}` | {matched}/{n} | {target}/{n} | {zero}/{n} | {shuffle}/{n} | {label}/{n} | {necessary} |".format(
                name=name,
                status=summary["status"],
                matched=summary["matched_correct"],
                target=summary["target_only_correct"],
                zero=summary["zero_source_correct"],
                shuffle=summary["shuffled_source_correct"],
                label=summary["label_shuffled_correct"],
                necessary=summary["clean_source_necessary"],
                n=payload["reference_n"],
            )
        )
    token_stats = payload["answer_window_stats"]
    lines.extend(
        [
            "",
            "## Answer Window Stats",
            "",
            f"- rows with detected final numeric answer: `{token_stats['detected_answer_rows']}/{payload['reference_n']}`",
            f"- average pre-answer token count: `{token_stats['avg_pre_answer_token_count']:.3f}`",
            f"- average post-answer token count: `{token_stats['avg_post_answer_token_count']:.3f}`",
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Decision",
            "",
            payload["decision"],
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze(
    *,
    source_model: str,
    target_model: str,
    eval_file: pathlib.Path,
    target_spec: syndrome.RowSpec,
    teacher_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    fallback_label: str,
    config: source_probe.ProbeConfig,
    min_numeric_coverage: int,
    device: str,
    max_new_tokens: int,
    residual_projection_dim: int,
    start_index: int,
    limit: int | None,
    run_date: str,
    output_json: pathlib.Path,
    output_md: pathlib.Path,
) -> dict[str, Any]:
    limited_debug_dir: pathlib.Path | None = None
    if limit is not None or start_index:
        limited_examples = load_generation(str(eval_file))[int(start_index) :]
        if limit is not None:
            limited_examples = limited_examples[: int(limit)]
        reference_ids = [_generation_example_id(example) for example in limited_examples]
        limit_label = "all" if limit is None else str(int(limit))
        limited_debug_dir = ROOT / ".debug" / f"{output_json.parent.name}_start{int(start_index)}_limit{limit_label}"
        target_spec = _limit_spec_to_ids(
            target_spec,
            reference_ids=reference_ids,
            output_dir=limited_debug_dir,
        )
        teacher_spec = _limit_spec_to_ids(
            teacher_spec,
            reference_ids=reference_ids,
            output_dir=limited_debug_dir,
        )
        candidate_specs = [
            _limit_spec_to_ids(
                spec,
                reference_ids=reference_ids,
                output_dir=limited_debug_dir,
            )
            for spec in candidate_specs
        ]
        target_set_path = _limit_target_set_to_ids(
            target_set_path,
            reference_ids=reference_ids,
            output_dir=limited_debug_dir,
        )
    features_by_window, feature_metadata, c2c_run_config = extract_preanswer_feature_sets(
        source_model=source_model,
        target_model=target_model,
        eval_file=eval_file,
        device=device,
        max_new_tokens=max_new_tokens,
        residual_projection_dim=residual_projection_dim,
        start_index=start_index,
        limit=limit,
    )
    window_payloads: dict[str, Any] = {}
    for window_name, features in features_by_window.items():
        window_metadata = []
        for row in feature_metadata:
            row_meta = dict(row)
            row_meta["feature_window"] = window_name
            if window_name == "post_answer_inclusive":
                feature_names = [
                    *(f"projector::{name}" for name in row_meta["components"]["projector"]["feature_names"]),
                    *(f"post::{name}" for name in row_meta["components"]["post_answer_logits"]["feature_names"]),
                ]
                row_meta["feature_names"] = feature_names
            window_metadata.append(row_meta)
        run_config = dict(c2c_run_config)
        run_config["feature_family"] = window_name
        payload = mechanism_probe.analyze_with_c2c_features(
            features=features,
            feature_metadata=window_metadata,
            c2c_run_config=run_config,
            target_spec=target_spec,
            teacher_spec=teacher_spec,
            candidate_specs=list(candidate_specs),
            target_set_path=target_set_path,
            fallback_label=fallback_label,
            config=config,
            min_numeric_coverage=min_numeric_coverage,
            run_date=run_date,
        )
        payload["config"]["feature_family"] = f"c2c_{window_name}_projector_and_logit_state"
        payload["interpretation"] = (
            "This is a C2C-derived proxy diagnostic. It trains leave-one-out "
            f"readouts from `{window_name}` features to C2C residue labels. "
            "It is not deployable unless a later source-side packet predicts "
            "the same pre-answer signal under source-private controls."
        )
        payload["feature_provenance"] = _feature_hash(features)
        window_payloads[window_name] = payload

    window_summaries = {
        name: _window_summary(payload)
        for name, payload in window_payloads.items()
    }
    answer_rows = [
        row for row in feature_metadata if row.get("answer_value") is not None
    ]
    status = _overall_status(
        window_payloads["pre_answer_exclusive"],
        window_payloads["post_answer_inclusive"],
    )
    if status == "pre_answer_c2c_state_proxy_clears_gate_not_deployable":
        decision = (
            "Promote this as a proxy capacity row only, then require a "
            "source-private predictor of the pre-answer state before any ICLR claim."
        )
    else:
        decision = (
            "Do not train a source predictor against this exact pre-answer summary "
            "unless a sharper state target is introduced. The current summary does "
            "not recover C2C-only clean rows beyond controls."
        )
    payload = {
        "date": run_date,
        "status": status,
        "reference_n": len(feature_metadata),
        "artifacts": {
            "output_json": _display_path(output_json),
            "output_md": _display_path(output_md),
            "limited_debug_dir": None if limited_debug_dir is None else _display_path(limited_debug_dir),
        },
        "c2c_run_config": c2c_run_config,
        "window_summaries": window_summaries,
        "answer_window_stats": {
            "detected_answer_rows": len(answer_rows),
            "avg_pre_answer_token_count": (
                sum(int(row["pre_answer_token_count"]) for row in feature_metadata)
                / max(len(feature_metadata), 1)
            ),
            "avg_post_answer_token_count": (
                sum(int(row["post_answer_token_count"]) for row in feature_metadata)
                / max(len(feature_metadata), 1)
            ),
        },
        "feature_metadata": feature_metadata,
        "windows": window_payloads,
        "interpretation": (
            "The pre-answer window excludes the token scores at and after the "
            "detected final numeric answer onset. The post-answer window includes "
            "the answer span as a leakage control. Both windows are C2C-derived "
            "teacher diagnostics, not source-private LatentWire packets."
        ),
        "decision": decision,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    _write_manifest(output_json.parent, payload, [output_json, output_md])
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--teacher", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target_self_repair")
    parser.add_argument("--moduli", default="2,3,5,7")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=14)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--residual-projection-dim", type=int, default=4)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = source_probe.ProbeConfig(
        moduli=tuple(int(value) for value in str(args.moduli).split(",") if value.strip()),
        probe_model="ridge",
        ridge_lambda=float(args.ridge_lambda),
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
    )
    payload = analyze(
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        eval_file=_resolve(args.eval_file),
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=list(args.candidate),
        target_set_path=_resolve(args.target_set_json),
        fallback_label=str(args.fallback_label),
        config=config,
        min_numeric_coverage=int(args.min_numeric_coverage),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        residual_projection_dim=int(args.residual_projection_dim),
        start_index=int(args.start_index),
        limit=args.limit,
        run_date=str(args.date),
        output_json=_resolve(args.output_json),
        output_md=_resolve(args.output_md),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_json": payload["artifacts"]["output_json"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
