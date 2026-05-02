#!/usr/bin/env python3
"""Probe whether C2C prefill mechanism traces predict the SVAMP32 syndrome.

This is a deployability diagnostic for the C2C-derived syndrome bound. It
extracts projector scalar/gate summaries after C2C prefill, before answer
generation, then reuses the strict target-candidate syndrome decoder and
source-destroying controls from the source-latent probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import date
from typing import Any

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.c2c_eval import (
    extract_c2c_prefill_trace_features,
    load_c2c_model,
)
from latent_bridge.evaluate import _generation_example_id, load_generation
from scripts import analyze_svamp32_source_latent_syndrome_probe as source_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _rewrite_status(status: str) -> str:
    return status.replace("source_latent", "c2c_mechanism")


@torch.no_grad()
def extract_mechanism_features(
    *,
    source_model: str,
    target_model: str,
    eval_file: pathlib.Path,
    device: str,
    max_new_tokens: int,
    residual_projection_dim: int,
    feature_family: str,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    examples = load_generation(str(eval_file))
    model, tokenizer, artifact = load_c2c_model(
        source_model=source_model,
        target_model=target_model,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for example in examples:
        feature, row_meta = extract_c2c_prefill_trace_features(
            model,
            tokenizer,
            example.prompt,
            device=device,
            residual_projection_dim=int(residual_projection_dim),
            feature_family=str(feature_family),
        )
        row_meta = dict(row_meta)
        row_meta["example_id"] = _generation_example_id(example)
        rows.append(feature)
        metadata.append(row_meta)

    run_config = {
        "source_model": source_model,
        "target_model": target_model,
        "eval_file": _display_path(eval_file),
        "device": device,
        "max_new_tokens": int(max_new_tokens),
        "residual_projection_dim": int(residual_projection_dim),
        "feature_family": str(feature_family),
        "published_repo_id": artifact.repo_id,
        "published_subdir": artifact.subdir,
        "published_config_path": artifact.config_path,
        "published_checkpoint_dir": artifact.checkpoint_dir,
        "local_root": artifact.local_root,
    }
    return torch.stack(rows, dim=0), metadata, run_config


def analyze_with_c2c_features(
    *,
    features: torch.Tensor,
    feature_metadata: list[dict[str, Any]],
    c2c_run_config: dict[str, Any],
    target_spec: syndrome.RowSpec,
    teacher_spec: syndrome.RowSpec,
    candidate_specs: list[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    fallback_label: str,
    config: source_probe.ProbeConfig,
    min_numeric_coverage: int,
    run_date: str,
) -> dict[str, Any]:
    payload = source_probe.analyze_with_features(
        features=features,
        feature_metadata=feature_metadata,
        target_spec=target_spec,
        teacher_spec=teacher_spec,
        candidate_specs=candidate_specs,
        target_set_path=target_set_path,
        fallback_label=fallback_label,
        config=config,
        min_numeric_coverage=min_numeric_coverage,
        run_date=run_date,
    )
    payload["status"] = _rewrite_status(str(payload["status"]))
    payload["interpretation"] = (
        "This probe trains leave-one-out classifiers from C2C prefill "
        "projector traces to the compact C2C residue syndrome. "
        "The feature extractor does not decode or parse C2C final answers; "
        "labels still come from C2C final numeric residues, so this remains a "
        "strict small-slice distillation diagnostic rather than a paper claim."
    )
    payload["c2c_run_config"] = dict(c2c_run_config)
    requested_family = str(c2c_run_config.get("feature_family", "summary_trace"))
    if requested_family == "token_layer_tail_residual":
        payload["config"]["feature_family"] = "c2c_prefill_token_layer_tail_residual"
    else:
        payload["config"]["feature_family"] = (
            "c2c_prefill_projector_residual_trace"
            if int(c2c_run_config.get("residual_projection_dim", 0)) <= 0
            else "c2c_prefill_projector_residual_trace_with_signed_projections"
        )
    feature_bytes = features.detach().cpu().contiguous().numpy().tobytes()
    payload["feature_provenance"] = {
        "shape": [int(dim) for dim in features.shape],
        "dtype": str(features.dtype),
        "sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "storage": "not_embedded_rerun_command_recorded",
    }
    payload["run"]["status"] = _rewrite_status(str(payload["run"]["status"]))
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    run = payload["run"]
    lines = [
        "# SVAMP32 C2C Mechanism Syndrome Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- feature family: `{payload['config']['feature_family']}`",
        f"- moduli: `{','.join(str(value) for value in payload['config']['moduli'])}`",
        f"- ridge lambda: `{payload['config']['ridge_lambda']}`",
        f"- teacher numeric coverage: `{payload['provenance']['teacher_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
        "",
        "## Summary",
        "",
        "| Condition | Correct | Clean Correct | Target-Self Correct |",
        "|---|---:|---:|---:|",
    ]
    for condition in (
        "matched",
        "zero_source",
        "shuffled_source",
        "label_shuffled",
        "target_only",
        "slots_only",
    ):
        summary = run["condition_summaries"][condition]
        lines.append(
            "| {condition} | {correct} | {clean} | {target_self} |".format(
                condition=condition,
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
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--teacher", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target_self_repair")
    parser.add_argument("--moduli", default="2,3,5,7")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--probe-model", choices=["ridge", "query_bottleneck"], default="ridge")
    parser.add_argument(
        "--feature-family",
        choices=["summary_trace", "token_layer_tail_residual"],
        default="summary_trace",
    )
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=14)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--query-slots", type=int, default=8)
    parser.add_argument("--query-epochs", type=int, default=80)
    parser.add_argument("--query-lr", type=float, default=1e-2)
    parser.add_argument("--query-weight-decay", type=float, default=1e-3)
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument(
        "--residual-projection-dim",
        type=int,
        default=0,
        help="Optional deterministic signed projection width per C2C residual tensor.",
    )
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    moduli = tuple(int(value) for value in str(args.moduli).split(",") if value.strip())
    config = source_probe.ProbeConfig(
        moduli=moduli,
        probe_model=str(args.probe_model),
        ridge_lambda=float(args.ridge_lambda),
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
        query_slots=int(args.query_slots),
        query_epochs=int(args.query_epochs),
        query_lr=float(args.query_lr),
        query_weight_decay=float(args.query_weight_decay),
        query_seed=int(args.query_seed),
    )
    features, feature_metadata, c2c_run_config = extract_mechanism_features(
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        eval_file=_resolve(args.eval_file),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        residual_projection_dim=int(args.residual_projection_dim),
        feature_family=str(args.feature_family),
    )
    payload = analyze_with_c2c_features(
        features=features,
        feature_metadata=feature_metadata,
        c2c_run_config=c2c_run_config,
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=list(args.candidate),
        target_set_path=_resolve(args.target_set_json),
        fallback_label=str(args.fallback_label),
        config=config,
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
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
