"""Metrics-only SSQ-LR all-layer local scout.

This script deliberately does not write tensor packets and cannot promote S1.
It reuses the Granite Tiny local execution path to scan recurrent-state bucket
metrics across many layers without adding large `.pt` artifacts to git.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from experimental.shared.hybrid_gate_evaluators import evaluate_ssq_lr_s1
from experimental.shared.hybrid_manifest_local_capture_runner import (
    DEFAULT_HF_HOME,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    GRANITE_TINY_REVISION,
    SSQ_BUCKETS,
    _bucket_tokenized,
    _cache_layers,
    _first_prompt_ids,
    _load_prompt,
    _load_tiny_model_and_tokenizer,
    _run_loaded_forward,
    _tokenize_prompt,
)
from experimental.shared.sensitivity_metrics import kurtosis, max_abs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/ssq_lr_all_layer_scout_20260507"


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _rms(tensor: torch.Tensor) -> float:
    values = tensor.float()
    return float(torch.sqrt(torch.mean(values * values)))


def _outlier_mass(tensor: torch.Tensor) -> float:
    values = tensor.float().reshape(-1).abs()
    if values.numel() == 0:
        return 0.0
    threshold = values.mean() + 3.0 * values.std(unbiased=False)
    return float(torch.mean((values > threshold).float()))


def _state_tensor_from_cache_layer(cache_layer: Any) -> torch.Tensor | None:
    state = getattr(cache_layer, "recurrent_states", None)
    if state is None:
        state = getattr(cache_layer, "ssm_states", None)
    if state is None:
        return None
    return state.detach().float().cpu()


def _row_from_state(
    *,
    prompt_id: str,
    layer: int,
    bucket: str,
    input_tokens: int,
    state: torch.Tensor,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "layer": layer,
        "layer_kind": "ssm",
        "state_tensor_kind": "recurrent_state",
        "position_bucket": bucket,
        "input_tokens": input_tokens,
        "max_abs": max_abs(state),
        "rms": _rms(state),
        "std": float(torch.std(state.float(), unbiased=False)),
        "kurtosis": kurtosis(state),
        "outlier_mass": _outlier_mass(state),
        "tensor_shape": [int(dim) for dim in state.shape],
    }


def _all_layer_rows_from_bucket_output(
    *,
    prompt_id: str,
    bucket: str,
    input_tokens: int,
    output: Any,
    layer_limit: int | None,
    layers: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    cache_layers = _cache_layers(output)
    wanted_layers = set(layers) if layers is not None else None
    rows: list[dict[str, Any]] = []
    recurrent_seen = 0
    for layer, cache_layer in enumerate(cache_layers):
        if wanted_layers is not None and layer not in wanted_layers:
            continue
        state = _state_tensor_from_cache_layer(cache_layer)
        if state is None:
            continue
        if layer_limit is not None and recurrent_seen >= layer_limit:
            break
        recurrent_seen += 1
        rows.append(
            _row_from_state(
                prompt_id=prompt_id,
                layer=layer,
                bucket=bucket,
                input_tokens=input_tokens,
                state=state,
            )
        )
    return rows


def _per_layer_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer_bucket: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_layer_bucket.setdefault(int(row["layer"]), {}).setdefault(str(row["position_bucket"]), []).append(row)
    readout: list[dict[str, Any]] = []
    for layer, buckets in sorted(by_layer_bucket.items()):
        first_rows = buckets.get("prefill_end", [])
        final_rows = buckets.get("final_minus_128", [])
        if not first_rows or not final_rows:
            continue
        max_ratio = _safe_ratio(_mean_field(final_rows, "max_abs"), _mean_field(first_rows, "max_abs"))
        std_ratio = _safe_ratio(_mean_field(final_rows, "std"), _mean_field(first_rows, "std"))
        kurtosis_ratio = _safe_ratio(_mean_field(final_rows, "kurtosis"), _mean_field(first_rows, "kurtosis"))
        readout.append(
            {
                "layer": layer,
                "max_abs_ratio": max_ratio,
                "std_ratio": std_ratio,
                "kurtosis_ratio": kurtosis_ratio,
                "selected_ratio": max(max_ratio, std_ratio),
                "local_pass": max(max_ratio, std_ratio) >= 2.0,
            }
        )
    return readout


def _mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(row[field]) for row in rows) / len(rows)) if rows else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0
    return float(numerator / denominator)


def _scout_decision(evaluation: dict[str, Any], *, layer_count: int) -> str:
    status = str(evaluation["gate_status"])
    if layer_count < 12:
        return f"RESOURCE_LIMITED_SCOUT_NOT_PROMOTABLE_{status}"
    return f"RESOURCE_LIMITED_ALL_LAYER_SCOUT_NOT_PROMOTABLE_{status}"


def run_scout(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_id: str = "hrsmoke_0001",
    prompt_ids: tuple[str, ...] | None = None,
    prompt_limit: int | None = None,
    prompt_path: Path = DEFAULT_PROMPTS,
    hf_home: Path = DEFAULT_HF_HOME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_input_tokens: int = 8,
    layer_limit: int | None = None,
    layers: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    _reset_output_dir(output_dir)
    if prompt_ids is None:
        prompt_ids = _first_prompt_ids(prompt_path, prompt_limit) if prompt_limit else (prompt_id,)
    tokenizer, model, load_seconds = _load_tiny_model_and_tokenizer(model_id=model_id, hf_home=hf_home)

    rows: list[dict[str, Any]] = []
    bucket_runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for selected_prompt_id in prompt_ids:
        prompt = _load_prompt(prompt_path, selected_prompt_id)
        tokenized = _tokenize_prompt(tokenizer, str(prompt["prompt"]), max_input_tokens)
        for bucket in SSQ_BUCKETS:
            bucket_tokenized = _bucket_tokenized(tokenized, bucket=bucket, max_input_tokens=max_input_tokens)
            bucket_output, execution, _ = _run_loaded_forward(model=model, tokenized=bucket_tokenized)
            input_tokens = int(bucket_tokenized["input_ids"].shape[1])
            rows.extend(
                _all_layer_rows_from_bucket_output(
                    prompt_id=selected_prompt_id,
                    bucket=bucket,
                    input_tokens=input_tokens,
                    output=bucket_output,
                    layer_limit=layer_limit,
                    layers=layers,
                )
            )
            bucket_runs.append(
                {
                    "prompt_id": selected_prompt_id,
                    "position_bucket": bucket,
                    "input_tokens": input_tokens,
                    "forward_seconds": float(execution["forward_seconds"]),
                }
            )

    evaluation = evaluate_ssq_lr_s1(rows)
    layer_ratios = _per_layer_ratios(rows)
    layer_count = int(evaluation["ssm_layer_count"])
    summary = {
        "surface": "ssq_lr_all_layer_scout",
        "decision": _scout_decision(evaluation, layer_count=layer_count),
        "model_id": model_id,
        "model_revision": GRANITE_TINY_REVISION,
        "prompt_ids": list(prompt_ids),
        "prompt_count": len(prompt_ids),
        "max_input_tokens": max_input_tokens,
        "layer_limit": layer_limit,
        "layers": list(layers) if layers is not None else None,
        "load_seconds": load_seconds,
        "elapsed_seconds": time.perf_counter() - started + load_seconds,
        "bucket_runs": bucket_runs,
        "row_count": len(rows),
        "evaluation": evaluation,
        "layer_ratios": layer_ratios,
        "claim_boundary": [
            "metrics-only resource-limited local scout",
            "not a tensor-provenance gate packet",
            "not SSQ-LR S1 promotion",
            "not GPU evidence",
            "not quality evidence",
        ],
    }
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": summary["decision"]}, sort_keys=True))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    evaluation = summary["evaluation"]
    lines = [
        "# SSQ-LR All-Layer Scout",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a metrics-only resource-limited local scout. It cannot promote SSQ-LR S1.",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Prompts: `{', '.join(summary['prompt_ids'])}`",
        f"- Max input tokens: `{summary['max_input_tokens']}`",
        f"- SSM layers scanned: `{evaluation['ssm_layer_count']}`",
        f"- Passing layers: `{evaluation['passing_layer_count']}` / `{evaluation['required_passing_layer_count']}`",
        f"- Selected S1 ratio: `{evaluation['selected_s1_ratio']:.6f}`",
        "",
        "| Layer | Max-abs ratio | Std ratio | Kurtosis ratio | Local pass |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in summary["layer_ratios"]:
        lines.append(
            "| {layer} | {max_abs_ratio:.4f} | {std_ratio:.4f} | {kurtosis_ratio:.4f} | `{local_pass}` |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-id", default="hrsmoke_0001")
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=8)
    parser.add_argument("--layer-limit", type=int)
    parser.add_argument(
        "--layers",
        help="Comma-separated cache-layer indices to scan instead of all recurrent layers.",
    )
    args = parser.parse_args()
    if args.layer_limit is not None and args.layer_limit < 1:
        raise ValueError("--layer-limit must be at least 1")
    layers = _parse_layers(args.layers) if args.layers else None
    run_scout(
        model_id=args.model_id,
        prompt_id=args.prompt_id,
        prompt_limit=args.prompt_limit,
        prompt_path=args.prompt_path,
        hf_home=args.hf_home,
        output_dir=args.output_dir,
        max_input_tokens=args.max_input_tokens,
        layer_limit=args.layer_limit,
        layers=layers,
    )


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("--layers must contain at least one layer index")
    if any(layer < 0 for layer in layers):
        raise ValueError("--layers must be non-negative")
    return tuple(dict.fromkeys(layers))


if __name__ == "__main__":
    main()
