"""Build a resource-limited HBSM B1 sensitivity packet from cached Granite Tiny.

This runner fills the HBSM row-packet template with actual local forward
sensitivity measurements. It is deliberately non-promoting: the default run
uses one short prompt and a small layer subset so we can validate the HBSM B1
artifact path on Mac before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experimental.shared.check_gate_packet import validate_gate_packet
from experimental.shared.fp4_simulator import simulate_mxfp4_e2m1
from experimental.shared.hybrid_manifest_local_capture_runner import (
    DEFAULT_CANONICAL_MODEL_ID,
    DEFAULT_HF_HOME,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    GRANITE_TINY_REVISION,
    _decoder_layers,
    _first_prompt_ids,
    _first_tensor,
    _load_prompt,
    _tokenize_prompt,
)
from experimental.shared.hybrid_trace_packet_builder import build_hbsm_packet
from experimental.shared.sensitivity_metrics import kurtosis, max_abs, symmetric_kl_from_logits


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hbsm_local_sensitivity_20260507"
HBSM_CONTROL_TYPES = (
    "perturbation_off",
    "random_flags",
    "layer_index",
    "parameter_count_norm",
    "kl_lens_rank",
    "activation_outlier",
)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_hbsm_template(
    manifest_dir: Path,
    *,
    canonical_model_id: str = DEFAULT_CANONICAL_MODEL_ID,
) -> dict[str, Any]:
    filename = f"hbsm__{canonical_model_id.replace('.', '-')}__row_packet_template.json"
    path = manifest_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing HBSM row template: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def select_hbsm_entries(
    template: dict[str, Any],
    *,
    prompt_id: str,
    layer_limit: int,
) -> list[dict[str, Any]]:
    """Select a layer-aligned resource-limited HBSM B1 row subset."""

    return select_hbsm_entries_for_prompts(
        template,
        prompt_ids=(prompt_id,),
        layer_limit=layer_limit,
    )


def select_hbsm_entries_for_prompts(
    template: dict[str, Any],
    *,
    prompt_ids: tuple[str, ...],
    layer_limit: int,
) -> list[dict[str, Any]]:
    """Select prompt-repeat, layer-aligned HBSM B1 rows plus shared controls."""

    if layer_limit < 2:
        raise ValueError("--layer-limit must be at least 2")
    if not prompt_ids:
        raise ValueError("at least one prompt_id is required")
    templates = template.get("hbsm_entry_templates")
    if not isinstance(templates, list):
        raise ValueError("HBSM template missing hbsm_entry_templates")
    prompt_set = set(prompt_ids)
    available_prompt_ids = {str(entry.get("prompt_id")) for entry in templates}
    missing_prompts = sorted(prompt_set - available_prompt_ids)
    if missing_prompts:
        raise ValueError(f"HBSM template missing prompt rows for {missing_prompts}")
    primary = [
        dict(entry)
        for entry in templates
        if str(entry["control_type"]) == "boundary_only" and str(entry["prompt_id"]) in prompt_set
    ]
    layers = sorted({int(entry["layer"]) for entry in primary})[:layer_limit]
    if len(layers) < layer_limit:
        raise ValueError(f"HBSM template only has {len(layers)} layers, requested {layer_limit}")
    selected_layer_set = set(layers)
    primary = [
        entry
        for entry in sorted(primary, key=lambda item: (str(item["prompt_id"]), int(item["layer"])))
        if int(entry["layer"]) in selected_layer_set
    ]
    for prompt_id in prompt_ids:
        prompt_rows = [entry for entry in primary if str(entry["prompt_id"]) == prompt_id]
        if len(prompt_rows) != layer_limit:
            raise ValueError(f"HBSM template does not cover {layer_limit} layers for {prompt_id}")
        flags = {bool(entry["boundary_flag"]) for entry in prompt_rows}
        if flags != {False, True}:
            raise ValueError("selected HBSM layers must include boundary and non-boundary rows")
    selected_layers = {int(entry["layer"]) for entry in primary}
    controls = [
        dict(entry)
        for entry in templates
        if str(entry["control_type"]) in HBSM_CONTROL_TYPES and int(entry["layer"]) in selected_layers
    ]
    by_control = {
        control_type: {int(entry["layer"]) for entry in controls if str(entry["control_type"]) == control_type}
        for control_type in HBSM_CONTROL_TYPES
    }
    for control_type, layers in by_control.items():
        if layers != selected_layers:
            raise ValueError(f"HBSM control {control_type} does not cover selected layers")
    return primary + sorted(
        controls,
        key=lambda entry: (str(entry["control_type"]), int(entry["layer"])),
    )


def _filled_metadata(
    template: dict[str, Any],
    *,
    entries: list[dict[str, Any]],
    max_input_tokens: int,
    block_size: int,
    layer_count: int,
    prompt_count: int,
) -> dict[str, Any]:
    metadata_template = template.get("metadata")
    if not isinstance(metadata_template, dict):
        raise ValueError("HBSM template missing metadata")
    metadata = {
        key: value
        for key, value in metadata_template.items()
        if key not in {"_template_only", "fill_before_use", "planned_entry_count"}
    }
    metadata.update(
        {
            "model_id": DEFAULT_CANONICAL_MODEL_ID,
            "canonical_model_id": DEFAULT_CANONICAL_MODEL_ID,
            "served_model_id": DEFAULT_MODEL_ID,
            "model_revision": GRANITE_TINY_REVISION,
            "tokenizer_revision": GRANITE_TINY_REVISION,
            "context_lengths": [max_input_tokens],
            "dtype": "float16",
            "device": "cpu",
            "command": "python -m experimental.shared.hbsm_local_sensitivity_runner",
            "resource_limit_note": (
                f"HBSM local sensitivity runner used {prompt_count} short prompt(s), "
                f"{layer_count} layer(s), max_input_tokens={max_input_tokens}, "
                f"and MXFP4 E2M1 block_size={block_size}; this validates "
                "forward-sensitivity plumbing only and cannot promote B1."
            ),
        }
    )
    metadata["filled_entry_count"] = len(entries)
    return metadata


def _load_tiny_model_and_tokenizer(
    *,
    model_id: str,
    hf_home: Path,
) -> tuple[Any, Any, float]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    return tokenizer, model, time.perf_counter() - load_started


def _replace_first_tensor(
    value: Any,
    replacement: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[Any, bool]:
    if torch.is_tensor(value):
        return replacement(value), True
    if isinstance(value, tuple):
        items = []
        replaced = False
        for item in value:
            if replaced:
                items.append(item)
                continue
            new_item, replaced = _replace_first_tensor(item, replacement)
            items.append(new_item)
        return tuple(items), replaced
    if isinstance(value, list):
        items = []
        replaced = False
        for item in value:
            if replaced:
                items.append(item)
                continue
            new_item, replaced = _replace_first_tensor(item, replacement)
            items.append(new_item)
        return items, replaced
    if isinstance(value, dict):
        copied = dict(value)
        for key in sorted(copied):
            new_item, replaced = _replace_first_tensor(copied[key], replacement)
            if replaced:
                copied[key] = new_item
                return copied, True
        return copied, False
    return value, False


def _install_activation_capture_hooks(
    model: Any,
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    layers = _decoder_layers(model)
    captures: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer_index in layer_indices:
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"requested layer {layer_index}, model has {len(layers)} layers")

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, layer: int = layer_index) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                raise ValueError(f"HBSM capture hook could not find output tensor for layer {layer}")
            captures[layer] = tensor.detach().float().cpu()

        handles.append(layers[layer_index].register_forward_hook(hook))
    return captures, handles


def _run_logits(
    model: Any,
    tokenized: dict[str, torch.Tensor],
    *,
    capture_layers: list[int] | None = None,
    perturb_layer: int | None = None,
    block_size: int = 32,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], float]:
    handles: list[Any] = []
    captures: dict[int, torch.Tensor] = {}
    if capture_layers:
        captures, capture_handles = _install_activation_capture_hooks(model, capture_layers)
        handles.extend(capture_handles)
    if perturb_layer is not None:
        layers = _decoder_layers(model)
        if perturb_layer < 0 or perturb_layer >= len(layers):
            raise ValueError(f"requested perturb_layer={perturb_layer}, model has {len(layers)} layers")

        def perturb_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            def quantize(tensor: torch.Tensor) -> torch.Tensor:
                return simulate_mxfp4_e2m1(tensor, block_size=block_size).dequantized.to(tensor.dtype)

            new_output, replaced = _replace_first_tensor(output, quantize)
            if not replaced:
                raise ValueError(f"HBSM perturbation hook could not find output tensor for layer {perturb_layer}")
            return new_output

        handles.append(layers[perturb_layer].register_forward_hook(perturb_hook))
    started = time.perf_counter()
    try:
        with torch.no_grad():
            output = model(**tokenized, use_cache=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    return output.logits[:, -1, :].detach().float().cpu(), captures, time.perf_counter() - started


def _layer_weight_stats(model: Any, layer_indices: list[int]) -> dict[int, dict[str, float]]:
    layers = _decoder_layers(model)
    stats: dict[int, dict[str, float]] = {}
    for layer_index in layer_indices:
        parameter_count = 0
        squared_sum = 0.0
        for parameter in layers[layer_index].parameters():
            values = parameter.detach().float()
            parameter_count += values.numel()
            squared_sum += float(torch.sum(values * values))
        stats[layer_index] = {
            "parameter_count": float(parameter_count),
            "weight_norm": math.sqrt(squared_sum),
        }
    return stats


def _normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: float((value - low) / (high - low)) for key, value in values.items()}


def _top_decile_layers(drift_by_layer: dict[int, float]) -> set[int]:
    count = math.ceil(0.10 * len(drift_by_layer)) if drift_by_layer else 0
    ranked = sorted(drift_by_layer, key=lambda layer: (-drift_by_layer[layer], layer))
    return set(ranked[:count])


def _random_top_layers(
    entries_by_layer: dict[int, dict[str, Any]],
    top_layers: set[int],
) -> set[int]:
    count = math.ceil(0.10 * len(entries_by_layer)) if entries_by_layer else 0
    non_boundary = [
        layer
        for layer, entry in sorted(entries_by_layer.items())
        if not bool(entry["boundary_flag"]) and layer not in top_layers
    ]
    candidates = non_boundary or [layer for layer in sorted(entries_by_layer) if layer not in top_layers]
    if len(candidates) < count:
        candidates = sorted(entries_by_layer)
    return set(candidates[:count])


def _fill_entries(
    selected_entries: list[dict[str, Any]],
    *,
    drift_by_layer: dict[int, float],
    drift_by_prompt_layer: dict[str, dict[int, float]] | None = None,
    weight_stats: dict[int, dict[str, float]],
    activation_stats: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    primary_by_layer = {
        int(entry["layer"]): entry
        for entry in selected_entries
        if str(entry["control_type"]) == "boundary_only"
    }
    top_layers = _top_decile_layers(drift_by_layer)
    random_layers = _random_top_layers(primary_by_layer, top_layers)
    layer_norm = _normalize({layer: float(layer) for layer in primary_by_layer})
    parameter_norm = _normalize({layer: weight_stats[layer]["parameter_count"] for layer in primary_by_layer})
    weight_norm = _normalize({layer: weight_stats[layer]["weight_norm"] for layer in primary_by_layer})
    drift_norm = _normalize(drift_by_layer)
    activation_norm = _normalize({layer: activation_stats[layer]["max_abs"] for layer in primary_by_layer})

    filled: list[dict[str, Any]] = []
    for entry in selected_entries:
        row = {
            key: value
            for key, value in entry.items()
            if key not in {"required_metric_fields"}
        }
        layer = int(row["layer"])
        control_type = str(row["control_type"])
        row["parameter_count"] = int(weight_stats[layer]["parameter_count"])
        row["weight_norm"] = float(weight_stats[layer]["weight_norm"])
        row["top_decile_flag"] = layer in top_layers
        row["random_top_decile"] = layer in random_layers
        if control_type == "boundary_only":
            prompt_id = str(row.get("prompt_id", ""))
            prompt_drift = (drift_by_prompt_layer or {}).get(prompt_id, {}).get(layer)
            row["kl_or_nll_drift"] = float(prompt_drift if prompt_drift is not None else drift_by_layer[layer])
            row["cheap_predictor"] = float(weight_norm[layer])
        elif control_type == "perturbation_off":
            row["kl_or_nll_drift"] = 0.0
            row["cheap_predictor"] = float(weight_norm[layer])
        else:
            row["kl_or_nll_drift"] = float(drift_by_layer[layer])
            if control_type == "random_flags":
                row["cheap_predictor"] = 1.0 if layer in random_layers else 0.0
            elif control_type == "layer_index":
                row["cheap_predictor"] = float(layer_norm[layer])
            elif control_type == "parameter_count_norm":
                row["cheap_predictor"] = float(parameter_norm[layer])
            elif control_type == "kl_lens_rank":
                row["cheap_predictor"] = float(drift_norm[layer])
            elif control_type == "activation_outlier":
                row["cheap_predictor"] = float(activation_norm[layer])
            else:
                row["cheap_predictor"] = float(weight_norm[layer])
        filled.append(row)
    return filled


def run_hbsm_sensitivity(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_id: str = "hrsmoke_0001",
    prompt_ids: tuple[str, ...] | None = None,
    prompt_limit: int = 1,
    prompt_path: Path = DEFAULT_PROMPTS,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_input_tokens: int = 8,
    layer_limit: int = 8,
    block_size: int = 32,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template = _load_hbsm_template(manifest_dir)
    if prompt_ids is None:
        prompt_ids = _first_prompt_ids(prompt_path, prompt_limit) if prompt_limit > 1 else (prompt_id,)
    selected_entries = select_hbsm_entries_for_prompts(
        template,
        prompt_ids=prompt_ids,
        layer_limit=layer_limit,
    )
    layer_indices = sorted(
        {
            int(entry["layer"])
            for entry in selected_entries
            if str(entry["control_type"]) == "boundary_only"
        }
    )
    tokenizer, model, load_seconds = _load_tiny_model_and_tokenizer(model_id=model_id, hf_home=hf_home)
    prompt_drift_by_layer: dict[str, dict[int, float]] = {}
    activation_stats_by_prompt: dict[str, dict[int, dict[str, float]]] = {}
    baseline_seconds_by_prompt: dict[str, float] = {}
    perturb_seconds: dict[str, dict[int, float]] = {}
    input_tokens_by_prompt: dict[str, int] = {}
    for current_prompt_id in prompt_ids:
        prompt = _load_prompt(prompt_path, current_prompt_id)
        tokenized = _tokenize_prompt(tokenizer, str(prompt["prompt"]), max_input_tokens)
        input_tokens_by_prompt[current_prompt_id] = int(tokenized["input_ids"].shape[1])
        baseline_logits, captures, baseline_seconds = _run_logits(
            model,
            tokenized,
            capture_layers=layer_indices,
            block_size=block_size,
        )
        baseline_seconds_by_prompt[current_prompt_id] = baseline_seconds
        missing_captures = set(layer_indices) - set(captures)
        if missing_captures:
            raise ValueError(f"missing baseline activation captures for layers {sorted(missing_captures)}")
        activation_stats_by_prompt[current_prompt_id] = {
            layer: {
                "max_abs": max_abs(tensor),
                "kurtosis": kurtosis(tensor),
            }
            for layer, tensor in captures.items()
        }
        prompt_drift_by_layer[current_prompt_id] = {}
        perturb_seconds[current_prompt_id] = {}
        for layer in layer_indices:
            candidate_logits, _, seconds = _run_logits(
                model,
                tokenized,
                perturb_layer=layer,
                block_size=block_size,
            )
            prompt_drift_by_layer[current_prompt_id][layer] = symmetric_kl_from_logits(
                baseline_logits,
                candidate_logits,
            )
            perturb_seconds[current_prompt_id][layer] = seconds
    drift_by_layer = {
        layer: float(
            sum(prompt_drift_by_layer[prompt][layer] for prompt in prompt_ids)
            / max(len(prompt_ids), 1)
        )
        for layer in layer_indices
    }
    activation_stats = {
        layer: {
            "max_abs": float(
                sum(activation_stats_by_prompt[prompt][layer]["max_abs"] for prompt in prompt_ids)
                / max(len(prompt_ids), 1)
            ),
            "kurtosis": float(
                sum(activation_stats_by_prompt[prompt][layer]["kurtosis"] for prompt in prompt_ids)
                / max(len(prompt_ids), 1)
            ),
        }
        for layer in layer_indices
    }
    weight_stats = _layer_weight_stats(model, layer_indices)
    filled_entries = _fill_entries(
        selected_entries,
        drift_by_layer=drift_by_layer,
        drift_by_prompt_layer=prompt_drift_by_layer,
        weight_stats=weight_stats,
        activation_stats=activation_stats,
    )
    row_packet = output_dir / "hbsm_row_packet.json"
    gate_packet = output_dir / "hbsm_gate_packet"
    _reset_output_dir(gate_packet)
    row_payload = {
        "metadata": _filled_metadata(
            template,
            entries=filled_entries,
            max_input_tokens=max_input_tokens,
            block_size=block_size,
            layer_count=len(layer_indices),
            prompt_count=len(prompt_ids),
        ),
        "hbsm_entries": filled_entries,
    }
    row_packet.write_text(json.dumps(row_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_hbsm_packet(row_packet, gate_packet)
    report = validate_gate_packet(gate_packet, mode="real", project="hbsm")
    summary = {
        "surface": "hbsm_local_sensitivity_runner",
        "decision": "RESOURCE_LIMITED_HBSM_B1_PACKET_WRITTEN_NOT_PROMOTABLE",
        "model_id": model_id,
        "prompt_id": prompt_ids[0],
        "prompt_ids": list(prompt_ids),
        "prompt_count": len(prompt_ids),
        "input_tokens_by_prompt": input_tokens_by_prompt,
        "input_tokens": max(input_tokens_by_prompt.values()) if input_tokens_by_prompt else 0,
        "layer_indices": layer_indices,
        "block_size": block_size,
        "load_seconds": load_seconds,
        "baseline_seconds": sum(baseline_seconds_by_prompt.values()),
        "baseline_seconds_by_prompt": baseline_seconds_by_prompt,
        "perturb_seconds": perturb_seconds,
        "drift_by_layer": drift_by_layer,
        "prompt_drift_by_layer": prompt_drift_by_layer,
        "activation_stats": activation_stats,
        "activation_stats_by_prompt": activation_stats_by_prompt,
        "row_packet": str(row_packet),
        "gate_packet": str(gate_packet),
        "checker_ok": bool(report["ok"]),
        "checker_decision": report["decision"],
        "checker_errors": report["errors"],
        "claim_boundary": [
            "resource-limited local HBSM sensitivity packet",
            "not HBSM B1 promotion",
            "not GPU evidence",
            "not quality benchmark evidence",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "checker_ok": report["ok"], "decision": report["decision"]}))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# HBSM Local Sensitivity Runner",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a resource-limited local HBSM B1 packet. It cannot promote B1.",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Prompts: `{summary.get('prompt_ids', [summary['prompt_id']])}`",
        f"- Input tokens: `{summary['input_tokens']}`",
        f"- Layers: `{summary['layer_indices']}`",
        f"- Load seconds: `{summary['load_seconds']:.2f}`",
        f"- Baseline forward seconds: `{summary['baseline_seconds']:.2f}`",
        f"- Checker OK: `{summary['checker_ok']}`",
        f"- Checker decision: `{summary['checker_decision']}`",
        "",
        "| Layer | Symmetric KL drift |",
        "|---:|---:|",
    ]
    for layer, drift in sorted((int(layer), float(value)) for layer, value in summary["drift_by_layer"].items()):
        lines.append(f"| {layer} | {drift:.8g} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-id", default="hrsmoke_0001")
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated prompt IDs. Overrides --prompt-limit.")
    parser.add_argument("--prompt-limit", type=int, default=1)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=8)
    parser.add_argument("--layer-limit", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    prompt_ids = tuple(item.strip() for item in args.prompt_ids.split(",") if item.strip()) if args.prompt_ids else None
    run_hbsm_sensitivity(
        model_id=args.model_id,
        prompt_id=args.prompt_id,
        prompt_ids=prompt_ids,
        prompt_limit=args.prompt_limit,
        prompt_path=args.prompt_path,
        manifest_dir=args.manifest_dir,
        hf_home=args.hf_home,
        output_dir=args.output_dir,
        max_input_tokens=args.max_input_tokens,
        layer_limit=args.layer_limit,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
