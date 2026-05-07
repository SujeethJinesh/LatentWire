"""Resource-limited SSQ-LR S2 state-quantization replay scout.

This runner mutates cached recurrent SSM states, replays a short continuation,
and measures continuation-NLL / BF16-argmax fidelity. It is not a task-accuracy
benchmark, not a native FP4/FP8 implementation, and not a promotable S2 packet.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch

from experimental.shared.followup_gate_contracts import evaluate_ssq_lr_s2
from experimental.shared.fp4_simulator import (
    _flatten_blocks,
    _restore_blocks,
    e2m1_codebook,
    simulate_fp8_e4m3,
    simulate_mxfp4_e2m1,
    simulate_symmetric_int,
)
from experimental.shared.hybrid_manifest_local_capture_runner import (
    DEFAULT_HF_HOME,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    GRANITE_TINY_REVISION,
    _first_prompt_ids,
    _load_prompt,
    _load_tiny_model_and_tokenizer,
    _tokenize_prompt,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/ssq_lr_s2_state_replay_scout_20260507"
DEFAULT_SOURCE_S1_PACKET = (
    ROOT
    / "experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/ssq_lr_gate_packet/summary.json"
)
DEFAULT_PREREGISTRATION = ROOT / "experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md"


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("layer list must contain at least one layer index")
    if any(layer < 0 for layer in layers):
        raise ValueError("layer list must contain non-negative layer indices")
    return tuple(dict.fromkeys(layers))


def _state(cache: Any, layer: int) -> torch.Tensor:
    state = getattr(cache.layers[layer], "recurrent_states", None)
    if state is None:
        state = getattr(cache.layers[layer], "ssm_states", None)
    if state is None:
        raise ValueError(f"cache layer {layer} lacks recurrent_states/ssm_states")
    return state


def _set_state(cache: Any, layer: int, value: torch.Tensor) -> None:
    if hasattr(cache.layers[layer], "recurrent_states"):
        cache.layers[layer].recurrent_states = value
    elif hasattr(cache.layers[layer], "ssm_states"):
        cache.layers[layer].ssm_states = value
    else:
        raise ValueError(f"cache layer {layer} lacks recurrent_states/ssm_states")


def _selected_state_bytes(cache: Any, layers: tuple[int, ...]) -> int:
    return int(sum(_state(cache, layer).numel() * 2 for layer in layers))


def _block_count(tensor: torch.Tensor, block_size: int) -> int:
    return int(math.ceil(tensor.numel() / block_size))


def _byte_plan(cache: Any, layers: tuple[int, ...], *, precision: str, block_size: int) -> dict[str, float]:
    numel = int(sum(_state(cache, layer).numel() for layer in layers))
    blocks = int(sum(_block_count(_state(cache, layer), block_size) for layer in layers))
    bf16 = numel * 2
    if precision == "bf16_noop":
        quantized = bf16
        scale = 0
        effective_bits = 16.0
    elif precision == "int3":
        quantized = int(math.ceil(numel * 3 / 8))
        scale = blocks * 2
        effective_bits = 3.0 + (scale * 8.0 / max(numel, 1))
    elif precision in {"int8", "fp8_e4m3"}:
        quantized = numel
        scale = blocks * 2 if precision == "int8" else 0
        effective_bits = 8.0 + (scale * 8.0 / max(numel, 1))
    elif precision == "mxfp4_e2m1":
        quantized = int(math.ceil(numel / 2))
        scale = blocks * 2
        effective_bits = 4.0 + (scale * 8.0 / max(numel, 1))
    elif precision.startswith("mixed_int3_mxfp4_low_error_"):
        fraction = _parse_mixed_int3_fraction(precision)
        quantized = 0
        metadata = 0
        for layer in layers:
            state = _state(cache, layer)
            layer_blocks = _block_count(state, block_size)
            int3_blocks = _mixed_int3_block_count(layer_blocks, fraction)
            int3_elements = min(state.numel(), int3_blocks * block_size)
            mxfp4_elements = state.numel() - int3_elements
            quantized += int(math.ceil((int3_elements * 3 + mxfp4_elements * 4) / 8))
            metadata += int(math.ceil(layer_blocks / 8))
        scale = blocks * 2
        effective_bits = ((quantized + scale + metadata) * 8.0) / max(numel, 1)
        return {
            "bf16_state_bytes": float(bf16),
            "quantized_state_bytes": float(quantized),
            "scale_bytes": float(scale),
            "metadata_bytes": float(metadata),
            "effective_bits": float(effective_bits),
        }
    elif precision in {"bf16_noop", "random_same_l2", "shuffled_scales"}:
        quantized = int(math.ceil(numel / 2))
        scale = blocks * 2
        effective_bits = 4.0 + (scale * 8.0 / max(numel, 1))
    else:
        raise ValueError(f"unknown precision {precision!r}")
    return {
        "bf16_state_bytes": float(bf16),
        "quantized_state_bytes": float(quantized),
        "scale_bytes": float(scale),
        "metadata_bytes": 0.0,
        "effective_bits": float(effective_bits),
    }


def _parse_mixed_int3_fraction(value: str) -> float:
    prefix = "mixed_int3_mxfp4_low_error_"
    if not value.startswith(prefix) or not value.endswith("pct"):
        raise ValueError(f"mixed precision identifier must be {prefix}<percent>pct, got {value!r}")
    percent = float(value.removeprefix(prefix).removesuffix("pct"))
    if percent <= 0.0 or percent >= 100.0:
        raise ValueError("mixed INT3 percentage must be in (0, 100)")
    return percent / 100.0


def _mixed_int3_block_count(blocks: int, fraction: float) -> int:
    return max(1, min(blocks, int(math.ceil(blocks * fraction))))


def _nll_and_top1(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll.mean()), torch.argmax(logits.float(), dim=-1)


def _run_continuation(
    *,
    model: Any,
    cache: Any,
    continuation_input_ids: torch.Tensor,
    targets: torch.Tensor,
    start_position: int,
) -> tuple[float, torch.Tensor]:
    logits: list[torch.Tensor] = []
    current_cache = cache
    for offset in range(int(continuation_input_ids.shape[1])):
        position = start_position + offset
        with torch.no_grad():
            output = model(
                input_ids=continuation_input_ids[:, offset : offset + 1],
                past_key_values=current_cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=continuation_input_ids.device),
            )
        logits.append(output.logits)
        current_cache = output.past_key_values
    return _nll_and_top1(torch.cat(logits, dim=1), targets)


def _quantize_mxfp4_shuffled_scales(tensor: torch.Tensor, *, block_size: int, seed: int) -> torch.Tensor:
    blocks, original_shape, pad = _flatten_blocks(tensor, block_size)
    codebook = e2m1_codebook(blocks.device).to(dtype=torch.float32)
    max_code = torch.max(torch.abs(codebook)).clamp_min(1e-8)
    scale = torch.amax(torch.abs(blocks), dim=1, keepdim=True).clamp_min(1e-8) / max_code
    normalized = blocks / scale
    indices = torch.argmin(torch.abs(normalized[..., None] - codebook), dim=-1)
    generator = torch.Generator(device=blocks.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(scale.shape[0], generator=generator, device=blocks.device)
    shuffled_scale = scale[permutation]
    dequantized = codebook[indices] * shuffled_scale
    return _restore_blocks(dequantized, original_shape, pad).to(tensor.dtype)


def _quantize_mixed_int3_mxfp4_low_error(
    tensor: torch.Tensor,
    *,
    block_size: int,
    int3_fraction: float,
) -> torch.Tensor:
    """Use INT3 for the lowest-error blocks and MXFP4 elsewhere.

    The selector uses only block-local quantization error against the current
    state tensor. It does not inspect downstream logits or labels, so it is a
    deployable allocation rule rather than an accuracy oracle.
    """

    int3 = simulate_symmetric_int(tensor, bits=3, block_size=block_size).dequantized
    mxfp4 = simulate_mxfp4_e2m1(tensor, block_size=block_size).dequantized
    original_blocks, original_shape, pad = _flatten_blocks(tensor, block_size)
    int3_blocks, _, _ = _flatten_blocks(int3, block_size)
    mxfp4_blocks, _, _ = _flatten_blocks(mxfp4, block_size)
    block_errors = torch.mean((int3_blocks.float() - original_blocks.float()) ** 2, dim=1)
    int3_count = _mixed_int3_block_count(int(block_errors.numel()), int3_fraction)
    selected = torch.zeros_like(block_errors, dtype=torch.bool)
    selected[torch.topk(block_errors, k=int3_count, largest=False).indices] = True
    mixed_blocks = torch.where(selected[:, None], int3_blocks, mxfp4_blocks)
    return _restore_blocks(mixed_blocks, original_shape, pad).to(tensor.dtype)


def _same_l2_noise(tensor: torch.Tensor, reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    error_norm = torch.linalg.norm((reference.float() - tensor.float()).reshape(-1))
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=torch.float32)
    noise_norm = torch.linalg.norm(noise.reshape(-1)).clamp_min(1e-8)
    return (tensor.float() + noise * (error_norm / noise_norm)).to(tensor.dtype)


def _apply_recipe(
    cache: Any,
    *,
    recipe_id: str,
    layers: tuple[int, ...],
    block_size: int,
    seed: int,
) -> None:
    for layer in layers:
        state = _state(cache, layer)
        if recipe_id.startswith("mixed_int3_mxfp4_low_error_"):
            quantized = _quantize_mixed_int3_mxfp4_low_error(
                state,
                block_size=block_size,
                int3_fraction=_parse_mixed_int3_fraction(recipe_id),
            )
        elif recipe_id.startswith("int3"):
            quantized = simulate_symmetric_int(state, bits=3, block_size=block_size).dequantized
        elif recipe_id.startswith("int8"):
            quantized = simulate_symmetric_int(state, bits=8, block_size=block_size).dequantized
        elif recipe_id.startswith("fp8"):
            quantized = simulate_fp8_e4m3(state).dequantized
        elif recipe_id.startswith("random_same_l2"):
            reference = simulate_mxfp4_e2m1(state, block_size=block_size).dequantized
            quantized = _same_l2_noise(state, reference, seed=seed + layer)
        elif recipe_id.startswith("shuffled_scales"):
            quantized = _quantize_mxfp4_shuffled_scales(state, block_size=block_size, seed=seed + layer)
        elif recipe_id.startswith("mxfp4") or "mxfp4" in recipe_id:
            quantized = simulate_mxfp4_e2m1(state, block_size=block_size).dequantized
        elif recipe_id.startswith("bf16_noop"):
            quantized = state.clone()
        else:
            raise ValueError(f"unknown recipe_id {recipe_id!r}")
        _set_state(cache, layer, quantized)


def _row(
    *,
    model_id: str,
    prompt_id: str,
    recipe_id: str,
    precision: str,
    scale_granularity: str,
    block_size: int,
    byte_plan: dict[str, float],
    bf16_nll: float,
    quantized_nll: float,
    argmax_agreement: float,
    control_type: str,
    bf16_noop_delta: float = 0.0,
) -> dict[str, Any]:
    accuracy_delta = float(1.0 - argmax_agreement)
    nll_delta = float(quantized_nll - bf16_nll)
    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "recipe_id": recipe_id,
        "precision": precision,
        "scale_granularity": scale_granularity,
        "block_size": block_size,
        **byte_plan,
        "bf16_accuracy": 1.0,
        "quantized_accuracy": float(argmax_agreement),
        "accuracy_delta_abs": accuracy_delta,
        "bf16_nll": bf16_nll,
        "quantized_nll": quantized_nll,
        "nll_delta": nll_delta,
        "nll_delta_abs": abs(nll_delta),
        "nll_delta_abs_ci_high": abs(nll_delta),
        "paired_ci_low": accuracy_delta,
        "paired_ci_high": accuracy_delta,
        "bf16_noop_delta": bf16_noop_delta,
        "control_type": control_type,
    }


def _candidate_recipe_specs(primary_layers: tuple[int, ...]) -> list[dict[str, Any]]:
    contiguous_control_layers = tuple(range(len(primary_layers)))
    return [
        {
            "recipe_id": "int3_primary_state_block_scaled",
            "precision": "int3",
            "control_type": "candidate_recipe",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax",
        },
        {
            "recipe_id": "mixed_int3_mxfp4_low_error_10pct",
            "precision": "mixed_int3_mxfp4_low_error_10pct",
            "control_type": "candidate_recipe",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax_with_int3_mask",
        },
        {
            "recipe_id": "mixed_int3_mxfp4_low_error_25pct",
            "precision": "mixed_int3_mxfp4_low_error_25pct",
            "control_type": "candidate_recipe",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax_with_int3_mask",
        },
        {
            "recipe_id": "int8_primary_state_block64",
            "precision": "int8",
            "control_type": "candidate_recipe",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax",
        },
        {
            "recipe_id": "fp8_e4m3_primary_state",
            "precision": "fp8_e4m3",
            "control_type": "candidate_recipe",
            "layers": primary_layers,
            "scale_granularity": "none_cast",
        },
        {
            "recipe_id": "mxfp4_primary_state_block64",
            "precision": "mxfp4_e2m1",
            "control_type": "candidate_recipe",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax",
        },
        {
            "recipe_id": "bf16_noop_primary_state",
            "precision": "bf16_noop",
            "control_type": "bf16_noop",
            "layers": primary_layers,
            "scale_granularity": "none",
        },
        {
            "recipe_id": "same_byte_uniform_contiguous_mxfp4",
            "precision": "mxfp4_e2m1",
            "control_type": "same_byte_uniform",
            "layers": contiguous_control_layers,
            "scale_granularity": "per_block_absmax",
        },
        {
            "recipe_id": "int8_state_baseline",
            "precision": "int8",
            "control_type": "int8_state",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax",
        },
        {
            "recipe_id": "fp8_state_baseline",
            "precision": "fp8_e4m3",
            "control_type": "fp8_state",
            "layers": primary_layers,
            "scale_granularity": "none_cast",
        },
        {
            "recipe_id": "mxfp4_state_baseline",
            "precision": "mxfp4_e2m1",
            "control_type": "mxfp4_state",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax",
        },
        {
            "recipe_id": "random_same_l2_mxfp4_error",
            "precision": "random_same_l2",
            "control_type": "random_same_l2",
            "layers": primary_layers,
            "scale_granularity": "matched_mxfp4_error_l2",
        },
        {
            "recipe_id": "shuffled_scales_mxfp4",
            "precision": "shuffled_scales",
            "control_type": "shuffled_scales",
            "layers": primary_layers,
            "scale_granularity": "per_block_absmax_shuffled",
        },
    ]


def _attach_conservative_quality_bounds(rows: list[dict[str, Any]]) -> None:
    """Attach prompt-paired conservative upper bounds per recipe/control group."""
    groups = sorted({(str(row["recipe_id"]), str(row["control_type"])) for row in rows})
    for recipe_id, control_type in groups:
        current = [
            row
            for row in rows
            if str(row["recipe_id"]) == recipe_id and str(row["control_type"]) == control_type
        ]
        accuracy_high = max(float(row["accuracy_delta_abs"]) for row in current)
        nll_high = max(abs(float(row["nll_delta"])) for row in current)
        for row in current:
            row["paired_ci_low"] = 0.0
            row["paired_ci_high"] = accuracy_high
            row["nll_delta_abs"] = abs(float(row["nll_delta"]))
            row["nll_delta_abs_ci_high"] = nll_high


def run_scout(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_limit: int = 4,
    prompt_path: Path = DEFAULT_PROMPTS,
    hf_home: Path = DEFAULT_HF_HOME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_input_tokens: int = 8,
    prefix_tokens: int = 4,
    primary_layers: tuple[int, ...] = (0, 12, 30),
    block_size: int = 64,
    seed: int = 1729,
) -> dict[str, Any]:
    if prefix_tokens < 1 or prefix_tokens >= max_input_tokens - 1:
        raise ValueError("prefix_tokens must leave at least two continuation tokens")
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    _reset_output_dir(output_dir)
    prompt_ids = _first_prompt_ids(prompt_path, prompt_limit)
    tokenizer, model, load_seconds = _load_tiny_model_and_tokenizer(model_id=model_id, hf_home=hf_home)
    specs = _candidate_recipe_specs(primary_layers)
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for prompt_index, prompt_id in enumerate(prompt_ids):
        prompt = _load_prompt(prompt_path, prompt_id)
        tokenized = _tokenize_prompt(tokenizer, str(prompt["prompt"]), max_input_tokens)
        input_ids = tokenized["input_ids"]
        prefix_ids = input_ids[:, :prefix_tokens]
        continuation_input_ids = input_ids[:, prefix_tokens : max_input_tokens - 1]
        targets = input_ids[:, prefix_tokens + 1 : max_input_tokens]
        if continuation_input_ids.numel() == 0 or targets.numel() == 0:
            continue
        with torch.no_grad():
            prefix_output = model(input_ids=prefix_ids, use_cache=True)
        bf16_cache = copy.deepcopy(prefix_output.past_key_values)
        bf16_nll, bf16_top1 = _run_continuation(
            model=model,
            cache=copy.deepcopy(bf16_cache),
            continuation_input_ids=continuation_input_ids,
            targets=targets,
            start_position=prefix_tokens,
        )
        bf16_state_bytes = _selected_state_bytes(bf16_cache, primary_layers)
        for spec_index, spec in enumerate(specs):
            recipe_cache = copy.deepcopy(bf16_cache)
            recipe_seed = seed + 1000 * prompt_index + spec_index
            _apply_recipe(
                recipe_cache,
                recipe_id=str(spec["recipe_id"]),
                layers=tuple(spec["layers"]),
                block_size=block_size,
                seed=recipe_seed,
            )
            quantized_nll, quantized_top1 = _run_continuation(
                model=model,
                cache=recipe_cache,
                continuation_input_ids=continuation_input_ids,
                targets=targets,
                start_position=prefix_tokens,
            )
            agreement = float(torch.mean((quantized_top1 == bf16_top1).float()))
            byte_plan = _byte_plan(
                bf16_cache,
                tuple(spec["layers"]),
                precision=str(spec["precision"]),
                block_size=block_size,
            )
            bf16_noop_delta = abs(quantized_nll - bf16_nll) if spec["control_type"] == "bf16_noop" else 0.0
            row = _row(
                model_id=model_id,
                prompt_id=prompt_id,
                recipe_id=str(spec["recipe_id"]),
                precision=str(spec["precision"]),
                scale_granularity=str(spec["scale_granularity"]),
                block_size=block_size,
                byte_plan=byte_plan,
                bf16_nll=bf16_nll,
                quantized_nll=quantized_nll,
                argmax_agreement=agreement,
                control_type=str(spec["control_type"]),
                bf16_noop_delta=bf16_noop_delta,
            )
            row["primary_layers"] = list(primary_layers)
            row["mutated_layers"] = list(spec["layers"])
            row["continuation_token_count"] = int(targets.numel())
            row["bf16_selected_state_bytes"] = bf16_state_bytes
            rows.append(row)
            run_rows.append(
                {
                    "prompt_id": prompt_id,
                    "recipe_id": str(spec["recipe_id"]),
                    "control_type": str(spec["control_type"]),
                    "nll_delta": row["nll_delta"],
                    "argmax_agreement": agreement,
                }
            )

    _attach_conservative_quality_bounds(rows)
    contract_eval = evaluate_ssq_lr_s2(rows)
    by_control = _summarize_by_control(rows)
    resource_limited_decision = f"RESOURCE_LIMITED_S2_SCOUT_NOT_PROMOTABLE_{contract_eval['gate_status']}"
    decision = str(contract_eval["gate_status"])
    config = {
        "gate_name": "ssq_lr_s2",
        "project": "ssq_lr",
        "source_gate_packet_sha256": _sha256(DEFAULT_SOURCE_S1_PACKET),
        "preregistration_sha256": _sha256(DEFAULT_PREREGISTRATION),
        "seed_list": [seed],
        "command": "python -m experimental.shared.ssq_lr_s2_state_replay_scout",
        "model_id": model_id,
        "model_revision": GRANITE_TINY_REVISION,
        "prompt_ids": list(prompt_ids),
        "max_input_tokens": max_input_tokens,
        "prefix_tokens": prefix_tokens,
        "primary_layers": list(primary_layers),
        "claim_boundary": [
            "resource-limited local continuation replay",
            "BF16-argmax fidelity proxy, not task accuracy",
            "simulated quantization only",
            "not GPU evidence",
            "not a promotable S2 packet",
        ],
    }
    summary = {
        "decision": decision,
        "resource_limited_decision": resource_limited_decision,
        "gate_name": contract_eval["gate_name"],
        "gate_status": contract_eval["gate_status"],
        "gate_pass": contract_eval["gate_pass"],
        **contract_eval,
        "row_count": len(rows),
        "prompt_count": len(prompt_ids),
        "contract_evaluation": contract_eval,
        "by_control": by_control,
        "elapsed_seconds": time.perf_counter() - started + load_seconds,
        "load_seconds": load_seconds,
        "run_rows": run_rows,
        "claim_boundary": config["claim_boundary"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_dir / "decision.md").write_text(
        "# SSQ-LR S2 State Replay Scout\n\n"
        f"`{decision}`\n\n"
        f"Resource-limited label: `{resource_limited_decision}`\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, sort_keys=True))
    return summary


def _summarize_by_control(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = sorted({str(row["control_type"]) for row in rows})
    readout: list[dict[str, Any]] = []
    for control in controls:
        current = [row for row in rows if str(row["control_type"]) == control]
        readout.append(
            {
                "control_type": control,
                "recipe_ids": sorted({str(row["recipe_id"]) for row in current}),
                "mean_nll_delta": _mean([float(row["nll_delta"]) for row in current]),
                "max_abs_nll_delta": max(abs(float(row["nll_delta"])) for row in current),
                "mean_accuracy_delta_abs": _mean([float(row["accuracy_delta_abs"]) for row in current]),
                "max_accuracy_delta_abs": max(float(row["accuracy_delta_abs"]) for row in current),
                "min_memory_reduction": min(
                    float(row["bf16_state_bytes"])
                    / (
                        float(row["quantized_state_bytes"])
                        + float(row["scale_bytes"])
                        + float(row["metadata_bytes"])
                    )
                    for row in current
                ),
            }
        )
    return readout


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summary_markdown(summary: dict[str, Any]) -> str:
    contract = summary["contract_evaluation"]
    lines = [
        "# SSQ-LR S2 State Replay Scout",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a resource-limited local continuation replay. It cannot promote S2.",
        "",
        f"- Prompts: `{summary['prompt_count']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Contract gate status: `{contract['gate_status']}`",
        f"- Selected recipe: `{contract['selected_recipe_id']}`",
        f"- Selected memory reduction: `{contract['selected_memory_reduction']:.3f}x`",
        f"- Selected accuracy delta high: `{contract['selected_accuracy_ci_high']:.6f}`",
        "",
        "| Control | Mean NLL delta | Max abs NLL delta | Max accuracy delta | Min memory reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_control"]:
        lines.append(
            "| {control_type} | {mean_nll_delta:.6f} | {max_abs_nll_delta:.6f} | {max_accuracy_delta_abs:.6f} | {min_memory_reduction:.3f}x |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--primary-layers", default="0,12,30")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    run_scout(
        model_id=args.model_id,
        prompt_limit=args.prompt_limit,
        prompt_path=args.prompt_path,
        hf_home=args.hf_home,
        output_dir=args.output_dir,
        max_input_tokens=args.max_input_tokens,
        prefix_tokens=args.prefix_tokens,
        primary_layers=_parse_layers(args.primary_layers),
        block_size=args.block_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
