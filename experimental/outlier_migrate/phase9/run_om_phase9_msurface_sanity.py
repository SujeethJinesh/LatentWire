#!/usr/bin/env python3
"""Run the Granite M-SURFACE two-trace hook sanity diagnostic."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
import traceback
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = "om_phase9_msurface_sanity_v1"
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
SOURCE_RUN = ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z"
SURFACES = (
    "mamba_out_projection_input",
    "attention_out_projection_input",
    "post_block_residual_block_output",
)
MAMBA_LAYERS = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39}
ATTENTION_LAYERS = {5, 15, 25, 35}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def selected_prompts(source_dir: Path, prompt_indices: tuple[int, ...]) -> list[dict[str, Any]]:
    manifest = load_json(source_dir / "prompt_manifest.json")
    by_index = {int(row["index"]): row for row in manifest["prompts"]}
    missing = [index for index in prompt_indices if index not in by_index]
    if missing:
        raise KeyError(f"prompt_manifest missing prompt indices {missing}")
    return [by_index[index] for index in prompt_indices]


def patch_granite_fast_path() -> None:
    try:
        import transformers.models.granitemoehybrid.modeling_granitemoehybrid as granite_mod

        granite_mod.is_fast_path_available = False
    except Exception:
        pass


def patch_hybrid_cache(model: Any, cache: Any) -> Any:
    if getattr(cache, "_latentwire_cache_patch", False):
        return cache
    config = getattr(model, "config", None)
    conv_states = getattr(cache, "conv_states", None)
    ssm_states = getattr(cache, "ssm_states", None)
    if config is None or not isinstance(conv_states, list) or not isinstance(ssm_states, list):
        return cache
    conv_kernel = getattr(config, "conv_kernel", None) or getattr(config, "mamba_d_conv", None)
    if conv_kernel is not None and not hasattr(cache, "conv_kernel_size"):
        cache.conv_kernel_size = int(conv_kernel)
    needed = [
        getattr(config, "mamba_num_heads", getattr(config, "mamba_n_heads", None)),
        getattr(config, "mamba_head_dim", getattr(config, "mamba_d_head", None)),
        getattr(config, "n_groups", getattr(config, "mamba_n_groups", None)),
        getattr(config, "ssm_state_size", getattr(config, "mamba_d_state", None)),
        conv_kernel,
    ]
    if all(value is not None for value in needed):
        heads, head_dim, groups, state_size, kernel = (int(value) for value in needed)
        conv_dim = heads * head_dim + 2 * groups * state_size
        for layer_idx, state_tensor in enumerate(list(conv_states)):
            if not hasattr(state_tensor, "ndim") or state_tensor.numel() == 0 or state_tensor.ndim != 3:
                continue
            if state_tensor.shape[1] == conv_dim and state_tensor.shape[2] == kernel:
                continue
            conv_states[layer_idx] = state_tensor.new_zeros((state_tensor.shape[0], conv_dim, kernel))

    def update_conv_state(self: Any, layer_idx: int, new_conv_state: Any, cache_init: bool = False) -> Any:
        target_device = self.conv_states[layer_idx].device
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(target_device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(target_device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self: Any, layer_idx: int, new_ssm_state: Any) -> Any:
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    cache.update_conv_state = types.MethodType(update_conv_state, cache)
    cache.update_ssm_state = types.MethodType(update_ssm_state, cache)
    cache._latentwire_cache_patch = True
    return cache


def tensor_from_hook_value(value: Any) -> Any:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        value = value[0]
    return value


def top_indices(tensor: Any) -> list[int]:
    import torch

    width = int(tensor.shape[-1])
    top_k = max(1, math.ceil(width * 0.01))
    indices = torch.topk(tensor, k=top_k, largest=True, sorted=True).indices
    return [int(index) for index in indices.tolist()]


def register_hooks(model: Any, state: dict[str, Any]) -> list[Any]:
    import torch

    handles: list[Any] = []
    layers = list(model.model.layers)

    def capture(surface: str, layer_index: int, layer_name: str, value: Any) -> None:
        if not state["capture_enabled"]:
            return
        tensor = tensor_from_hook_value(value)
        if not torch.is_tensor(tensor) or tensor.ndim < 2:
            return
        last = tensor[:, -1, :] if tensor.ndim == 3 else tensor
        mags = last.detach().abs().to(torch.float32).cpu()
        state["records"].append((surface, layer_index, layer_name, mags))

    for layer_index, layer in enumerate(layers):
        layer_name = f"model.layers.{layer_index}"

        def post_hook(_module: Any, _inputs: Any, output: Any, *, idx: int = layer_index, name: str = layer_name) -> None:
            capture("post_block_residual_block_output", idx, name, output)

        handles.append(layer.register_forward_hook(post_hook))
        if layer_index in MAMBA_LAYERS and getattr(layer, "mamba", None) is not None:

            def mamba_hook(_module: Any, inputs: Any, *, idx: int = layer_index) -> None:
                capture("mamba_out_projection_input", idx, f"model.layers.{idx}.mamba.out_proj", inputs)

            handles.append(layer.mamba.out_proj.register_forward_pre_hook(mamba_hook))
        if layer_index in ATTENTION_LAYERS and getattr(layer, "self_attn", None) is not None:

            def attn_hook(_module: Any, inputs: Any, *, idx: int = layer_index) -> None:
                capture("attention_out_projection_input", idx, f"model.layers.{idx}.self_attn.o_proj", inputs)

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(attn_hook))
    return handles


def run_capture(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    positions: tuple[int, ...],
    max_new_tokens: int,
    output_path: Path,
    run_events_path: Path,
) -> dict[str, Any]:
    import torch

    forward_parameters = set(__import__("inspect").signature(model.forward).parameters)
    cache_input_name = "cache_params" if "cache_params" in forward_parameters else "past_key_values"

    def normalize_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
        if cache_input_name == "cache_params" and "past_key_values" in model_inputs:
            model_inputs["cache_params"] = model_inputs.pop("past_key_values")
        if "cache_params" in model_inputs:
            model_inputs["cache_params"] = patch_hybrid_cache(model, model_inputs["cache_params"])
        return model_inputs

    def output_cache(outputs: Any) -> Any:
        return getattr(outputs, "past_key_values", None) or getattr(outputs, "cache_params", None)

    state: dict[str, Any] = {"capture_enabled": False, "records": []}
    handles = register_hooks(model, state)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prompt_events: list[dict[str, Any]] = []
    rows_written = 0
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as handle, torch.inference_mode():
            for prompt in prompts:
                batch_indices = [int(prompt["index"])]
                text = shared.make_prompt_text(str(prompt["prompt"]))
                encoded = tokenizer([text], padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                cache_position = torch.arange(input_ids.shape[1], device=device)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                )
                outputs = model(**normalize_inputs(model_inputs))
                past_key_values = output_cache(outputs)
                if past_key_values is None:
                    raise RuntimeError("model did not return generation cache")
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                first_eos = None
                for decode_position in range(1, max_new_tokens + 1):
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    state["capture_enabled"] = decode_position in positions
                    state["records"] = []
                    cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids=next_token[:, None],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    outputs = model(**normalize_inputs(model_inputs))
                    past_key_values = output_cache(outputs)
                    if state["capture_enabled"]:
                        for surface, layer_index, layer_name, magnitudes in state["records"]:
                            row = {
                                "schema_version": f"{SCHEMA_VERSION}_surface_row",
                                "prompt_index": int(prompt["index"]),
                                "prompt_id": prompt["prompt_id"],
                                "decode_position": decode_position,
                                "surface": surface,
                                "layer_index": layer_index,
                                "layer_name": layer_name,
                                "channel_count": int(magnitudes.shape[-1]),
                                "top1pct_channels": top_indices(magnitudes[0]),
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            rows_written += 1
                    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id) and first_eos is None:
                        first_eos = decode_position
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                prompt_events.append(
                    {
                        "prompt_index": int(prompt["index"]),
                        "prompt_id": prompt["prompt_id"],
                        "input_token_count": int(encoded["attention_mask"][0].sum().item()),
                        "first_eos_decode_position": first_eos,
                    }
                )
                run_events_path.open("a", encoding="utf-8").write(
                    json.dumps({"created_at_utc": shared.utc_now(), "event": "completed_prompt", "prompt_index": int(prompt["index"])}, sort_keys=True)
                    + "\n"
                )
    finally:
        for handle in handles:
            handle.remove()
    return {
        "schema_version": f"{SCHEMA_VERSION}_surface_manifest",
        "created_at_utc": shared.utc_now(),
        "artifact": "surface_topk_rows.jsonl.gz",
        "artifact_sha256": shared.file_sha256(output_path),
        "row_count": rows_written,
        "positions": list(positions),
        "surfaces": list(SURFACES),
        "prompt_events": prompt_events,
    }


def iter_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_metrics(rows: list[dict[str, Any]], positions: tuple[int, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_key: dict[tuple[int, str, int], dict[int, set[int]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["prompt_index"]), str(row["surface"]), int(row["layer_index"]))
        by_key[key][int(row["decode_position"])] = {int(value) for value in row["top1pct_channels"]}
    per_unit: list[dict[str, Any]] = []
    base_position, final_position = positions[0], positions[-1]
    for (prompt_index, surface, layer_index), by_position in sorted(by_key.items()):
        if base_position not in by_position or final_position not in by_position:
            continue
        base = by_position[base_position]
        final = by_position[final_position]
        leaving = 1.0 - (len(base & final) / len(base))
        per_unit.append(
            {
                "prompt_index": prompt_index,
                "surface": surface,
                "layer_index": layer_index,
                "base_position": base_position,
                "final_position": final_position,
                "strict_set_leaving": leaving,
                "top1_count": len(base),
            }
        )
    by_surface: dict[str, list[float]] = defaultdict(list)
    post_by_prompt_layer: dict[tuple[int, int], float] = {}
    for row in per_unit:
        by_surface[row["surface"]].append(float(row["strict_set_leaving"]))
        if row["surface"] == "post_block_residual_block_output":
            post_by_prompt_layer[(int(row["prompt_index"]), int(row["layer_index"]))] = float(row["strict_set_leaving"])
    surface_metrics: dict[str, Any] = {}
    for surface, values in sorted(by_surface.items()):
        paired_deltas = [
            float(row["strict_set_leaving"]) - post_by_prompt_layer[(int(row["prompt_index"]), int(row["layer_index"]))]
            for row in per_unit
            if row["surface"] == surface and (int(row["prompt_index"]), int(row["layer_index"])) in post_by_prompt_layer
        ]
        surface_metrics[surface] = {
            "unit_count": len(values),
            "mean_strict_set_leaving": mean(values),
            "median_strict_set_leaving": median(values),
            "mean_internal_minus_post_block_same_layer": mean(paired_deltas) if paired_deltas else None,
            "median_internal_minus_post_block_same_layer": median(paired_deltas) if paired_deltas else None,
        }
    decision = "KILL_OR_DEFER_SURFACE_NO_LOWER_DRIFT"
    reasons: list[str] = []
    for surface, metrics in surface_metrics.items():
        if surface == "post_block_residual_block_output":
            continue
        mean_leaving = float(metrics["mean_strict_set_leaving"])
        mean_delta = metrics["mean_internal_minus_post_block_same_layer"]
        if mean_leaving < 0.30 or (mean_delta is not None and mean_delta <= -0.15):
            decision = "PROMOTE_SURFACE_DIAGNOSTIC"
            reasons.append(f"{surface} passes drift gate: mean={mean_leaving:.6f}, delta={mean_delta}")
    if not reasons:
        reasons.append("no internal surface had mean strict leaving <0.30 or <=-0.15 paired delta vs post-block")
    return (
        {
            "schema_version": f"{SCHEMA_VERSION}_metrics",
            "surface_metrics": surface_metrics,
            "decision": decision,
            "decision_reasons": reasons,
            "per_unit_count": len(per_unit),
        },
        per_unit,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--prompt-indices", type=parse_csv_ints, default=(0, 1))
    parser.add_argument("--positions", type=parse_csv_ints, default=(100, 20000))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    run_id = args.run_id or f"om_phase9_msurface_granite_sanity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.results_dir / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")
    try:
        source_dir = args.source_run_dir.resolve()
        prompts = selected_prompts(source_dir, args.prompt_indices)
        model_provenance = load_json(source_dir / "model_provenance.json")
        shared.write_json(
            run_dir / "config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_config",
                "run_id": run_id,
                "source_run_dir": str(args.source_run_dir),
                "results_dir": str(args.results_dir),
                "prompt_indices": list(args.prompt_indices),
                "positions": list(args.positions),
                "max_new_tokens": args.max_new_tokens or max(args.positions),
                "dtype": args.dtype,
                "device": args.device,
            },
        )
        (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(sys.argv) + "\n", encoding="utf-8")
        shutil.copy2(source_dir / "model_provenance.json", run_dir / "model_provenance.json")
        shared.write_json(run_dir / "prompt_manifest.json", {"schema_version": f"{SCHEMA_VERSION}_prompt_manifest", "prompts": prompts})
        patch_granite_fast_path()
        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        surface_path = run_dir / "surface_topk_rows.jsonl.gz"
        manifest = run_capture(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            positions=args.positions,
            max_new_tokens=args.max_new_tokens or max(args.positions),
            output_path=surface_path,
            run_events_path=run_events_path,
        )
        shared.write_json(run_dir / "surface_manifest.json", manifest)
        metrics, per_unit = build_metrics(list(iter_rows(surface_path)), args.positions)
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "decision.json", {"schema_version": f"{SCHEMA_VERSION}_decision", "decision": metrics["decision"], "reasons": metrics["decision_reasons"]})
        write_csv(run_dir / "surface_unit_metrics.csv", per_unit)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        print(json.dumps({"run_dir": str(run_dir), "decision": metrics["decision"], "reasons": metrics["decision_reasons"]}, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        shared.write_json(
            run_dir / "infra_error.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_infra_error",
                "created_at_utc": shared.utc_now(),
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
