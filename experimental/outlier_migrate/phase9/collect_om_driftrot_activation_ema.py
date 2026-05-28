#!/usr/bin/env python3
"""Collect long-decode input activation EMA for residual-correction candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_paroquant_baseline as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = "om_driftrot_activation_ema_v1"
DEFAULT_BASE_RUN_DIR = ROOT / "experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z"
DEFAULT_RESIDUAL_CACHE = ROOT / "artifacts/rot_resid_correction/residual_cache_granite_tight_20260528T2025Z/residual_column_cache.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/rot_resid_correction"


def parse_indices(raw: str) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("at least one prompt index is required")
    return indices


def select_module_names(cache_path: Path, top_tensors: int) -> list[str]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    tensors = sorted(cache["tensors"], key=lambda item: float(item["residual_norm_sq_total"]), reverse=True)
    names: list[str] = []
    for item in tensors:
        name = str(item["name"])
        if name == "model.embed_tokens":
            continue
        names.append(name)
        if len(names) >= top_tensors:
            break
    return names


def build_prompts(prompt_indices: list[int]) -> list[dict[str, Any]]:
    prompt_manifest, reasons = __import__(
        "experimental.outlier_migrate.phase9.run_om_paroquant_baseline",
        fromlist=["build_prompt_manifest"],
    ).build_prompt_manifest(checker.DEFAULT_PROMPT_FILE)
    if reasons:
        raise RuntimeError(f"canonical prompt manifest failed validation: {reasons}")
    prompts = [row for row in prompt_manifest["prompts"] if int(row["index"]) in set(prompt_indices)]
    if [int(row["index"]) for row in prompts] != prompt_indices:
        raise RuntimeError("selected prompt indices are not present in canonical order")
    return prompts


def collect_activation_ema(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    module_names: list[str],
    max_new_tokens: int,
    batch_size: int,
    use_float16_autocast: bool,
    run_events_path: Path,
) -> dict[str, Any]:
    import torch

    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise RuntimeError(f"requested module hooks missing: {missing}")

    autocast_enabled = bool(use_float16_autocast and torch.cuda.is_available())
    cache_dtype = torch.float16 if autocast_enabled else None
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model, cache_dtype=cache_dtype)
    previous_fast_paths = phase4_runner.set_autocast_sensitive_fast_paths(False) if autocast_enabled else {}
    score_start = checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1
    score_end = checker.SCORING_POSITION
    hook_enabled = {"position": None}
    accum: dict[str, dict[str, Any]] = {
        name: {"sum_sq": None, "count": 0, "samples": 0} for name in module_names
    }

    def make_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            position = hook_enabled["position"]
            if position is None or not (score_start <= int(position) <= score_end):
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim == 0:
                return
            x = x.detach().float()
            if x.shape[-1] <= 0:
                return
            reduce_dims = tuple(range(x.ndim - 1))
            values = (x * x).sum(dim=reduce_dims).double()
            item = accum[name]
            if item["sum_sq"] is None:
                item["sum_sq"] = torch.zeros_like(values, dtype=torch.float64, device="cpu")
            item["sum_sq"].add_(values.cpu())
            item["count"] += int(x.numel() // x.shape[-1])
            item["samples"] += 1

        return hook

    handles = [modules[name].register_forward_pre_hook(make_hook(name)) for name in module_names]
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                batch_indices = [int(item["index"]) for item in batch]
                texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
                encoded = tokenizer(texts, padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                target_tensor = torch.tensor(
                    [target_tokens[index][:max_new_tokens] for index in batch_indices],
                    dtype=torch.long,
                    device=device,
                )
                with torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                    cache_position = torch.arange(input_ids.shape[1], device=device)
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    hook_enabled["position"] = 0
                    outputs = model(**normalize_cache_inputs(model_inputs))
                    past_key_values = output_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError("model did not return cache while collecting activation EMA")
                    logits = outputs.logits[:, -1, :]
                    for decode_position in range(1, max_new_tokens + 1):
                        current_target = target_tensor[:, decode_position - 1]
                        if decode_position == max_new_tokens:
                            break
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                            ],
                            dim=1,
                        )
                        cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                        model_inputs = model.prepare_inputs_for_generation(
                            input_ids=current_target[:, None],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            use_cache=True,
                        )
                        hook_enabled["position"] = decode_position
                        outputs = model(**normalize_cache_inputs(model_inputs))
                        past_key_values = output_cache(outputs)
                        if past_key_values is None:
                            raise RuntimeError(f"model dropped cache at decode position {decode_position}")
                        logits = outputs.logits[:, -1, :]
                hook_enabled["position"] = None
                run_events_path.open("a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "completed_activation_batch",
                            "prompt_indices": batch_indices,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                del encoded, input_ids, attention_mask, target_tensor, cache_position, model_inputs, outputs, past_key_values, logits
                phase4_runner.release_model_memory()
    finally:
        hook_enabled["position"] = None
        for handle in handles:
            handle.remove()
        phase4_runner.restore_autocast_sensitive_fast_paths(previous_fast_paths)

    modules_out: dict[str, Any] = {}
    for name, item in accum.items():
        sum_sq = item["sum_sq"]
        count = int(item["count"])
        if sum_sq is None or count == 0:
            modules_out[name] = {"count": count, "samples": int(item["samples"]), "ema_sq": []}
            continue
        ema = (sum_sq / float(count)).tolist()
        modules_out[name] = {
            "count": count,
            "samples": int(item["samples"]),
            "ema_sq": [float(value) for value in ema],
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_activation_ema",
        "created_at_utc": shared.utc_now(),
        "score_start": score_start,
        "score_end": score_end,
        "modules": modules_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"activation_ema_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN_DIR)
    parser.add_argument("--residual-cache", type=Path, default=DEFAULT_RESIDUAL_CACHE)
    parser.add_argument("--prompt-indices", type=parse_indices, default=[4])
    parser.add_argument("--top-tensors", type=int, default=8)
    parser.add_argument("--model-id", default=checker.MODEL_ID)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")
    module_names = select_module_names(args.residual_cache, args.top_tensors)
    prompts = build_prompts(args.prompt_indices)
    target_tokens_all = phase4_runner.load_trace_tokens(args.base_run_dir / "bf16_traces.jsonl.gz")
    target_tokens = {index: target_tokens_all[index] for index in args.prompt_indices}
    model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
    if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
        raise SystemExit("Granite model snapshot missing or mismatch")
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
    activation_ema = collect_activation_ema(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        target_tokens=target_tokens,
        module_names=module_names,
        max_new_tokens=checker.SCORING_POSITION,
        batch_size=args.batch_size,
        use_float16_autocast=True,
        run_events_path=run_events_path,
    )
    shared.write_json(
        run_dir / "config.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_config",
            "model_id": args.model_id,
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
            "base_run_dir": str(args.base_run_dir),
            "residual_cache": str(args.residual_cache),
            "prompt_indices": args.prompt_indices,
            "module_names": module_names,
            "top_tensors": args.top_tensors,
            "dtype": args.dtype,
        },
    )
    shared.write_json(run_dir / "model_provenance.json", model_provenance)
    shared.write_json(run_dir / "activation_ema.json", activation_ema)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    print(json.dumps({"run_dir": str(run_dir), "module_count": len(module_names), "prompt_indices": args.prompt_indices}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
