from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_bridge.evaluate import (
    _generation_example_id,
    _generation_match,
    _generation_metrics,
    _normalize_generation_text,
    _source_reasoning_prompt,
    load_generation,
    write_prediction_records,
    write_prediction_sidecar,
)


def _ensure_kvcomm_repo_on_path() -> pathlib.Path:
    repo_root = pathlib.Path("references/repos/KVComm").resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"KVComm repo clone not found at {repo_root}")
    _ensure_kvcomm_compat_patch(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _ensure_kvcomm_compat_patch(repo_root: pathlib.Path) -> None:
    model_attn_path = repo_root / "model_attn.py"
    models_path = repo_root / "models.py"
    marker_ok = (
        "Qwen3AttentionTracer" in model_attn_path.read_text(encoding="utf-8")
        and "_match_kv_shape" in models_path.read_text(encoding="utf-8")
    )
    if marker_ok:
        return
    patch_path = pathlib.Path(__file__).resolve().parent / "patches" / "kvcomm_qwen3_compat.patch"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing KVComm compatibility patch at {patch_path}")
    subprocess.run(
        ["git", "-C", str(repo_root), "apply", str(patch_path)],
        check=True,
    )


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _chat_input_ids(tokenizer, prompt: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)


def _build_prompts(example, source_reasoning_mode: str) -> tuple[str, str]:
    source_base = getattr(example, "source_question", "") or example.prompt
    source_prompt = _source_reasoning_prompt(source_base, source_reasoning_mode)
    target_prompt = example.prompt
    return source_prompt, target_prompt


def _resolve_selected_layers(layer_ranking: list[int], top_layers_fraction: float) -> list[int]:
    if not layer_ranking:
        return []
    topk = max(1, int(round(len(layer_ranking) * float(top_layers_fraction))))
    return list(layer_ranking[:topk])


def load_kvcomm_models(
    *,
    source_model: str,
    target_model: str,
    device: str,
    dtype: str,
):
    torch_dtype = _torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": {"": device},
        "torch_dtype": torch_dtype,
        "attn_implementation": "sdpa",
    }
    model_A = AutoModelForCausalLM.from_pretrained(source_model, **model_kwargs)
    model_B = AutoModelForCausalLM.from_pretrained(target_model, **model_kwargs)
    model_A.eval()
    model_B.eval()
    model_A.name = source_model
    model_B.name = target_model
    return model_A, model_B, tokenizer


@torch.no_grad()
def rank_layers_from_calibration(
    *,
    model_A,
    model_B,
    tokenizer,
    calibration_examples: list[Any],
    device: str,
    source_reasoning_mode: str,
) -> list[int]:
    _ensure_kvcomm_repo_on_path()
    from layer_importance import calc_layer_importance, get_layer_ranking  # type: ignore
    from models import CVCommunicator  # type: ignore

    if hasattr(model_A.config, "num_hidden_layers"):
        num_layers = int(model_A.config.num_hidden_layers)
    else:
        num_layers = int(model_A.config.text_config.num_hidden_layers)

    communicator = CVCommunicator(
        model_A,
        model_B,
        layer_from=0,
        layer_to=num_layers - 1,
        layers_list=list(range(num_layers)),
        apply_attn_tracer=True,
        shift_back=False,
    ).to(device)

    layer_importance_total: dict[int, list[float]] = defaultdict(list)
    for ex in calibration_examples:
        source_prompt, target_prompt = _build_prompts(ex, source_reasoning_mode)
        input_ids_A = _chat_input_ids(tokenizer, source_prompt, device)
        input_ids_B = _chat_input_ids(tokenizer, target_prompt, device)
        out_A = model_A(
            input_ids=input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
            use_cache=True,
            return_dict=True,
        )
        communicator(
            input_ids=input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=out_A.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        communicator.calc_attn_weights_from_qk()
        layer_importance_total = calc_layer_importance(
            communicator.B_attn_weights,
            model_A.name,
            layer_importance_total,
        )

    cfg = SimpleNamespace(alpha=1.0, mu=0.5, sigma=10.0)
    ranking = get_layer_ranking(layer_importance_total, cfg)
    return [int(layer) for layer in ranking.tolist()]


@torch.no_grad()
def evaluate_kvcomm_generation(
    *,
    model_A,
    model_B,
    tokenizer,
    examples: list[Any],
    device: str,
    max_new_tokens: int,
    selected_layers: list[int],
    source_reasoning_mode: str,
) -> dict[str, Any]:
    _ensure_kvcomm_repo_on_path()
    from models import CVCommunicator  # type: ignore

    communicator = CVCommunicator(
        model_A,
        model_B,
        layer_from=0,
        layer_to=max(selected_layers) if selected_layers else 0,
        layers_list=selected_layers,
        apply_attn_tracer=False,
        shift_back=False,
    ).to(device)

    records: list[dict[str, Any]] = []
    correct = 0
    total_generated_tokens = 0
    total_elapsed_sec = 0.0
    total_ttft_sec = 0.0

    for idx, ex in enumerate(examples):
        source_prompt, target_prompt = _build_prompts(ex, source_reasoning_mode)
        input_ids_A = _chat_input_ids(tokenizer, source_prompt, device)
        input_ids_B = _chat_input_ids(tokenizer, target_prompt, device)

        out_A = model_A(
            input_ids=input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
            use_cache=True,
            return_dict=True,
        )
        start = time.perf_counter()
        output = communicator.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=out_A.past_key_values,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        elapsed = time.perf_counter() - start
        context_length = int(input_ids_B.shape[-1])
        prediction = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()
        is_correct = _generation_match(prediction, ex.answers)

        correct += int(is_correct)
        generated_tokens = int(max(output.shape[-1] - context_length, 0))
        total_generated_tokens += generated_tokens
        total_elapsed_sec += float(elapsed)
        total_ttft_sec += float(elapsed)
        records.append(
            {
                "index": idx,
                "example_id": _generation_example_id(ex),
                "method": "kvcomm",
                "prediction": prediction,
                "answer": ex.answers,
                "correct": bool(is_correct),
                "generated_tokens": generated_tokens,
                "latency_sec": float(elapsed),
                "normalized_prediction": _normalize_generation_text(prediction),
                "selected_layers": list(selected_layers),
            }
        )

    metrics = _generation_metrics(
        correct=correct,
        num_examples=len(examples),
        total_generated_tokens=total_generated_tokens,
        total_ttft_sec=total_ttft_sec,
        total_elapsed_sec=total_elapsed_sec,
    )
    return {
        "records": records,
        "metrics": {f"kvcomm_{k}": v for k, v in metrics.items()},
    }


def run_kvcomm_generation_eval(
    *,
    source_model: str,
    target_model: str,
    calibration_file: str,
    eval_file: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    source_reasoning_mode: str,
    top_layers_grid: list[float],
    prediction_output: str | None = None,
) -> dict[str, Any]:
    _ensure_kvcomm_repo_on_path()
    calibration_examples = load_generation(calibration_file)
    eval_examples = load_generation(eval_file)
    model_A, model_B, tokenizer = load_kvcomm_models(
        source_model=source_model,
        target_model=target_model,
        device=device,
        dtype=dtype,
    )
    layer_ranking = rank_layers_from_calibration(
        model_A=model_A,
        model_B=model_B,
        tokenizer=tokenizer,
        calibration_examples=calibration_examples,
        device=device,
        source_reasoning_mode=source_reasoning_mode,
    )

    sweep: list[dict[str, Any]] = []
    best_fraction = top_layers_grid[0]
    best_metric = float("-inf")
    for fraction in top_layers_grid:
        selected_layers = _resolve_selected_layers(layer_ranking, fraction)
        result = evaluate_kvcomm_generation(
            model_A=model_A,
            model_B=model_B,
            tokenizer=tokenizer,
            examples=calibration_examples,
            device=device,
            max_new_tokens=max_new_tokens,
            selected_layers=selected_layers,
            source_reasoning_mode=source_reasoning_mode,
        )
        accuracy = float(result["metrics"]["kvcomm_accuracy"])
        sweep.append(
            {
                "top_layers_fraction": float(fraction),
                "selected_layer_count": len(selected_layers),
                "selected_layers": selected_layers,
                "accuracy": accuracy,
            }
        )
        if accuracy > best_metric:
            best_metric = accuracy
            best_fraction = fraction

    selected_layers = _resolve_selected_layers(layer_ranking, best_fraction)
    held_out = evaluate_kvcomm_generation(
        model_A=model_A,
        model_B=model_B,
        tokenizer=tokenizer,
        examples=eval_examples,
        device=device,
        max_new_tokens=max_new_tokens,
        selected_layers=selected_layers,
        source_reasoning_mode=source_reasoning_mode,
    )
    run_config = {
        "baseline": "kvcomm",
        "source_model": source_model,
        "target_model": target_model,
        "calibration_file": calibration_file,
        "eval_file": eval_file,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": int(max_new_tokens),
        "source_reasoning_mode": source_reasoning_mode,
        "top_layers_grid": [float(x) for x in top_layers_grid],
        "best_top_layers_fraction": float(best_fraction),
        "selected_layers": selected_layers,
        "layer_ranking": layer_ranking,
        "calibration_sweep": sweep,
    }
    if prediction_output:
        write_prediction_records(prediction_output, held_out["records"])
        write_prediction_sidecar(prediction_output, held_out["records"], held_out["metrics"], run_config)
    return {
        "records": held_out["records"],
        "metrics": held_out["metrics"],
        "run_config": run_config,
    }


def _parse_grid(raw: str) -> list[float]:
    return [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--top-layers-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--prediction-output", required=True)
    args = parser.parse_args()

    payload = run_kvcomm_generation_eval(
        source_model=args.source_model,
        target_model=args.target_model,
        calibration_file=args.calibration_file,
        eval_file=args.eval_file,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        source_reasoning_mode=args.source_reasoning_mode,
        top_layers_grid=_parse_grid(args.top_layers_grid),
        prediction_output=args.prediction_output,
    )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
