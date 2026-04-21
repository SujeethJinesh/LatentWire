#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import torch
from transformers import QuantizedCache, pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
KVPRESS_ROOT = ROOT / "references" / "repos" / "kvpress"
if str(KVPRESS_ROOT) not in sys.path:
    sys.path.insert(0, str(KVPRESS_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kvpress  # noqa: F401  # registers the custom pipeline
from kvpress import ExpectedAttentionPress
from kvpress.presses import base_press as base_press_mod
from kvpress import pipeline as kvpress_pipeline_mod
from kvpress import utils as kvpress_utils

from latent_bridge.evaluate import (
    _generation_match,
    _generation_metrics,
    _generation_example_id,
    default_device,
    load_generation,
)


def _patch_kvpress_compat() -> None:
    """Make the vendored KVPress work with the current transformers cache API."""

    def extract_keys_and_values(cache, layer_idx: int):
        if isinstance(cache, QuantizedCache):
            if hasattr(cache, "layers"):
                layer = cache.layers[layer_idx]
                return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        if hasattr(cache, "layers"):
            return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    def forward_hook(self, module, input, kwargs: dict, output):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if cache is None:
            raise KeyError("Expected `past_key_values` or `past_key_value` in attention hook kwargs")
        q_len = hidden_states.shape[1]
        if kwargs["cache_position"][-1] > q_len:
            return output
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
        if isinstance(cache, QuantizedCache) and hasattr(cache, "layers"):
            layer = cache.layers[module.layer_idx]
            layer._quantized_keys = layer._quantize(keys, axis=layer.axis_key)
            layer._quantized_values = layer._quantize(values, axis=layer.axis_value)
            layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            layer.cumulative_length = keys.shape[2]
        elif hasattr(cache, "layers"):
            layer = cache.layers[module.layer_idx]
            layer.keys = keys
            layer.values = values
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values
        return output

    def remove_answer_from_cache(self, cache, cache_seq_lengths: list[int]):
        if hasattr(cache, "layers"):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.layers[layer_idx].keys = cache.layers[layer_idx].keys[:, :, :sequence_length]
                cache.layers[layer_idx].values = cache.layers[layer_idx].values[:, :, :sequence_length]
        else:
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :sequence_length]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :sequence_length]

        if isinstance(cache, QuantizedCache) and hasattr(cache, "layers"):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.layers[layer_idx]._quantized_keys = cache.layers[layer_idx]._quantized_keys[:, :, :sequence_length]
                cache.layers[layer_idx]._quantized_values = cache.layers[layer_idx]._quantized_values[
                    :, :, :sequence_length
                ]

    kvpress_utils.extract_keys_and_values = extract_keys_and_values
    base_press_mod.extract_keys_and_values = extract_keys_and_values
    base_press_mod.BasePress.forward_hook = forward_hook
    kvpress_pipeline_mod.KVPressTextGenerationPipeline._remove_answer_from_cache = remove_answer_from_cache


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vendored KVPress on our JSONL generation slices.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--eval-file", required=True)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--press", default="expected_attention", choices=["none", "expected_attention"])
    p.add_argument("--compression-ratio", type=float, default=0.5)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prediction-output")
    return p.parse_args()


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def main() -> None:
    args = _parse_args()
    _patch_kvpress_compat()
    examples = load_generation(args.eval_file)
    if args.limit is not None:
        examples = examples[: max(int(args.limit), 0)]
    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model,
        device=args.device,
        torch_dtype=_torch_dtype(args.dtype),
    )
    press = None
    if args.press == "expected_attention":
        press = ExpectedAttentionPress(compression_ratio=float(args.compression_ratio))

    records: list[dict[str, Any]] = []
    correct = 0
    total_generated_tokens = 0
    total_ttft_sec = 0.0
    total_elapsed_sec = 0.0

    for example in examples:
        t0 = time.perf_counter()
        out = pipe(
            example.prompt,
            question="",
            press=press,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=bool(args.enable_thinking),
        )
        elapsed = time.perf_counter() - t0
        prediction = out["answer"]
        is_correct = _generation_match(prediction, example.answers)
        correct += int(is_correct)
        total_elapsed_sec += elapsed
        total_generated_tokens += max(0, len(prediction.split()))
        record = {
            "id": _generation_example_id(example),
            "prompt": example.prompt,
            "answers": example.answers,
            "prediction": prediction,
            "correct": bool(is_correct),
            "press": args.press,
            "compression_ratio": None if press is None else float(args.compression_ratio),
        }
        records.append(record)

    metrics = _generation_metrics(
        correct=correct,
        num_examples=len(examples),
        total_generated_tokens=total_generated_tokens,
        total_ttft_sec=total_ttft_sec,
        total_elapsed_sec=total_elapsed_sec,
    )
    summary = {
        "model": args.model,
        "eval_file": args.eval_file,
        "press": args.press,
        "compression_ratio": None if press is None else float(args.compression_ratio),
        "enable_thinking": bool(args.enable_thinking),
        "limit": args.limit,
        **metrics,
    }
    print(json.dumps(summary, indent=2))

    if args.prediction_output:
        out_path = pathlib.Path(args.prediction_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
