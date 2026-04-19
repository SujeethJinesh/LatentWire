from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

import torch

from latent_bridge.baselines import C2CAdapter
from latent_bridge.evaluate import (
    _generation_example_id,
    _generation_match,
    _generation_metrics,
    _normalize_generation_text,
    GenerationExample,
    load_generation,
    write_prediction_records,
    write_prediction_sidecar,
)


def load_c2c_model(
    *,
    source_model: str,
    target_model: str,
    device: str,
    max_new_tokens: int,
):
    repo_root = pathlib.Path("references/repos/C2C").resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"C2C repo clone not found at {repo_root}")
    sys.path.insert(0, str(repo_root))

    from rosetta.utils.evaluate import load_rosetta_model  # type: ignore

    artifact = C2CAdapter.prepare_published_artifact(
        source_model,
        target_model,
        download=True,
    )
    config_path = C2CAdapter.local_config_path(artifact)
    checkpoint_dir = C2CAdapter.local_checkpoint_dir(artifact)
    if config_path is None or checkpoint_dir is None:
        raise RuntimeError("Published C2C artifact did not resolve to local paths")

    bundle_config = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    model_config = {
        "model_name": "Rosetta",
        "rosetta_config": {
            "base_model": bundle_config["model"]["base_model"],
            "teacher_model": bundle_config["model"]["teacher_model"],
            "is_do_alignment": bundle_config["model"].get("is_do_alignment", False),
            "alignment_strategy": bundle_config["model"].get("alignment_strategy", "first"),
            "checkpoints_dir": checkpoint_dir,
        },
    }
    generation_config = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
    }
    model, tokenizer = load_rosetta_model(
        model_config,
        eval_config={},
        device=torch.device(device),
        generation_config=generation_config,
    )
    return model, tokenizer, artifact


def build_c2c_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


@torch.no_grad()
def generate_c2c_text(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
) -> tuple[str, int, float]:
    messages = build_c2c_messages(prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    instruction_index = (
        torch.tensor([1, 0], dtype=torch.long)
        .repeat(max(prompt_len - 1, 1), 1)
        .unsqueeze(0)
        .to(device)
    )
    label_index = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0).to(device)
    kv_cache_index = [instruction_index, label_index]

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        kv_cache_index=kv_cache_index,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.perf_counter() - start
    generated = outputs[0, prompt_len:]
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return content, int(generated.shape[0]), float(elapsed)


def run_c2c_generation_eval(
    *,
    source_model: str,
    target_model: str,
    eval_file: str,
    device: str,
    max_new_tokens: int,
    limit: int | None = None,
    prediction_output: str | None = None,
) -> dict[str, Any]:
    examples = load_generation(eval_file)
    if limit is not None:
        examples = examples[:limit]
    model, tokenizer, artifact = load_c2c_model(
        source_model=source_model,
        target_model=target_model,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    records: list[dict[str, Any]] = []
    correct = 0
    total_generated_tokens = 0
    total_elapsed_sec = 0.0
    total_ttft_sec = 0.0
    for idx, ex in enumerate(examples):
        prediction, generated_tokens, elapsed = generate_c2c_text(
            model,
            tokenizer,
            ex.prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        is_correct = _generation_match(prediction, ex.answers)
        correct += int(is_correct)
        total_generated_tokens += int(generated_tokens)
        total_elapsed_sec += float(elapsed)
        total_ttft_sec += float(elapsed)
        records.append(
            {
                "index": idx,
                "example_id": _generation_example_id(ex),
                "method": "c2c",
                "prediction": prediction,
                "answer": ex.answers,
                "correct": bool(is_correct),
                "generated_tokens": int(generated_tokens),
                "latency_sec": float(elapsed),
                "normalized_prediction": _normalize_generation_text(prediction),
            }
        )

    metrics = _generation_metrics(
        correct=correct,
        num_examples=len(examples),
        total_generated_tokens=total_generated_tokens,
        total_ttft_sec=total_ttft_sec,
        total_elapsed_sec=total_elapsed_sec,
    )
    results = {f"c2c_{k}": v for k, v in metrics.items()}
    run_config = {
        "baseline": "c2c",
        "source_model": source_model,
        "target_model": target_model,
        "eval_file": eval_file,
        "device": device,
        "max_new_tokens": int(max_new_tokens),
        "limit": limit,
        "published_repo_id": artifact.repo_id,
        "published_subdir": artifact.subdir,
        "published_config_path": artifact.config_path,
        "published_checkpoint_dir": artifact.checkpoint_dir,
        "local_root": artifact.local_root,
    }
    if prediction_output:
        write_prediction_records(prediction_output, records)
        write_prediction_sidecar(prediction_output, records, results, run_config)
    return {
        "records": records,
        "metrics": results,
        "run_config": run_config,
    }
