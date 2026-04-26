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


def build_c2c_kv_cache_index(prompt_len: int, *, device: str | torch.device) -> list[torch.Tensor]:
    instruction_index = (
        torch.tensor([1, 0], dtype=torch.long)
        .repeat(max(int(prompt_len) - 1, 1), 1)
        .unsqueeze(0)
        .to(device)
    )
    label_index = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0).to(device)
    return [instruction_index, label_index]


def c2c_trace_feature_names(projector_count: int, *, residual_projection_dim: int = 0) -> list[str]:
    names: list[str] = []
    for idx in range(int(projector_count)):
        prefix = f"projector_{idx:02d}"
        for tensor_name in ("key_scalar", "value_scalar"):
            for stat in ("mean", "std", "min", "max"):
                names.append(f"{prefix}.{tensor_name}.{stat}")
        names.extend(
            [
                f"{prefix}.key_gate_logit",
                f"{prefix}.value_gate_logit",
                f"{prefix}.key_gate_active",
                f"{prefix}.value_gate_active",
            ]
        )
        for tensor_name in ("key_residual", "value_residual"):
            for stat in (
                "source_l2",
                "target_l2",
                "delta_l2",
                "delta_to_target_ratio",
                "delta_mean_abs",
                "delta_max_abs",
                "output_mean_abs",
                "delta_target_cosine",
                "tail_delta_l2",
                "tail_delta_to_target_ratio",
            ):
                names.append(f"{prefix}.{tensor_name}.{stat}")
            for proj_idx in range(int(residual_projection_dim)):
                names.append(f"{prefix}.{tensor_name}.delta_projection_{proj_idx:03d}")
            for proj_idx in range(int(residual_projection_dim)):
                names.append(f"{prefix}.{tensor_name}.tail_delta_projection_{proj_idx:03d}")
    return names


def _signed_bucket_projection(values: torch.Tensor, dim: int, *, salt: int) -> torch.Tensor:
    flat = values.detach().float().reshape(-1)
    if dim <= 0:
        return torch.empty((0,), dtype=torch.float32, device=flat.device)
    if flat.numel() == 0:
        return torch.zeros((dim,), dtype=torch.float32, device=flat.device)
    idx = torch.arange(flat.numel(), device=flat.device, dtype=torch.long)
    buckets = (idx * 1103515245 + int(salt) * 12345) % int(dim)
    signs = torch.where(
        ((idx * 214013 + int(salt) * 2531011) & 1) == 0,
        torch.ones_like(flat),
        -torch.ones_like(flat),
    )
    out = torch.zeros((dim,), dtype=torch.float32, device=flat.device)
    out.scatter_add_(0, buckets, flat * signs)
    return out / float(flat.numel() ** 0.5)


def _trace_tensor_stats(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    projection_dim: int = 0,
    projection_salt: int = 0,
) -> dict[str, Any]:
    source_f = source.detach().float()
    target_f = target.detach().float()
    output_f = output.detach().float()
    delta_f = output_f - target_f
    source_l2 = float(torch.linalg.vector_norm(source_f).item())
    target_l2 = float(torch.linalg.vector_norm(target_f).item())
    delta_l2 = float(torch.linalg.vector_norm(delta_f).item())
    output_mean_abs = float(output_f.abs().mean().item())
    delta_mean_abs = float(delta_f.abs().mean().item())
    delta_max_abs = float(delta_f.abs().max().item()) if delta_f.numel() else 0.0
    denom = max(target_l2, 1e-12)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            delta_f.reshape(1, -1),
            target_f.reshape(1, -1),
            dim=1,
        )
        .cpu()
        .item()
    )
    tail_delta = delta_f[:, :, -1:, :]
    tail_target = target_f[:, :, -1:, :]
    tail_delta_l2 = float(torch.linalg.vector_norm(tail_delta).item())
    tail_target_l2 = float(torch.linalg.vector_norm(tail_target).item())
    stats: dict[str, Any] = {
        "source_l2": source_l2,
        "target_l2": target_l2,
        "delta_l2": delta_l2,
        "delta_to_target_ratio": float(delta_l2 / denom),
        "delta_mean_abs": delta_mean_abs,
        "delta_max_abs": delta_max_abs,
        "output_mean_abs": output_mean_abs,
        "delta_target_cosine": cosine,
        "tail_delta_l2": tail_delta_l2,
        "tail_delta_to_target_ratio": float(tail_delta_l2 / max(tail_target_l2, 1e-12)),
    }
    if projection_dim > 0:
        stats["delta_projection"] = [
            float(value)
            for value in _signed_bucket_projection(
                delta_f,
                int(projection_dim),
                salt=int(projection_salt),
            )
            .detach()
            .cpu()
            .tolist()
        ]
        stats["tail_delta_projection"] = [
            float(value)
            for value in _signed_bucket_projection(
                tail_delta,
                int(projection_dim),
                salt=int(projection_salt) + 1009,
            )
            .detach()
            .cpu()
            .tolist()
        ]
    return stats


def install_c2c_projector_trace_hooks(model, *, residual_projection_dim: int = 0) -> None:
    for projector_idx, projector in enumerate(getattr(model, "projector_list", [])):
        if (
            getattr(projector, "_latentwire_trace_wrapped", False)
            and getattr(projector, "_latentwire_trace_projection_dim", 0) == int(residual_projection_dim)
        ):
            continue
        original_forward = getattr(projector, "_latentwire_original_forward", projector.forward)

        def traced_forward(
            source_kv,
            target_kv,
            *args,
            _original=original_forward,
            _projector=projector,
            _projector_idx=projector_idx,
            **kwargs,
        ):
            output_key, output_value = _original(source_kv, target_kv, *args, **kwargs)
            try:
                source_key, source_value = source_kv
                target_key, target_value = target_kv
                _projector.last_latentwire_trace = {
                    "key_residual": _trace_tensor_stats(
                        source=source_key,
                        target=target_key,
                        output=output_key,
                        projection_dim=int(residual_projection_dim),
                        projection_salt=int(_projector_idx) * 2,
                    ),
                    "value_residual": _trace_tensor_stats(
                        source=source_value,
                        target=target_value,
                        output=output_value,
                        projection_dim=int(residual_projection_dim),
                        projection_salt=int(_projector_idx) * 2 + 1,
                    ),
                }
            except Exception:
                _projector.last_latentwire_trace = {}
            return output_key, output_value

        projector._latentwire_original_forward = original_forward
        projector.forward = traced_forward
        projector._latentwire_trace_wrapped = True
        projector._latentwire_trace_projection_dim = int(residual_projection_dim)


def summarize_c2c_projector_trace(
    model,
    *,
    residual_projection_dim: int = 0,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    features: list[float] = []
    metadata: list[dict[str, Any]] = []
    for idx, projector in enumerate(getattr(model, "projector_list", [])):
        key_scalar = getattr(projector, "last_norm_key_scalar", None)
        value_scalar = getattr(projector, "last_norm_value_scalar", None)
        key_gate_logit = float(getattr(projector, "last_key_gate_logit", 0.0))
        value_gate_logit = float(getattr(projector, "last_value_gate_logit", 0.0))
        projector_meta: dict[str, Any] = {
            "projector_index": int(idx),
            "has_trace": key_scalar is not None and value_scalar is not None,
            "key_gate_logit": key_gate_logit,
            "value_gate_logit": value_gate_logit,
            "key_gate_active": bool(key_gate_logit > 0.0),
            "value_gate_active": bool(value_gate_logit > 0.0),
        }
        for name, tensor in (("key_scalar", key_scalar), ("value_scalar", value_scalar)):
            if tensor is None:
                values = torch.zeros((1,), dtype=torch.float32)
                shape: list[int] = []
            else:
                values = tensor.detach().float().reshape(-1).cpu()
                shape = [int(dim) for dim in tensor.shape]
            std = values.std(unbiased=False) if values.numel() > 1 else torch.tensor(0.0)
            stats = {
                "mean": float(values.mean().item()),
                "std": float(std.item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "shape": shape,
            }
            features.extend([stats["mean"], stats["std"], stats["min"], stats["max"]])
            projector_meta[name] = stats
        features.extend(
            [
                key_gate_logit,
                value_gate_logit,
                float(key_gate_logit > 0.0),
                float(value_gate_logit > 0.0),
            ]
        )
        residual_trace = getattr(projector, "last_latentwire_trace", {}) or {}
        for name in ("key_residual", "value_residual"):
            stats = dict(residual_trace.get(name, {}))
            for stat in (
                "source_l2",
                "target_l2",
                "delta_l2",
                "delta_to_target_ratio",
                "delta_mean_abs",
                "delta_max_abs",
                "output_mean_abs",
                "delta_target_cosine",
                "tail_delta_l2",
                "tail_delta_to_target_ratio",
            ):
                features.append(float(stats.get(stat, 0.0)))
            delta_projection = list(stats.get("delta_projection", []))[: int(residual_projection_dim)]
            delta_projection.extend([0.0] * max(int(residual_projection_dim) - len(delta_projection), 0))
            tail_delta_projection = list(stats.get("tail_delta_projection", []))[
                : int(residual_projection_dim)
            ]
            tail_delta_projection.extend(
                [0.0] * max(int(residual_projection_dim) - len(tail_delta_projection), 0)
            )
            for value in delta_projection:
                features.append(float(value))
            for value in tail_delta_projection:
                features.append(float(value))
            projector_meta[name] = stats
        metadata.append(projector_meta)
    return torch.tensor(features, dtype=torch.float32), metadata


@torch.no_grad()
def extract_c2c_prefill_trace_features(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    residual_projection_dim: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    messages = build_c2c_messages(prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    install_c2c_projector_trace_hooks(model, residual_projection_dim=int(residual_projection_dim))
    model(
        **inputs,
        kv_cache_index=build_c2c_kv_cache_index(prompt_len, device=device),
        use_cache=True,
    )
    features, projector_metadata = summarize_c2c_projector_trace(
        model,
        residual_projection_dim=int(residual_projection_dim),
    )
    metadata = {
        "formatted_prompt_tokens": prompt_len,
        "feature_dim": int(features.numel()),
        "feature_family": "c2c_prefill_projector_residual_trace",
        "feature_names": c2c_trace_feature_names(
            len(projector_metadata),
            residual_projection_dim=int(residual_projection_dim),
        ),
        "residual_projection_dim": int(residual_projection_dim),
        "projectors": projector_metadata,
    }
    return features, metadata


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

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        kv_cache_index=build_c2c_kv_cache_index(prompt_len, device=device),
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
