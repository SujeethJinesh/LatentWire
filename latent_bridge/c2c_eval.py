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


def _dynamic_cache_layer_pairs(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return populated key/value layer tensors across old and new cache APIs."""
    if cache is None:
        return []
    if hasattr(cache, "layers"):
        pairs = []
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is not None and value is not None:
                pairs.append((key, value))
        return pairs
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return [
            (key, value)
            for key, value in zip(cache.key_cache, cache.value_cache)
            if key is not None and value is not None
        ]
    return []


def install_c2c_dynamic_cache_compatibility_shim(wrapper_module: Any | None = None) -> None:
    """Adapt vendored C2C's old DynamicCache usage to current Transformers caches.

    The published C2C wrapper was written against a Transformers DynamicCache
    API with mutable ``key_cache``/``value_cache`` lists. Current Transformers
    stores tensors in ``cache.layers`` instead. We install a narrow shim here so
    LatentWire can run the vendored wrapper without editing the reference clone.
    """
    from transformers.cache_utils import DynamicCache

    if not getattr(DynamicCache, "_latentwire_old_cache_api_shim", False):

        def _key_cache(self):
            return [key for key, _ in _dynamic_cache_layer_pairs(self)]

        def _value_cache(self):
            return [value for _, value in _dynamic_cache_layer_pairs(self)]

        def _getitem(self, layer_idx: int):
            return _dynamic_cache_layer_pairs(self)[layer_idx]

        DynamicCache.key_cache = property(_key_cache)  # type: ignore[attr-defined]
        DynamicCache.value_cache = property(_value_cache)  # type: ignore[attr-defined]
        DynamicCache.__getitem__ = _getitem  # type: ignore[method-assign]
        DynamicCache._latentwire_old_cache_api_shim = True  # type: ignore[attr-defined]

    if wrapper_module is None:
        return

    if getattr(wrapper_module, "_latentwire_dynamic_cache_shim", False):
        return

    def clone_kv_cache(kv_cache: Any):
        if kv_cache is None:
            return None
        pairs = [
            (key.clone().detach(), value.clone().detach())
            for key, value in _dynamic_cache_layer_pairs(kv_cache)
        ]
        return DynamicCache(pairs)

    def hybrid_to_dynamic(hybrid_cache: Any):
        if hybrid_cache is None:
            return None
        if isinstance(hybrid_cache, DynamicCache):
            return hybrid_cache
        pairs = [
            (key.clone().detach(), value.clone().detach())
            for key, value in _dynamic_cache_layer_pairs(hybrid_cache)
        ]
        if pairs:
            return DynamicCache(pairs)
        raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")

    wrapper_module.clone_kv_cache = clone_kv_cache
    wrapper_module.hybrid_to_dynamic = hybrid_to_dynamic
    wrapper_module._latentwire_dynamic_cache_shim = True


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
    from rosetta.model import wrapper as rosetta_wrapper  # type: ignore

    install_c2c_dynamic_cache_compatibility_shim(rosetta_wrapper)

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


def _tail_local_tokens(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_f = source.detach().float()
    target_f = target.detach().float()
    output_f = output.detach().float()
    delta_f = output_f - target_f
    tensors = {
        "source": source_f,
        "target": target_f,
        "output": output_f,
        "delta": delta_f,
    }
    return {
        name: tensor[:, :, -1, :].reshape(-1).detach().cpu()
        for name, tensor in tensors.items()
    }


def _projector_scalar_trace(projector) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, tensor in (
        ("key_scalar", getattr(projector, "last_norm_key_scalar", None)),
        ("value_scalar", getattr(projector, "last_norm_value_scalar", None)),
    ):
        if tensor is None:
            values = torch.zeros((1,), dtype=torch.float32)
            shape: list[int] = []
        else:
            values = tensor.detach().float().reshape(-1).cpu()
            shape = [int(dim) for dim in tensor.shape]
        std = values.std(unbiased=False) if values.numel() > 1 else torch.tensor(0.0)
        out[name] = {
            "mean": float(values.mean().item()),
            "std": float(std.item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "shape": shape,
        }
    key_gate_logit = float(getattr(projector, "last_key_gate_logit", 0.0))
    value_gate_logit = float(getattr(projector, "last_value_gate_logit", 0.0))
    out.update(
        {
            "key_gate_logit": key_gate_logit,
            "value_gate_logit": value_gate_logit,
            "key_gate_active": bool(key_gate_logit > 0.0),
            "value_gate_active": bool(value_gate_logit > 0.0),
        }
    )
    return out


def reset_c2c_projector_trace_history(model, *, enabled: bool = True) -> None:
    for projector in getattr(model, "projector_list", []):
        projector.latentwire_trace_history = []
        projector._latentwire_record_trace_history = bool(enabled)


def stop_c2c_projector_trace_history(model) -> None:
    for projector in getattr(model, "projector_list", []):
        projector._latentwire_record_trace_history = False


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
                _projector.last_latentwire_local_tokens = {
                    "key": _tail_local_tokens(
                        source=source_key,
                        target=target_key,
                        output=output_key,
                    ),
                    "value": _tail_local_tokens(
                        source=source_value,
                        target=target_value,
                        output=output_value,
                    ),
                }
                if getattr(_projector, "_latentwire_record_trace_history", False):
                    history = getattr(_projector, "latentwire_trace_history", [])
                    entry = {
                        "call_index": len(history),
                        "source_seq_len": int(source_key.shape[-2]),
                        "target_seq_len": int(target_key.shape[-2]),
                        **_projector_scalar_trace(_projector),
                        "key_residual": _projector.last_latentwire_trace["key_residual"],
                        "value_residual": _projector.last_latentwire_trace["value_residual"],
                    }
                    history.append(entry)
                    _projector.latentwire_trace_history = history
            except Exception:
                _projector.last_latentwire_trace = {}
                _projector.last_latentwire_local_tokens = {}
            return output_key, output_value

        projector._latentwire_original_forward = original_forward
        projector.forward = traced_forward
        projector._latentwire_trace_wrapped = True
        projector._latentwire_trace_projection_dim = int(residual_projection_dim)


def _flatten_numeric_trace(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        out[prefix] = float(value)
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if isinstance(item, (int, float, bool)):
                out[f"{prefix}_{idx:03d}"] = float(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "shape":
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric_trace(next_prefix, item, out)


def _history_aggregates(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(std.item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "first": float(tensor[0].item()),
        "last": float(tensor[-1].item()),
        "delta": float((tensor[-1] - tensor[0]).item()),
    }


def summarize_c2c_projector_generation_history(model) -> tuple[torch.Tensor, dict[str, Any]]:
    features: list[float] = []
    feature_names: list[str] = []
    projector_metadata: list[dict[str, Any]] = []
    aggregate_names = ("mean", "std", "min", "max", "first", "last", "delta")
    for idx, projector in enumerate(getattr(model, "projector_list", [])):
        history = list(getattr(projector, "latentwire_trace_history", []) or [])
        projector_metadata.append(
            {
                "projector_index": int(idx),
                "history_length": int(len(history)),
            }
        )
        flattened_rows: list[dict[str, float]] = []
        keys: set[str] = set()
        for entry in history:
            flat: dict[str, float] = {}
            _flatten_numeric_trace("", entry, flat)
            flattened_rows.append(flat)
            keys.update(flat)
        for key in sorted(keys):
            values = [float(row.get(key, 0.0)) for row in flattened_rows]
            aggregates = _history_aggregates(values)
            for aggregate_name in aggregate_names:
                feature_names.append(f"projector_{idx:02d}.history.{key}.{aggregate_name}")
                features.append(float(aggregates[aggregate_name]))
    if not features:
        raise ValueError("No C2C generation trace history was recorded")
    tensor = torch.tensor(features, dtype=torch.float32)
    metadata = {
        "feature_family": "c2c_generation_projector_trace_history",
        "feature_dim": int(tensor.numel()),
        "feature_names": feature_names,
        "aggregate_names": list(aggregate_names),
        "projectors": projector_metadata,
    }
    return tensor, metadata


def _logit_step_stats(logits: torch.Tensor, generated_token: int | None) -> dict[str, float]:
    values = logits.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "top1_logit": 0.0,
            "top2_logit": 0.0,
            "top1_prob": 0.0,
            "top2_prob": 0.0,
            "top_margin": 0.0,
            "entropy": 0.0,
            "generated_logit": 0.0,
            "generated_rank_frac": 1.0,
        }
    std = values.std(unbiased=False) if values.numel() > 1 else torch.tensor(0.0)
    topk = torch.topk(values, k=min(2, int(values.numel())))
    top1_logit = float(topk.values[0].item())
    top2_logit = float(topk.values[1].item()) if topk.values.numel() > 1 else top1_logit
    probs = torch.softmax(values, dim=0)
    top1_prob = float(probs[int(topk.indices[0].item())].item())
    top2_prob = float(probs[int(topk.indices[1].item())].item()) if topk.indices.numel() > 1 else top1_prob
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-30))).sum()).item())
    generated_logit = 0.0
    generated_rank_frac = 1.0
    if generated_token is not None and 0 <= int(generated_token) < values.numel():
        token = int(generated_token)
        generated_logit = float(values[token].item())
        rank = int(torch.sum(values > values[token]).item())
        generated_rank_frac = float(rank / max(int(values.numel()) - 1, 1))
    return {
        "mean": float(values.mean().item()),
        "std": float(std.item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "top1_logit": top1_logit,
        "top2_logit": top2_logit,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "top_margin": float(top1_logit - top2_logit),
        "entropy": entropy,
        "generated_logit": generated_logit,
        "generated_rank_frac": generated_rank_frac,
    }


def summarize_c2c_generation_score_history(
    scores: list[torch.Tensor],
    generated_tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    step_rows: list[dict[str, float]] = []
    for step_idx, logits in enumerate(scores):
        token = None
        if generated_tokens.numel() > step_idx:
            token = int(generated_tokens.reshape(-1)[step_idx].item())
        step_rows.append(_logit_step_stats(logits[0], token))
    feature_names: list[str] = []
    features: list[float] = []
    aggregate_names = ("mean", "std", "min", "max", "first", "last", "delta")
    for key in sorted(step_rows[0]) if step_rows else []:
        aggregates = _history_aggregates([float(row[key]) for row in step_rows])
        for aggregate_name in aggregate_names:
            feature_names.append(f"generation_logits.{key}.{aggregate_name}")
            features.append(float(aggregates[aggregate_name]))
    tensor = torch.tensor(features, dtype=torch.float32)
    metadata = {
        "feature_family": "c2c_generation_target_logit_history",
        "feature_dim": int(tensor.numel()),
        "feature_names": feature_names,
        "step_count": int(len(step_rows)),
        "aggregate_names": list(aggregate_names),
    }
    return tensor, metadata


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


def summarize_c2c_projector_local_tokens(model) -> tuple[torch.Tensor, dict[str, Any]]:
    raw_tokens: list[torch.Tensor] = []
    token_names: list[str] = []
    raw_hidden_dims: list[int] = []
    for idx, projector in enumerate(getattr(model, "projector_list", [])):
        local_trace = getattr(projector, "last_latentwire_local_tokens", {}) or {}
        for stream_name in ("key", "value"):
            stream_tokens = local_trace.get(stream_name, {}) or {}
            for tensor_name in ("source", "target", "output", "delta"):
                token = stream_tokens.get(tensor_name)
                if token is None:
                    continue
                token = token.detach().float().reshape(-1).cpu()
                raw_tokens.append(token)
                raw_hidden_dims.append(int(token.numel()))
                token_names.append(f"projector_{idx:02d}.{stream_name}.{tensor_name}.tail")
    if not raw_tokens:
        raise ValueError("No C2C local token traces were recorded")
    hidden_dim = max(raw_hidden_dims)
    tokens = [
        torch.nn.functional.pad(token, (0, int(hidden_dim) - int(token.numel())))
        if int(token.numel()) < int(hidden_dim)
        else token
        for token in raw_tokens
    ]
    stacked = torch.stack(tokens, dim=0).contiguous()
    metadata = {
        "feature_family": "c2c_prefill_token_layer_tail_residual",
        "feature_token_shape": [int(stacked.shape[0]), int(stacked.shape[1])],
        "token_names": token_names,
        "raw_hidden_dims": raw_hidden_dims,
        "padded_hidden_dim": int(hidden_dim),
        "token_count": int(stacked.shape[0]),
        "hidden_dim": int(stacked.shape[1]),
    }
    return stacked.reshape(-1), metadata


@torch.no_grad()
def extract_c2c_prefill_trace_features(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    residual_projection_dim: int = 0,
    feature_family: str = "summary_trace",
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
    if feature_family == "token_layer_tail_residual":
        features, token_metadata = summarize_c2c_projector_local_tokens(model)
        metadata = {
            "formatted_prompt_tokens": prompt_len,
            "feature_dim": int(features.numel()),
            "residual_projection_dim": int(residual_projection_dim),
            **token_metadata,
        }
        return features, metadata
    if feature_family != "summary_trace":
        raise ValueError(f"Unsupported C2C feature family: {feature_family!r}")
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
def extract_c2c_generation_trace_features(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
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
    reset_c2c_projector_trace_history(model, enabled=True)
    outputs = model.generate(
        **inputs,
        kv_cache_index=build_c2c_kv_cache_index(prompt_len, device=device),
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        return_dict_in_generate=True,
        output_scores=True,
    )
    stop_c2c_projector_trace_history(model)
    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs["sequences"]
    score_list = list(outputs.scores or []) if hasattr(outputs, "scores") else list(outputs.get("scores") or [])
    generated = sequences[0, prompt_len:]
    projector_features, projector_metadata = summarize_c2c_projector_generation_history(model)
    score_features, score_metadata = summarize_c2c_generation_score_history(score_list, generated)
    features = torch.cat([projector_features, score_features], dim=0)
    metadata = {
        "formatted_prompt_tokens": prompt_len,
        "generated_tokens": int(generated.shape[0]),
        "decoded_prediction": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "residual_projection_dim": int(residual_projection_dim),
        "feature_family": "c2c_generation_projector_and_logit_trace_history",
        "feature_dim": int(features.numel()),
        "components": {
            "projector": projector_metadata,
            "target_logits": score_metadata,
        },
        "feature_names": [
            *(f"projector::{name}" for name in projector_metadata["feature_names"]),
            *(f"logits::{name}" for name in score_metadata["feature_names"]),
        ],
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
