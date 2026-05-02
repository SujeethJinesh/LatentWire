from __future__ import annotations

"""ARC hidden/query common-basis source-private packet gate.

This gate asks whether an alternate source model's hidden/query state can be
converted into the same public Fourier/anchor receiver coordinates that made
the Qwen ARC packet work.  It trains only on validation disagreement rows and
evaluates once on frozen test disagreement rows from a source-family
falsification gate.
"""

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_fourier_anchor_syndrome_gate as fourier_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_seed_stability as seed_stability  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_hidden_query_common_basis_gate_20260502_tinyllama_disagreement"
)
DEFAULT_TRAIN_ANCHORS = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_SOURCE_FAMILY_GATE_DIR = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu"
)
DEFAULT_SOURCE_MODEL = "auto"
DEFAULT_SEEDS = (47, 53, 59, 61, 67)
DEFAULT_PCA_DIMS = (16, 32, 64, 96)
DEFAULT_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_VIEWS = ("hidden_residual", "query_residual", "hidden_query_residual")
STRICT_QWEN_DELTA = 0.02
STRICT_CACHED_DELTA = 0.02


@dataclasses.dataclass(frozen=True)
class HiddenQueryState:
    hidden: np.ndarray
    query: np.ndarray
    metadata: dict[str, Any]


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_optional_json(path: pathlib.Path | str) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "source"


def _source_cache_prefix(source_family: str) -> str:
    slug = _slug(source_family)
    if slug.startswith("tinyllama"):
        return "tinyllama"
    if slug.startswith("phi3"):
        return "phi3"
    if slug.startswith("qwen2_5_1_5b"):
        return "qwen25_15b"
    return slug


def _source_cache_contract(source_family_gate_dir: pathlib.Path, source_model: str) -> dict[str, Any]:
    """Resolve alternate-source cache/model metadata from a falsification gate."""

    source_family_gate_dir = _resolve(source_family_gate_dir)
    audit = _read_optional_json(source_family_gate_dir / "source_cache_audit.json")
    validation_cache = audit.get("alt_validation_cache") or audit.get("alternate_validation_cache")
    test_cache = audit.get("alt_test_cache") or audit.get("alternate_test_cache")
    if validation_cache is None:
        validation_cache = source_family_gate_dir / "tinyllama_validation" / "source_prediction_cache.jsonl"
    if test_cache is None:
        test_cache = source_family_gate_dir / "tinyllama_test" / "source_prediction_cache.jsonl"
    audit_source_model = str(audit.get("alternate_source_model") or "").strip()
    resolved_source_model = audit_source_model if str(source_model).lower() == "auto" and audit_source_model else str(source_model)
    if not resolved_source_model or resolved_source_model.lower() == "auto":
        raise ValueError(
            f"source model was 'auto' but {source_family_gate_dir / 'source_cache_audit.json'} "
            "does not declare alternate_source_model"
        )
    source_family = str(audit.get("alternate_source_family") or _infer_source_family_from_cache(validation_cache) or "source")
    return {
        "source_family": source_family,
        "source_cache_prefix": _source_cache_prefix(source_family),
        "source_model": resolved_source_model,
        "source_model_requested": str(source_model),
        "validation_source_cache": _resolve(validation_cache),
        "test_source_cache": _resolve(test_cache),
        "source_cache_audit": audit,
        "source_cache_audit_path": source_family_gate_dir / "source_cache_audit.json",
    }


def _infer_source_family_from_cache(path: pathlib.Path | str) -> str | None:
    resolved = _resolve(path)
    if not resolved.exists():
        return None
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row.get("source_family")
            if family:
                return str(family)
            return None
    return None


def _content_digest(rows: list[arc_gate.ArcRow]) -> str:
    payload = [
        {
            "row_id": row.row_id,
            "content_id": row.content_id,
            "question": row.question,
            "choices": list(row.choices),
        }
        for row in rows
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one float is required")
    return values


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one view is required")
    return values


def _model_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("could not locate transformer layers for query extraction")


def _attention_module(layer: Any) -> Any:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        attn = getattr(layer, "attention", None)
    if attn is None:
        raise ValueError("could not locate attention module for query extraction")
    return attn


def _query_projection(attn_mod: Any, hidden: Any, *, head_dim: int | None = None) -> Any:
    q_proj = getattr(attn_mod, "q_proj", None)
    if q_proj is not None:
        return q_proj(hidden)
    qkv_proj = getattr(attn_mod, "qkv_proj", None)
    if qkv_proj is not None:
        qkv = qkv_proj(hidden)
        num_heads = getattr(attn_mod, "num_heads", None) or getattr(attn_mod, "num_attention_heads", None)
        if head_dim is None:
            head_dim = getattr(attn_mod, "head_dim", None)
        if num_heads is None or head_dim is None:
            raise ValueError("qkv projection needs num_heads/head_dim metadata")
        return qkv[..., : int(num_heads) * int(head_dim)]
    raise ValueError("attention module must expose q_proj or qkv_proj")


def _hidden_cache_metadata(
    *,
    rows: list[arc_gate.ArcRow],
    model_path: str,
    dtype: str,
    max_length: int,
    prompt_mode: str,
    hidden_layer: int,
    query_layer: int,
) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "row_ids": [row.row_id for row in rows],
        "content_digest": _content_digest(rows),
        "model_path": model_path,
        "dtype": dtype,
        "max_length": max_length,
        "prompt_mode": prompt_mode,
        "hidden_layer": int(hidden_layer),
        "query_layer": int(query_layer),
    }


def _load_hidden_query_cache(
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    expected: dict[str, Any],
) -> HiddenQueryState | None:
    if not npz_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None
    with np.load(npz_path) as data:
        hidden = np.asarray(data["hidden"], dtype=np.float64)
        query = np.asarray(data["query"], dtype=np.float64)
    return HiddenQueryState(
        hidden=hidden,
        query=query,
        metadata=metadata | {"cache_hit": True, "cache_npz": _display_path(npz_path), "cache_meta": _display_path(meta_path)},
    )


def _write_hidden_query_cache(
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    state: HiddenQueryState,
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, hidden=state.hidden.astype(np.float32), query=state.query.astype(np.float32))
    meta_path.write_text(json.dumps(state.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_hidden_query_state(
    rows: list[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    prompt_mode: str,
    hidden_layer: int,
    query_layer: int,
    local_files_only: bool,
) -> HiddenQueryState:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    resolved_device = "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if (
        isinstance(getattr(config, "rope_scaling", None), dict)
        and config.rope_scaling.get("rope_type") == "default"
        and "type" not in config.rope_scaling
        and getattr(config, "rope_parameters", None) is None
    ):
        config.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=arc_gate._torch_dtype(dtype),
    ).to(resolved_device)
    model.eval()
    base_model = getattr(model, "model", None)
    layers = _model_layers(model)
    query_layer_index = int(query_layer)
    if query_layer_index < 0:
        query_layer_index = len(layers) + query_layer_index
    if query_layer_index < 0 or query_layer_index >= len(layers):
        raise ValueError(f"query layer {query_layer} is out of range for {len(layers)} layers")
    attn_mod = _attention_module(layers[query_layer_index])

    hidden_rows: list[list[np.ndarray]] = []
    query_rows: list[list[np.ndarray]] = []
    hidden_dim: int | None = None
    query_dim: int | None = None
    max_choices = max(len(row.choices) for row in rows)
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            prompt = arc_gate._lm_choice_prompt(row, prompt_mode=prompt_mode)
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            texts = [prompt + " " + choice for choice in row.choices]
            padding_mode: bool | str = True
            tokenizer_max_length = max_length
            if str(resolved_device).startswith("mps"):
                raw_lengths = tokenizer(texts, padding=False, truncation=True, max_length=max_length)["input_ids"]
                row_max_length = max(len(input_ids) for input_ids in raw_lengths)
                tokenizer_max_length = min(max_length, int(math.ceil(row_max_length / 32.0) * 32))
                padding_mode = "max_length"
            encoded = tokenizer(
                texts,
                padding=padding_mode,
                truncation=True,
                max_length=tokenizer_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            if base_model is not None:
                output = base_model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            else:
                output = model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            hidden_states = output.hidden_states
            hidden_tensor = hidden_states[hidden_layer]
            query_input = hidden_states[query_layer_index]
            query_tensor = _query_projection(attn_mod, query_input)
            attention = encoded["attention_mask"].bool()
            row_hidden: list[np.ndarray] = []
            row_query: list[np.ndarray] = []
            for choice_index in range(len(row.choices)):
                choice_mask = attention[choice_index].clone()
                choice_mask[: min(prompt_len, choice_mask.shape[0])] = False
                if not bool(choice_mask.any()):
                    choice_mask = attention[choice_index]
                hidden_feature = hidden_tensor[choice_index][choice_mask].mean(dim=0).detach().cpu().numpy().astype(np.float64)
                query_feature = query_tensor[choice_index][choice_mask].mean(dim=0).detach().cpu().numpy().astype(np.float64)
                hidden_norm = float(np.linalg.norm(hidden_feature))
                query_norm = float(np.linalg.norm(query_feature))
                if hidden_norm > 0:
                    hidden_feature = hidden_feature / hidden_norm
                if query_norm > 0:
                    query_feature = query_feature / query_norm
                hidden_dim = int(hidden_feature.shape[0])
                query_dim = int(query_feature.shape[0])
                row_hidden.append(hidden_feature)
                row_query.append(query_feature)
            hidden_rows.append(row_hidden)
            query_rows.append(row_query)

    if hidden_dim is None or query_dim is None:
        raise ValueError("no hidden/query features were extracted")
    hidden = np.zeros((len(rows), max_choices, hidden_dim), dtype=np.float64)
    query = np.zeros((len(rows), max_choices, query_dim), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for choice_index in range(len(row.choices)):
            hidden[row_index, choice_index] = hidden_rows[row_index][choice_index]
            query[row_index, choice_index] = query_rows[row_index][choice_index]
    return HiddenQueryState(
        hidden=hidden,
        query=query,
        metadata={
            "kind": "local_causal_lm_hidden_query_choice_mean",
            "model_path": model_path,
            "device": str(resolved_device),
            "dtype": dtype,
            "max_length": max_length,
            "prompt_mode": prompt_mode,
            "hidden_layer": int(hidden_layer),
            "query_layer": int(query_layer),
            "query_layer_index": int(query_layer_index),
            "hidden_dim": int(hidden_dim),
            "query_dim": int(query_dim),
            "latency_s": float(time.perf_counter() - started),
            "cache_hit": False,
        },
    )


def _hidden_query_state(
    rows: list[arc_gate.ArcRow],
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    prompt_mode: str,
    hidden_layer: int,
    query_layer: int,
    local_files_only: bool,
) -> HiddenQueryState:
    expected = _hidden_cache_metadata(
        rows=rows,
        model_path=model_path,
        dtype=dtype,
        max_length=max_length,
        prompt_mode=prompt_mode,
        hidden_layer=hidden_layer,
        query_layer=query_layer,
    )
    cached = _load_hidden_query_cache(npz_path=npz_path, meta_path=meta_path, expected=expected)
    if cached is not None:
        return cached
    state = _extract_hidden_query_state(
        rows,
        model_path=model_path,
        device=device,
        dtype=dtype,
        max_length=max_length,
        prompt_mode=prompt_mode,
        hidden_layer=hidden_layer,
        query_layer=query_layer,
        local_files_only=local_files_only,
    )
    metadata = expected | state.metadata | {"created_utc": dt.datetime.now(dt.UTC).isoformat()}
    state = HiddenQueryState(hidden=state.hidden, query=state.query, metadata=metadata)
    _write_hidden_query_cache(npz_path=npz_path, meta_path=meta_path, state=state)
    return HiddenQueryState(
        hidden=state.hidden,
        query=state.query,
        metadata=metadata | {"cache_npz": _display_path(npz_path), "cache_meta": _display_path(meta_path)},
    )


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_source_predictions(path: pathlib.Path | str) -> dict[str, int]:
    predictions: dict[str, int] = {}
    for row in _read_jsonl(path):
        forbidden = set(row.get("forbidden_source_fields", ()))
        if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
            raise ValueError(f"source cache row {row.get('row_id')} is missing forbidden-field contract")
        predictions[str(row["content_id"])] = int(row["source_selected_index"])
    if not predictions:
        raise ValueError(f"{path} contained no source prediction rows")
    return predictions


def _load_disagreement_row_ids(
    *,
    path: pathlib.Path,
    split: str,
    seed: int,
    limit: int | None,
) -> list[str]:
    row_ids: list[str] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        if row.get("split") != split or int(row.get("seed", -1)) != int(seed):
            continue
        if row.get("condition") != arc_gate.MATCHED_CONDITION:
            continue
        row_id = str(row["row_id"])
        if row_id in seen:
            continue
        seen.add(row_id)
        row_ids.append(row_id)
        if limit is not None and len(row_ids) >= limit:
            break
    if not row_ids:
        raise ValueError(f"no disagreement rows found for split={split!r} seed={seed}")
    return row_ids


def _filter_rows_by_ids(rows: list[arc_gate.ArcRow], row_ids: list[str]) -> list[arc_gate.ArcRow]:
    by_id = {row.row_id: row for row in rows}
    missing = [row_id for row_id in row_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"missing {len(missing)} disagreement row ids in split rows")
    return [by_id[row_id] for row_id in row_ids]


def _source_predictions(rows: list[arc_gate.ArcRow], cache: dict[str, int]) -> list[int]:
    output: list[int] = []
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        if row.content_id not in cache:
            missing.append(row.content_id)
            continue
        prediction = int(cache[row.content_id])
        if prediction < 0 or prediction >= len(row.choices):
            invalid.append(row.content_id)
            continue
        output.append(prediction)
    if missing or invalid:
        raise ValueError(f"source prediction mismatch: missing={len(missing)} invalid={len(invalid)}")
    return output


def _baseline_correctness(
    *,
    path: pathlib.Path,
    split: str,
    seed: int,
    condition: str,
) -> dict[str, bool]:
    correctness: dict[str, bool] = {}
    for row in _read_jsonl(path):
        if row.get("split") == split and int(row.get("seed", -1)) == int(seed) and row.get("condition") == condition:
            correctness[str(row["row_id"])] = bool(row["correct"])
    if not correctness:
        raise ValueError(f"no baseline rows for split={split} seed={seed} condition={condition}")
    return correctness


def _flat_to_rows(rows: list[arc_gate.ArcRow], flat: np.ndarray) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    cursor = 0
    for row in rows:
        next_cursor = cursor + len(row.choices)
        output.append(np.asarray(flat[cursor:next_cursor], dtype=np.float64))
        cursor = next_cursor
    if cursor != flat.shape[0]:
        raise ValueError("flat candidate feature count did not match rows")
    return output


def _row_flat_indices(rows: list[arc_gate.ArcRow], row_indices: np.ndarray) -> np.ndarray:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for row in rows:
        end = cursor + len(row.choices)
        offsets.append((cursor, end))
        cursor = end
    return np.concatenate(
        [np.arange(offsets[int(index)][0], offsets[int(index)][1], dtype=np.int64) for index in row_indices]
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1e-12), out=np.zeros_like(values), where=norms > 0)


def _flat_source_view(
    *,
    rows: list[arc_gate.ArcRow],
    state: HiddenQueryState,
    view: str,
) -> np.ndarray:
    flat: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        count = len(row.choices)
        hidden = np.asarray(state.hidden[row_index, :count], dtype=np.float64)
        query = np.asarray(state.query[row_index, :count], dtype=np.float64)
        hidden_residual = hidden - np.mean(hidden, axis=0, keepdims=True)
        query_residual = query - np.mean(query, axis=0, keepdims=True)
        if view == "hidden_residual":
            row_features = hidden_residual
        elif view == "query_residual":
            row_features = query_residual
        elif view == "hidden_query_residual":
            row_features = np.concatenate([hidden_residual, query_residual], axis=1)
        elif view == "hidden_abs_query_residual":
            row_features = np.concatenate([hidden, hidden_residual, query_residual], axis=1)
        else:
            raise ValueError(f"unknown source view {view!r}")
        flat.extend(_normalize_rows(row_features))
    return np.asarray(flat, dtype=np.float64)


def _fit_pca_ridge_map(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    fit_flat_indices: np.ndarray,
    pca_dim: int,
    ridge: float,
) -> dict[str, Any]:
    x_fit_raw = source_features[fit_flat_indices].astype(np.float64)
    y_fit = target_features[fit_flat_indices].astype(np.float64)
    mean = np.mean(x_fit_raw, axis=0)
    scale = np.std(x_fit_raw, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_fit = (x_fit_raw - mean) / scale
    _, singular_values, vt = np.linalg.svd(x_fit, full_matrices=False)
    dims = min(int(pca_dim), vt.shape[0])
    components = vt[:dims].astype(np.float64)
    z_fit = x_fit @ components.T
    z = np.concatenate([np.ones((z_fit.shape[0], 1), dtype=np.float64), z_fit], axis=1)
    reg = float(ridge) * np.eye(z.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(z.T @ z + reg, z.T @ y_fit)
    pred = z @ weights
    cos = _rowwise_cosine(pred, y_fit)
    return {
        "mean": mean,
        "scale": scale,
        "components": components,
        "weights": weights,
        "pca_dim": int(dims),
        "ridge": float(ridge),
        "fit_candidate_rows": int(len(fit_flat_indices)),
        "source_dim": int(source_features.shape[1]),
        "target_dim": int(target_features.shape[1]),
        "singular_values_top8": [float(value) for value in singular_values[:8]],
        "fit_alignment_cosine_mean": float(np.mean(cos)),
        "fit_alignment_cosine_p10": float(np.percentile(cos, 10)),
    }


def _apply_pca_ridge_map(source_features: np.ndarray, mapper: dict[str, Any]) -> np.ndarray:
    x = (source_features - mapper["mean"]) / mapper["scale"]
    z = x @ mapper["components"].T
    z = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    return z @ mapper["weights"]


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numer = np.sum(left * right, axis=1)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numer, np.maximum(denom, 1e-12), out=np.zeros_like(numer), where=denom > 0)


def _paired_delta_ci(
    experimental_correct: np.ndarray,
    baseline_correct: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    if experimental_correct.shape != baseline_correct.shape:
        raise ValueError("paired arrays must have matching shape")
    delta = experimental_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    if samples <= 0:
        return {
            "mean_delta": float(np.mean(delta)),
            "ci95_low": float(np.mean(delta)),
            "ci95_high": float(np.mean(delta)),
        }
    rng = np.random.default_rng(seed)
    indices = np.arange(len(delta), dtype=np.int64)
    boot = np.empty(int(samples), dtype=np.float64)
    for sample_index in range(int(samples)):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        boot[sample_index] = float(np.mean(delta[sampled]))
    return {
        "mean_delta": float(np.mean(delta)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def _matched_correct_by_row(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {str(row["row_id"]): bool(row["correct"]) for row in rows if row.get("condition") == arc_gate.MATCHED_CONDITION}


def _summarize_against_external(
    *,
    split: str,
    seed: int,
    prediction_rows: list[dict[str, Any]],
    qwen_disagreement_path: pathlib.Path,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    has_overlap: bool,
) -> dict[str, Any]:
    base = seed_stability._summarize_seed(
        seed=seed,
        rows=prediction_rows,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=has_overlap,
    )
    matched = _matched_correct_by_row(prediction_rows)
    qwen = _baseline_correctness(
        path=qwen_disagreement_path,
        split=split,
        seed=seed,
        condition="qwen_substituted_packet",
    )
    cached = _baseline_correctness(
        path=qwen_disagreement_path,
        split=split,
        seed=seed,
        condition=arc_gate.MATCHED_CONDITION,
    )
    row_ids = [row_id for row_id in matched if row_id in qwen and row_id in cached]
    if len(row_ids) != len(matched):
        raise ValueError("external baselines do not cover all matched rows")
    exp = np.asarray([matched[row_id] for row_id in row_ids], dtype=bool)
    qwen_values = np.asarray([qwen[row_id] for row_id in row_ids], dtype=bool)
    cached_values = np.asarray([cached[row_id] for row_id in row_ids], dtype=bool)
    qwen_ci = _paired_delta_ci(exp, qwen_values, seed=seed + 5001, samples=bootstrap_samples)
    cached_ci = _paired_delta_ci(exp, cached_values, seed=seed + 5002, samples=bootstrap_samples)
    return base | {
        "qwen_substituted_accuracy": float(np.mean(qwen_values)),
        "cached_tiny_packet_accuracy": float(np.mean(cached_values)),
        "cached_source_packet_accuracy": float(np.mean(cached_values)),
        "matched_minus_qwen_substituted": qwen_ci["mean_delta"],
        "matched_minus_cached_tiny_packet": cached_ci["mean_delta"],
        "matched_minus_cached_source_packet": cached_ci["mean_delta"],
        "paired_ci95_vs_qwen_substituted": qwen_ci,
        "paired_ci95_vs_cached_tiny_packet": cached_ci,
        "paired_ci95_vs_cached_source_packet": cached_ci,
    }


def _aggregate_external(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = seed_stability._aggregate(per_seed)
    for key in (
        "qwen_substituted_accuracy",
        "cached_tiny_packet_accuracy",
        "cached_source_packet_accuracy",
        "matched_minus_qwen_substituted",
        "matched_minus_cached_tiny_packet",
        "matched_minus_cached_source_packet",
    ):
        values = [float(row[key]) for row in per_seed]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_min"] = float(np.min(values))
        aggregate[f"{key}_max"] = float(np.max(values))
    aggregate["paired_ci95_low_vs_qwen_substituted_min"] = float(
        min(row["paired_ci95_vs_qwen_substituted"]["ci95_low"] for row in per_seed)
    )
    aggregate["paired_ci95_low_vs_cached_tiny_packet_min"] = float(
        min(row["paired_ci95_vs_cached_tiny_packet"]["ci95_low"] for row in per_seed)
    )
    aggregate["paired_ci95_low_vs_cached_source_packet_min"] = float(
        min(row["paired_ci95_vs_cached_source_packet"]["ci95_low"] for row in per_seed)
    )
    return aggregate


def _subset_rows_and_features(
    *,
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    receiver_features: np.ndarray,
    mapped_features: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[list[arc_gate.ArcRow], list[int], np.ndarray, np.ndarray]:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    flat_indices = _row_flat_indices(rows, row_indices)
    sub_rows = [rows[int(index)] for index in row_indices]
    sub_predictions = [source_predictions[int(index)] for index in row_indices]
    return sub_rows, sub_predictions, receiver_features[flat_indices], mapped_features[flat_indices]


def _evaluate_features(
    *,
    split: str,
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    mapped_features: np.ndarray,
    receiver_features: np.ndarray,
    qwen_disagreement_path: pathlib.Path,
    index_prior: list[float],
    seeds: tuple[int, ...],
    budget_bytes: int,
    code_dim: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    has_overlap: bool,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_residuals = _flat_to_rows(rows, mapped_features)
    receiver_residuals = arc_gate._candidate_residuals(rows, receiver_features)
    per_seed: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        projection = arc_gate._projection_matrix(receiver_features.shape[1], code_dim, seed=seed + 171)
        prediction_rows = arc_gate._rows_for_predictions(
            eval_rows=rows,
            residuals=source_residuals,
            decode_residuals=receiver_residuals,
            source_predictions=source_predictions,
            projection=projection,
            budget_bytes=budget_bytes,
            index_prior=index_prior,
            seed=seed + 911,
        )
        per_seed.append(
            _summarize_against_external(
                split=split,
                seed=seed,
                prediction_rows=prediction_rows,
                qwen_disagreement_path=qwen_disagreement_path,
                bootstrap_samples=bootstrap_samples,
                min_lift_over_target=min_lift_over_target,
                min_gap_over_control=min_gap_over_control,
                min_gap_over_text=min_gap_over_text,
                has_overlap=has_overlap,
            )
            | {"variant": variant}
        )
        for row in prediction_rows:
            all_rows.append({**row, "split": split, "seed": seed, "variant": variant})
    return per_seed, _aggregate_external(per_seed), all_rows


def _roll_candidate_features(rows: list[arc_gate.ArcRow], flat_features: np.ndarray) -> np.ndarray:
    rolled: list[np.ndarray] = []
    cursor = 0
    for row in rows:
        count = len(row.choices)
        block = np.asarray(flat_features[cursor : cursor + count], dtype=np.float64)
        rolled.append(np.roll(block, shift=-1, axis=0))
        cursor += count
    return np.concatenate(rolled, axis=0)


def _permute_feature_columns(features: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(features.shape[1])
    if features.shape[1] > 1 and np.all(permutation == np.arange(features.shape[1])):
        permutation = np.roll(permutation, 1)
    return features[:, permutation]


def _dev_row_split(rows: list[arc_gate.ArcRow], *, dev_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) < 4:
        raise ValueError("need at least four rows for train/dev split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    dev_count = max(1, int(round(len(rows) * float(dev_fraction))))
    dev = np.sort(order[:dev_count])
    fit = np.sort(order[dev_count:])
    if len(fit) == 0:
        raise ValueError("fit split is empty")
    return fit.astype(np.int64), dev.astype(np.int64)


def _target_spectra(
    *,
    rows: list[arc_gate.ArcRow],
    anchor_texts: list[str],
    anchor_count: int,
    spectral_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, receiver_features, metadata = fourier_gate._fourier_pair_features_for_variant(
        eval_rows=rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        variant=fourier_gate.MATCHED_VARIANT,
    )
    return receiver_features, metadata


def _target_residual_flat(rows: list[arc_gate.ArcRow], features: np.ndarray) -> np.ndarray:
    return np.concatenate(arc_gate._candidate_residuals(rows, features), axis=0)


def _write_frontier_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "view",
        "pca_dim",
        "ridge",
        "dev_accuracy_mean",
        "dev_qwen_substituted_accuracy_mean",
        "dev_cached_tiny_packet_accuracy_mean",
        "dev_matched_minus_qwen_substituted_mean",
        "dev_matched_minus_cached_tiny_packet_mean",
        "dev_paired_ci95_low_vs_qwen_substituted_min",
        "dev_matched_minus_target_mean",
        "fit_alignment_cosine_mean",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_summary_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC Hidden/Query Common-Basis Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected view: `{h['selected_view']}`",
        f"- selected pca/ridge: `{h['selected_pca_dim']}` / `{h['selected_ridge']}`",
        f"- validation disagreement rows: `{h['validation_disagreement_rows']}`",
        f"- test disagreement rows: `{h['test_disagreement_rows']}`",
        f"- test matched mean: `{h['test_matched_accuracy_mean']:.6f}`",
        f"- test Qwen-substituted mean: `{h['test_qwen_substituted_accuracy_mean']:.6f}`",
        f"- test cached source packet mean: `{h['test_cached_tiny_packet_accuracy_mean']:.6f}`",
        f"- test delta vs Qwen-sub: `{h['test_matched_minus_qwen_substituted_mean']:.6f}`",
        f"- test min CI95 low vs Qwen-sub: `{h['test_paired_ci95_low_vs_qwen_substituted_min']:.6f}`",
        f"- candidate-roll control mean: `{h['candidate_roll_matched_accuracy_mean']:.6f}`",
        f"- spectral-permutation control mean: `{h['spectral_permutation_matched_accuracy_mean']:.6f}`",
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_anchor_path: pathlib.Path = DEFAULT_TRAIN_ANCHORS,
    validation_path: pathlib.Path = DEFAULT_VALIDATION,
    test_path: pathlib.Path = DEFAULT_TEST,
    source_family_gate_dir: pathlib.Path = DEFAULT_SOURCE_FAMILY_GATE_DIR,
    source_model: str = DEFAULT_SOURCE_MODEL,
    source_device: str = "auto_cpu",
    source_dtype: str = "float32",
    source_max_length: int = 256,
    source_prompt_mode: str = "qa",
    hidden_layer: int = -1,
    query_layer: int = -1,
    local_files_only: bool = True,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    selection_seed: int = 18013,
    dev_fraction: float = 0.25,
    train_disagreement_limit: int | None = None,
    test_disagreement_limit: int | None = None,
    source_views: tuple[str, ...] = DEFAULT_VIEWS,
    pca_dims: tuple[int, ...] = DEFAULT_PCA_DIMS,
    ridges: tuple[float, ...] = DEFAULT_RIDGES,
    budget_bytes: int = 12,
    anchor_count: int = 384,
    spectral_dim: int = 96,
    code_dim: int = 96,
    bootstrap_samples: int = 500,
    min_lift_over_target: float = 0.02,
    min_gap_over_control: float = 0.02,
    min_gap_over_text: float = 0.0,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_anchor_path = _resolve(train_anchor_path)
    validation_path = _resolve(validation_path)
    test_path = _resolve(test_path)
    source_family_gate_dir = _resolve(source_family_gate_dir)
    qwen_disagreement_path = source_family_gate_dir / "qwen_disagreement_predictions.jsonl"
    source_contract = _source_cache_contract(source_family_gate_dir, source_model)
    source_model_resolved = str(source_contract["source_model"])
    source_cache_prefix = str(source_contract["source_cache_prefix"])
    validation_source_cache = pathlib.Path(source_contract["validation_source_cache"])
    test_source_cache = pathlib.Path(source_contract["test_source_cache"])
    validation_hidden_npz = output_dir / f"{source_cache_prefix}_validation_disagreement_hidden_query_cache.npz"
    validation_hidden_meta = output_dir / f"{source_cache_prefix}_validation_disagreement_hidden_query_cache.json"
    test_hidden_npz = output_dir / f"{source_cache_prefix}_test_disagreement_hidden_query_cache.npz"
    test_hidden_meta = output_dir / f"{source_cache_prefix}_test_disagreement_hidden_query_cache.json"

    train_anchor_rows = arc_gate._load_rows(train_anchor_path)
    validation_full = arc_gate._load_rows(validation_path)
    test_full = arc_gate._load_rows(test_path)
    selection_seed_value = int(seeds[0])
    validation_ids = _load_disagreement_row_ids(
        path=qwen_disagreement_path,
        split="validation",
        seed=selection_seed_value,
        limit=train_disagreement_limit,
    )
    test_ids = _load_disagreement_row_ids(
        path=qwen_disagreement_path,
        split="test",
        seed=selection_seed_value,
        limit=test_disagreement_limit,
    )
    validation_rows = _filter_rows_by_ids(validation_full, validation_ids)
    test_rows = _filter_rows_by_ids(test_full, test_ids)
    train_content = {row.content_id for row in validation_rows}
    test_content = {row.content_id for row in test_rows}
    overlap = sorted(train_content & test_content)

    validation_source_predictions = _source_predictions(validation_rows, _read_source_predictions(validation_source_cache))
    test_source_predictions = _source_predictions(test_rows, _read_source_predictions(test_source_cache))

    validation_state = _hidden_query_state(
        validation_rows,
        npz_path=validation_hidden_npz,
        meta_path=validation_hidden_meta,
        model_path=source_model_resolved,
        device=source_device,
        dtype=source_dtype,
        max_length=source_max_length,
        prompt_mode=source_prompt_mode,
        hidden_layer=hidden_layer,
        query_layer=query_layer,
        local_files_only=local_files_only,
    )
    test_state = _hidden_query_state(
        test_rows,
        npz_path=test_hidden_npz,
        meta_path=test_hidden_meta,
        model_path=source_model_resolved,
        device=source_device,
        dtype=source_dtype,
        max_length=source_max_length,
        prompt_mode=source_prompt_mode,
        hidden_layer=hidden_layer,
        query_layer=query_layer,
        local_files_only=local_files_only,
    )

    anchor_texts = arc_gate._choice_pair_texts(train_anchor_rows)
    validation_receiver_features, validation_basis_meta = _target_spectra(
        rows=validation_rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
    )
    test_receiver_features, test_basis_meta = _target_spectra(
        rows=test_rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
    )
    validation_target_residual_flat = _target_residual_flat(validation_rows, validation_receiver_features)
    test_index_prior = arc_gate._index_prior(train_anchor_rows)
    fit_rows, dev_rows = _dev_row_split(validation_rows, dev_fraction=dev_fraction, seed=selection_seed)
    fit_flat = _row_flat_indices(validation_rows, fit_rows)

    frontier_rows: list[dict[str, Any]] = []
    mapped_validation_by_candidate: dict[tuple[str, int, float], np.ndarray] = {}
    selected_key: tuple[str, int, float] | None = None
    best_sort_key: tuple[float, float, float, float] | None = None
    for view in source_views:
        validation_source_features = _flat_source_view(rows=validation_rows, state=validation_state, view=view)
        for pca_dim in pca_dims:
            for ridge in ridges:
                mapper = _fit_pca_ridge_map(
                    source_features=validation_source_features,
                    target_features=validation_target_residual_flat,
                    fit_flat_indices=fit_flat,
                    pca_dim=int(pca_dim),
                    ridge=float(ridge),
                )
                mapped_validation = _apply_pca_ridge_map(validation_source_features, mapper)
                mapped_validation_by_candidate[(view, int(pca_dim), float(ridge))] = mapped_validation
                dev_subset = _subset_rows_and_features(
                    rows=validation_rows,
                    source_predictions=validation_source_predictions,
                    receiver_features=validation_receiver_features,
                    mapped_features=mapped_validation,
                    row_indices=dev_rows,
                )
                dev_per_seed, dev_aggregate, _ = _evaluate_features(
                    split="validation",
                    rows=dev_subset[0],
                    source_predictions=dev_subset[1],
                    mapped_features=dev_subset[3],
                    receiver_features=dev_subset[2],
                    qwen_disagreement_path=qwen_disagreement_path,
                    index_prior=test_index_prior,
                    seeds=seeds,
                    budget_bytes=budget_bytes,
                    code_dim=code_dim,
                    bootstrap_samples=bootstrap_samples,
                    min_lift_over_target=min_lift_over_target,
                    min_gap_over_control=min_gap_over_control,
                    min_gap_over_text=min_gap_over_text,
                    has_overlap=bool(overlap),
                    variant="dev_candidate",
                )
                row = {
                    "view": view,
                    "pca_dim": int(mapper["pca_dim"]),
                    "ridge": float(ridge),
                    "dev_accuracy_mean": dev_aggregate["matched_accuracy_mean"],
                    "dev_qwen_substituted_accuracy_mean": dev_aggregate["qwen_substituted_accuracy_mean"],
                    "dev_cached_tiny_packet_accuracy_mean": dev_aggregate["cached_tiny_packet_accuracy_mean"],
                    "dev_matched_minus_qwen_substituted_mean": dev_aggregate[
                        "matched_minus_qwen_substituted_mean"
                    ],
                    "dev_matched_minus_cached_tiny_packet_mean": dev_aggregate[
                        "matched_minus_cached_tiny_packet_mean"
                    ],
                    "dev_paired_ci95_low_vs_qwen_substituted_min": dev_aggregate[
                        "paired_ci95_low_vs_qwen_substituted_min"
                    ],
                    "dev_matched_minus_target_mean": dev_aggregate["matched_minus_target_mean"],
                    "fit_alignment_cosine_mean": mapper["fit_alignment_cosine_mean"],
                    "fit_alignment_cosine_p10": mapper["fit_alignment_cosine_p10"],
                    "dev_per_seed": dev_per_seed,
                    "selected": False,
                }
                sort_key = (
                    row["dev_matched_minus_qwen_substituted_mean"],
                    row["dev_paired_ci95_low_vs_qwen_substituted_min"],
                    row["dev_matched_minus_cached_tiny_packet_mean"],
                    row["dev_accuracy_mean"],
                )
                if best_sort_key is None or sort_key > best_sort_key:
                    best_sort_key = sort_key
                    selected_key = (view, int(mapper["pca_dim"]), float(ridge))
                frontier_rows.append(row)
    if selected_key is None:
        raise ValueError("no selected candidate")
    for row in frontier_rows:
        row["selected"] = (
            row["view"] == selected_key[0]
            and int(row["pca_dim"]) == int(selected_key[1])
            and float(row["ridge"]) == float(selected_key[2])
        )

    selected_view, selected_pca_dim, selected_ridge = selected_key
    selected_validation_source = _flat_source_view(rows=validation_rows, state=validation_state, view=selected_view)
    selected_test_source = _flat_source_view(rows=test_rows, state=test_state, view=selected_view)
    final_mapper = _fit_pca_ridge_map(
        source_features=selected_validation_source,
        target_features=validation_target_residual_flat,
        fit_flat_indices=np.arange(selected_validation_source.shape[0], dtype=np.int64),
        pca_dim=selected_pca_dim,
        ridge=selected_ridge,
    )
    selected_test_mapped = _apply_pca_ridge_map(selected_test_source, final_mapper)
    test_per_seed, test_aggregate, test_prediction_rows = _evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=selected_test_mapped,
        receiver_features=test_receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=test_index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=bool(overlap),
        variant="matched_hidden_query_common_basis",
    )

    candidate_roll_per_seed, candidate_roll_aggregate, candidate_roll_rows = _evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=_roll_candidate_features(test_rows, selected_test_mapped),
        receiver_features=test_receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=test_index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=bool(overlap),
        variant="candidate_roll_hidden_query_control",
    )
    spectral_control_per_seed, spectral_control_aggregate, spectral_control_rows = _evaluate_features(
        split="test",
        rows=test_rows,
        source_predictions=test_source_predictions,
        mapped_features=selected_test_mapped,
        receiver_features=_permute_feature_columns(test_receiver_features, seed=selection_seed + 41),
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=test_index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=bool(overlap),
        variant="receiver_spectral_permutation_control",
    )

    strict_qwen_gate = bool(
        test_aggregate["matched_minus_qwen_substituted_min"] >= STRICT_QWEN_DELTA
        and test_aggregate["paired_ci95_low_vs_qwen_substituted_min"] > 0.0
    )
    strict_cached_gate = bool(
        test_aggregate["matched_minus_cached_tiny_packet_min"] >= STRICT_CACHED_DELTA
        and test_aggregate["paired_ci95_low_vs_cached_tiny_packet_min"] > 0.0
    )
    candidate_roll_gate = bool(
        candidate_roll_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
    )
    spectral_control_gate = bool(
        spectral_control_aggregate["matched_accuracy_mean"] <= test_aggregate["qwen_substituted_accuracy_mean"] + 0.005
    )
    pass_gate = bool(strict_qwen_gate and strict_cached_gate and candidate_roll_gate and spectral_control_gate and not overlap)
    selected_frontier = next(row for row in frontier_rows if row["selected"])
    headline = {
        "validation_disagreement_rows": len(validation_rows),
        "test_disagreement_rows": len(test_rows),
        "selected_view": selected_view,
        "selected_pca_dim": selected_pca_dim,
        "selected_ridge": selected_ridge,
        "selected_dev_delta_vs_qwen_substituted": selected_frontier["dev_matched_minus_qwen_substituted_mean"],
        "selected_dev_ci95_low_vs_qwen_substituted": selected_frontier[
            "dev_paired_ci95_low_vs_qwen_substituted_min"
        ],
        "test_matched_accuracy_mean": test_aggregate["matched_accuracy_mean"],
        "test_matched_accuracy_min": test_aggregate["matched_accuracy_min"],
        "test_qwen_substituted_accuracy_mean": test_aggregate["qwen_substituted_accuracy_mean"],
        "test_cached_tiny_packet_accuracy_mean": test_aggregate["cached_tiny_packet_accuracy_mean"],
        "test_matched_minus_qwen_substituted_mean": test_aggregate[
            "matched_minus_qwen_substituted_mean"
        ],
        "test_matched_minus_qwen_substituted_min": test_aggregate["matched_minus_qwen_substituted_min"],
        "test_paired_ci95_low_vs_qwen_substituted_min": test_aggregate[
            "paired_ci95_low_vs_qwen_substituted_min"
        ],
        "test_matched_minus_cached_tiny_packet_mean": test_aggregate[
            "matched_minus_cached_tiny_packet_mean"
        ],
        "test_paired_ci95_low_vs_cached_tiny_packet_min": test_aggregate[
            "paired_ci95_low_vs_cached_tiny_packet_min"
        ],
        "candidate_roll_matched_accuracy_mean": candidate_roll_aggregate["matched_accuracy_mean"],
        "spectral_permutation_matched_accuracy_mean": spectral_control_aggregate["matched_accuracy_mean"],
        "strict_qwen_gate": strict_qwen_gate,
        "strict_cached_gate": strict_cached_gate,
        "candidate_roll_gate": candidate_roll_gate,
        "spectral_control_gate": spectral_control_gate,
        "train_test_content_overlap_count": len(overlap),
        "source_hidden_query_train_cache_hit": bool(validation_state.metadata.get("cache_hit", False)),
        "source_hidden_query_test_cache_hit": bool(test_state.metadata.get("cache_hit", False)),
    }
    lay_explanation = (
        f"We take only the examples where {source_contract['source_family']} and Qwen chose different answers.  "
        "The source model's internal hidden/query vectors are compressed into the same small public coordinate system used by the ARC "
        f"packet method, then the receiver tries to answer from a {budget_bytes}-byte packet.  The key comparison is not "
        "against doing nothing; it is against simply using the stronger Qwen packet on those same examples."
    )
    interpretation = (
        "This is a strict source-family common-basis gate.  A pass would mean source hidden/query state "
        "can be translated into useful fixed-byte public-basis evidence beyond both the cached source packet "
        "and the Qwen-substituted packet.  A failure weakens this ARC branch and says the current mapping is "
        "not yet a real cross-family latent language, even though it may remain useful as a falsification and "
        "systems accounting artifact."
    )
    payload = {
        "gate": "source_private_arc_challenge_hidden_query_common_basis_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass requires the selected hidden/query common-basis packet to beat Qwen-substituted packets "
            "and cached source packets by at least 0.02 on every seed, with paired CI95 lower bound above "
            "zero vs both baselines. Candidate-roll and receiver-spectral-permutation controls must not exceed "
            "Qwen-substituted accuracy by more than 0.005."
        ),
        "headline": headline,
        "frontier_rows": frontier_rows,
        "test_per_seed": test_per_seed,
        "test_aggregate": test_aggregate,
        "candidate_roll_per_seed": candidate_roll_per_seed,
        "candidate_roll_aggregate": candidate_roll_aggregate,
        "spectral_permutation_per_seed": spectral_control_per_seed,
        "spectral_permutation_aggregate": spectral_control_aggregate,
        "fit_dev_split": {
            "selection_seed": selection_seed,
            "dev_fraction": dev_fraction,
            "fit_rows": int(len(fit_rows)),
            "dev_rows": int(len(dev_rows)),
            "fit_row_ids_sha256": hashlib.sha256(
                "\n".join(validation_rows[int(index)].row_id for index in fit_rows).encode("utf-8")
            ).hexdigest(),
            "dev_row_ids_sha256": hashlib.sha256(
                "\n".join(validation_rows[int(index)].row_id for index in dev_rows).encode("utf-8")
            ).hexdigest(),
        },
        "final_mapper": {
            key: value
            for key, value in final_mapper.items()
            if key not in {"mean", "scale", "components", "weights"}
        },
        "basis_contract": {
            "target_basis": "public train-anchor relative coordinates followed by orthonormal low-frequency DCT-II",
            "anchor_count": anchor_count,
            "spectral_dim": spectral_dim,
            "code_dim": code_dim,
            "budget_bytes": budget_bytes,
            "validation_basis_metadata": validation_basis_meta,
            "test_basis_metadata": test_basis_meta,
        },
        "method_contract": {
            "source_family": source_contract["source_family"],
            "source_model": source_model_resolved,
            "source_model_requested": source_contract["source_model_requested"],
            "source_inputs_at_eval": ["question", "choices"],
            "forbidden_eval_source_inputs": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            "source_hidden_query_raw_transmitted": False,
            "source_text_transmitted": False,
            "source_kv_transmitted": False,
            "packet_format": f"{budget_bytes}-byte sparse signed projection packet emitted from train-only mapped source hidden/query features",
            "native_gpu_claims_allowed": False,
        },
        "source_hidden_query_metadata": {
            "validation": validation_state.metadata,
            "test": test_state.metadata,
        },
        "inputs": {
            "train_anchor_path": _display_path(train_anchor_path),
            "train_anchor_sha256": _sha256_file(train_anchor_path),
            "validation_path": _display_path(validation_path),
            "validation_sha256": _sha256_file(validation_path),
            "test_path": _display_path(test_path),
            "test_sha256": _sha256_file(test_path),
            "source_family_gate_dir": _display_path(source_family_gate_dir),
            "qwen_disagreement_predictions": _display_path(qwen_disagreement_path),
            "qwen_disagreement_predictions_sha256": _sha256_file(qwen_disagreement_path),
            "validation_source_cache": _display_path(validation_source_cache),
            "validation_source_cache_sha256": _sha256_file(validation_source_cache),
            "test_source_cache": _display_path(test_source_cache),
            "test_source_cache_sha256": _sha256_file(test_source_cache),
            "source_cache_audit": _display_path(source_contract["source_cache_audit_path"]),
        },
        "source_cache_contract": {
            "source_family": source_contract["source_family"],
            "source_cache_prefix": source_cache_prefix,
            "source_model": source_model_resolved,
            "source_model_requested": source_contract["source_model_requested"],
            "validation_source_cache": _display_path(validation_source_cache),
            "test_source_cache": _display_path(test_source_cache),
            "validation_hidden_query_cache_npz": _display_path(validation_hidden_npz),
            "validation_hidden_query_cache_meta": _display_path(validation_hidden_meta),
            "test_hidden_query_cache_npz": _display_path(test_hidden_npz),
            "test_hidden_query_cache_meta": _display_path(test_hidden_meta),
        },
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": budget_bytes,
            "framed_record_bytes_per_request": budget_bytes + 3,
            "logical_test_raw_payload_bytes_total": int(budget_bytes * len(test_rows)),
            "logical_test_framed_record_bytes_total": int((budget_bytes + 3) * len(test_rows)),
            "total_wall_time_s": float(time.perf_counter() - started),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "native_gpu_claims_allowed": False,
        },
        "lay_explanation": lay_explanation,
        "interpretation": interpretation,
    }
    json_path = output_dir / "arc_challenge_hidden_query_common_basis_gate.json"
    md_path = output_dir / "arc_challenge_hidden_query_common_basis_gate.md"
    frontier_path = output_dir / "dev_frontier.csv"
    predictions_path = output_dir / "test_predictions.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_markdown(md_path, payload)
    _write_frontier_csv(frontier_path, frontier_rows)
    _write_jsonl(predictions_path, [*test_prediction_rows, *candidate_roll_rows, *spectral_control_rows])
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "pass_gate": payload["pass_gate"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (
                json_path,
                md_path,
                frontier_path,
                predictions_path,
                validation_hidden_npz,
                validation_hidden_meta,
                test_hidden_npz,
                test_hidden_meta,
            )
        ],
        "inputs": payload["inputs"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# ARC Hidden/Query Common-Basis Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- selected view: `{selected_view}`",
                f"- test delta vs Qwen-sub: `{headline['test_matched_minus_qwen_substituted_mean']:.6f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-anchor-path", type=pathlib.Path, default=DEFAULT_TRAIN_ANCHORS)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--source-family-gate-dir", type=pathlib.Path, default=DEFAULT_SOURCE_FAMILY_GATE_DIR)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--source-dtype", default="float32")
    parser.add_argument("--source-max-length", type=int, default=256)
    parser.add_argument("--source-prompt-mode", default="qa")
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--query-layer", type=int, default=-1)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--seeds", type=_parse_int_tuple, default=DEFAULT_SEEDS)
    parser.add_argument("--selection-seed", type=int, default=18013)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--train-disagreement-limit", type=int, default=None)
    parser.add_argument("--test-disagreement-limit", type=int, default=None)
    parser.add_argument("--source-views", type=_parse_str_tuple, default=DEFAULT_VIEWS)
    parser.add_argument("--pca-dims", type=_parse_int_tuple, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--anchor-count", type=int, default=384)
    parser.add_argument("--spectral-dim", type=int, default=96)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-lift-over-target", type=float, default=0.02)
    parser.add_argument("--min-gap-over-control", type=float, default=0.02)
    parser.add_argument("--min-gap-over-text", type=float, default=0.0)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_anchor_path=args.train_anchor_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        source_family_gate_dir=args.source_family_gate_dir,
        source_model=args.source_model,
        source_device=args.source_device,
        source_dtype=args.source_dtype,
        source_max_length=args.source_max_length,
        source_prompt_mode=args.source_prompt_mode,
        hidden_layer=args.hidden_layer,
        query_layer=args.query_layer,
        local_files_only=not args.allow_downloads,
        seeds=args.seeds,
        selection_seed=args.selection_seed,
        dev_fraction=args.dev_fraction,
        train_disagreement_limit=args.train_disagreement_limit,
        test_disagreement_limit=args.test_disagreement_limit,
        source_views=args.source_views,
        pca_dims=args.pca_dims,
        ridges=args.ridges,
        budget_bytes=args.budget_bytes,
        anchor_count=args.anchor_count,
        spectral_dim=args.spectral_dim,
        code_dim=args.code_dim,
        bootstrap_samples=args.bootstrap_samples,
        min_lift_over_target=args.min_lift_over_target,
        min_gap_over_control=args.min_gap_over_control,
        min_gap_over_text=args.min_gap_over_text,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
