"""No-weight-download eligibility packet for live hybrid experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_model_eligibility_20260506"
DEFAULT_CACHE_DIR = ROOT / ".debug/hf_home"
DEFAULT_MODELS = (
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-small-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
)
ARCHITECTURE_MAPS = ROOT / "experimental/shared/results/hybrid_architecture_maps_20260506/architecture_maps.json"


def _local_cache_dir(cache_root: Path, model_id: str) -> Path:
    owner, name = model_id.split("/", 1)
    return cache_root / "hub" / f"models--{owner}--{name}"


def _has_cached_weights(local_dir: Path) -> bool:
    if not local_dir.exists():
        return False
    for snapshot in (local_dir / "snapshots").glob("*"):
        if any(snapshot.rglob("*.safetensors")) or any(snapshot.rglob("pytorch_model*.bin")):
            return True
    return False


def _size_gb(bytes_value: int | None) -> float | None:
    if bytes_value is None:
        return None
    return bytes_value / (1024**3)


def _architecture_hash(model_id: str) -> str | None:
    if not ARCHITECTURE_MAPS.exists():
        return None
    maps = json.loads(ARCHITECTURE_MAPS.read_text(encoding="utf-8"))
    slug = model_id.split("/", 1)[-1].lower()
    normalized_slug = slug.removesuffix("-fp8")
    for row in maps:
        row_id = str(row.get("model_id", "")).lower()
        normalized_row_id = row_id.removesuffix("-fp8")
        if (
            row_id == slug
            or row_id.endswith(slug)
            or slug.endswith(row_id)
            or normalized_row_id.endswith(normalized_slug)
            or normalized_slug.endswith(normalized_row_id)
        ):
            return str(row.get("config_sha256"))
    return None


def _probe_transformers(model_id: str) -> tuple[bool, bool, str]:
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        config_status = f"config:{type(config).__name__}:{getattr(config, 'model_type', 'unknown')}"
        config_ok = True
    except Exception as exc:  # pragma: no cover - network/library dependent
        config_status = f"config_error:{type(exc).__name__}:{str(exc)[:160]}"
        config_ok = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        tok_status = f"tokenizer:{type(tokenizer).__name__}"
        tokenizer_ok = True
    except Exception as exc:  # pragma: no cover - network/library dependent
        tok_status = f"tokenizer_error:{type(exc).__name__}:{str(exc)[:160]}"
        tokenizer_ok = False
    return config_ok, tokenizer_ok, f"{config_status}; {tok_status}"


def inspect_model(api: HfApi, model_id: str, cache_root: Path) -> dict[str, Any]:
    info = api.model_info(model_id, files_metadata=True)
    siblings = list(info.siblings or [])
    total_bytes = sum(int(getattr(sibling, "size", 0) or 0) for sibling in siblings)
    safetensor_bytes = sum(
        int(getattr(sibling, "size", 0) or 0)
        for sibling in siblings
        if str(getattr(sibling, "rfilename", "")).endswith(".safetensors")
    )
    config_files = sorted(
        str(getattr(sibling, "rfilename", ""))
        for sibling in siblings
        if str(getattr(sibling, "rfilename", "")).endswith(("config.json", "tokenizer_config.json"))
    )
    tags = sorted(str(tag) for tag in (info.tags or []))
    model_type = "unknown"
    for tag in tags:
        if tag in {"granitemoehybrid", "qwen3_next"}:
            model_type = tag
            break
    local_dir = _local_cache_dir(cache_root, model_id)
    has_local_snapshot = local_dir.exists() and any((local_dir / "snapshots").glob("*"))
    weights_cached = _has_cached_weights(local_dir)
    config_ok, tokenizer_ok, support_status = _probe_transformers(model_id)
    mamba_ssm_installed = importlib.util.find_spec("mamba_ssm") is not None
    requires_mamba_ssm = model_type in {"granitemoehybrid", "qwen3_next"}
    estimated_weight_gb = _size_gb(safetensor_bytes or total_bytes)
    mac_trace_decision = "BLOCKED_NOT_CACHED"
    if weights_cached:
        mac_trace_decision = "POSSIBLE_LOCAL_CACHE_CHECK_REQUIRED"
    if estimated_weight_gb is not None and estimated_weight_gb > 24.0:
        mac_trace_decision = "GPU_RECOMMENDED_SIZE"
    if not weights_cached:
        mac_trace_decision = "BLOCKED_NOT_CACHED"
    elif requires_mamba_ssm and not mamba_ssm_installed:
        mac_trace_decision = "PREFLIGHT_BLOCKED_MISSING_RUNTIME"
    estimated_mac_feasible = bool(estimated_weight_gb is not None and estimated_weight_gb <= 16.0)
    can_load_weights_local_only = weights_cached
    eligible_for_ssq_lr = weights_cached and config_ok and model_type in {"granitemoehybrid", "qwen3_next"}
    eligible_for_horn = eligible_for_ssq_lr
    eligible_for_hbsm = eligible_for_ssq_lr
    blocking_reason = "none"
    if not weights_cached:
        blocking_reason = "weights not present in repo-local HF cache"
    elif requires_mamba_ssm and not mamba_ssm_installed:
        blocking_reason = "mamba_ssm runtime not installed"
    elif not config_ok or not tokenizer_ok:
        blocking_reason = "transformers config/tokenizer probe failed"

    return {
        "model_id": model_id,
        "sha": info.sha,
        "architecture_map_hash": _architecture_hash(model_id),
        "library_name": info.library_name,
        "pipeline_tag": info.pipeline_tag,
        "tags": tags,
        "model_type_tag": model_type,
        "hf_config_available": "config.json" in config_files,
        "tokenizer_available": any("tokenizer" in name for name in config_files),
        "transformers_support_status": support_status,
        "config_files": config_files,
        "file_count": len(siblings),
        "download_size_bytes": safetensor_bytes or total_bytes,
        "total_gb": _size_gb(total_bytes),
        "safetensors_gb": _size_gb(safetensor_bytes),
        "local_cache_dir": str(local_dir),
        "weights_cached": weights_cached,
        "has_local_snapshot": has_local_snapshot,
        "requires_remote_code": False,
        "requires_mamba_ssm": requires_mamba_ssm,
        "mamba_ssm_installed": mamba_ssm_installed,
        "can_instantiate_config_only": config_ok,
        "can_load_tokenizer": tokenizer_ok,
        "can_load_weights_local_only": can_load_weights_local_only,
        "estimated_mac_feasible": estimated_mac_feasible,
        "eligible_for_ssq_lr": eligible_for_ssq_lr,
        "eligible_for_horn": eligible_for_horn,
        "eligible_for_hbsm": eligible_for_hbsm,
        "blocking_reason": blocking_reason,
        "mac_trace_decision": mac_trace_decision,
    }


def write_packet(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_root: Path = DEFAULT_CACHE_DIR,
    model_ids: tuple[str, ...] = DEFAULT_MODELS,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    api = HfApi()
    rows = [inspect_model(api, model_id, cache_root) for model_id in model_ids]
    decision = "PREFLIGHT_BLOCKED_NO_LOCAL_WEIGHTS"
    if any(row["mac_trace_decision"] == "POSSIBLE_LOCAL_CACHE_CHECK_REQUIRED" for row in rows):
        decision = "LOCAL_CACHE_CANDIDATE_REQUIRES_TRACE_RUN"
    if any(row["mac_trace_decision"] == "PREFLIGHT_BLOCKED_MISSING_RUNTIME" for row in rows):
        decision = "PREFLIGHT_BLOCKED_MISSING_RUNTIME"

    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model_ids": list(model_ids),
                "hf_home": str(cache_root),
                "command": "python -m experimental.shared.hybrid_model_eligibility",
                "download_policy": "metadata-only; no model weights downloaded",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "surface": "hf_metadata_hybrid_model_eligibility",
                "decision": decision,
                "row_count": len(rows),
                "rows": [
                    {
                        "model_id": row["model_id"],
                        "model_type_tag": row["model_type_tag"],
                        "architecture_map_hash": row["architecture_map_hash"],
                        "safetensors_gb": row["safetensors_gb"],
                        "has_local_snapshot": row["has_local_snapshot"],
                        "mac_trace_decision": row["mac_trace_decision"],
                        "blocking_reason": row["blocking_reason"],
                    }
                    for row in rows
                ],
                "claim_boundary": ["metadata-only", "not model evidence", "not GPU evidence"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    lines = [
        "# Hybrid Model Eligibility",
        "",
        "Metadata-only preflight for HybridKernel, SSQ-LR, HORN, and HBSM.",
        "No model weights are downloaded by this packet.",
        "",
        "| Model | Type tag | Architecture hash | Safetensors GB | Local weights | Decision |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        gb = row["safetensors_gb"]
        gb_text = "unknown" if gb is None else f"{gb:.2f}"
        lines.append(
            "| {model_id} | {model_type_tag} | `{arch}` | {gb} | {local} | `{decision}` |".format(
                model_id=row["model_id"],
                model_type_tag=row["model_type_tag"],
                arch=str(row["architecture_map_hash"])[:12] if row["architecture_map_hash"] else "none",
                gb=gb_text,
                local="yes" if row["weights_cached"] else "no",
                decision=row["mac_trace_decision"],
            )
        )
    lines.extend(
        [
            "",
            f"Decision: `{decision}`.",
            "",
            "A real SSQ-LR/HORN/HBSM trace packet still requires loaded model weights,",
            "hooked recurrent state or boundary activations, and the strict real-packet checker.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "decision.md").write_text(
        "# Hybrid Model Eligibility Decision\n\n"
        f"`{decision}`\n\n"
        "This is a metadata-only blocker/provenance packet. It does not promote any branch.\n"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    args = parser.parse_args()
    rows = write_packet(
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        model_ids=tuple(args.model_ids) if args.model_ids else DEFAULT_MODELS,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "models": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
