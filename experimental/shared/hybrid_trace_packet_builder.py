"""Build strict real gate packets from saved hybrid trace outputs.

The builder consumes tensor packets written by `activation_dumper.py`. It does
not run models. This keeps the 5090 workflow short: dump tensors once, then use
this script locally or on the node to produce checker-compatible SSQ-LR/HORN
packets. HBSM consumes a JSON row packet because its first real gate is a
forward sensitivity table rather than a raw tensor summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from experimental.shared.activation_dumper import load_tensor_manifest, load_tensor_packet
from experimental.shared.hybrid_gate_evaluators import (
    evaluate_hbsm_b1,
    evaluate_horn_h1,
    evaluate_ssq_lr_s1,
)
from experimental.shared.sensitivity_metrics import kurtosis, max_abs

TEMPLATE_MARKERS = ("TO_FILL_BEFORE_CAPTURE", "TEMPLATE_ONLY")
ARCHITECTURE_MAPS_PATH = (
    Path(__file__).resolve().parent / "results/hybrid_architecture_maps_20260506/architecture_maps.json"
)


def _rms(tensor: torch.Tensor) -> float:
    values = tensor.float()
    return float(torch.sqrt(torch.mean(values * values)))


def _outlier_mass(tensor: torch.Tensor) -> float:
    values = tensor.float().reshape(-1).abs()
    threshold = values.mean() + 3.0 * values.std(unbiased=False)
    if values.numel() == 0:
        return 0.0
    return float(torch.mean((values > threshold).float()))


def _tensor_lookup_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _lookup_tensor(
    tensors: dict[str, torch.Tensor],
    manifest: dict[str, dict[str, object]],
    name: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    lookup_name = name if name in tensors else _tensor_lookup_name(name)
    if lookup_name in tensors:
        tensor = tensors[lookup_name]
        provenance = dict(manifest.get(lookup_name, {}))
        if not provenance:
            raise KeyError(f"tensor {name!r} is missing from tensor_manifest.json")
        return tensor, provenance
    raise KeyError(f"tensor {name!r} not found in packet")


def _tensor_provenance_fields(
    *,
    row_tensor_name: str,
    source_tensor_name: str,
    provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "tensor_name": row_tensor_name,
        "tensor_source_name": source_tensor_name,
        "tensor_storage_name": str(provenance.get("storage_name", "")),
        "tensor_sha256": str(provenance.get("sha256", "")),
        "tensor_dtype": str(provenance.get("dtype", "")),
        "tensor_shape": list(provenance.get("shape", [])),
    }


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _contains_template_marker(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in TEMPLATE_MARKERS)
    if isinstance(value, dict):
        return any(_contains_template_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_template_marker(item) for item in value)
    return False


def _model_aliases() -> dict[str, str]:
    if not ARCHITECTURE_MAPS_PATH.exists():
        return {}
    payload = json.loads(ARCHITECTURE_MAPS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    aliases: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        canonical = str(row.get("model_id", "")).strip()
        if not canonical:
            continue
        aliases[canonical] = canonical
        row_aliases = row.get("model_id_aliases", [])
        if isinstance(row_aliases, list):
            for alias in row_aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    aliases[alias_text] = canonical
    return aliases


def _base_config(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("_template_only") is True:
        raise ValueError("metadata is a capture template; fill it before building a packet")
    if _contains_template_marker(metadata):
        raise ValueError("metadata still contains capture template markers")
    required = [
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "prompt_source",
        "prompt_ids_hash",
        "seed_list",
        "context_lengths",
        "dtype",
        "device",
        "command",
        "architecture_map_hash",
        "trace_plan_hash",
    ]
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"metadata missing required fields: {', '.join(missing)}")
    config = {field: metadata[field] for field in required}
    aliases = _model_aliases()
    original_model_id = str(config["model_id"])
    canonical_model_id = aliases.get(original_model_id, original_model_id)
    if canonical_model_id != original_model_id:
        config["served_model_id"] = metadata.get("served_model_id", original_model_id)
        config["model_id"] = canonical_model_id
    elif "served_model_id" in metadata:
        config["served_model_id"] = metadata["served_model_id"]
    config["canonical_model_id"] = canonical_model_id
    if "trace_plan_path" in metadata:
        config["trace_plan_path"] = metadata["trace_plan_path"]
    if "resource_limit_note" in metadata:
        config["resource_limit_note"] = metadata["resource_limit_note"]
    return config


def _packet_decision(config: dict[str, Any], gate_status: object) -> str:
    status = str(gate_status)
    if "resource_limit_note" in config:
        return f"RESOURCE_LIMITED_NOT_PROMOTABLE_{status}"
    return status


def _write_packet(
    output_dir: Path,
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    surface: str,
    decision: str,
    claim_boundary: list[str],
    summary_extra: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    summary = {
        "seed": config["seed_list"][0] if config["seed_list"] else None,
        "surface": surface,
        "decision": decision,
        "row_count": len(rows),
        "rows": rows,
        "claim_boundary": claim_boundary,
    }
    if summary_extra:
        summary.update(summary_extra)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(
        _summary_markdown(surface=surface, decision=decision, rows=rows, summary_extra=summary_extra or {})
    )
    (output_dir / "decision.md").write_text(
        f"# Real Trace Packet Decision\n\n`{decision}`\n\n"
        "This packet was built from saved tensors and must still pass the project gate interpretation.\n"
    )


def _copy_tensor_artifacts(tensor_packet: Path, output_dir: Path) -> None:
    """Copy the saved tensor files needed to make row hashes reviewable."""

    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "tensor_manifest.json"):
        source = tensor_packet / name
        if not source.is_file():
            raise FileNotFoundError(f"tensor packet missing required artifact: {source}")
        shutil.copy2(source, tensor_dir / name)
    for source in sorted(tensor_packet.glob("*.pt")):
        shutil.copy2(source, tensor_dir / source.name)


def _copy_hbsm_row_artifact(row_packet: Path, output_dir: Path) -> str:
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    storage_name = "hbsm_row_packet.json"
    target = evidence_dir / storage_name
    shutil.copy2(row_packet, target)
    digest = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "source_row_packet": {
            "original_name": row_packet.name,
            "sha256": digest,
            "storage_name": storage_name,
        }
    }
    (evidence_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return digest


def _summary_markdown(
    *,
    surface: str,
    decision: str,
    rows: list[dict[str, Any]],
    summary_extra: dict[str, Any],
) -> str:
    lines = [
        f"# {surface}",
        "",
        f"Decision: `{decision}`.",
        "",
        f"Rows: `{len(rows)}`.",
        "",
        "Aggregate gate fields:",
    ]
    for key in sorted(summary_extra):
        lines.append(f"- `{key}`: `{summary_extra[key]}`")
    lines += [
        "",
        "This packet contains saved-tensor measurements only. It is not GPU throughput, HBM, or latency evidence.",
    ]
    return "\n".join(lines) + "\n"


def build_ssq_lr_packet(tensor_packet: Path, output_dir: Path) -> list[dict[str, Any]]:
    tensors, metadata = load_tensor_packet(tensor_packet)
    manifest = load_tensor_manifest(tensor_packet)
    config = _base_config(metadata)
    entries = metadata.get("ssq_lr_entries")
    if not isinstance(entries, list):
        raise ValueError("metadata must contain list field ssq_lr_entries")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        tensor_name = str(entry["tensor"])
        tensor, provenance = _lookup_tensor(tensors, manifest, tensor_name)
        rows.append(
            {
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "prompt_id": str(entry["prompt_id"]),
                "layer": int(entry["layer"]),
                "layer_kind": str(entry["layer_kind"]),
                "position_bucket": str(entry["position_bucket"]),
                "state_tensor_kind": str(entry["state_tensor_kind"]),
                "state_shape": list(tensor.shape),
                "max_abs": max_abs(tensor),
                "rms": _rms(tensor),
                "std": float(torch.std(tensor.float(), unbiased=False)),
                "kurtosis": kurtosis(tensor),
                "outlier_mass": _outlier_mass(tensor),
                "control_type": str(entry.get("control_type", "bf16_no_quant")),
                **_tensor_provenance_fields(
                    row_tensor_name=tensor_name,
                    source_tensor_name=str(provenance.get("original_name", tensor_name)),
                    provenance=provenance,
                ),
            }
        )
    summary_extra = evaluate_ssq_lr_s1(rows)
    decision = _packet_decision(config, summary_extra["gate_status"])
    _write_packet(
        output_dir,
        config=config,
        rows=rows,
        surface="real_ssq_lr_s1_tensor_packet",
        decision=decision,
        claim_boundary=["saved tensor trace", "not GPU evidence"],
        summary_extra=summary_extra,
    )
    _copy_tensor_artifacts(tensor_packet, output_dir)
    return rows


def build_horn_packet(tensor_packet: Path, output_dir: Path) -> list[dict[str, Any]]:
    tensors, metadata = load_tensor_packet(tensor_packet)
    manifest = load_tensor_manifest(tensor_packet)
    config = _base_config(metadata)
    entries = metadata.get("horn_entries")
    if not isinstance(entries, list):
        raise ValueError("metadata must contain list field horn_entries")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row_tensor_name = str(entry["tensor"])
        source_tensor_name = str(entry.get("tensor_alias_of", row_tensor_name))
        tensor, provenance = _lookup_tensor(tensors, manifest, source_tensor_name)
        row = {
            "model_id": config["model_id"],
            "prompt_id": str(entry["prompt_id"]),
            "prompt_cluster_id": str(entry.get("prompt_cluster_id") or entry["prompt_id"]),
            "layer_left": int(entry["layer_left"]),
            "layer_right": int(entry["layer_right"]),
            "direction": str(entry["direction"]),
            "matched_boundary_direction": str(
                entry.get("matched_boundary_direction", entry["direction"])
            ),
            "boundary_index": int(entry["boundary_index"]),
            "pre_norm_position": str(entry["pre_norm_position"]),
            "post_norm_position": str(entry["post_norm_position"]),
            "max_abs": max_abs(tensor),
            "rms": _rms(tensor),
            "kurtosis": kurtosis(tensor),
            "control_type": str(entry["control_type"]),
            **_tensor_provenance_fields(
                row_tensor_name=row_tensor_name,
                source_tensor_name=str(provenance.get("original_name", source_tensor_name)),
                provenance=provenance,
            ),
        }
        if "tensor_alias_of" in entry:
            row["tensor_alias_of"] = str(entry["tensor_alias_of"])
        rows.append(row)
    summary_extra = evaluate_horn_h1(rows)
    decision = _packet_decision(config, summary_extra["gate_status"])
    _write_packet(
        output_dir,
        config=config,
        rows=rows,
        surface="real_horn_h1_tensor_packet",
        decision=decision,
        claim_boundary=["saved tensor trace", "not GPU evidence"],
        summary_extra=summary_extra,
    )
    _copy_tensor_artifacts(tensor_packet, output_dir)
    return rows


def build_hbsm_packet(row_packet: Path, output_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads(row_packet.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HBSM row packet must be a JSON object")
    if payload.get("_template_only") is True or _contains_template_marker(payload):
        raise ValueError("HBSM row packet is a capture template; fill it before building a packet")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("HBSM row packet must contain object field metadata")
    config = _base_config(metadata)
    entries = payload.get("hbsm_entries")
    if not isinstance(entries, list):
        raise ValueError("HBSM row packet must contain list field hbsm_entries")
    config["source_row_packet_sha256"] = "sha256:" + hashlib.sha256(row_packet.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"HBSM entry {index} must be an object")
        rows.append(
            {
                "model_id": config["model_id"],
                "prompt_id": str(entry["prompt_id"]),
                "layer": int(entry["layer"]),
                "boundary_flag": _require_bool(entry["boundary_flag"], "boundary_flag"),
                "precision_perturbation": str(entry["precision_perturbation"]),
                "kl_or_nll_drift": float(entry["kl_or_nll_drift"]),
                "cheap_predictor": float(entry["cheap_predictor"]),
                "parameter_count": int(entry["parameter_count"]),
                "weight_norm": float(entry["weight_norm"]),
                "top_decile_flag": _require_bool(entry["top_decile_flag"], "top_decile_flag"),
                "random_top_decile": _require_bool(entry["random_top_decile"], "random_top_decile"),
                "train_test_split": str(entry["train_test_split"]),
                "control_type": str(entry["control_type"]),
            }
        )
    summary_extra = evaluate_hbsm_b1(rows)
    decision = _packet_decision(config, summary_extra["gate_status"])
    _write_packet(
        output_dir,
        config=config,
        rows=rows,
        surface="real_hbsm_b1_sensitivity_packet",
        decision=decision,
        claim_boundary=["saved forward sensitivity rows", "not GPU evidence"],
        summary_extra=summary_extra,
    )
    _copy_hbsm_row_artifact(row_packet, output_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=("ssq_lr", "horn", "hbsm"), required=True)
    parser.add_argument("--tensor-packet", type=Path)
    parser.add_argument("--row-packet", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.project == "ssq_lr":
        if args.tensor_packet is None:
            raise SystemExit("--tensor-packet is required for ssq_lr")
        rows = build_ssq_lr_packet(args.tensor_packet, args.output_dir)
    elif args.project == "horn":
        if args.tensor_packet is None:
            raise SystemExit("--tensor-packet is required for horn")
        rows = build_horn_packet(args.tensor_packet, args.output_dir)
    else:
        if args.row_packet is None:
            raise SystemExit("--row-packet is required for hbsm")
        rows = build_hbsm_packet(args.row_packet, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
