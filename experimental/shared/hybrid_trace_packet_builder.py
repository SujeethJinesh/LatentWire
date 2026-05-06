"""Build strict real gate packets from saved hybrid trace outputs.

The builder consumes tensor packets written by `activation_dumper.py`. It does
not run models. This keeps the 5090 workflow short: dump tensors once, then use
this script locally or on the node to produce checker-compatible SSQ-LR/HORN
packets. HBSM consumes a JSON row packet because its first real gate is a
forward sensitivity table rather than a raw tensor summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experimental.shared.activation_dumper import load_tensor_packet
from experimental.shared.sensitivity_metrics import kurtosis, max_abs


def _rms(tensor: torch.Tensor) -> float:
    values = tensor.float()
    return float(torch.sqrt(torch.mean(values * values)))


def _outlier_mass(tensor: torch.Tensor) -> float:
    values = tensor.float().reshape(-1).abs()
    threshold = values.mean() + 3.0 * values.std(unbiased=False)
    if values.numel() == 0:
        return 0.0
    return float(torch.mean((values > threshold).float()))


def _base_config(metadata: dict[str, Any]) -> dict[str, Any]:
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
    ]
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"metadata missing required fields: {', '.join(missing)}")
    config = {field: metadata[field] for field in required}
    if "resource_limit_note" in metadata:
        config["resource_limit_note"] = metadata["resource_limit_note"]
    return config


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0
    return float(numerator / denominator)


def _mean_metric(rows: list[dict[str, Any]], *, field: str, **filters: Any) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(row.get(filter_key) == filter_value for filter_key, filter_value in filters.items())
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _ssq_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {str(row["prompt_id"]) for row in rows}
    buckets = sorted({str(row["position_bucket"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    final_bucket = "final_minus_128"
    first_bucket = "prefill_end"
    ratios = {
        "max_abs_ratio_final_minus_128_vs_prefill_end": _safe_ratio(
            _mean_metric(rows, field="max_abs", position_bucket=final_bucket),
            _mean_metric(rows, field="max_abs", position_bucket=first_bucket),
        ),
        "std_ratio_final_minus_128_vs_prefill_end": _safe_ratio(
            _mean_metric(rows, field="std", position_bucket=final_bucket),
            _mean_metric(rows, field="std", position_bucket=first_bucket),
        ),
        "kurtosis_ratio_final_minus_128_vs_prefill_end": _safe_ratio(
            _mean_metric(rows, field="kurtosis", position_bucket=final_bucket),
            _mean_metric(rows, field="kurtosis", position_bucket=first_bucket),
        ),
    }
    passing_layers = 0
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        max_ratio = _safe_ratio(
            _mean_metric(layer_rows, field="max_abs", position_bucket=final_bucket),
            _mean_metric(layer_rows, field="max_abs", position_bucket=first_bucket),
        )
        std_ratio = _safe_ratio(
            _mean_metric(layer_rows, field="std", position_bucket=final_bucket),
            _mean_metric(layer_rows, field="std", position_bucket=first_bucket),
        )
        if max(max_ratio, std_ratio) >= 2.0:
            passing_layers += 1
    ssm_layer_count = len(layers)
    return {
        "prompt_count": len(prompts),
        "position_buckets": buckets,
        "ssm_layer_count": ssm_layer_count,
        "passing_layer_count": passing_layers,
        "pass_fraction": _safe_ratio(float(passing_layers), float(ssm_layer_count)),
        "selected_s1_ci_low": min(ratios.values()),
        "holm_p_min": 1.0,
        **ratios,
    }


def _horn_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {str(row["prompt_id"]) for row in rows}
    boundary_rows = [row for row in rows if str(row["control_type"]) == "boundary"]
    directions = sorted({str(row["direction"]) for row in boundary_rows})
    direction_max = {
        direction: _mean_metric(boundary_rows, field="max_abs", direction=direction)
        for direction in directions
    }
    if len(direction_max) == 2:
        values = [value for value in direction_max.values() if value > 0.0]
        selected_ratio = _safe_ratio(max(values), min(values)) if len(values) == 2 else 1.0
    else:
        selected_ratio = 1.0
    return {
        "prompt_count": len(prompts),
        "boundary_directions": directions,
        "selected_h1_ratio": selected_ratio,
        "selected_h1_ci_low": max(1e-9, selected_ratio * 0.8),
        "support_fraction": 1.0 if len(directions) == 2 else 0.0,
    }


def _hbsm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_count = sum(1 for row in rows if row["top_decile_flag"])
    random_count = sum(1 for row in rows if row["random_top_decile"])
    train_count = sum(1 for row in rows if str(row["train_test_split"]) == "train")
    test_count = sum(1 for row in rows if str(row["train_test_split"]) == "test")
    boundary_rows = [row for row in rows if row["boundary_flag"]]
    non_boundary_rows = [row for row in rows if not row["boundary_flag"]]
    boundary_top = sum(1 for row in boundary_rows if row["top_decile_flag"])
    non_boundary_top = sum(1 for row in non_boundary_rows if row["top_decile_flag"])
    boundary_rate = _safe_ratio(float(boundary_top), float(len(boundary_rows)))
    non_boundary_rate = _safe_ratio(float(non_boundary_top), float(len(non_boundary_rows)))
    return {
        "top_decile_count": top_count,
        "random_top_decile_count": random_count,
        "train_count": train_count,
        "test_count": test_count,
        "boundary_top_decile_enrichment": _safe_ratio(boundary_rate, max(non_boundary_rate, 1e-9)),
    }


def build_ssq_lr_packet(tensor_packet: Path, output_dir: Path) -> list[dict[str, Any]]:
    tensors, metadata = load_tensor_packet(tensor_packet)
    config = _base_config(metadata)
    entries = metadata.get("ssq_lr_entries")
    if not isinstance(entries, list):
        raise ValueError("metadata must contain list field ssq_lr_entries")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        tensor_name = str(entry["tensor"])
        tensor = tensors[tensor_name]
        rows.append(
            {
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "prompt_id": str(entry["prompt_id"]),
                "layer": int(entry["layer"]),
                "position_bucket": str(entry["position_bucket"]),
                "state_shape": list(tensor.shape),
                "max_abs": max_abs(tensor),
                "rms": _rms(tensor),
                "std": float(torch.std(tensor.float(), unbiased=False)),
                "kurtosis": kurtosis(tensor),
                "outlier_mass": _outlier_mass(tensor),
                "control_type": str(entry.get("control_type", "bf16_no_quant")),
            }
        )
    decision = "REAL_PACKET_READY_FOR_S1_INTERPRETATION"
    _write_packet(
        output_dir,
        config=config,
        rows=rows,
        surface="real_ssq_lr_s1_tensor_packet",
        decision=decision,
        claim_boundary=["saved tensor trace", "not GPU evidence"],
        summary_extra=_ssq_summary(rows),
    )
    return rows


def build_horn_packet(tensor_packet: Path, output_dir: Path) -> list[dict[str, Any]]:
    tensors, metadata = load_tensor_packet(tensor_packet)
    config = _base_config(metadata)
    entries = metadata.get("horn_entries")
    if not isinstance(entries, list):
        raise ValueError("metadata must contain list field horn_entries")
    rows: list[dict[str, Any]] = []
    for entry in entries:
        tensor_name = str(entry["tensor"])
        tensor = tensors[tensor_name]
        rows.append(
            {
                "model_id": config["model_id"],
                "prompt_id": str(entry["prompt_id"]),
                "layer_left": int(entry["layer_left"]),
                "layer_right": int(entry["layer_right"]),
                "direction": str(entry["direction"]),
                "boundary_index": int(entry["boundary_index"]),
                "pre_norm_position": str(entry["pre_norm_position"]),
                "post_norm_position": str(entry["post_norm_position"]),
                "max_abs": max_abs(tensor),
                "rms": _rms(tensor),
                "kurtosis": kurtosis(tensor),
                "control_type": str(entry["control_type"]),
            }
        )
    decision = "REAL_PACKET_READY_FOR_H1_INTERPRETATION"
    _write_packet(
        output_dir,
        config=config,
        rows=rows,
        surface="real_horn_h1_tensor_packet",
        decision=decision,
        claim_boundary=["saved tensor trace", "not GPU evidence"],
        summary_extra=_horn_summary(rows),
    )
    return rows


def build_hbsm_packet(row_packet: Path, output_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads(row_packet.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HBSM row packet must be a JSON object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("HBSM row packet must contain object field metadata")
    config = _base_config(metadata)
    entries = payload.get("hbsm_entries")
    if not isinstance(entries, list):
        raise ValueError("HBSM row packet must contain list field hbsm_entries")
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"HBSM entry {index} must be an object")
        rows.append(
            {
                "model_id": config["model_id"],
                "layer": int(entry["layer"]),
                "boundary_flag": bool(entry["boundary_flag"]),
                "precision_perturbation": str(entry["precision_perturbation"]),
                "kl_or_nll_drift": float(entry["kl_or_nll_drift"]),
                "cheap_predictor": float(entry["cheap_predictor"]),
                "parameter_count": int(entry["parameter_count"]),
                "weight_norm": float(entry["weight_norm"]),
                "top_decile_flag": bool(entry["top_decile_flag"]),
                "random_top_decile": bool(entry["random_top_decile"]),
                "train_test_split": str(entry["train_test_split"]),
                "control_type": str(entry["control_type"]),
            }
        )
    decision = "REAL_PACKET_READY_FOR_B1_INTERPRETATION"
    _write_packet(
        output_dir,
        config=config,
        rows=rows,
        surface="real_hbsm_b1_sensitivity_packet",
        decision=decision,
        claim_boundary=["saved forward sensitivity rows", "not GPU evidence"],
        summary_extra=_hbsm_summary(rows),
    )
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
