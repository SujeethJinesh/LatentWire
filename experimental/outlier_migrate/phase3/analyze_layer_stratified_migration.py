#!/usr/bin/env python3
"""Layer-stratified migration analysis for existing OutlierMigrate packets."""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_MD = ROOT / "experimental/outlier_migrate/phase3/results/layer_stratified_migration.md"
OUTPUT_JSON = ROOT / "experimental/outlier_migrate/phase3/results/layer_stratified_migration.json"

RUNS = [
    (
        "Phase 0 Granite-4.0-H-Tiny",
        ROOT / "experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z",
    ),
    (
        "Phase 1 Granite-4.0-H-Small",
        ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
    ),
    (
        "Phase 2 Nemotron-3-Nano partial",
        ROOT / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
    ),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def iter_activation_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def layer_types_from_config(config: dict[str, Any]) -> tuple[list[str], str]:
    if isinstance(config.get("layer_types"), list):
        return [normalize_layer_type(str(item)) for item in config["layer_types"]], "config.layer_types"
    pattern = config.get("hybrid_override_pattern")
    if isinstance(pattern, str):
        return [normalize_pattern_char(char) for char in pattern], "config.hybrid_override_pattern"
    count = int(config.get("num_hidden_layers", 0))
    return ["unknown"] * count, "fallback_unknown"


def normalize_pattern_char(char: str) -> str:
    if char == "M":
        return "ssm_mamba"
    if char == "*":
        return "attention"
    if char == "-":
        return "mlp"
    return "moe"


def normalize_layer_type(layer_type: str) -> str:
    lowered = layer_type.lower()
    if lowered in {"mamba", "ssm"}:
        return "ssm_mamba"
    if lowered == "attention":
        return "attention"
    if lowered in {"moe", "expert"}:
        return "moe"
    if lowered in {"mlp", "ffn"}:
        return "mlp"
    return lowered or "unknown"


def classify_layers(run_dir: Path) -> dict[int, str]:
    provenance = load_json(run_dir / "model_provenance.json")
    config_path = Path(provenance["snapshot_path"]) / "config.json"
    config = load_json(config_path)
    layer_types, source = layer_types_from_config(config)
    manifest = load_json(run_dir / "activation_magnitude_manifest.json")
    layer_count = int(manifest["layer_count"])
    if len(layer_types) < layer_count:
        layer_types = [*layer_types, *(["unknown"] * (layer_count - len(layer_types)))]
    mapping = {layer_index: layer_types[layer_index] for layer_index in range(layer_count)}
    return {
        "mapping": mapping,
        "source": source,
        "model_type": str(config.get("model_type", "unknown")),
        "has_moe_config": bool(config.get("num_local_experts") or config.get("n_routed_experts")),
    }


def load_by_trace_layer(path: Path) -> dict[int, dict[int, dict[int, list[float]]]]:
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in iter_activation_rows(path):
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    return by_trace_layer


def select_top_by_layer(
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]], base_position: int
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    layer_indices = sorted({layer for trace in by_trace_layer.values() for layer in trace})
    for layer_index in layer_indices:
        vectors = [
            by_trace_layer[prompt_index][layer_index][base_position]
            for prompt_index in sorted(by_trace_layer)
            if base_position in by_trace_layer[prompt_index].get(layer_index, {})
        ]
        if not vectors:
            continue
        channel_count = len(vectors[0])
        top_k = max(1, math.ceil(0.01 * channel_count))
        means = [mean(vector[channel] for vector in vectors) for channel in range(channel_count)]
        selected[layer_index] = top_channels(means, top_k)
    return selected


def summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    positions = [int(value) for value in metrics["positions"]]
    base_position = positions[0]
    final_position = positions[-1]
    by_trace_layer = load_by_trace_layer(run_dir / "activation_magnitudes.jsonl.gz")
    top_by_layer = select_top_by_layer(by_trace_layer, base_position)
    classification = classify_layers(run_dir)
    layer_type_by_index: dict[int, str] = classification["mapping"]

    per_layer: dict[int, dict[str, Any]] = {}
    for layer_index, original_top_set in sorted(top_by_layer.items()):
        original_values: list[float] = []
        strict_left_values: list[float] = []
        stayed_moved_values: list[float] = []
        top_k = len(original_top_set)
        top_boundary_rank = top_k - 1
        for prompt_index in sorted(by_trace_layer):
            layer_positions = by_trace_layer[prompt_index].get(layer_index, {})
            if base_position not in layer_positions or final_position not in layer_positions:
                continue
            base_ranks = ranks_desc(layer_positions[base_position])
            final_ranks = ranks_desc(layer_positions[final_position])
            strict_prompt_top_set = top_channels(layer_positions[base_position], top_k)
            original_migrated = 0
            strict_left = 0
            stayed_moved = 0
            for channel in original_top_set:
                rank_delta = abs(final_ranks[channel] - base_ranks[channel])
                if rank_delta > 2:
                    original_migrated += 1
            for channel in strict_prompt_top_set:
                rank_delta = abs(final_ranks[channel] - base_ranks[channel])
                if final_ranks[channel] > top_boundary_rank:
                    strict_left += 1
                elif rank_delta > 2:
                    stayed_moved += 1
            original_values.append(original_migrated / top_k)
            strict_left_values.append(strict_left / top_k)
            stayed_moved_values.append(stayed_moved / top_k)
        per_layer[layer_index] = {
            "layer_index": layer_index,
            "layer_type": layer_type_by_index.get(layer_index, "unknown"),
            "trace_count": len(original_values),
            "top_1_percent_count": top_k,
            "top_1_percent_boundary_rank_zero_based": top_boundary_rank,
            "original_migration_fraction": mean(original_values),
            "strict_set_leaving_fraction": mean(strict_left_values),
            "within_set_rank_shuffling_fraction": mean(stayed_moved_values),
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_layer.values():
        by_type[row["layer_type"]].append(row)

    type_summary = {}
    for layer_type, rows in sorted(by_type.items()):
        type_summary[layer_type] = {
            "layer_count": len(rows),
            "original_migration_fraction_mean": mean(row["original_migration_fraction"] for row in rows),
            "strict_set_leaving_fraction_mean": mean(row["strict_set_leaving_fraction"] for row in rows),
            "within_set_rank_shuffling_fraction_mean": mean(
                row["within_set_rank_shuffling_fraction"] for row in rows
            ),
        }

    return {
        "label": label,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "model_id": metrics["model_id"],
        "model_snapshot_commit": metrics.get("model_snapshot_commit"),
        "positions": positions,
        "base_position": base_position,
        "final_position": final_position,
        "classification_source": classification["source"],
        "model_type": classification["model_type"],
        "has_moe_config": classification["has_moe_config"],
        "overall_migration_fraction": metrics["migration_fraction"],
        "bootstrap_ci95": metrics["bootstrap_ci95"],
        "type_summary": type_summary,
        "per_layer": [per_layer[index] for index in sorted(per_layer)],
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_markdown(summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# OutlierMigrate Layer-Stratified Migration Analysis",
        "",
        "Generated from existing Phase 0/1/2 activation packets. No Phase 3 intervention data is used.",
        "",
        "Definitions:",
        "- **strict set-leaving**: a channel in the position-100 top-1% set has final rank greater than the zero-based top-1% boundary rank.",
        "- **within-set rank shuffling**: a channel stays inside the final top-1% set but moves by more than two rank positions.",
        "- **original migration**: the preregistered metric, which counts any top-1% channel whose final rank differs by more than two positions.",
        "",
        "Granite exposes Mamba-vs-attention mixer layer types; its MoE feed-forward is configured across blocks and is not separately isolatable from layer-output rows. Nemotron-3 exposes Mamba, attention, and MoE-only blocks via `hybrid_override_pattern`.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['label']}",
                "",
                f"- Run dir: `{summary['run_dir']}`",
                f"- Model: `{summary['model_id']}`",
                f"- Positions: `{summary['positions']}`",
                f"- Overall original migration fraction: `{fmt(summary['overall_migration_fraction'])}` "
                f"(CI95 `{fmt(summary['bootstrap_ci95']['ci95_low'])}`, `{fmt(summary['bootstrap_ci95']['ci95_high'])}`)",
                f"- Layer classification: `{summary['classification_source']}`",
                "",
                "| Layer type | Layers | Strict set-leaving | Within-set shuffling | Original migration |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for layer_type, row in summary["type_summary"].items():
            lines.append(
                f"| `{layer_type}` | {row['layer_count']} | "
                f"{fmt(row['strict_set_leaving_fraction_mean'])} | "
                f"{fmt(row['within_set_rank_shuffling_fraction_mean'])} | "
                f"{fmt(row['original_migration_fraction_mean'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- The strict set-leaving fraction is the most relevant number for static protected-channel lists, because it counts channels that leave the protected top-1% set entirely.",
            "- Within-set shuffling remains relevant for scale refresh and ordering-sensitive schemes, but it is not by itself evidence that a static top-1% membership list has failed.",
            "- Layer-type summaries average per-layer means equally, so very wide MoE blocks do not dominate the aggregate solely by parameter count.",
            "",
        ]
    )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summaries = [summarize_run(label, run_dir) for label, run_dir in RUNS]
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summaries)
    print(json.dumps({"wrote": [str(OUTPUT_MD), str(OUTPUT_JSON)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
