#!/usr/bin/env python3
"""Experiment D: formalize OutlierMigrate decomposition across landed packets."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "experimental/outlier_migrate/decomposition_analysis"
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260512
TOP_FRACTION = 0.01
RANK_DELTA = 2


@dataclass(frozen=True)
class Packet:
    key: str
    phase_label: str
    model_lineage: str
    run_dir: Path


PACKETS = [
    Packet(
        key="phase0_granite_tiny",
        phase_label="Phase 0 Granite-4.0-H-Tiny",
        model_lineage="mamba2_hybrid_granite",
        run_dir=ROOT / "experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z",
    ),
    Packet(
        key="phase1_granite_small",
        phase_label="Phase 1 Granite-4.0-H-Small",
        model_lineage="mamba2_hybrid_granite",
        run_dir=ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
    ),
    Packet(
        key="phase2_nemotron3",
        phase_label="Phase 2 Nemotron-3-Nano partial",
        model_lineage="mamba2_hybrid_nemotron",
        run_dir=ROOT / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
    ),
    Packet(
        key="phase5p_transformer",
        phase_label="Phase 5' pure-Transformer control",
        model_lineage="pure_transformer_rope",
        run_dir=ROOT / "experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z",
    ),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def bootstrap_ci(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(mean(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def pearson(x_values: list[float], y_values: list[float]) -> dict[str, float | None]:
    if len(x_values) < 3 or len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return {"r": None, "pvalue": None}
    result = stats.pearsonr(x_values, y_values)
    return {"r": float(result.statistic), "pvalue": float(result.pvalue)}


def classify_layer_types(packet: Packet, model_provenance: dict[str, Any], layer_count: int) -> list[str]:
    snapshot = Path(model_provenance.get("snapshot_path") or "")
    config_path = snapshot / "config.json"
    if config_path.is_file():
        config = load_json(config_path)
        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and len(layer_types) == layer_count:
            return [str(item) for item in layer_types]
        pattern = config.get("hybrid_override_pattern")
        if isinstance(pattern, str) and len(pattern) == layer_count:
            mapping = {
                "M": "mamba",
                "*": "attention",
                "E": "moe_expert",
            }
            return [mapping.get(char, f"unknown_{char}") for char in pattern]
    if packet.model_lineage == "pure_transformer_rope":
        return ["attention"] * layer_count
    return ["unknown"] * layer_count


def load_packet(packet: Packet) -> dict[str, Any]:
    metrics = load_json(packet.run_dir / "metrics.json")
    manifest = load_json(packet.run_dir / "activation_magnitude_manifest.json")
    prompt_manifest = load_json(packet.run_dir / "prompt_manifest.json")
    model_provenance = load_json(packet.run_dir / "model_provenance.json")
    rows = list(iter_rows(packet.run_dir / "activation_magnitudes.jsonl.gz"))
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    positions = tuple(int(pos) for pos in metrics["positions"])
    layer_count = int(metrics["layer_count"])
    prompt_events = {
        int(row["prompt_index"]): row for row in manifest.get("prompt_events", []) if "prompt_index" in row
    }
    prompts = {
        int(row["index"]): row for row in prompt_manifest.get("prompts", []) if isinstance(row, dict) and "index" in row
    }
    return {
        "packet": packet,
        "metrics": metrics,
        "manifest": manifest,
        "prompt_manifest": prompt_manifest,
        "model_provenance": model_provenance,
        "by_trace_layer": by_trace_layer,
        "positions": positions,
        "layer_count": layer_count,
        "layer_types": classify_layer_types(packet, model_provenance, layer_count),
        "prompt_events": prompt_events,
        "prompts": prompts,
    }


def strict_components_for_layer(base: list[float], target: list[float]) -> dict[str, float]:
    top_k = max(1, math.ceil(len(base) * TOP_FRACTION))
    top_boundary = top_k - 1
    base_ranks = ranks_desc(base)
    target_ranks = ranks_desc(target)
    selected = [channel for channel, rank in enumerate(base_ranks) if rank <= top_boundary]
    left = 0
    within = 0
    original = 0
    for channel in selected:
        delta = abs(target_ranks[channel] - base_ranks[channel])
        if delta > RANK_DELTA:
            original += 1
        if target_ranks[channel] > top_boundary:
            left += 1
        elif delta > RANK_DELTA:
            within += 1
    denom = len(selected)
    return {
        "left_set_fraction": left / denom,
        "within_set_rank_shuffle_fraction": within / denom,
        "strict_original_fraction": original / denom,
    }


def compute_kendall(packet_data: dict[str, Any]) -> dict[str, Any]:
    packet: Packet = packet_data["packet"]
    by_trace_layer = packet_data["by_trace_layer"]
    positions = packet_data["positions"]
    layer_types = packet_data["layer_types"]
    rows: list[dict[str, Any]] = []
    trace_position_values: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for prompt_index in sorted(by_trace_layer):
        for layer_index in sorted(by_trace_layer[prompt_index]):
            base = by_trace_layer[prompt_index][layer_index][positions[0]]
            base_ranks = ranks_desc(base)
            for position in positions[1:]:
                target = by_trace_layer[prompt_index][layer_index][position]
                target_ranks = ranks_desc(target)
                tau_result = stats.kendalltau(base_ranks, target_ranks)
                tau = float(tau_result.statistic)
                item = {
                    "packet": packet.key,
                    "phase_label": packet.phase_label,
                    "model_lineage": packet.model_lineage,
                    "prompt_index": prompt_index,
                    "layer_index": layer_index,
                    "layer_type": layer_types[layer_index],
                    "position": position,
                    "kendall_tau": tau,
                    "pvalue": float(tau_result.pvalue),
                }
                rows.append(item)
                trace_position_values[prompt_index][position].append(tau)
    aggregate: list[dict[str, Any]] = []
    for position in positions[1:]:
        trace_means = [
            float(mean(trace_position_values[prompt_index][position]))
            for prompt_index in sorted(trace_position_values)
        ]
        aggregate.append(
            {
                "packet": packet.key,
                "phase_label": packet.phase_label,
                "position": position,
                "mean_kendall_tau": float(mean(trace_means)),
                "bootstrap_ci95": bootstrap_ci(trace_means),
                "bootstrap_unit": "trace mean over layers",
                "trace_count": len(trace_means),
            }
        )
    return {"rows": rows, "aggregate_by_position": aggregate}


def compute_components(packet_data: dict[str, Any]) -> dict[str, Any]:
    packet: Packet = packet_data["packet"]
    by_trace_layer = packet_data["by_trace_layer"]
    positions = packet_data["positions"]
    metrics = packet_data["metrics"]
    layer_types = packet_data["layer_types"]
    final_position = positions[-1]
    trace_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    layer_type_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for prompt_index in sorted(by_trace_layer):
        trace_values: dict[str, list[float]] = defaultdict(list)
        for layer_index in sorted(by_trace_layer[prompt_index]):
            components = strict_components_for_layer(
                by_trace_layer[prompt_index][layer_index][positions[0]],
                by_trace_layer[prompt_index][layer_index][final_position],
            )
            layer_type = layer_types[layer_index]
            for key, value in components.items():
                trace_values[key].append(value)
                layer_type_values[layer_type][key].append(value)
            layer_rows.append(
                {
                    "packet": packet.key,
                    "prompt_index": prompt_index,
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    **components,
                }
            )
        trace_rows.append(
            {
                "packet": packet.key,
                "prompt_index": prompt_index,
                **{key: float(mean(values)) for key, values in trace_values.items()},
            }
        )
    aggregate: dict[str, Any] = {
        "packet": packet.key,
        "phase_label": packet.phase_label,
        "model_lineage": packet.model_lineage,
        "run_dir": str(packet.run_dir.relative_to(ROOT)),
        "base_position": positions[0],
        "final_position": final_position,
        "top_channel_fraction": TOP_FRACTION,
        "rank_delta_strictly_greater_than": RANK_DELTA,
        "gate_metric_migration_fraction": float(metrics["migration_fraction"]),
        "gate_metric_ci95": metrics["bootstrap_ci95"],
    }
    for key in [
        "left_set_fraction",
        "within_set_rank_shuffle_fraction",
        "strict_original_fraction",
    ]:
        values = [float(row[key]) for row in trace_rows]
        aggregate[key] = float(mean(values))
        aggregate[f"{key}_ci95"] = bootstrap_ci(values)
    layer_type_breakdown = []
    for layer_type, values_by_key in sorted(layer_type_values.items()):
        item = {"packet": packet.key, "layer_type": layer_type, "row_count": len(next(iter(values_by_key.values())))}
        for key, values in values_by_key.items():
            item[key] = float(mean(values))
            item[f"{key}_ci95"] = bootstrap_ci([float(v) for v in values])
        layer_type_breakdown.append(item)
    return {
        "aggregate": aggregate,
        "trace_rows": trace_rows,
        "layer_rows": layer_rows,
        "layer_type_breakdown": layer_type_breakdown,
    }


def prompt_features(packet_data: dict[str, Any], prompt_index: int) -> dict[str, Any]:
    prompts = packet_data["prompts"]
    events = packet_data["prompt_events"]
    prompt = prompts.get(prompt_index, {})
    event = events.get(prompt_index, {})
    answer = str(prompt.get("answer", ""))
    source_file = str(prompt.get("source_file", ""))
    return {
        "prompt_index": prompt_index,
        "prompt_char_count": len(str(prompt.get("prompt", ""))),
        "input_token_count": event.get("input_token_count"),
        "answer_digit_count": len(answer),
        "source_file_is_part_ii": 1 if source_file.endswith("II.jsonl") else 0,
        "bf16_perplexity_at_scoring_position": None,
        "bf16_perplexity_unavailable_reason": (
            "migration characterization packets capture activation magnitudes but do not score BF16 perplexity"
        ),
    }


def regress(y: list[float], x_rows: list[dict[str, Any]], predictor_names: list[str]) -> dict[str, Any]:
    usable: list[tuple[float, list[float]]] = []
    for target, row in zip(y, x_rows):
        predictors: list[float] = []
        ok = True
        for name in predictor_names:
            value = row.get(name)
            if value is None:
                ok = False
                break
            predictors.append(float(value))
        if ok and math.isfinite(target):
            usable.append((float(target), predictors))
    if len(usable) <= len(predictor_names) + 1:
        return {"n": len(usable), "status": "insufficient_rows"}
    y_arr = np.array([item[0] for item in usable], dtype=float)
    x_arr = np.array([item[1] for item in usable], dtype=float)
    means = x_arr.mean(axis=0)
    stds = x_arr.std(axis=0)
    stds[stds == 0.0] = 1.0
    x_std = (x_arr - means) / stds
    design = np.column_stack([np.ones(len(y_arr)), x_std])
    beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    pred = design @ beta
    ss_res = float(((y_arr - pred) ** 2).sum())
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    return {
        "n": len(usable),
        "predictors": predictor_names,
        "intercept": float(beta[0]),
        "standardized_coefficients": {
            name: float(value) for name, value in zip(predictor_names, beta[1:])
        },
        "r_squared": None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot,
    }


def compute_regression(packet_data: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    trace_rows = components["trace_rows"]
    feature_rows = [prompt_features(packet_data, int(row["prompt_index"])) for row in trace_rows]
    predictor_names = [
        "prompt_index",
        "prompt_char_count",
        "input_token_count",
        "answer_digit_count",
        "source_file_is_part_ii",
    ]
    outputs = {}
    for target_name in [
        "strict_original_fraction",
        "left_set_fraction",
        "within_set_rank_shuffle_fraction",
    ]:
        y = [float(row[target_name]) for row in trace_rows]
        outputs[target_name] = {
            "regression": regress(y, feature_rows, predictor_names),
            "pearson_correlations": {
                name: pearson(
                    [float(row[name]) for row in feature_rows if row.get(name) is not None],
                    [float(trace_rows[i][target_name]) for i, row in enumerate(feature_rows) if row.get(name) is not None],
                )
                for name in predictor_names
            },
        }
    return {
        "packet": packet_data["packet"].key,
        "phase_label": packet_data["packet"].phase_label,
        "feature_rows": feature_rows,
        "targets": outputs,
        "omitted_predictors": {
            "bf16_perplexity_at_scoring_position": (
                "Unavailable in Phase 0/1/2/5' activation-migration packets; should be added to future scoring packets "
                "rather than inferred post hoc."
            )
        },
    }


def write_component_markdown(path: Path, components: list[dict[str, Any]], kendall: list[dict[str, Any]]) -> None:
    lines = [
        "# OutlierMigrate Experiment D: Decomposition Formalization",
        "",
        "## Definitions",
        "",
        "- Strict set-leaving: a channel with rank inside the top-1% boundary at decode position 100 has rank outside that boundary at the final decode position.",
        "- Within-set rank shuffling: the channel remains inside the final top-1% set but moves by more than 2 rank positions.",
        "- Gate migration: the preregistered original migration metric from each packet's `metrics.json`.",
        "",
        "## Component Summary",
        "",
        "| Packet | Gate migration | Gate CI95 | Strict set-leaving | Within-set shuffling | Strict original |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for payload in components:
        agg = payload["aggregate"]
        lines.append(
            "| {phase} | {gate:.12f} | [{glo:.12f}, {ghi:.12f}] | {left:.12f} [{llo:.12f}, {lhi:.12f}] | {within:.12f} [{wlo:.12f}, {whi:.12f}] | {orig:.12f} [{olo:.12f}, {ohi:.12f}] |".format(
                phase=agg["phase_label"],
                gate=agg["gate_metric_migration_fraction"],
                glo=agg["gate_metric_ci95"]["ci95_low"],
                ghi=agg["gate_metric_ci95"]["ci95_high"],
                left=agg["left_set_fraction"],
                llo=agg["left_set_fraction_ci95"]["ci95_low"],
                lhi=agg["left_set_fraction_ci95"]["ci95_high"],
                within=agg["within_set_rank_shuffle_fraction"],
                wlo=agg["within_set_rank_shuffle_fraction_ci95"]["ci95_low"],
                whi=agg["within_set_rank_shuffle_fraction_ci95"]["ci95_high"],
                orig=agg["strict_original_fraction"],
                olo=agg["strict_original_fraction_ci95"]["ci95_low"],
                ohi=agg["strict_original_fraction_ci95"]["ci95_high"],
            )
        )
    lines.extend([
        "",
        "## Kendall Tau Summary",
        "",
        "| Packet | Position | Mean Kendall tau | 95% bootstrap CI |",
        "| --- | ---: | ---: | ---: |",
    ])
    for item in kendall:
        ci = item["bootstrap_ci95"]
        lines.append(
            f"| {item['phase_label']} | {item['position']} | {item['mean_kendall_tau']:.12f} | "
            f"[{ci['ci95_low']:.12f}, {ci['ci95_high']:.12f}] |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Phase 5' shows that high rank migration is not confined to the measured Mamba-2 hybrid models. The decomposition remains useful for static-protection methods because strict set-leaving directly measures channels that leave a fixed protected set.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packet_payloads = [load_packet(packet) for packet in PACKETS if packet.run_dir.is_dir()]
    all_kendall_rows: list[dict[str, Any]] = []
    kendall_aggregates: list[dict[str, Any]] = []
    component_payloads: list[dict[str, Any]] = []
    regression_payloads: list[dict[str, Any]] = []
    cross_tab: list[dict[str, Any]] = []
    for packet_data in packet_payloads:
        kendall_payload = compute_kendall(packet_data)
        all_kendall_rows.extend(kendall_payload["rows"])
        kendall_aggregates.extend(kendall_payload["aggregate_by_position"])
        components = compute_components(packet_data)
        component_payloads.append(components)
        regression_payloads.append(compute_regression(packet_data, components))
        cross_tab.extend(components["layer_type_breakdown"])

    write_json(
        args.output_dir / "kendall_tau_by_position.json",
        {
            "schema_version": "om_experiment_d_kendall_tau_v1",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "rows": all_kendall_rows,
            "aggregate_by_position": kendall_aggregates,
        },
    )
    write_json(
        args.output_dir / "component_decomposition.json",
        {
            "schema_version": "om_experiment_d_component_decomposition_v1",
            "top_channel_fraction": TOP_FRACTION,
            "rank_delta_strictly_greater_than": RANK_DELTA,
            "packets": component_payloads,
        },
    )
    write_component_markdown(
        args.output_dir / "component_decomposition.md",
        component_payloads,
        kendall_aggregates,
    )
    write_json(
        args.output_dir / "cross_tabulation.json",
        {
            "schema_version": "om_experiment_d_cross_tabulation_v1",
            "description": "Layer-type x decomposition-component table.",
            "rows": cross_tab,
            "layer_type_note": (
                "Granite layer types come from config.layer_types. Nemotron-3 uses config.hybrid_override_pattern "
                "with M=mamba, *=attention, E=moe_expert. Pure Transformer layers are attention."
            ),
        },
    )
    write_json(
        args.output_dir / "trace_difficulty_regression.json",
        {
            "schema_version": "om_experiment_d_trace_difficulty_regression_v1",
            "packets": regression_payloads,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "packets": [payload["packet"].key for payload in packet_payloads],
                "kendall_rows": len(all_kendall_rows),
                "cross_tab_rows": len(cross_tab),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
