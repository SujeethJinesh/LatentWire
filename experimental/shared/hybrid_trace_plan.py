"""Build deterministic trace-collection plans for active hybrid gates.

The plan is not model evidence. It tells a Mac or 5090 tensor-dump run exactly
which prompts, layers, boundaries, and controls must be captured before the
strict packet builders can produce SSQ-LR, HORN, or HBSM real gate packets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = ROOT / "experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl"
DEFAULT_ARCHITECTURE_MAPS = (
    ROOT / "experimental/shared/results/hybrid_architecture_maps_20260506/architecture_maps.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_trace_plan_20260507"
SSQ_BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"{path} line {line_number} missing prompt_id")
            prompts.append(row)
    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


def _load_architecture_maps(path: Path) -> list[dict[str, Any]]:
    maps = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(maps, list) or not maps:
        raise ValueError(f"{path} must contain a non-empty architecture map list")
    return maps


def _architecture_hash(model_map: dict[str, Any]) -> str:
    config_sha = str(model_map.get("config_sha256", "")).strip().lower()
    if not config_sha:
        raise ValueError(f"model map {model_map.get('model_id')} missing config_sha256")
    return f"sha256:{config_sha}"


def _ssm_layers(model_map: dict[str, Any]) -> list[int]:
    return [
        index
        for index, kind in enumerate(model_map.get("layer_kinds", []))
        if str(kind).lower() == "ssm"
    ]


def _ssq_state_tensor_kind(model_map: dict[str, Any]) -> str:
    model_type = str(model_map.get("model_type", "")).lower()
    if model_type == "granitemoehybrid":
        return "mamba2_recurrent_state"
    return "recurrent_ssm_state"


def _non_boundary_pairs(model_map: dict[str, Any]) -> list[dict[str, Any]]:
    kinds = [str(kind) for kind in model_map.get("layer_kinds", [])]
    boundary_pairs = {
        (int(boundary["left_layer"]), int(boundary["right_layer"]))
        for boundary in model_map.get("boundaries", [])
    }
    pairs: list[dict[str, Any]] = []
    for left, (left_kind, right_kind) in enumerate(zip(kinds, kinds[1:])):
        right = left + 1
        if (left, right) in boundary_pairs:
            continue
        pairs.append(
            {
                "layer_left": left,
                "layer_right": right,
                "direction": f"{left_kind}->{right_kind}",
                "left_kind": left_kind,
                "right_kind": right_kind,
            }
        )
    return pairs


def _choose_non_boundary_control(
    non_boundary_pairs: list[dict[str, Any]],
    *,
    matched_boundary_direction: str,
) -> dict[str, Any]:
    if not non_boundary_pairs:
        raise ValueError("architecture map has no adjacent non-boundary controls")
    wanted_left = matched_boundary_direction.split("->", maxsplit=1)[0]
    for pair in non_boundary_pairs:
        if pair["left_kind"] == wanted_left:
            return pair
    return non_boundary_pairs[0]


def _build_ssq_lr_rows(
    *,
    prompts: list[dict[str, Any]],
    model_map: dict[str, Any],
) -> list[dict[str, Any]]:
    model_id = str(model_map["model_id"])
    state_tensor_kind = _ssq_state_tensor_kind(model_map)
    layer_kinds = [str(kind) for kind in model_map.get("layer_kinds", [])]
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        for layer in _ssm_layers(model_map):
            layer_kind = layer_kinds[layer] if layer < len(layer_kinds) else "ssm"
            for bucket in SSQ_BUCKETS:
                rows.append(
                    {
                        "project": "ssq_lr",
                        "gate": "S1",
                        "model_id": model_id,
                        "architecture_map_hash": _architecture_hash(model_map),
                        "prompt_id": prompt_id,
                        "layer": layer,
                        "layer_kind": layer_kind,
                        "position_bucket": bucket,
                        "state_tensor_kind": state_tensor_kind,
                        "control_type": "bf16_no_quant",
                        "tensor_name": f"ssq_lr/{model_id}/{prompt_id}/layer_{layer}/{bucket}",
                        "capture": "recurrent SSM/Mamba2 state after state update",
                    }
                )
    return rows


def _build_horn_rows(
    *,
    prompts: list[dict[str, Any]],
    model_map: dict[str, Any],
) -> list[dict[str, Any]]:
    model_id = str(model_map["model_id"])
    non_boundary_pairs = _non_boundary_pairs(model_map)
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        prompt_cluster_id = str(prompt.get("prompt_cluster_id") or prompt.get("task") or prompt_id)
        for boundary in model_map.get("boundaries", []):
            direction = str(boundary["direction"])
            base = {
                "project": "horn",
                "gate": "H1a/H1",
                "model_id": model_id,
                "architecture_map_hash": _architecture_hash(model_map),
                "prompt_id": prompt_id,
                "prompt_cluster_id": prompt_cluster_id,
                "boundary_index": int(boundary["boundary_index"]),
                "layer_left": int(boundary["left_layer"]),
                "layer_right": int(boundary["right_layer"]),
                "pre_norm_position": "post_left_layer_norm_or_residual_output",
                "post_norm_position": "pre_right_layer_input",
            }
            rows.append(
                {
                    **base,
                    "control_type": "boundary",
                    "direction": direction,
                    "matched_boundary_direction": direction,
                    "tensor_name": (
                        f"horn/{model_id}/{prompt_id}/boundary_{boundary['boundary_index']}/observed"
                    ),
                    "capture": "activation crossing observed attention/SSM boundary",
                }
            )
            flipped = "ssm->attention" if direction == "attention->ssm" else "attention->ssm"
            rows.append(
                {
                    **base,
                    "control_type": "permuted_direction",
                    "direction": flipped,
                    "matched_boundary_direction": flipped,
                    "tensor_alias_of": (
                        f"horn/{model_id}/{prompt_id}/boundary_{boundary['boundary_index']}/observed"
                    ),
                    "tensor_name": (
                        f"horn/{model_id}/{prompt_id}/boundary_{boundary['boundary_index']}/permuted_label"
                    ),
                    "capture": "reuse observed boundary tensor; flip only direction label in metadata",
                }
            )
            pair = _choose_non_boundary_control(
                non_boundary_pairs,
                matched_boundary_direction=direction,
            )
            rows.append(
                {
                    "project": "horn",
                    "gate": "H1a/H1",
                    "model_id": model_id,
                    "architecture_map_hash": _architecture_hash(model_map),
                    "prompt_id": prompt_id,
                    "boundary_index": int(boundary["boundary_index"]),
                    "layer_left": int(pair["layer_left"]),
                    "layer_right": int(pair["layer_right"]),
                    "direction": pair["direction"],
                    "matched_boundary_direction": direction,
                    "pre_norm_position": "matched_non_boundary_left_output",
                    "post_norm_position": "matched_non_boundary_right_input",
                    "control_type": "non_boundary",
                    "tensor_name": (
                        f"horn/{model_id}/{prompt_id}/non_boundary_"
                        f"{pair['layer_left']}_{pair['layer_right']}/boundary_{boundary['boundary_index']}"
                    ),
                    "capture": "activation crossing adjacent non-boundary control pair matched to one observed boundary",
                }
            )
    return rows


def _build_hbsm_rows(
    *,
    prompts: list[dict[str, Any]],
    model_map: dict[str, Any],
) -> list[dict[str, Any]]:
    model_id = str(model_map["model_id"])
    boundary_layers = {
        int(boundary["left_layer"])
        for boundary in model_map.get("boundaries", [])
    } | {
        int(boundary["right_layer"])
        for boundary in model_map.get("boundaries", [])
    }
    rows: list[dict[str, Any]] = []
    layer_kinds = [str(kind) for kind in model_map.get("layer_kinds", [])]
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        for layer, kind in enumerate(layer_kinds):
            rows.append(
                {
                    "project": "hbsm",
                    "gate": "B1",
                    "model_id": model_id,
                    "architecture_map_hash": _architecture_hash(model_map),
                    "prompt_id": prompt_id,
                    "layer": layer,
                    "layer_kind": kind,
                    "boundary_flag": layer in boundary_layers,
                    "precision_perturbation": "fp4_sim_weight_or_activation_perturbation",
                    "train_test_split": "train" if layer % 2 == 0 else "test",
                    "control_type": "boundary_only",
                    "capture": "forward sensitivity row with kl_or_nll_drift and cheap predictor fields",
                }
            )
    for control_type in (
        "perturbation_off",
        "random_flags",
        "layer_index",
        "parameter_count_norm",
        "kl_lens_rank",
        "activation_outlier",
    ):
        for layer, kind in enumerate(layer_kinds):
            rows.append(
                {
                    "project": "hbsm",
                    "gate": "B1",
                    "model_id": model_id,
                    "architecture_map_hash": _architecture_hash(model_map),
                    "prompt_id": f"control_{control_type}",
                    "layer": layer,
                    "layer_kind": kind,
                    "boundary_flag": layer in boundary_layers,
                    "precision_perturbation": control_type,
                    "train_test_split": "control",
                    "control_type": control_type,
                    "capture": "layer-aligned B1 comparator/control row required before packet promotion",
                }
            )
    return rows


def build_trace_plan(
    *,
    prompts_path: Path = DEFAULT_PROMPTS,
    architecture_maps_path: Path = DEFAULT_ARCHITECTURE_MAPS,
) -> dict[str, Any]:
    prompts = _load_prompts(prompts_path)
    maps = _load_architecture_maps(architecture_maps_path)
    project_rows = {"ssq_lr": [], "horn": [], "hbsm": []}
    for model_map in maps:
        project_rows["ssq_lr"].extend(_build_ssq_lr_rows(prompts=prompts, model_map=model_map))
        project_rows["horn"].extend(_build_horn_rows(prompts=prompts, model_map=model_map))
        project_rows["hbsm"].extend(_build_hbsm_rows(prompts=prompts, model_map=model_map))
    return {
        "config": {
            "prompt_source": str(prompts_path),
            "prompt_count": len(prompts),
            "prompt_ids_hash": f"sha256:{_sha256(prompts_path)}",
            "architecture_maps": str(architecture_maps_path),
            "architecture_maps_hash": f"sha256:{_sha256(architecture_maps_path)}",
            "model_ids": [str(row["model_id"]) for row in maps],
            "decision": "TRACE_PLAN_READY_NOT_MODEL_EVIDENCE",
            "claim_boundary": ["trace-plan-only", "not model evidence", "not GPU evidence"],
        },
        "rows": project_rows,
    }


def write_trace_plan(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prompts_path: Path = DEFAULT_PROMPTS,
    architecture_maps_path: Path = DEFAULT_ARCHITECTURE_MAPS,
) -> dict[str, Any]:
    plan = build_trace_plan(prompts_path=prompts_path, architecture_maps_path=architecture_maps_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts = {}
    trace_plan_hashes = {}
    for project, rows in plan["rows"].items():
        row_counts[project] = len(rows)
        project_path = output_dir / f"{project}_trace_plan.jsonl"
        project_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        trace_plan_hashes[project] = f"sha256:{_sha256(project_path)}"
    config = dict(plan["config"])
    config["trace_plan_hashes"] = trace_plan_hashes
    summary = {
        "surface": "hybrid_trace_collection_plan",
        "decision": "TRACE_PLAN_READY_NOT_MODEL_EVIDENCE",
        "row_counts": row_counts,
        "trace_plan_hashes": trace_plan_hashes,
        "model_count": len(plan["config"]["model_ids"]),
        "prompt_count": plan["config"]["prompt_count"],
        "claim_boundary": plan["config"]["claim_boundary"],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "decision.md").write_text(
        "# Hybrid Trace Plan Decision\n\n"
        "`TRACE_PLAN_READY_NOT_MODEL_EVIDENCE`\n\n"
        "This artifact enumerates required SSQ-LR, HORN, and HBSM trace rows. "
        "It is not model evidence and contains no activations, SSM state, quality "
        "metrics, latency, or GPU evidence.\n"
    )
    lines = [
        "# Hybrid Trace Collection Plan",
        "",
        "This artifact enumerates the next admissible Mac/local trace captures for SSQ-LR, HORN, and HBSM.",
        "It is not model evidence; it is an execution checklist for producing real packets.",
        "",
        "| Project | Planned rows | Trace-plan hash |",
        "|---|---:|---|",
    ]
    for project, count in sorted(row_counts.items()):
        lines.append(f"| `{project}` | {count} | `{trace_plan_hashes[project][:19]}...` |")
    lines.extend(
        [
            "",
            "## Use",
            "",
            "Use these JSONL files to populate `ssq_lr_entries`, `horn_entries`, or HBSM row packets,",
            "then run `experimental.shared.hybrid_trace_packet_builder` and `check_gate_packet --mode real`.",
            "",
            "## Claim Boundary",
            "",
            "A complete trace plan only reduces operational ambiguity. It cannot promote any gate.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--architecture-maps", type=Path, default=DEFAULT_ARCHITECTURE_MAPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = write_trace_plan(
        output_dir=args.output_dir,
        prompts_path=args.prompts,
        architecture_maps_path=args.architecture_maps,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
