"""Build explicit hybrid layer/boundary maps from local model configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experimental.shared.boundary_inspector import LayerKind, boundaries_from_named_layers


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "experimental/hybridkernel/phase0/configs"
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_architecture_maps_20260506"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _infer_layer_kinds(config: dict[str, Any]) -> list[LayerKind]:
    layer_types = config.get("layer_types")
    if layer_types:
        kinds: list[LayerKind] = []
        for layer_type in layer_types:
            if layer_type == "attention":
                kinds.append(LayerKind.ATTENTION)
            elif layer_type in {"mamba", "ssm", "linear", "gated_delta", "gateddeltanet"}:
                kinds.append(LayerKind.SSM)
            elif layer_type in {"mlp", "ffn"}:
                kinds.append(LayerKind.MLP)
            else:
                kinds.append(LayerKind.OTHER)
        return kinds

    n_layers = int(config.get("num_hidden_layers", 0))
    interval = int(config.get("full_attention_interval", 0) or 0)
    if config.get("model_type") == "qwen3_next" and interval > 0:
        return [LayerKind.ATTENTION if (idx + 1) % interval == 0 else LayerKind.SSM for idx in range(n_layers)]
    return [LayerKind.OTHER for _ in range(n_layers)]


def build_map(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    kinds = _infer_layer_kinds(config)
    named_layers = [(f"layers.{idx}", kind) for idx, kind in enumerate(kinds)]
    boundaries = boundaries_from_named_layers(named_layers)
    direction_counts: dict[str, int] = {}
    for boundary in boundaries:
        direction_counts[boundary.direction] = direction_counts.get(boundary.direction, 0) + 1

    return {
        "config": config_path.name,
        "config_sha256": _sha256(config_path),
        "model_id": config_path.name.removesuffix(".config.json"),
        "architecture": ",".join(config.get("architectures", [])),
        "model_type": config.get("model_type", ""),
        "hidden_size": int(config.get("hidden_size", 0)),
        "num_hidden_layers": len(kinds),
        "layer_kinds": [kind.value for kind in kinds],
        "boundary_count": len(boundaries),
        "direction_counts": direction_counts,
        "boundaries": [
            {
                "boundary_index": index,
                "left_layer": int(boundary.left_name.split(".")[-1]),
                "right_layer": int(boundary.right_name.split(".")[-1]),
                "direction": boundary.direction,
                "left_kind": boundary.left.value,
                "right_kind": boundary.right.value,
            }
            for index, boundary in enumerate(boundaries)
        ],
    }


def _packet_rows(maps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_map in maps:
        layers = list(model_map["layer_kinds"])
        model_id = str(model_map["model_id"])
        config_hash = str(model_map["config_sha256"])
        for layer_index, kind in enumerate(layers):
            rows.append(
                {
                    "row_type": "layer",
                    "model_id": model_id,
                    "model_revision": config_hash,
                    "layer_index": layer_index,
                    "module_name": f"layers.{layer_index}",
                    "class_name": f"config::{kind}",
                    "module_kind": kind,
                    "has_recurrent_state": kind == LayerKind.SSM.value,
                    "boundary_index": None,
                    "direction": None,
                    "left_module": None,
                    "right_module": None,
                    "pre_norm_position": "unknown_config_only",
                    "post_norm_position": "unknown_config_only",
                    "map_source": "local_config",
                    "confidence": "config_derived",
                    "notes": "layer kind inferred from local config only",
                }
            )
        boundary_pairs = {(row["left_layer"], row["right_layer"]) for row in model_map["boundaries"]}
        for boundary in model_map["boundaries"]:
            rows.append(
                {
                    "row_type": "boundary",
                    "model_id": model_id,
                    "model_revision": config_hash,
                    "layer_index": boundary["left_layer"],
                    "module_name": f"layers.{boundary['left_layer']}->{boundary['right_layer']}",
                    "class_name": "config::boundary",
                    "module_kind": "boundary",
                    "has_recurrent_state": boundary["left_kind"] == LayerKind.SSM.value
                    or boundary["right_kind"] == LayerKind.SSM.value,
                    "boundary_index": boundary["boundary_index"],
                    "direction": boundary["direction"],
                    "left_module": f"layers.{boundary['left_layer']}",
                    "right_module": f"layers.{boundary['right_layer']}",
                    "pre_norm_position": "unknown_config_only",
                    "post_norm_position": "unknown_config_only",
                    "map_source": "local_config",
                    "confidence": "config_derived",
                    "notes": "admissible HORN/HBSM boundary id; activations not measured",
                }
            )
            rows.append(
                {
                    "row_type": "permuted_direction_control",
                    "model_id": model_id,
                    "model_revision": config_hash,
                    "layer_index": boundary["left_layer"],
                    "module_name": f"layers.{boundary['left_layer']}->{boundary['right_layer']}",
                    "class_name": "config::boundary_permutation",
                    "module_kind": "boundary",
                    "has_recurrent_state": boundary["left_kind"] == LayerKind.SSM.value
                    or boundary["right_kind"] == LayerKind.SSM.value,
                    "boundary_index": boundary["boundary_index"],
                    "direction": "attention->ssm"
                    if boundary["direction"] == "ssm->attention"
                    else "ssm->attention",
                    "left_module": f"layers.{boundary['left_layer']}",
                    "right_module": f"layers.{boundary['right_layer']}",
                    "pre_norm_position": "unknown_config_only",
                    "post_norm_position": "unknown_config_only",
                    "map_source": "local_config",
                    "confidence": "negative_control",
                    "notes": "direction-label permutation control; not a measured boundary",
                }
            )
        for left_index, (left_kind, right_kind) in enumerate(zip(layers, layers[1:])):
            if (left_index, left_index + 1) in boundary_pairs:
                continue
            rows.append(
                {
                    "row_type": "non_boundary_control",
                    "model_id": model_id,
                    "model_revision": config_hash,
                    "layer_index": left_index,
                    "module_name": f"layers.{left_index}->layers.{left_index + 1}",
                    "class_name": "config::non_boundary",
                    "module_kind": f"{left_kind}->{right_kind}",
                    "has_recurrent_state": left_kind == LayerKind.SSM.value or right_kind == LayerKind.SSM.value,
                    "boundary_index": None,
                    "direction": f"{left_kind}->{right_kind}",
                    "left_module": f"layers.{left_index}",
                    "right_module": f"layers.{left_index + 1}",
                    "pre_norm_position": "unknown_config_only",
                    "post_norm_position": "unknown_config_only",
                    "map_source": "local_config",
                    "confidence": "negative_control",
                    "notes": "adjacent non-boundary control; activations not measured",
                }
            )
    return rows


def write_maps(config_dir: Path = DEFAULT_CONFIG_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = [build_map(path) for path in sorted(config_dir.glob("*.config.json"))]
    rows = _packet_rows(maps)
    (output_dir / "architecture_maps.json").write_text(json.dumps(maps, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "config_dir": str(config_dir),
                "source_configs": [row["config"] for row in maps],
                "command": "python -m experimental.shared.hybrid_architecture_maps",
                "claim_boundary": ["config-only", "not model evidence", "not GPU evidence"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "surface": "config_only_hybrid_architecture_map",
                "decision": "CONFIG_ONLY_READY_FOR_TRACE_PACKET_PROVENANCE",
                "model_count": len(maps),
                "row_count": len(rows),
                "rows": [
                    {
                        "model_id": row["model_id"],
                        "row_type": row["row_type"],
                        "boundary_index": row["boundary_index"],
                        "direction": row["direction"],
                    }
                    for row in rows
                ],
                "claim_boundary": ["config-only", "not model evidence", "not GPU evidence"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "decision.md").write_text(
        "# Hybrid Architecture Map Decision\n\n"
        "`CONFIG_ONLY_READY_FOR_TRACE_PACKET_PROVENANCE`\n\n"
        "This packet supplies explicit layer kinds, boundary IDs, direction-label "
        "permutation controls, and non-boundary adjacent controls for future "
        "SSQ-LR/HORN/HBSM trace packets. It contains no activations or SSM state "
        "and cannot promote any branch by itself.\n"
    )
    lines = [
        "# Hybrid Architecture Maps",
        "",
        "Explicit layer and boundary maps for SSQ-LR, HORN, and HBSM real trace packets.",
        "This artifact is config-only: it does not contain activations, SSM state, quality metrics, or GPU evidence.",
        "",
        "| Model | Layers | Boundaries | Direction counts | Config hash |",
        "|---|---:|---:|---|---|",
    ]
    for row in maps:
        lines.append(
            "| {model_id} | {num_hidden_layers} | {boundary_count} | `{direction_counts}` | `{hash}` |".format(
                model_id=row["model_id"],
                num_hidden_layers=row["num_hidden_layers"],
                boundary_count=row["boundary_count"],
                direction_counts=json.dumps(row["direction_counts"], sort_keys=True),
                hash=str(row["config_sha256"])[:12],
            )
        )
    lines.extend(
        [
            "",
            "Rows are also written to `raw_rows.jsonl` with `layer`, `boundary`,",
            "`non_boundary_control`, and `permuted_direction_control` row types.",
            "",
            "## Claim Boundary",
            "",
            "These maps can validate boundary IDs and architecture provenance in future real trace packets.",
            "They cannot promote SSQ-LR, HORN, or HBSM without measured model states or activations.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    maps = write_maps(config_dir=args.config_dir, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "models": len(maps)}, sort_keys=True))


if __name__ == "__main__":
    main()
