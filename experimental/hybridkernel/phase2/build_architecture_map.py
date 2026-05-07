"""Build a Mac-local architecture map for HybridKernel Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "experimental/hybridkernel/phase0/configs"
OUT_DIR = ROOT / "experimental/hybridkernel/phase2"


def _read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_layer_types(config: dict[str, Any]) -> list[str]:
    if config.get("layer_types"):
        return list(config["layer_types"])
    n_layers = int(config["num_hidden_layers"])
    interval = int(config.get("full_attention_interval", 0) or 0)
    if config.get("model_type") == "qwen3_next" and interval > 0:
        return ["attention" if (idx + 1) % interval == 0 else "linear" for idx in range(n_layers)]
    pattern = config.get("hybrid_override_pattern")
    if config.get("model_type") == "nemotron_h" and isinstance(pattern, str):
        mapping = {
            "M": "mamba",
            "*": "attention",
            "-": "mlp",
            "E": "moe",
        }
        return [mapping.get(char, "unknown") for char in pattern]
    return ["unknown"] * n_layers


def _transitions(layer_types: list[str]) -> int:
    return sum(1 for left, right in zip(layer_types, layer_types[1:]) if left != right)


def _boundary_indices(layer_types: list[str], config: dict[str, Any]) -> list[int]:
    if config.get("model_type") == "nemotron_h":
        return [
            idx
            for idx, (left, right) in enumerate(zip(layer_types, layer_types[1:]))
            if left != right and (left == "attention" or right == "attention")
        ]
    return [
        idx
        for idx, (left, right) in enumerate(zip(layer_types, layer_types[1:]))
        if left != right
    ]


def _model_id_from_config_path(config_path: Path) -> str:
    stem = config_path.name.removesuffix(".config.json")
    if stem.startswith("ibm-granite-"):
        return "ibm-granite/" + stem.replace("ibm-granite-", "granite-", 1)
    if stem == "qwen3-next-80b-a3b-instruct":
        return "Qwen/Qwen3-Next-80B-A3B-Instruct"
    if stem == "nvidia-nemotron-nano-9b-v2":
        return "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    return stem


def _row(config_path: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    layers = _infer_layer_types(config)
    hidden_size = int(config["hidden_size"])
    dtype_bytes = 2 if str(config.get("torch_dtype", "")).lower() in {"bfloat16", "float16"} else 4
    all_transitions = _transitions(layers)
    boundaries = _boundary_indices(layers, config)
    read_write_bytes_per_transition = hidden_size * dtype_bytes * 2
    transition_bytes_per_token = len(boundaries) * read_write_bytes_per_transition
    all_layer_stream_bytes = len(layers) * read_write_bytes_per_transition
    activation_fraction = transition_bytes_per_token / all_layer_stream_bytes if all_layer_stream_bytes else 0.0
    recovered_fraction_if_60pct = activation_fraction * 0.60
    return {
        "model": _model_id_from_config_path(config_path),
        "config": config_path.name,
        "architecture": ",".join(config.get("architectures", [])),
        "model_type": config.get("model_type", ""),
        "hidden_size": hidden_size,
        "num_layers": len(layers),
        "layer_types": layers,
        "attention_like_layers": sum(1 for layer in layers if layer == "attention"),
        "ssm_or_linear_layers": sum(1 for layer in layers if layer in {"mamba", "linear"}),
        "mlp_or_moe_layers": sum(1 for layer in layers if layer in {"mlp", "moe"}),
        "all_layer_type_transitions": all_transitions,
        "boundary_indices": boundaries,
        "boundary_transitions": len(boundaries),
        "dtype_bytes": dtype_bytes,
        "transition_bytes_per_token": transition_bytes_per_token,
        "all_layer_stream_bytes_per_token": all_layer_stream_bytes,
        "activation_stream_fraction": activation_fraction,
        "recovered_fraction_if_60pct": recovered_fraction_if_60pct,
        "phase2_decision": "pass_theoretical_gate" if recovered_fraction_if_60pct >= 0.03 else "below_3pct_gate",
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# HybridKernel Phase 2 Architecture Map",
        "",
        "This Mac-local gate estimates whether layer-type boundary movement is large enough to justify GPU profiling later.",
        "It is an activation-stream upper-bound calculation, not an end-to-end latency measurement.",
        "",
        "| Config | Layers | Attn | SSM/linear | Boundaries | Boundary bytes/token | Stream fraction | 60% recovered | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {num_layers} | {attention_like_layers} | {ssm_or_linear_layers} | "
            "{boundary_transitions} | {transition_bytes_per_token} | {activation_stream_fraction:.1%} | "
            "{recovered_fraction_if_60pct:.1%} | {phase2_decision} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Granite 4.0 H Tiny/Small, Qwen3-Next, and the preregistered Nemotron Nano 9B v2 replacement clear the >=3% theoretical activation-stream gate under this upper-bound model.",
            "This keeps HybridKernel alive only for native NVIDIA/vLLM profiling; the source/control audits and integration map are complete enough for Mac-local work.",
            "This map does not prove an end-to-end GPU speedup, and the next gate is to determine whether the apparent boundary cost survives actual vLLM/vendor implementation details in server-side Nsight traces.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_row(path) for path in sorted(CONFIG_DIR.glob("*.config.json"))]
    (OUT_DIR / "architecture_map.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_markdown(rows, OUT_DIR / "architecture_map.md")
    print(f"Wrote {OUT_DIR / 'architecture_map.md'}")


if __name__ == "__main__":
    main()
