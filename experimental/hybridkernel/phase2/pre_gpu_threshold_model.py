"""Pre-GPU threshold model for HybridKernel."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/hybridkernel/phase2"

BOUNDARY_OVERHEAD_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]
RECOVERY_RATIOS = [0.30, 0.60, 0.80]


def _load_rows() -> list[dict[str, object]]:
    return json.loads((OUT_DIR / "architecture_map.json").read_text(encoding="utf-8"))


def _run() -> dict[str, object]:
    rows = []
    for row in _load_rows():
        stream_fraction = float(row["activation_stream_fraction"])
        for overhead in BOUNDARY_OVERHEAD_RATIOS:
            for recovery in RECOVERY_RATIOS:
                proxy_gain = stream_fraction * overhead * recovery
                rows.append(
                    {
                        "config": row["config"],
                        "stream_fraction": stream_fraction,
                        "boundary_overhead_ratio": overhead,
                        "recovery_ratio": recovery,
                        "proxy_end_to_end_gain": proxy_gain,
                        "clears_3pct_gate": proxy_gain >= 0.03,
                    }
                )
    minimum_overhead_for_3pct_at_60pct = {
        str(row["config"]): 0.03 / (float(row["activation_stream_fraction"]) * 0.60)
        for row in _load_rows()
    }
    return {
        "rows": rows,
        "minimum_boundary_overhead_ratio_for_3pct_gain_at_60pct_recovery": minimum_overhead_for_3pct_at_60pct,
        "status": _status(minimum_overhead_for_3pct_at_60pct),
    }


def _status(minimum_overheads: dict[str, float]) -> str:
    best = min(minimum_overheads.values())
    if best <= 0.10:
        return "ALIVE; one config clears pre-GPU threshold if >=10% of boundary stream is avoidable."
    if best <= 0.25:
        return "WEAKLY ALIVE; needs unusually high avoidable boundary overhead before GPU work is justified."
    return "WEAKENED; required avoidable overhead is too high without native profiler evidence."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# HybridKernel Pre-GPU Threshold Model",
        "",
        f"Status: **{result['status']}**",
        "",
        "This model asks what fraction of the layer-boundary activation stream must be actual avoidable overhead before a fused boundary kernel could plausibly clear a 3% end-to-end gate.",
        "It is not a GPU benchmark and not a latency result.",
        "",
        "## Required Avoidable Boundary Fraction",
        "",
        "Assuming 60% recovery of truly avoidable boundary overhead:",
        "",
        "| Config | Required avoidable boundary fraction for 3% proxy gain |",
        "|---|---:|",
    ]
    for config, value in result["minimum_boundary_overhead_ratio_for_3pct_gain_at_60pct_recovery"].items():
        lines.append(f"| {config} | {value:.1%} |")
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "| Config | Boundary stream fraction | Avoidable overhead | Recovery | Proxy gain | Clears 3%? |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["rows"]:
        if row["recovery_ratio"] != 0.60:
            continue
        lines.append(
            "| {config} | {stream_fraction:.1%} | {boundary_overhead_ratio:.0%} | {recovery_ratio:.0%} | {proxy_end_to_end_gain:.1%} | {clears} |".format(
                clears="yes" if row["clears_3pct_gate"] else "no",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This weakens Mac-only implementation. Granite requires roughly 25% of boundary traffic to be genuinely avoidable at 60% recovery to clear a 3% proxy gain.",
            "Qwen3-Next is closer, requiring roughly 10.4%, but its linear-attention/Gated-DeltaNet boundary is less directly matched to the Granite Mamba2 fusion idea.",
            "Before NVIDIA GPU work, the only useful local action is to prepare the profiler runbook and exact counters to verify avoidable boundary overhead.",
        ]
    )
    (OUT_DIR / "pre_gpu_threshold_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = _run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pre_gpu_threshold_model.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps(result["minimum_boundary_overhead_ratio_for_3pct_gain_at_60pct_recovery"], indent=2))


if __name__ == "__main__":
    main()
