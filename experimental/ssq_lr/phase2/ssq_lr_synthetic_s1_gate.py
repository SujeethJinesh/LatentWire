"""Synthetic SSQ-LR S1 gate.

This no-download gate validates the SSM-state heterogeneity metrics and artifact
packet shape before running on real hybrid-model state dumps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experimental.shared.sensitivity_metrics import kurtosis, max_abs


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1"


def _make_states(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layers, samples, channels = 6, 64, 128
    base = torch.randn(layers, samples, channels, generator=generator)
    outlier = torch.zeros_like(base)
    outlier[:, :, :8] = torch.randn(layers, samples, 8, generator=generator) * 4.0
    return {
        "early": 0.45 * base,
        "middle": 0.85 * base + 0.35 * outlier,
        "late": 1.30 * base + outlier,
    }


def run_gate(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    states = _make_states(seed)
    rows = []
    for position, state in states.items():
        rows.append(
            {
                "position": position,
                "max_abs": max_abs(state),
                "std": float(torch.std(state.float())),
                "kurtosis": kurtosis(state),
            }
        )

    early = next(row for row in rows if row["position"] == "early")
    late = next(row for row in rows if row["position"] == "late")
    max_abs_ratio = late["max_abs"] / early["max_abs"]
    std_ratio = late["std"] / early["std"]
    kurtosis_ratio = late["kurtosis"] / early["kurtosis"]
    decision = "SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT" if max_abs_ratio >= 2.0 or std_ratio >= 2.0 else "SYNTHETIC_FAIL"

    summary = {
        "seed": seed,
        "surface": "synthetic_ssm_state_heterogeneity",
        "decision": decision,
        "rows": rows,
        "max_abs_ratio_late_vs_early": max_abs_ratio,
        "std_ratio_late_vs_early": std_ratio,
        "kurtosis_ratio_late_vs_early": kurtosis_ratio,
        "claim_boundary": ["synthetic-only", "not model evidence", "not GPU evidence"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "config.json").write_text(json.dumps({"seed": seed, "layers": 6, "samples": 64, "channels": 128}, indent=2) + "\n")
    (output_dir / "decision.md").write_text(
        f"# SSQ-LR Synthetic S1 Decision\n\n`{decision}`\n\n"
        f"- late/early max-abs ratio: `{max_abs_ratio:.3f}`\n"
        f"- late/early std ratio: `{std_ratio:.3f}`\n"
        f"- late/early kurtosis ratio: `{kurtosis_ratio:.3f}`\n\n"
        "Synthetic-only: this validates artifact mechanics and does not promote the branch.\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_gate(seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
