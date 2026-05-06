"""Synthetic HORN H1 gate for directional boundary metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experimental.shared.sensitivity_metrics import kurtosis, max_abs


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/horn/phase2/results/horn_synthetic_h1"


def _boundary_tensor(generator: torch.Generator, *, scale: float, outlier_scale: float) -> torch.Tensor:
    tensor = scale * torch.randn(48, 128, generator=generator)
    tensor[:, :8] += outlier_scale * torch.randn(48, 8, generator=generator)
    return tensor


def run_gate(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for layer in range(8):
        direction = "attention->ssm" if layer % 2 == 0 else "ssm->attention"
        if direction == "attention->ssm":
            activation = _boundary_tensor(generator, scale=0.7, outlier_scale=0.8)
        else:
            activation = _boundary_tensor(generator, scale=0.7, outlier_scale=4.0)
        rows.append(
            {
                "layer": layer,
                "direction": direction,
                "max_abs": max_abs(activation),
                "kurtosis": kurtosis(activation),
            }
        )

    grouped: dict[str, list[dict[str, float | str | int]]] = {"attention->ssm": [], "ssm->attention": []}
    for row in rows:
        grouped[str(row["direction"])].append(row)
    attn_to_ssm_max = torch.tensor([float(row["max_abs"]) for row in grouped["attention->ssm"]]).median().item()
    ssm_to_attn_max = torch.tensor([float(row["max_abs"]) for row in grouped["ssm->attention"]]).median().item()
    attn_to_ssm_kurt = torch.tensor([float(row["kurtosis"]) for row in grouped["attention->ssm"]]).median().item()
    ssm_to_attn_kurt = torch.tensor([float(row["kurtosis"]) for row in grouped["ssm->attention"]]).median().item()
    max_ratio = ssm_to_attn_max / attn_to_ssm_max
    kurtosis_ratio = ssm_to_attn_kurt / attn_to_ssm_kurt
    decision = "SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT" if max_ratio >= 3.0 or kurtosis_ratio >= 2.0 else "SYNTHETIC_FAIL"

    summary = {
        "seed": seed,
        "surface": "synthetic_directional_boundary_outliers",
        "decision": decision,
        "rows": rows,
        "ssm_to_attention_over_attention_to_ssm_max_ratio": max_ratio,
        "ssm_to_attention_over_attention_to_ssm_kurtosis_ratio": kurtosis_ratio,
        "claim_boundary": ["synthetic-only", "not model evidence", "not GPU evidence"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "config.json").write_text(json.dumps({"seed": seed, "layers": 8, "samples": 48, "channels": 128}, indent=2) + "\n")
    (output_dir / "decision.md").write_text(
        f"# HORN Synthetic H1 Decision\n\n`{decision}`\n\n"
        f"- max ratio: `{max_ratio:.3f}`\n"
        f"- kurtosis ratio: `{kurtosis_ratio:.3f}`\n\n"
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
