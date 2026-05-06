"""Synthetic HBSM B1/B2 gate for sensitivity and cheap predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experimental.shared.fp4_simulator import simulate_mxfp4_e2m1
from experimental.shared.sensitivity_metrics import kurtosis, rel_l2, spearman_rank_correlation


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/hbsm/phase2/results/hbsm_synthetic_b1"


def run_gate(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    sensitivities = []
    predictors = []
    for layer in range(12):
        boundary_flag = layer in {2, 5, 8, 11}
        weight = torch.randn(96, 96, generator=generator)
        if boundary_flag:
            weight[:8] *= 5.0
            weight[:, :4] *= 2.5
        cheap_predictor = kurtosis(weight)
        quantized = simulate_mxfp4_e2m1(weight, block_size=32).dequantized
        sensitivity = rel_l2(weight, quantized) * (1.0 + 0.015 * cheap_predictor) * (1.3 if boundary_flag else 1.0)
        rows.append(
            {
                "layer": layer,
                "boundary_flag": boundary_flag,
                "cheap_predictor_kurtosis": cheap_predictor,
                "simulated_sensitivity": sensitivity,
            }
        )
        sensitivities.append(sensitivity)
        predictors.append(cheap_predictor)

    sensitivity_tensor = torch.tensor(sensitivities)
    predictor_tensor = torch.tensor(predictors)
    rho = spearman_rank_correlation(predictor_tensor, sensitivity_tensor)
    top_decile_cut = torch.quantile(sensitivity_tensor, 0.90).item()
    boundary_top_decile_hits = sum(
        1 for row in rows if bool(row["boundary_flag"]) and float(row["simulated_sensitivity"]) >= top_decile_cut
    )
    decision = "SYNTHETIC_PASS_REAL_LAYER_SENSITIVITY_NEXT" if rho >= 0.6 and boundary_top_decile_hits >= 1 else "SYNTHETIC_FAIL"

    summary = {
        "seed": seed,
        "surface": "synthetic_layer_sensitivity",
        "decision": decision,
        "rows": rows,
        "spearman_rho_kurtosis_vs_sensitivity": rho,
        "boundary_top_decile_hits": boundary_top_decile_hits,
        "claim_boundary": ["synthetic-only", "not model evidence", "not GPU evidence"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "config.json").write_text(json.dumps({"seed": seed, "layers": 12, "shape": [96, 96]}, indent=2) + "\n")
    (output_dir / "decision.md").write_text(
        f"# HBSM Synthetic B1/B2 Decision\n\n`{decision}`\n\n"
        f"- Spearman rho: `{rho:.3f}`\n"
        f"- boundary top-decile hits: `{boundary_top_decile_hits}`\n\n"
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
