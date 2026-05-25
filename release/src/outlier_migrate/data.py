"""Configuration and frozen paper-claim data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PAPER_CLAIMS: dict[str, dict[str, Any]] = {
    "set_leaving_granite_small": {"value": 0.566234756098, "tolerance": 1e-12},
    "set_leaving_nemotron": {"value": 0.533713200380, "tolerance": 1e-12},
    "set_leaving_deepseek": {"value": 0.670572916667, "tolerance": 1e-12},
    "set_leaving_falcon": {"value": 0.673611111111, "tolerance": 1e-12},
    "phase4_static_union_median": {"value": 0.0, "tolerance": 1e-12},
    "phase4_no_gap_fraction": {"value": 0.375, "tolerance": 1e-12},
    "m2_median": {"value": -0.866837313391, "tolerance": 1e-12},
    "m2_random_margin": {"value": -0.667548290168, "tolerance": 1e-12},
    "m10_median": {"value": 0.234447927834, "tolerance": 1e-12},
    "m10_random_margin": {"value": -0.761487459046, "tolerance": 1e-12},
    "m11_median": {"value": 0.048299284138, "tolerance": 1e-12},
    "m18_activation_k_median": {"value": -0.343590844126, "tolerance": 1e-12},
    "decdec_median": {"value": -0.070034781258, "tolerance": 1e-12},
    "m11b_granite_top5": {"value": 0.449284091125, "tolerance": 1e-12},
    "m11b_nemotron_top5": {"value": 0.456736183270, "tolerance": 1e-12},
    "m11b_nemotron_top10": {"value": 0.814739798903, "tolerance": 1e-12},
    "m26_median": {"value": 0.177615807648, "tolerance": 1e-12},
    "paroquant_median": {"value": 0.753776848891, "tolerance": 1e-12},
    "kl_static_mean": {"value": 0.150302750276, "tolerance": 1e-12},
    "kl_decdec_mean": {"value": 0.132701884446, "tolerance": 1e-12},
    "kl_m11_mean": {"value": 0.131406585383, "tolerance": 1e-12},
    "fft_entropy": {"value": 0.85, "tolerance": 0.02},
    "fft_autocorr_tokens": {"value": 100, "tolerance": 5},
    "component_granite_attention": {"value": 0.557165, "tolerance": 1e-6},
    "component_granite_ssm": {"value": 0.567243, "tolerance": 1e-6},
    "component_nemotron_attention": {"value": 0.563014, "tolerance": 1e-6},
    "component_nemotron_moe": {"value": 0.526503, "tolerance": 1e-6},
    "component_nemotron_ssm": {"value": 0.533280, "tolerance": 1e-6},
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML reproduction config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return data


def claim_subset(keys: list[str]) -> dict[str, dict[str, Any]]:
    """Return selected frozen paper claims."""

    missing = [key for key in keys if key not in PAPER_CLAIMS]
    if missing:
        raise KeyError(f"Unknown paper claims: {missing}")
    return {key: PAPER_CLAIMS[key] for key in keys}


def verify_claims(claims: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Verify that observed and expected values are within tolerance."""

    results: dict[str, bool] = {}
    for key, claim in claims.items():
        observed = float(claim.get("observed", claim["value"]))
        expected = float(claim["value"])
        tolerance = float(claim["tolerance"])
        results[key] = abs(observed - expected) <= tolerance
    return results


def build_reproduction_payload(args: Any, claims: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a dry-run or fast-verification payload.

    Full GPU reproduction is intentionally not simulated by the release
    scripts. Running without ``--dry-run`` or ``--fast-verify`` fails loudly so
    reviewers cannot mistake frozen-claim validation for a full model run.
    """

    if getattr(args, "dry_run", False):
        return {
            "mode": "dry_run",
            "config_valid": True,
            "full_reproduction_available": False,
        }
    if not getattr(args, "fast_verify", False):
        raise RuntimeError(
            "Full-fidelity GPU reproduction is not implemented in release/. "
            "Use --fast-verify for the verified claim replay, or use the "
            "archived experimental packet runners for full model execution."
        )
    return {
        "mode": "fast_verify",
        "claims": claims,
        "verified": verify_claims(claims),
    }


def write_result(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON result file with deterministic formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
