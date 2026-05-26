#!/usr/bin/env python3
"""Recompute method-table BCa CIs and Holm-Bonferroni corrections."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PHASE9 = ROOT / "experimental/outlier_migrate/phase9"
RESULTS = PHASE9 / "results"
PREREG = PHASE9 / "preregister_om_stage1_e15_bca_holm.md"
SCHEMA_VERSION = "om_stage1_e15_bca_holm_v1"
DEFAULT_SEED = 20260615
DEFAULT_SAMPLES = 10_000

DEFAULT_PACKETS = {
    "m2_granite": RESULTS / "om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z",
    "m10_granite": RESULTS / "om_phase9_m10_granite_small_vac12_20260515T085800Z",
    "m11_granite": RESULTS / "om_phase9_m11_granite_small_vac12_20260516T010728Z",
    "m11b_granite": RESULTS / "om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z",
    "m11b_nemotron": RESULTS / "om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z",
    "m18_granite": RESULTS / "om_phase9_m18_granite_small_vac12_20260516T193500Z",
    "decdec_granite": RESULTS / "om_phase9_decdec_granite_small_vac12_20260517T141500Z",
    "m26_granite": RESULTS / "om_phase9_m26_granite_small_vac12_20260518T203000Z",
    "paroquant_granite": RESULTS / "om_paroquant_granite_small_20260520T1555Z",
}


class MissingInputError(RuntimeError):
    """Raised when exact per-trace recovery values are unavailable."""


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write stable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for a file."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def parse_packet(value: str) -> tuple[str, Path]:
    """Parse a LABEL=PATH packet argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("--packet entries must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("packet label must be non-empty")
    return label, Path(raw_path).expanduser()


def selected_regimes(filters: list[str]) -> dict[str, set[str]]:
    """Parse optional LABEL:REGIME filters."""
    parsed: dict[str, set[str]] = {}
    for item in filters:
        if ":" not in item:
            raise argparse.ArgumentTypeError("--regime entries must be LABEL:REGIME")
        label, regime = item.split(":", 1)
        parsed.setdefault(label, set()).add(regime)
    return parsed


def values_from_rows(packet_dir: Path) -> tuple[dict[str, list[float]], str]:
    """Extract exact per-trace recovery values from per_trace_metrics.json."""
    path = packet_dir / "per_trace_metrics.json"
    if not path.is_file():
        return {}, ""
    rows = load_json(path).get("traces", [])
    values: dict[str, list[float]] = {}
    for row in rows:
        recoveries = row.get("recoveries", {})
        if not isinstance(recoveries, dict):
            continue
        for regime, value in recoveries.items():
            if value is not None:
                values.setdefault(str(regime), []).append(float(value))
    return {key: val for key, val in values.items() if val}, "per_trace_metrics.json"


def values_from_bootstrap(packet_dir: Path) -> tuple[dict[str, list[float]], str]:
    """Extract preserved exact per-trace recovery values from bootstrap_ci.json."""
    path = packet_dir / "bootstrap_ci.json"
    if not path.is_file():
        return {}, ""
    results = load_json(path).get("results_by_regime", {})
    values: dict[str, list[float]] = {}
    for regime, summary in results.items():
        raw = summary.get("per_trace_recovery_included") if isinstance(summary, dict) else None
        if isinstance(raw, list) and raw:
            values[str(regime)] = [float(value) for value in raw]
    return values, "bootstrap_ci.json"


def packet_values(packet_dir: Path) -> tuple[dict[str, list[float]], str]:
    """Return exact per-trace recovery values and source file."""
    values, source = values_from_rows(packet_dir)
    if values:
        return values, source
    return values_from_bootstrap(packet_dir)


def percentile(sorted_values: list[float], q: float) -> float:
    """Return an interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    q = min(1.0, max(0.0, q))
    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low))


def bootstrap_medians(values: list[float], *, samples: int, seed: int) -> list[float]:
    """Return bootstrap medians for a fixed seed."""
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(float(median(sample)))
    return stats


def bca_interval(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    """Compute a two-sided 95% BCa interval for the median."""
    if len(values) < 2:
        raise MissingInputError("BCa median interval requires at least two per-trace values")
    observed = float(median(values))
    boot = bootstrap_medians(values, samples=samples, seed=seed)
    boot_sorted = sorted(boot)
    normal = NormalDist()
    less = sum(1 for stat in boot if stat < observed)
    prop = min(1.0 - 0.5 / samples, max(0.5 / samples, less / samples))
    z0 = normal.inv_cdf(prop)
    jack = [float(median(values[:idx] + values[idx + 1 :])) for idx in range(len(values))]
    jack_mean = sum(jack) / len(jack)
    numerator = sum((jack_mean - val) ** 3 for val in jack)
    denominator = 6.0 * (sum((jack_mean - val) ** 2 for val in jack) ** 1.5)
    warning = None
    acceleration = 0.0
    if denominator == 0.0:
        warning = "jackknife acceleration degenerate; used acceleration=0"
    else:
        acceleration = numerator / denominator

    def adjusted(alpha: float) -> float:
        z_alpha = normal.inv_cdf(alpha)
        denom = 1.0 - acceleration * (z0 + z_alpha)
        if denom == 0.0:
            return alpha
        return normal.cdf(z0 + (z0 + z_alpha) / denom)

    low_q = adjusted(0.025)
    high_q = adjusted(0.975)
    p_positive = (1 + sum(1 for stat in boot if stat <= 0.0)) / (samples + 1)
    return {
        "median_recovery": observed,
        "bca_ci95": {"ci95_low": percentile(boot_sorted, low_q), "ci95_high": percentile(boot_sorted, high_q)},
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "bca": {"z0": z0, "acceleration": acceleration, "low_quantile": low_q, "high_quantile": high_q},
        "one_sided_p_median_gt_0": p_positive,
        "warning": warning,
    }


def holm_bonferroni(tests: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni correction to one-sided p-values."""
    ordered = sorted(tests, key=lambda row: float(row["p_value"]))
    adjusted: list[dict[str, Any]] = []
    running_adj = 0.0
    stopped = False
    m = len(ordered)
    for rank, row in enumerate(ordered, start=1):
        multiplier = m - rank + 1
        p_value = float(row["p_value"])
        running_adj = max(running_adj, min(1.0, multiplier * p_value))
        threshold = alpha / multiplier
        reject = (not stopped) and p_value <= threshold
        if not reject:
            stopped = True
        item = dict(row)
        item.update({"holm_rank": rank, "holm_threshold": threshold, "holm_adjusted_p": running_adj, "holm_reject": reject})
        adjusted.append(item)
    return sorted(adjusted, key=lambda row: (row["packet"], row["regime"]))


def collect(packet_args: list[tuple[str, Path]], regime_filters: dict[str, set[str]], *, samples: int, seed: int) -> dict[str, Any]:
    """Collect per-trace values and compute E15 statistics."""
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    tests: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []

    for label, raw_path in packet_args:
        packet_dir = raw_path if raw_path.is_absolute() else (ROOT / raw_path)
        if not packet_dir.is_dir():
            missing.append({"packet": label, "path": str(packet_dir), "reason": "packet directory not found"})
            continue
        values_by_regime, source = packet_values(packet_dir)
        wanted = regime_filters.get(label, set(values_by_regime))
        inputs.append({"label": label, "path": str(packet_dir), "source": source or None, "available_regimes": sorted(values_by_regime)})
        for regime in sorted(wanted):
            values = values_by_regime.get(regime)
            if not values:
                missing.append({"packet": label, "path": str(packet_dir), "regime": regime, "reason": "missing exact per-trace recovery values"})
                continue
            try:
                stats = bca_interval(values, samples=samples, seed=seed)
            except MissingInputError as exc:
                missing.append({"packet": label, "path": str(packet_dir), "regime": regime, "reason": str(exc)})
                continue
            stats["included_trace_count"] = len(values)
            stats["per_trace_source"] = source
            results.setdefault(label, {})[regime] = stats
            tests.append({"packet": label, "regime": regime, "p_value": stats["one_sided_p_median_gt_0"]})

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": file_sha256(PREREG) if PREREG.is_file() else None,
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "test_family": "method_table_median_recovery_gt_0",
        "inputs": inputs,
        "results_by_packet": results,
        "holm_bonferroni": {"alpha": 0.05, "tests": holm_bonferroni(tests, alpha=0.05)},
        "missing_inputs": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", action="append", default=[], type=parse_packet, help="Additional LABEL=PATH packet input.")
    parser.add_argument("--no-default-packets", action="store_true", help="Use only packets passed with --packet.")
    parser.add_argument("--regime", action="append", default=[], help="Restrict a packet to LABEL:REGIME. May repeat.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned analysis without writing output.")
    parser.add_argument("--output", type=Path, default=RESULTS / f"om_stage1_e15_bca_holm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}" / "e15_bca_holm.json")
    args = parser.parse_args(argv)

    packets = [] if args.no_default_packets else list(DEFAULT_PACKETS.items())
    packets.extend(args.packet)
    if not packets:
        raise SystemExit("no packets specified")
    payload = collect(packets, selected_regimes(args.regime), samples=args.samples, seed=args.seed)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        write_json(args.output if args.output.is_absolute() else ROOT / args.output, payload)
        print(args.output)
    return 2 if payload["missing_inputs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
