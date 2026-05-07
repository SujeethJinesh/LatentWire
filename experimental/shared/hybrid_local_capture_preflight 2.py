"""Local readiness preflight for real hybrid trace capture.

This packet checks whether the current Mac can move from trace templates to
real SSQ-LR/HORN/HBSM tensor capture. It does not run models and it does not
promote any gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_local_capture_preflight_20260507"
DEFAULT_ARCHITECTURE_MAPS = ROOT / "experimental/shared/results/hybrid_architecture_maps_20260506/architecture_maps.json"
DEFAULT_CAPTURE_SUMMARY = ROOT / "experimental/shared/results/hybrid_capture_manifests_20260507/summary.json"
DEFAULT_ELIGIBILITY_SUMMARY = ROOT / "experimental/shared/results/hybrid_model_eligibility_20260506/summary.json"
REQUIRED_BASE_PACKAGES = ("torch", "transformers", "huggingface_hub")
OPTIONAL_RUNTIME_PACKAGES = ("mamba_ssm", "vllm")
DEFAULT_MAC_WEIGHT_BUDGET_GB = 24.0


@dataclass(frozen=True)
class CacheProbe:
    cache_root: Path
    model_id: str
    model_cache_dir: Path
    snapshot_count: int
    weight_file_count: int
    weight_bytes: int

    @property
    def has_weights(self) -> bool:
        return self.weight_file_count > 0


def _default_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser())
    roots.extend([ROOT / ".debug/hf_home", Path.home() / ".cache/huggingface"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return tuple(unique)


def _model_cache_dir(cache_root: Path, model_id: str) -> Path:
    if "/" not in model_id:
        return cache_root / "hub" / f"models--{model_id}"
    owner, name = model_id.split("/", 1)
    base = cache_root if cache_root.name == "hub" else cache_root / "hub"
    return base / f"models--{owner}--{name}"


def _snapshot_weight_files(model_cache_dir: Path) -> list[Path]:
    if not model_cache_dir.exists():
        return []
    files: list[Path] = []
    for snapshot in sorted((model_cache_dir / "snapshots").glob("*")):
        files.extend(sorted(snapshot.rglob("*.safetensors")))
        files.extend(sorted(snapshot.rglob("pytorch_model*.bin")))
    return files


def _probe_cache(cache_root: Path, model_id: str) -> CacheProbe:
    model_cache = _model_cache_dir(cache_root, model_id)
    snapshots = sorted((model_cache / "snapshots").glob("*")) if model_cache.exists() else []
    weight_files = _snapshot_weight_files(model_cache)
    weight_bytes = sum(path.stat().st_size for path in weight_files if path.is_file())
    return CacheProbe(
        cache_root=cache_root,
        model_id=model_id,
        model_cache_dir=model_cache,
        snapshot_count=len(snapshots),
        weight_file_count=len(weight_files),
        weight_bytes=weight_bytes,
    )


def _package_status(package_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for package_name in package_names:
        spec = importlib.util.find_spec(package_name)
        status[package_name] = {
            "installed": spec is not None,
            "origin": None if spec is None else str(spec.origin),
        }
    return status


def _transformers_model_class_status(aliases: list[str]) -> tuple[bool, str]:
    """Check whether local Transformers can instantiate the hybrid class.

    This is intentionally config/class-only. It does not load weights.
    """

    errors: list[str] = []
    for alias in aliases:
        try:
            config = AutoConfig.from_pretrained(alias, local_files_only=True, trust_remote_code=False)
            model_class = AutoModelForCausalLM._model_mapping[type(config)]
            return True, f"{alias}:{model_class.__module__}.{model_class.__name__}"
        except Exception as exc:  # pragma: no cover - depends on local HF cache details
            errors.append(f"{alias}:{type(exc).__name__}:{str(exc)[:120]}")
    return False, "; ".join(errors)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_rows_from_inputs(architecture_maps: Path, capture_summary: Path) -> list[dict[str, Any]]:
    maps = _load_json(architecture_maps)
    capture = _load_json(capture_summary)
    required_models: set[str] = set()
    counts = capture.get("counts", {})
    if isinstance(counts, dict):
        for project_counts in counts.values():
            if isinstance(project_counts, dict):
                required_models.update(str(model_id) for model_id in project_counts)

    rows: list[dict[str, Any]] = []
    for row in maps:
        model_id = str(row.get("model_id", ""))
        if model_id not in required_models:
            continue
        aliases = [str(alias) for alias in row.get("model_id_aliases", []) if str(alias)]
        rows.append(
            {
                "model_id": model_id,
                "model_type": row.get("model_type"),
                "architecture": row.get("architecture"),
                "architecture_map_hash": row.get("config_sha256"),
                "hidden_size": row.get("hidden_size"),
                "num_hidden_layers": row.get("num_hidden_layers"),
                "boundary_count": row.get("boundary_count"),
                "direction_counts": row.get("direction_counts", {}),
                "aliases": aliases or [model_id],
                "project_entry_counts": {
                    project: project_counts.get(model_id)
                    for project, project_counts in counts.items()
                    if isinstance(project_counts, dict) and model_id in project_counts
                },
            }
        )
    return rows


def _estimated_sizes_by_model(eligibility_summary: Path) -> dict[str, float]:
    if not eligibility_summary.exists():
        return {}
    payload = _load_json(eligibility_summary)
    sizes: dict[str, float] = {}
    for row in payload.get("rows", []):
        model_id = str(row.get("model_id", ""))
        size = row.get("safetensors_gb")
        if isinstance(size, int | float):
            canonical = model_id.split("/", 1)[-1].lower()
            canonical = canonical.replace("_", "-")
            canonical = canonical.removesuffix("-fp8")
            sizes[model_id] = float(size)
            sizes[canonical] = float(size)
    return sizes


def _memory_budget_gb(requested_budget_gb: float | None) -> float:
    if requested_budget_gb is not None:
        return requested_budget_gb
    return min(DEFAULT_MAC_WEIGHT_BUDGET_GB, psutil.virtual_memory().total / (1024**3) * 0.60)


def _decide_model(
    *,
    weights_cached: bool,
    estimated_weight_gb: float | None,
    missing_base_packages: list[str],
    transformers_model_class_available: bool,
    mac_weight_budget_gb: float,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if missing_base_packages:
        blockers.append("missing base packages: " + ", ".join(missing_base_packages))
    if not weights_cached:
        blockers.append("no local cached model weights found")
    if estimated_weight_gb is not None and estimated_weight_gb > mac_weight_budget_gb:
        blockers.append(
            f"estimated weights {estimated_weight_gb:.2f} GB exceed Mac capture budget {mac_weight_budget_gb:.2f} GB"
        )
    if not transformers_model_class_available:
        blockers.append("local transformers cannot instantiate the hybrid model class from cached config")

    if not blockers:
        return "LOCAL_CAPTURE_READY_NOT_EVIDENCE", blockers
    if missing_base_packages or not transformers_model_class_available:
        return "LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE", blockers
    if not weights_cached:
        return "LOCAL_CAPTURE_BLOCKED_MODEL_CACHE_NOT_EVIDENCE", blockers
    return "LOCAL_CAPTURE_GPU_RECOMMENDED_NOT_EVIDENCE", blockers


def build_preflight_rows(
    *,
    architecture_maps: Path = DEFAULT_ARCHITECTURE_MAPS,
    capture_summary: Path = DEFAULT_CAPTURE_SUMMARY,
    eligibility_summary: Path = DEFAULT_ELIGIBILITY_SUMMARY,
    cache_roots: tuple[Path, ...] | None = None,
    mac_weight_budget_gb: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_roots = cache_roots or _default_cache_roots()
    package_names = REQUIRED_BASE_PACKAGES + OPTIONAL_RUNTIME_PACKAGES
    package_status = _package_status(package_names)
    missing_base = [name for name in REQUIRED_BASE_PACKAGES if not package_status[name]["installed"]]
    sizes = _estimated_sizes_by_model(eligibility_summary)
    budget_gb = _memory_budget_gb(mac_weight_budget_gb)

    rows: list[dict[str, Any]] = []
    for model in _model_rows_from_inputs(architecture_maps, capture_summary):
        aliases = list(model["aliases"])
        probes = [_probe_cache(cache_root, alias) for cache_root in cache_roots for alias in aliases]
        weighted_probes = [probe for probe in probes if probe.has_weights]
        weights_cached = bool(weighted_probes)
        canonical = str(model["model_id"])
        estimated_weight_gb = sizes.get(canonical) or sizes.get(canonical.lower())
        if estimated_weight_gb is None:
            for alias in aliases:
                alias_key = str(alias)
                alias_slug = alias_key.split("/", 1)[-1].lower()
                estimated_weight_gb = sizes.get(alias_key) or sizes.get(alias_key.lower()) or sizes.get(alias_slug)
                if estimated_weight_gb is not None:
                    break
        if estimated_weight_gb is None and weighted_probes:
            estimated_weight_gb = max(probe.weight_bytes for probe in weighted_probes) / (1024**3)
        class_available, class_status = _transformers_model_class_status(aliases)
        decision, blockers = _decide_model(
            weights_cached=weights_cached,
            estimated_weight_gb=estimated_weight_gb,
            missing_base_packages=missing_base,
            transformers_model_class_available=class_available,
            mac_weight_budget_gb=budget_gb,
        )
        rows.append(
            {
                **model,
                "cache_aliases_checked": aliases,
                "cache_roots_checked": [str(root) for root in cache_roots],
                "weights_cached": weights_cached,
                "cached_weight_locations": [
                    {
                        "cache_root": str(probe.cache_root),
                        "model_id": probe.model_id,
                        "model_cache_dir": str(probe.model_cache_dir),
                        "snapshot_count": probe.snapshot_count,
                        "weight_file_count": probe.weight_file_count,
                        "weight_gb_on_disk": probe.weight_bytes / (1024**3),
                    }
                    for probe in weighted_probes
                ],
                "estimated_weight_gb": estimated_weight_gb,
                "mac_weight_budget_gb": budget_gb,
                "transformers_model_class_available": class_available,
                "transformers_model_class_status": class_status,
                "optional_runtime_packages_missing": [
                    name for name in OPTIONAL_RUNTIME_PACKAGES if not package_status[name]["installed"]
                ],
                "required_for_projects": sorted(str(project) for project in model["project_entry_counts"]),
                "decision": decision,
                "blocking_reasons": blockers,
            }
        )

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "mac_weight_budget_gb": budget_gb,
        "packages": package_status,
        "claim_boundary": ["preflight-only", "not model evidence", "not GPU evidence"],
    }
    return rows, environment


def _overall_decision(rows: list[dict[str, Any]]) -> str:
    decisions = {str(row["decision"]) for row in rows}
    if "LOCAL_CAPTURE_READY_NOT_EVIDENCE" in decisions:
        return "LOCAL_CAPTURE_READY_NOT_EVIDENCE"
    if "LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE" in decisions:
        return "LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE"
    if "LOCAL_CAPTURE_GPU_RECOMMENDED_NOT_EVIDENCE" in decisions:
        return "LOCAL_CAPTURE_GPU_RECOMMENDED_NOT_EVIDENCE"
    return "LOCAL_CAPTURE_BLOCKED_MODEL_CACHE_NOT_EVIDENCE"


def write_packet(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    architecture_maps: Path = DEFAULT_ARCHITECTURE_MAPS,
    capture_summary: Path = DEFAULT_CAPTURE_SUMMARY,
    eligibility_summary: Path = DEFAULT_ELIGIBILITY_SUMMARY,
    cache_roots: tuple[Path, ...] | None = None,
    mac_weight_budget_gb: float | None = None,
) -> dict[str, Any]:
    rows, environment = build_preflight_rows(
        architecture_maps=architecture_maps,
        capture_summary=capture_summary,
        eligibility_summary=eligibility_summary,
        cache_roots=cache_roots,
        mac_weight_budget_gb=mac_weight_budget_gb,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = _overall_decision(rows)
    summary = {
        "surface": "hybrid_local_capture_preflight",
        "decision": decision,
        "row_count": len(rows),
        "rows": rows,
        "environment": environment,
        "claim_boundary": environment["claim_boundary"],
        "next_step": _next_step(decision),
    }
    config = {
        "architecture_maps": str(architecture_maps),
        "capture_summary": str(capture_summary),
        "eligibility_summary": str(eligibility_summary),
        "cache_roots": [str(root) for root in (cache_roots or _default_cache_roots())],
        "command": "python -m experimental.shared.hybrid_local_capture_preflight",
        "download_policy": "local filesystem inspection only; no downloads",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary))
    (output_dir / "decision.md").write_text(
        "# Local Capture Preflight Decision\n\n"
        f"`{decision}`\n\n"
        "This packet is preflight-only. It cannot promote SSQ-LR, HORN, or HBSM.\n"
    )
    return summary


def _next_step(decision: str) -> str:
    if decision == "LOCAL_CAPTURE_READY_NOT_EVIDENCE":
        return "Run the first real SSQ-LR/HORN/HBSM tensor capture from the frozen manifest."
    if decision == "LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE":
        return "Fix the missing base package or Transformers model-class support in the repo-local venv, then rerun this preflight."
    if decision == "LOCAL_CAPTURE_GPU_RECOMMENDED_NOT_EVIDENCE":
        return "Use this packet to prepare a GPU-node capture runbook; do not try to promote Mac-only metadata."
    return "Populate a repo-local HF cache or use a GPU node with cached weights, then rerun this preflight."


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Local Capture Preflight",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is not model evidence and cannot promote SSQ-LR, HORN, or HBSM.",
        "",
        f"Next step: {summary['next_step']}",
        "",
        "| Model | Projects | Cached weights | Transformers class | Est. GB | Decision | Blockers |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in summary["rows"]:
        est = row.get("estimated_weight_gb")
        est_text = "unknown" if est is None else f"{float(est):.2f}"
        blockers = "; ".join(row.get("blocking_reasons", [])) or "none"
        lines.append(
            "| {model} | {projects} | {cached} | {cls} | {est} | `{decision}` | {blockers} |".format(
                model=row["model_id"],
                projects=", ".join(row["required_for_projects"]),
                cached="yes" if row["weights_cached"] else "no",
                cls="yes" if row["transformers_model_class_available"] else "no",
                est=est_text,
                decision=row["decision"],
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{summary['environment']['python']}`",
            f"- Torch: `{summary['environment']['torch_version']}`",
            f"- MPS available: `{summary['environment']['mps_available']}`",
            f"- Mac weight budget GB: `{summary['environment']['mac_weight_budget_gb']:.2f}`",
            "",
            "## Package Status",
            "",
        ]
    )
    for package, status in summary["environment"]["packages"].items():
        lines.append(f"- `{package}`: {'installed' if status['installed'] else 'missing'}")
    lines.extend(
        [
            "",
            "`mamba_ssm` and `vllm` are recorded as optional local/GPU runtime packages here;",
            "they are not hard blockers when cached configs map to native `transformers` hybrid classes.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--architecture-maps", type=Path, default=DEFAULT_ARCHITECTURE_MAPS)
    parser.add_argument("--capture-summary", type=Path, default=DEFAULT_CAPTURE_SUMMARY)
    parser.add_argument("--eligibility-summary", type=Path, default=DEFAULT_ELIGIBILITY_SUMMARY)
    parser.add_argument("--cache-root", type=Path, action="append", dest="cache_roots")
    parser.add_argument("--mac-weight-budget-gb", type=float)
    args = parser.parse_args()

    summary = write_packet(
        output_dir=args.output_dir,
        architecture_maps=args.architecture_maps,
        capture_summary=args.capture_summary,
        eligibility_summary=args.eligibility_summary,
        cache_roots=tuple(args.cache_roots) if args.cache_roots else None,
        mac_weight_budget_gb=args.mac_weight_budget_gb,
    )
    print(json.dumps({"decision": summary["decision"], "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
