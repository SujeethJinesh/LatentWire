#!/usr/bin/env python3
"""Inventory non-confirm CPU-safe caches and write the cheap-exhaustion report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 5 * 1024 * 1024
CONFIRM_RE = re.compile(r"confirm|confirmation", re.I)
TEXT_EXTS = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".md"}
HARD_EXTS = {".pt", ".bin", ".safetensors", ".npy", ".npz", ".pkl", ".gz", ".pdf", ".png", ".jpg", ".jpeg"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def path_has_confirm(path: Path) -> bool:
    return bool(CONFIRM_RE.search(path.as_posix()))


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def walk_keys(value: Any, keys: Counter[str], limit: int = 2500) -> None:
    if sum(keys.values()) >= limit:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            keys[str(key)] += 1
            walk_keys(child, keys, limit)
    elif isinstance(value, list):
        for child in value[:50]:
            walk_keys(child, keys, limit)


def preview_jsonl_keys(path: Path) -> tuple[Counter[str], int]:
    keys: Counter[str] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            if rows <= 50:
                try:
                    walk_keys(json.loads(line), keys)
                except Exception:
                    pass
    return keys, rows


def preview_csv_keys(path: Path) -> tuple[Counter[str], int]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        keys = Counter(str(name) for name in (reader.fieldnames or []))
        rows = sum(1 for _ in reader)
    return keys, rows


def row_count_from_json(obj: Any) -> int | None:
    if isinstance(obj, list):
        return len(obj)
    if not isinstance(obj, dict):
        return None
    for key in ("traces", "rows", "examples", "prompts", "entries"):
        if isinstance(obj.get(key), list):
            return len(obj[key])
    if isinstance(obj.get("scores"), dict):
        return len(obj["scores"])
    return None


def regimes_from_json(path: Path, obj: Any) -> set[str]:
    regimes: set[str] = set()
    if path.parent.name == "score_cache" and path.suffix == ".json":
        regimes.add(path.stem)
    if isinstance(obj, dict):
        for key in ("results_by_regime", "regime_summary"):
            if isinstance(obj.get(key), dict):
                regimes.update(str(name) for name in obj[key].keys())
        if isinstance(obj.get("regime"), str):
            regimes.add(str(obj["regime"]))
        if isinstance(obj.get("controls"), dict):
            regimes.update(str(name) for name in obj["controls"].keys())
    return regimes


def model_hint(path: Path, obj: Any | None, text_lower: str) -> str | None:
    model_id = obj.get("model_id") if isinstance(obj, dict) else None
    if model_id:
        haystack = str(model_id).lower()
    else:
        # Use the path for provenance files that do not carry model_id. Do not
        # use arbitrary text content: preregistration filenames can mention
        # multiple models and would contaminate run-level model assignment.
        haystack = path.as_posix().lower()
    if "granite" in haystack:
        return "granite"
    if "deepseek" in haystack:
        return "deepseek"
    if "falcon" in haystack:
        return "falcon"
    if "qwen" in haystack:
        return "qwen"
    return None


def feature_flags(path: Path, keys: Counter[str], regimes: set[str], text_lower: str) -> dict[str, bool]:
    key_text = " ".join(k.lower() for k in keys)
    regime_text = " ".join(sorted(regimes)).lower()
    combined = f"{path.as_posix().lower()} {key_text} {regime_text}"
    key_only = key_text
    return {
        "row_id": any(token in key_only for token in ("row_id", "prompt_id", "prompt_index", "trace_id", "example_id")),
        "drift_or_kl_feature": any(token in key_only for token in ("mean_kl", "max_kl", "decode_position", "channel_magnitudes", "activation_magnitudes", "trajectory")),
        "outcome_label": any(token in key_only for token in ("recoveries", "recovery", "static_gap", "correct", "accuracy", "policy_uplift", "difficulty")),
        "warmup_policy": any(token in combined for token in ("warmup", "ce13")),
        "required_policy_set": all(token in combined for token in ("paroquant", "survival", "reject")) and ("c_a1" in combined or "tight_clip" in combined),
        "givens_pairing": any(token in combined for token in ("givens", "co_drift", "codrift", "co-drift")),
        "budget_policy": "budget" in combined and any(token in combined for token in ("paroquant", "static", "m11b", "decdec", "tight_clip", "c_a1")),
        "score_cache": "/score_cache/" in path.as_posix(),
        "tight_clip": "tight_clip" in combined or "scale-clip" in text_lower,
        "paroquant": "paroquant" in combined,
        "decdec_or_osc": any(token in combined for token in ("decdec", "osc")),
    }


def inspect_file(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    size = path.stat().st_size
    base = {
        "path": rel,
        "bytes": size,
        "status": "scanned",
        "row_count": None,
        "keys": [],
        "regimes": [],
        "model_hint": None,
        "features": {},
    }
    if path_has_confirm(path):
        base["status"] = "excluded_confirm_path"
        base["path"] = "REDACTED_HELDOUT_PATH"
        return base
    if path.suffix.lower() in HARD_EXTS or "/caches/" in path.as_posix() or path.as_posix().startswith("caches/"):
        base["status"] = "excluded_binary_or_cache"
        return base
    if path.suffix.lower() not in TEXT_EXTS:
        base["status"] = "excluded_non_text"
        return base
    if size > MAX_TEXT_BYTES:
        base["status"] = "excluded_too_large"
        return base

    text = path.read_text(encoding="utf-8", errors="replace")
    if CONFIRM_RE.search(text):
        base["status"] = "excluded_embedded_confirm"
        return base

    keys: Counter[str] = Counter()
    row_count: int | None = None
    obj: Any | None = None
    if path.suffix == ".json":
        obj = read_json(path)
        walk_keys(obj, keys)
        row_count = row_count_from_json(obj)
    elif path.suffix == ".jsonl":
        keys, row_count = preview_jsonl_keys(path)
    elif path.suffix == ".csv":
        keys, row_count = preview_csv_keys(path)
    else:
        keys = Counter()

    regimes = regimes_from_json(path, obj)
    flags = feature_flags(path, keys, regimes, text.lower())
    base.update(
        {
            "row_count": row_count,
            "keys": sorted(keys)[:120],
            "regimes": sorted(regimes),
            "model_hint": model_hint(path, obj, text.lower()),
            "features": flags,
        }
    )
    return base


def discover_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(files))


def run_dir_for(path: str) -> str | None:
    parts = Path(path).parts
    if "results" not in parts:
        return None
    idx = parts.index("results")
    if idx + 1 >= len(parts):
        return None
    return "/".join(parts[: idx + 2])


def build_run_inventory(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["status"] != "scanned":
            continue
        run = run_dir_for(record["path"])
        if not run:
            continue
        info = runs.setdefault(
            run,
            {
                "run_dir": run,
                "models": set(),
                "regimes": set(),
                "score_cache_regimes": set(),
                "rows_by_file": {},
                "has_tight_clip": False,
                "has_paroquant": False,
                "has_drift_residual": False,
                "has_decdec_or_osc": False,
            },
        )
        if record.get("model_hint"):
            info["models"].add(record["model_hint"])
        info["regimes"].update(record.get("regimes", []))
        if "/score_cache/" in record["path"]:
            info["score_cache_regimes"].update(record.get("regimes", []))
        if record.get("row_count") is not None:
            info["rows_by_file"][record["path"]] = record["row_count"]
        features = record.get("features", {})
        info["has_tight_clip"] = info["has_tight_clip"] or bool(features.get("tight_clip"))
        info["has_paroquant"] = info["has_paroquant"] or bool(features.get("paroquant"))
        info["has_drift_residual"] = info["has_drift_residual"] or "driftrot_residual" in " ".join(record.get("regimes", []))
        info["has_decdec_or_osc"] = info["has_decdec_or_osc"] or bool(features.get("decdec_or_osc"))
    return {
        run: {
            **info,
            "models": sorted(info["models"]),
            "regimes": sorted(info["regimes"]),
            "score_cache_regimes": sorted(info["score_cache_regimes"]),
        }
        for run, info in sorted(runs.items())
    }


def method_verdicts(records: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scanned = [record for record in records if record["status"] == "scanned"]
    cu1_candidates = [
        record["path"]
        for record in scanned
        if record["features"].get("row_id")
        and record["features"].get("drift_or_kl_feature")
        and record["features"].get("outcome_label")
    ]
    cw1_candidates = [record["path"] for record in scanned if record["features"].get("required_policy_set")]
    ce1_candidates = [record["path"] for record in scanned if record["features"].get("givens_pairing")]
    cc1_candidates = [record["path"] for record in scanned if record["features"].get("budget_policy")]

    c_a1 = c_a1_manifest_status(runs, records)
    c_d1_runs = [run for run, info in runs.items() if info["has_decdec_or_osc"] or any("decdec" in r for r in info["regimes"])]

    return {
        "C_U1_drift_as_signal_router": {
            "verdict": "PARKED",
            "promotion_allowed": False,
            "candidate_files": cu1_candidates[:12],
            "blocker": "No non-confirm cache has row-level drift/KL trajectory features paired with row-level recovery/difficulty/uplift labels.",
            "required_native_cache": "For each Granite/DeepSeek/Falcon row: prompt_id, split, receiver confidence, KL/drift trajectory by position/layer, warmup activation features, static_gap/recoverable_gap, and per-policy uplift labels.",
        },
        "C_W1_fixed_library_warmup_selector": {
            "verdict": "PARKED",
            "promotion_allowed": False,
            "candidate_files": cw1_candidates[:12],
            "blocker": "No non-confirm cache exposes same-row outcomes for the fixed library ParoQuant/C_A1/survival/reject policies.",
            "required_native_cache": "Same prompt_id rows for paroquant, tight_clip_c_a1, survival_core, and reject/no_answer policies with dev/gate split and row-safe outcomes.",
        },
        "CE1_codrift_givens_pairing": {
            "verdict": "PARKED",
            "promotion_allowed": False,
            "candidate_files": ce1_candidates[:12],
            "blocker": "No non-confirm cache contains co-drift/Givens pairing fields joined to outcome labels.",
            "required_native_cache": "Per-row paired-channel/Givens metadata, co-drift score, row_id, static_gap, and policy recovery for the paired intervention and its controls.",
        },
        "C_C1_budget_router": {
            "verdict": "PARKED",
            "promotion_allowed": False,
            "candidate_files": cc1_candidates[:12],
            "blocker": "No non-confirm cache contains budget-conditioned same-row outcomes across usable policies.",
            "required_native_cache": "Per-row budget, policy_id, byte/compute cost, ParoQuant/static/tight-clip outcomes, reject option, and no-gap denominator fields.",
        },
        "C_A1_cvar_evt_clip_grid": c_a1,
        "C_A2_horizon_rotation": {
            "verdict": "PARKED_TESTS_ONLY",
            "promotion_allowed": False,
            "blocker": "No explicit C_A2 orthogonality plus full-precision-equivalence test artifact was found in the non-confirm scan.",
            "required_native_cache": "Tests must pass before even a 2-bin C_A2 screen; no GPU queue item should be emitted from this pass.",
        },
        "C_D1_osc_decdec_vs_drift_defense": {
            "verdict": "CACHE_PRESENT_UNDERPOWERED_SINGLE_MODEL" if c_d1_runs else "PARKED",
            "promotion_allowed": False,
            "candidate_runs": c_d1_runs[:12],
            "blocker": "Cached DecDEC/KL evidence is Granite-only and 12-row/8-included with wide CIs; usable as a defense note, not a positive method.",
            "required_native_cache": "Same-row OSC/DecDEC stress surfaces joined to drift/KL and outcome labels across at least two models.",
        },
    }


def prompt_ids_from_run(run_dir: Path) -> set[str]:
    per_trace = run_dir / "per_trace_metrics.json"
    if not per_trace.is_file() or path_has_confirm(per_trace):
        return set()
    text = per_trace.read_text(encoding="utf-8", errors="replace")
    if CONFIRM_RE.search(text):
        return set()
    obj = json.loads(text)
    return {str(row.get("prompt_id", row.get("prompt_index"))) for row in obj.get("traces", [])}


def c_a1_manifest_status(runs: dict[str, dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"baseline": [], "tight_clip": []})
    unsafe_runs = {
        run
        for record in records
        if record["status"] in {"excluded_confirm_path", "excluded_embedded_confirm"}
        for run in [run_dir_for(record["path"])]
        if run
    }
    for run, info in runs.items():
        if run in unsafe_runs:
            continue
        models = info.get("models") or [None]
        score_regimes = set(info.get("score_cache_regimes", [])) | set(info.get("regimes", []))
        for model in models:
            if model not in {"granite", "deepseek", "falcon"}:
                continue
            if "paroquant_w4a16" in score_regimes and info["has_paroquant"] and not info["has_tight_clip"]:
                by_model[model]["baseline"].append(run)
            if "paroquant_w4a16" in score_regimes and info["has_tight_clip"] and not info["has_drift_residual"]:
                by_model[model]["tight_clip"].append(run)

    pair_status: dict[str, Any] = {}
    for model in ("granite", "deepseek", "falcon"):
        baselines = by_model[model]["baseline"]
        tight = by_model[model]["tight_clip"]
        best_common = 0
        best_pair: list[str] | None = None
        for base_run in baselines:
            base_ids = prompt_ids_from_run(ROOT / base_run)
            for tight_run in tight:
                tight_ids = prompt_ids_from_run(ROOT / tight_run)
                common = len(base_ids & tight_ids)
                if common > best_common:
                    best_common = common
                    best_pair = [base_run, tight_run]
        pair_status[model] = {
            "baseline_runs": baselines,
            "tight_clip_runs": tight,
            "best_same_row_pair": best_pair,
            "same_row_count": best_common,
        }

    embedded_confirm = [
        record["path"]
        for record in records
        if record["status"] == "excluded_embedded_confirm" and "driftrot_granite_clip_tight" in record["path"]
    ]
    complete = all(pair_status[model]["same_row_count"] >= 12 for model in ("granite", "deepseek", "falcon"))
    return {
        "verdict": "PARKED_NEEDS_NATIVE_PAIRING",
        "promotion_allowed": False,
        "manifest_complete": complete,
        "models": pair_status,
        "excluded_granite_tight_clip_embedded_confirm_files": embedded_confirm[:20],
        "blocker": "DeepSeek/Falcon have non-confirm 12-row baseline-vs-tight-clip pairs; Granite tight-clip candidates are embedded-confirm or diagnostic residual subsets, so the three-model C_A1 manifest is incomplete.",
        "exact_gpu_row_materialization_command": (
            "local_runner enqueue channel_set_c_a1_pair_materialization "
            "--models granite,deepseek,falcon --split dev,gate "
            "--prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl "
            "--policies paroquant_baseline,tight_clip_c_a1 "
            "--scale-clip-min 0.5 --scale-clip-max 2.0 "
            "--require-same-row --write-access-manifest --fail-on-confirm "
            "--out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}"
        ),
    }


def summarize_latentwire() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for summary in sorted((ROOT / "results/mps_first").glob("L_*/summary.json")):
        payload = read_json(summary) or {}
        out[payload.get("id", summary.parent.name)] = {
            "verdict": payload.get("verdict"),
            "status": payload.get("status"),
            "promotion_allowed": payload.get("promotion_allowed", False),
            "achieved_n": payload.get("achieved_n"),
            "blocker": payload.get("blocker") or payload.get("reason"),
            "exact_command": payload.get("exact_command"),
        }
    return out


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Cheap Exhaustion Report",
        "",
        f"- created_utc: `{summary['created_utc']}`",
        f"- git_head: `{summary['git_head']}`",
        f"- wall_clock_seconds: `{summary['wall_clock_seconds']:.3f}`",
        f"- files_seen: `{summary['inventory']['files_seen']}`",
        f"- files_scanned: `{summary['inventory']['status_counts'].get('scanned', 0)}`",
        f"- confirm_path_excluded: `{summary['inventory']['status_counts'].get('excluded_confirm_path', 0)}`",
        f"- embedded_confirm_excluded: `{summary['inventory']['status_counts'].get('excluded_embedded_confirm', 0)}`",
        f"- gpu_foreground_empty: `{summary['gpu_foreground_empty']}`",
        "",
        "## Verdicts",
        "",
        "| method | verdict | promotion_allowed | blocker |",
        "|---|---:|---:|---|",
    ]
    for method, payload in summary["method_verdicts"].items():
        blocker = str(payload.get("blocker", "")).replace("\n", " ")
        lines.append(f"| `{method}` | `{payload.get('verdict')}` | `{payload.get('promotion_allowed')}` | {blocker} |")
    lines.extend(["", "## C_A1 Manifest Status", ""])
    c_a1 = summary["method_verdicts"]["C_A1_cvar_evt_clip_grid"]
    lines.append(f"- manifest_complete: `{c_a1['manifest_complete']}`")
    for model, payload in c_a1["models"].items():
        lines.append(
            f"- {model}: same_row_count=`{payload['same_row_count']}`, "
            f"baseline_runs=`{len(payload['baseline_runs'])}`, tight_clip_runs=`{len(payload['tight_clip_runs'])}`"
        )
    lines.append(f"- exact_gpu_row_materialization_command: `{c_a1['exact_gpu_row_materialization_command']}`")
    if c_a1["excluded_granite_tight_clip_embedded_confirm_files"]:
        lines.append("- excluded Granite tight-clip files with embedded confirm:")
        for path in c_a1["excluded_granite_tight_clip_embedded_confirm_files"][:12]:
            lines.append(f"  - `{path}`")

    lines.extend(["", "## LatentWire CPU-Safe Status", ""])
    for method, payload in summary["latentwire"].items():
        blocker = str(payload.get("blocker", "")).replace("\n", " ")
        lines.append(
            f"- `{method}`: verdict=`{payload.get('verdict')}`, status=`{payload.get('status')}`, "
            f"promotion_allowed=`{payload.get('promotion_allowed')}`, achieved_n=`{payload.get('achieved_n')}`. {blocker}"
        )

    lines.extend(["", "## Required Native Cache Requirements", ""])
    for method, payload in summary["method_verdicts"].items():
        if "required_native_cache" in payload:
            lines.append(f"- `{method}`: {payload['required_native_cache']}")

    lines.extend(["", "## Next 6 Commands", ""])
    for idx, command in enumerate(summary["next_6_commands"], start=1):
        lines.append(f"{idx}. `{command}`")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    start = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (ROOT / "results/cheap_exhaustion" / stamp)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists():
        raise SystemExit(f"write-once output exists: {out_dir}")
    out_dir.mkdir(parents=True)

    roots = [
        ROOT / "experimental/outlier_migrate",
        ROOT / "results/mps_first",
        ROOT / "results/overnight_v3",
    ]
    files = discover_files([root for root in roots if root.exists()])
    records = [inspect_file(path) for path in files]
    runs = build_run_inventory(records)
    verdicts = method_verdicts(records, runs)
    status_counts = Counter(record["status"] for record in records)
    gpu_foreground = read_json(ROOT / "queues/gpu_foreground.yaml")
    # YAML is not JSON; use a conservative text check for the one field we need.
    gpu_foreground_text = (ROOT / "queues/gpu_foreground.yaml").read_text(encoding="utf-8", errors="replace")
    gpu_foreground_empty = "foreground: []" in gpu_foreground_text
    summary = {
        "schema_version": "cheap_exhaustion_scan_v1",
        "created_utc": utc_now(),
        "git_head": git_head(),
        "wall_clock_seconds": time.perf_counter() - start,
        "inventory": {
            "files_seen": len(files),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "method_verdicts": verdicts,
        "latentwire": summarize_latentwire(),
        "gpu_foreground_empty": gpu_foreground_empty,
        "gpu_foreground_parse_json": gpu_foreground,
        "next_6_commands": [
            "venv_arm64/bin/python scripts/check_review_packet.py review_packet.zip",
            "sed -n '1,260p' dashboard/cheap_exhaustion_report.md",
            "sed -n '1,260p' dashboard/c_a1_gpu_backfill_runbook.md",
            "sed -n '1,220p' queues/gpu_backfill.yaml",
            "sed -n '1,80p' queues/gpu_foreground.yaml",
            verdicts["C_A1_cvar_evt_clip_grid"]["exact_gpu_row_materialization_command"],
        ],
    }
    (out_dir / "cache_inventory.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, ROOT / "dashboard/cheap_exhaustion_report.md")
    print(f"wrote {out_dir.relative_to(ROOT)}/summary.json")
    print(f"wrote dashboard/cheap_exhaustion_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
