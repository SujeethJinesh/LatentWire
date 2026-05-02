#!/usr/bin/env python3
"""Rank source-contrastive target sets as durable method-gate surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TargetSetSpec:
    label: str
    path: pathlib.Path
    role: str = "candidate"
    note: str = ""


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _parse_target_set_spec(raw: str) -> TargetSetSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Expected label=path=...,role=...,note=... spec: {raw!r}"
        )
    label, rest = raw.split("=", 1)
    fields: dict[str, str] = {}
    for item in rest.split(","):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value in {raw!r}: {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    if not label or not fields.get("path"):
        raise argparse.ArgumentTypeError(f"Spec needs label and path: {raw!r}")
    return TargetSetSpec(
        label=label,
        path=_resolve(fields["path"]),
        role=fields.get("role", "candidate"),
        note=fields.get("note", ""),
    )


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_sorted_list(payload: dict[str, Any], key: str) -> list[str]:
    return sorted(str(item) for item in payload.get("ids", {}).get(key, []))


def _duplicate_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for item in items:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return sorted(dupes)


def _summaries(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = payload.get("summaries", {})
    target = dict(summaries.get("target", {}))
    source = dict(summaries.get("source", {}))
    controls = {
        str(label): dict(summary)
        for label, summary in dict(summaries.get("controls", {})).items()
    }
    baselines = {
        str(label): dict(summary)
        for label, summary in dict(summaries.get("baselines", {})).items()
    }
    return {
        "target": target,
        "source": source,
        "controls": controls,
        "baselines": baselines,
    }


def _numeric_min(summary: dict[str, Any]) -> int:
    values: list[int] = []
    for item in [summary.get("target", {}), summary.get("source", {})]:
        if "numeric_coverage" in item:
            values.append(int(item["numeric_coverage"]))
    for group in [summary.get("controls", {}), summary.get("baselines", {})]:
        for item in group.values():
            if "numeric_coverage" in item:
                values.append(int(item["numeric_coverage"]))
    return min(values) if values else 0


def analyze_target_set(
    spec: TargetSetSpec,
    *,
    min_clean_source_only: int,
    min_numeric_coverage: int,
    require_exact_ordered_id_parity: bool,
) -> dict[str, Any]:
    payload = _read_json(spec.path)
    reference_ids = [str(item) for item in payload.get("reference_ids", [])]
    duplicates = _duplicate_items(reference_ids)
    if duplicates:
        raise ValueError(f"{spec.label}: duplicate reference_ids: {duplicates}")
    if payload.get("reference_n") is not None and int(payload["reference_n"]) != len(reference_ids):
        raise ValueError(
            f"{spec.label}: reference_n={payload['reference_n']} but "
            f"reference_ids has {len(reference_ids)} rows"
        )

    ids = {
        "source_only": _as_sorted_list(payload, "source_only"),
        "clean_source_only": _as_sorted_list(payload, "clean_source_only"),
        "target_only": _as_sorted_list(payload, "target_only"),
        "control_union": _as_sorted_list(payload, "control_union"),
        "baseline_union": _as_sorted_list(payload, "baseline_union"),
    }
    clean_leaks = sorted(
        set(ids["clean_source_only"])
        & (set(ids["control_union"]) | set(ids["baseline_union"]))
    )
    if clean_leaks:
        raise ValueError(f"{spec.label}: clean_source_only leaks controls/baselines: {clean_leaks}")

    summaries = _summaries(payload)
    provenance = dict(payload.get("provenance", {}))
    exact_ordered_id_parity = bool(provenance.get("exact_ordered_id_parity"))
    numeric_min = _numeric_min(summaries)
    counts = dict(payload.get("counts", {}))
    clean_count = len(ids["clean_source_only"])
    source_only_count = len(ids["source_only"])
    target_correct = int(counts.get("target_correct", summaries["target"].get("correct", 0) or 0))
    oracle_count = int(counts.get("target_or_source_oracle", target_correct))
    oracle_gain = oracle_count - target_correct
    invalid_reasons: list[str] = []
    if require_exact_ordered_id_parity and not exact_ordered_id_parity:
        invalid_reasons.append("missing_exact_ordered_id_parity")
    if numeric_min and numeric_min < min_numeric_coverage:
        invalid_reasons.append("numeric_coverage_below_threshold")

    if invalid_reasons:
        decision = "invalid"
    elif clean_count >= min_clean_source_only and oracle_gain >= min_clean_source_only:
        decision = "primary_ready"
    elif clean_count > 0:
        decision = "weak_clean_headroom"
    else:
        decision = "no_clean_headroom"

    rank_components = {
        "decision_priority": {
            "primary_ready": 3,
            "weak_clean_headroom": 2,
            "no_clean_headroom": 1,
            "invalid": 0,
        }[decision],
        "clean_source_only": clean_count,
        "oracle_gain_vs_target": oracle_gain,
        "source_only": source_only_count,
        "target_preservation_pool": len(ids["target_only"]),
        "numeric_min": numeric_min,
    }
    return {
        "label": spec.label,
        "role": spec.role,
        "decision": decision,
        "invalid_reasons": invalid_reasons,
        "note": spec.note,
        "target_set": {
            "path": _display_path(spec.path),
            "sha256": _sha256(spec.path),
            "status": payload.get("status"),
            "schema": "source_contrastive_target_set",
        },
        "id_provenance": {
            "reference_n": len(reference_ids),
            "exact_ordered_id_parity": exact_ordered_id_parity,
            "duplicate_reference_ids": duplicates,
        },
        "summaries": summaries,
        "ids": ids,
        "counts": {
            "target_correct": target_correct,
            "source_correct": int(counts.get("source_correct", summaries["source"].get("correct", 0) or 0)),
            "source_only": source_only_count,
            "clean_source_only": clean_count,
            "target_only": len(ids["target_only"]),
            "control_union": len(ids["control_union"]),
            "baseline_union": len(ids["baseline_union"]),
            "target_or_source_oracle": oracle_count,
            "oracle_gain_vs_target": oracle_gain,
        },
        "rank": {
            "score_components": rank_components,
        },
    }


def _rank_key(surface: dict[str, Any]) -> tuple[Any, ...]:
    components = surface["rank"]["score_components"]
    return (
        int(components["decision_priority"]),
        int(components["clean_source_only"]),
        int(components["oracle_gain_vs_target"]),
        int(components["source_only"]),
        int(components["numeric_min"]),
        surface["label"],
    )


def _overlap_matrix(surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in surfaces:
        left_ids = set(left["ids"]["clean_source_only"])
        for right in surfaces:
            right_ids = set(right["ids"]["clean_source_only"])
            shared = sorted(left_ids & right_ids)
            rows.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "clean_source_only_overlap_count": len(shared),
                    "clean_source_only_overlap_ids": shared,
                }
            )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Durable Source-Surface Ranking",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload['git_commit']}`",
        f"- min clean source-only: `{payload['config']['min_clean_source_only']}`",
        f"- min numeric coverage: `{payload['config']['min_numeric_coverage']}`",
        "",
        "| Rank | Surface | Role | Decision | Clean | Source-only | Oracle gain | Target | Source | Notes |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for surface in payload["ranked_surfaces"]:
        target = surface["summaries"]["target"]
        source = surface["summaries"]["source"]
        lines.append(
            f"| {surface['rank']['index']} | `{surface['label']}` | `{surface['role']}` | "
            f"`{surface['decision']}` | {surface['counts']['clean_source_only']} | "
            f"{surface['counts']['source_only']} | {surface['counts']['oracle_gain_vs_target']} | "
            f"{target.get('correct', 0)}/{target.get('n', surface['id_provenance']['reference_n'])} | "
            f"{source.get('correct', 0)}/{source.get('n', surface['id_provenance']['reference_n'])} | "
            f"{surface.get('note', '')} |"
        )
    lines.extend(["", "## Top Clean IDs", ""])
    for surface in payload["ranked_surfaces"][:3]:
        ids = ", ".join(f"`{item}`" for item in surface["ids"]["clean_source_only"]) or "none"
        lines.append(f"- `{surface['label']}`: {ids}")
    lines.extend(["", "## Next Gate", "", payload["recommended_next_gate"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    surfaces = [
        analyze_target_set(
            spec,
            min_clean_source_only=int(args.min_clean_source_only),
            min_numeric_coverage=int(args.min_numeric_coverage),
            require_exact_ordered_id_parity=bool(args.require_exact_ordered_id_parity),
        )
        for spec in args.target_set
    ]
    ranked = sorted(surfaces, key=_rank_key, reverse=True)
    for index, surface in enumerate(ranked, start=1):
        surface["rank"]["index"] = index
        surface["rank"]["rank_score"] = list(_rank_key(surface))

    top = ranked[0] if ranked else None
    status = (
        "primary_surface_selected"
        if top and top["decision"] == "primary_ready"
        else "no_primary_surface_selected"
    )
    recommended = (
        f"Run the next method gate on `{top['label']}` first, then replay on the "
        "canonical holdout with identical controls."
        if status == "primary_surface_selected"
        else "Materialize a stronger exact-ID source surface before another method run."
    )
    payload = {
        "schema_version": "source_surface_ranking.v1",
        "date": str(args.date),
        "status": status,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "config": {
            "min_clean_source_only": int(args.min_clean_source_only),
            "min_numeric_coverage": int(args.min_numeric_coverage),
            "require_exact_ordered_id_parity": bool(args.require_exact_ordered_id_parity),
        },
        "surfaces": surfaces,
        "ranked_surfaces": ranked,
        "overlap_matrix": _overlap_matrix(surfaces),
        "recommended_next_gate": recommended,
    }
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": status, "top": top["label"] if top else None}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-set", action="append", type=_parse_target_set_spec, default=[])
    parser.add_argument("--min-clean-source-only", type=int, default=5)
    parser.add_argument("--min-numeric-coverage", type=int, default=0)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--require-exact-ordered-id-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    if not args.target_set:
        parser.error("at least one --target-set is required")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
