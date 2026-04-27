#!/usr/bin/env python3
"""Audit target-side candidate-pool headroom for source-side communication."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


@dataclass(frozen=True)
class SurfaceSpec:
    label: str
    path: pathlib.Path
    role: str
    note: str


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _parse_surface_spec(raw: str) -> SurfaceSpec:
    label, rest = raw.split("=", 1)
    fields: dict[str, str] = {}
    for part in rest.split(","):
        key, value = part.split("=", 1)
        fields[key] = value
    if "path" not in fields:
        raise ValueError(f"surface spec missing path: {raw!r}")
    return SurfaceSpec(
        label=label,
        path=_resolve(fields["path"]),
        role=fields.get("role", ""),
        note=fields.get("note", ""),
    )


def _correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("correct"))


def _candidate_values(surface: decoder.Surface, example_id: str) -> set[str]:
    return {candidate.value for candidate in decoder._candidate_pool(surface, example_id)}


def _audit_surface(spec: SurfaceSpec) -> dict[str, Any]:
    surface = decoder._load_surface(spec.label, spec.path)
    clean_ids = decoder._source_ids(surface, "clean_source_only") or decoder._source_ids(surface, "clean_residual_targets")
    target_self_ids = decoder._source_ids(surface, "target_self_repair")
    rows: list[dict[str, Any]] = []
    target_correct = 0
    source_correct = 0
    oracle_correct = 0
    clean_gold_in_pool: list[str] = []
    clean_oracle_gain: list[str] = []
    target_self_gold_in_pool = 0

    for example_id in surface.reference_ids:
        target_row = surface.records_by_label["target"].get(example_id)
        source_row = surface.records_by_label["source"].get(example_id)
        gold = decoder._gold_numeric(surface, example_id)
        values = _candidate_values(surface, example_id)
        target_ok = _correct(target_row)
        source_ok = _correct(source_row)
        pool_has_gold = gold in values
        oracle_ok = target_ok or pool_has_gold
        target_correct += int(target_ok)
        source_correct += int(source_ok)
        oracle_correct += int(oracle_ok)
        if example_id in clean_ids and pool_has_gold:
            clean_gold_in_pool.append(example_id)
            if not target_ok:
                clean_oracle_gain.append(example_id)
        if example_id in target_self_ids and pool_has_gold:
            target_self_gold_in_pool += 1
        rows.append(
            {
                "example_id": example_id,
                "target_correct": target_ok,
                "source_correct": source_ok,
                "gold": gold,
                "pool_has_gold": pool_has_gold,
                "target_side_pool_values": sorted(values),
                "clean_source_only": example_id in clean_ids,
                "target_self_repair": example_id in target_self_ids,
            }
        )

    n = len(surface.reference_ids)
    return {
        "label": spec.label,
        "role": spec.role,
        "note": spec.note,
        "path": _display_path(spec.path),
        "path_sha256": _sha256_file(spec.path),
        "n": n,
        "target_correct": target_correct,
        "source_correct": source_correct,
        "target_side_oracle_correct": oracle_correct,
        "target_side_oracle_gain": oracle_correct - target_correct,
        "clean_source_only_count": len(clean_ids),
        "clean_gold_in_target_side_pool": len(clean_gold_in_pool),
        "clean_gold_in_target_side_pool_ids": sorted(clean_gold_in_pool),
        "clean_oracle_gain_ids": sorted(clean_oracle_gain),
        "target_self_count": len(target_self_ids),
        "target_self_gold_in_target_side_pool": target_self_gold_in_pool,
        "rows": rows,
    }


def _rank_key(surface: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(surface["clean_gold_in_target_side_pool"]),
        int(surface["target_side_oracle_gain"]),
        int(surface["target_side_oracle_correct"]),
        -int(surface["target_correct"]),
    )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Target-Side Candidate Headroom Audit",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        "",
        "| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, surface in enumerate(payload["surfaces"], start=1):
        lines.append(
            "| {rank} | `{label}` | `{role}` | {target}/{n} | {source}/{n} | {oracle}/{n} | {gain} | {clean}/{clean_total} | {note} |".format(
                rank=rank,
                label=surface["label"],
                role=surface["role"],
                target=surface["target_correct"],
                source=surface["source_correct"],
                oracle=surface["target_side_oracle_correct"],
                gain=surface["target_side_oracle_gain"],
                clean=surface["clean_gold_in_target_side_pool"],
                clean_total=surface["clean_source_only_count"],
                n=surface["n"],
                note=surface["note"],
            )
        )
    lines.extend(["", "## Clean IDs In Target-Side Pool", ""])
    for surface in payload["surfaces"]:
        ids = ", ".join(f"`{example_id}`" for example_id in surface["clean_gold_in_target_side_pool_ids"]) or "none"
        lines.append(f"- `{surface['label']}`: {ids}")
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-set", action="append", type=_parse_surface_spec, required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/analyze_target_side_candidate_headroom.py", *argv]

    surfaces = sorted((_audit_surface(spec) for spec in args.target_set), key=_rank_key, reverse=True)
    top = surfaces[0] if surfaces else None
    next_gate = (
        "No target-side candidate-pool surface was provided."
        if top is None
        else (
            f"Use `{top['label']}` only if the next method can target "
            f"{top['clean_gold_in_target_side_pool']} clean IDs already present in the target-side pool; "
            "otherwise switch to a new candidate-surface generator."
        )
    )
    payload = {
        "date": args.date,
        "status": "target_side_candidate_headroom_ranked" if top else "no_surfaces",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "surfaces": surfaces,
        "next_gate": next_gate,
    }

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "top": top["label"] if top else None}, indent=2))
    return payload


if __name__ == "__main__":
    main()
