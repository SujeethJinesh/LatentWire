#!/usr/bin/env python3
"""Compare a candidate-pool reachability audit against a baseline audit."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ids(payload: dict[str, Any], key: str) -> set[str]:
    return {str(item) for item in payload.get(key, [])}


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate Pool Reachability Comparison",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- baseline: `{payload['baseline_path']}`",
        f"- candidate: `{payload['candidate_path']}`",
        "",
        "## Summary",
        "",
        f"- baseline oracle: `{payload['baseline_sample_oracle_correct']}/{payload['reference_n']}`",
        f"- candidate oracle: `{payload['candidate_sample_oracle_correct']}/{payload['reference_n']}`",
        f"- candidate minus baseline oracle: `{payload['candidate_minus_baseline_oracle']}`",
        f"- new candidate oracle IDs: `{payload['new_oracle_count']}`",
        f"- lost baseline oracle IDs: `{payload['lost_oracle_count']}`",
        f"- new C2C clean residual IDs: `{payload['new_c2c_clean_residual_count']}`",
        f"- candidate C2C clean residual in pool: `{payload['candidate_c2c_clean_residual_in_pool']}/{payload['candidate_c2c_clean_residual_total']}`",
        "",
        "## IDs",
        "",
        "- new oracle IDs: " + (", ".join(f"`{item}`" for item in payload["new_oracle_ids"]) or "none"),
        "- lost oracle IDs: " + (", ".join(f"`{item}`" for item in payload["lost_oracle_ids"]) or "none"),
        "- new C2C clean residual IDs: "
        + (", ".join(f"`{item}`" for item in payload["new_c2c_clean_residual_ids"]) or "none"),
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-reachability", required=True)
    parser.add_argument("--candidate-reachability", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    raw_argv = sys.argv if argv is None else ["scripts/compare_candidate_pool_reachability.py", *argv]

    baseline_path = _resolve(args.baseline_reachability)
    candidate_path = _resolve(args.candidate_reachability)
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)

    baseline_oracle = _ids(baseline, "sample_oracle_ids")
    candidate_oracle = _ids(candidate, "sample_oracle_ids")
    baseline_c2c_clean = _ids(baseline, "c2c_clean_residual_in_pool_ids")
    candidate_c2c_clean = _ids(candidate, "c2c_clean_residual_in_pool_ids")
    new_oracle = sorted(candidate_oracle - baseline_oracle)
    lost_oracle = sorted(baseline_oracle - candidate_oracle)
    new_c2c_clean = sorted(candidate_c2c_clean - baseline_c2c_clean)

    if new_c2c_clean:
        decision = "Pass: candidate pool adds C2C-clean residual reachability beyond the baseline pool."
    elif len(candidate_oracle) > len(baseline_oracle):
        decision = "Weak pass: candidate pool improves total oracle but not C2C-clean residual reachability."
    else:
        decision = "Fail: candidate pool does not improve oracle reachability over the baseline pool."

    payload = {
        "date": args.date,
        "status": "candidate_pool_reachability_compared",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "baseline_path": _display(baseline_path),
        "candidate_path": _display(candidate_path),
        "reference_n": int(candidate.get("reference_n", baseline.get("reference_n", 0))),
        "baseline_sample_oracle_correct": int(baseline.get("sample_oracle_correct", len(baseline_oracle))),
        "candidate_sample_oracle_correct": int(candidate.get("sample_oracle_correct", len(candidate_oracle))),
        "candidate_minus_baseline_oracle": int(len(candidate_oracle) - len(baseline_oracle)),
        "new_oracle_count": len(new_oracle),
        "new_oracle_ids": new_oracle,
        "lost_oracle_count": len(lost_oracle),
        "lost_oracle_ids": lost_oracle,
        "new_c2c_clean_residual_count": len(new_c2c_clean),
        "new_c2c_clean_residual_ids": new_c2c_clean,
        "candidate_c2c_clean_residual_in_pool": int(candidate.get("c2c_clean_residual_in_pool", len(candidate_c2c_clean))),
        "candidate_c2c_clean_residual_total": int(candidate.get("c2c_clean_residual_total", 0)),
        "decision": decision,
    }

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "decision": payload["decision"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
