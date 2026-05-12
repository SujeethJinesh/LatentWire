#!/usr/bin/env python3
"""Check OutlierMigrate Phase 5'' Qwen3.6 packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase5_double_prime/results"
SCHEMA_VERSION = "om_phase5dp_qwen36_v1"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_SNAPSHOT_COMMIT = "995ad96eacd98c81ed38be0c5b274b04031597b0"

DYNAMIC = "DYNAMIC_REGIME_QWEN36"
STATIC = "STATIC_REGIME_QWEN36"
AMBIGUOUS = "AMBIGUOUS_QWEN36"
FAIL_INFRA = "FAIL_INFRA_QWEN36"

REQUIRED_FILES = [
    "environment.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "capability_probe.json",
    "infra_error.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no Phase 5'' result dirs under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in HASHED_FILES:
        item = by_path.get(rel)
        path = run_dir / rel
        if item is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if not path.is_file():
            infra.append(f"hashed artifact missing on disk: {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required artifact: {rel}")
    try:
        environment = load_json(run_dir / "environment.json")
        model = load_json(run_dir / "model_provenance.json")
        prompt = load_json(run_dir / "prompt_manifest.json")
        command = load_json(run_dir / "command_metadata.json")
        probe = load_json(run_dir / "capability_probe.json")
        infra_error = load_json(run_dir / "infra_error.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
    except Exception as exc:
        infra.append(f"bad JSON artifacts: {exc!r}")
        environment = model = prompt = command = probe = infra_error = artifact_hashes = {}

    if environment.get("schema_version") != f"{SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if model.get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if model.get("hf_snapshot_commit") != MODEL_SNAPSHOT_COMMIT:
        infra.append("model snapshot commit mismatch")
    if prompt.get("prompt_sha256") != "sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e":
        infra.append("prompt SHA mismatch")
    if command.get("branch") != "outlier_migrate_phase5_double_prime_qwen36":
        infra.append("command branch mismatch")
    if probe.get("activation_capture_accessible") is not False:
        infra.append("capability_probe.activation_capture_accessible must be false for this infra packet")
    if "per_layer_activation_capture_unavailable" not in probe.get("failure_classification", ""):
        infra.append("capability_probe failure classification mismatch")
    if not infra_error.get("reasons"):
        infra.append("infra_error.reasons must be non-empty")
    if isinstance(artifact_hashes, dict):
        validate_artifact_hashes(run_dir, artifact_hashes, infra)

    result = {
        "decision": FAIL_INFRA,
        "run_dir": str(run_dir),
        "artifact_complete": not infra,
        "reasons": infra if infra else list(infra_error.get("reasons", [])),
        "model_id": MODEL_ID,
        "model_snapshot_commit": MODEL_SNAPSHOT_COMMIT,
        "activation_capture_accessible": False,
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(
        run_dir / "artifact_check.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_artifact_check",
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "required_files": REQUIRED_FILES,
            "artifact_complete": result["artifact_complete"],
            "reasons": result["reasons"],
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
