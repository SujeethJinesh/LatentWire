"""Validate HybridKernel quality-smoke artifacts before citing speed rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


QUALITY_SMOKE_VERSION = "hybridkernel_quality_smoke_v1"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_ACCURACY_DROP = 0.01
MAX_LENGTH_DRIFT_FRACTION = 0.10


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_relative(base: Path, value: object, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty relative path")
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label} must stay inside the repository")
        return None
    resolved = base / path
    if not resolved.is_file():
        errors.append(f"{label} does not exist: {value}")
        return None
    return resolved


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def check_quality_smoke(path: Path, *, repo_root: Path | None = None) -> dict[str, object]:
    repo_root = repo_root or Path.cwd()
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "FAIL", "errors": [f"quality smoke JSON is invalid: {exc}"]}
    if not isinstance(payload, dict):
        return {"status": "FAIL", "errors": ["quality smoke JSON must be an object"]}
    if payload.get("quality_smoke_version") != QUALITY_SMOKE_VERSION:
        errors.append(f"quality_smoke_version must be {QUALITY_SMOKE_VERSION!r}")

    prompt_path = _resolve_relative(repo_root, payload.get("prompt_file"), "prompt_file", errors)
    prompt_sha = str(payload.get("prompt_file_sha256", "")).strip()
    if not SHA256_PATTERN.match(prompt_sha):
        errors.append("prompt_file_sha256 must be sha256:<64 lowercase hex chars>")
    elif prompt_path is not None and _file_sha256(prompt_path) != prompt_sha:
        errors.append("prompt_file_sha256 must match prompt_file")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("quality smoke must contain non-empty rows")
        rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index} must be an object")
            continue
        for field in ("model", "stock_mode", "prototype_mode"):
            if not isinstance(row.get(field), str) or not str(row.get(field)).strip():
                errors.append(f"row {index} {field} must be filled")
        if not _positive_int(row.get("prompt_count")) or int(row.get("prompt_count", 0)) < 12:
            errors.append(f"row {index} prompt_count must be at least 12")
        mismatch_count = row.get("normalized_answer_mismatch_count")
        if not isinstance(mismatch_count, int) or isinstance(mismatch_count, bool) or mismatch_count != 0:
            errors.append(f"row {index} normalized_answer_mismatch_count must be 0")
        accuracy_delta = _number(row.get("accuracy_delta_prototype_minus_stock"))
        if accuracy_delta is None:
            errors.append(f"row {index} accuracy_delta_prototype_minus_stock must be numeric")
        elif accuracy_delta < -MAX_ACCURACY_DROP:
            errors.append(f"row {index} accuracy drop exceeds {MAX_ACCURACY_DROP:.2f}")
        length_drift = _number(row.get("mean_output_length_drift_fraction"))
        if length_drift is None:
            errors.append(f"row {index} mean_output_length_drift_fraction must be numeric")
        elif abs(length_drift) > MAX_LENGTH_DRIFT_FRACTION:
            errors.append(f"row {index} output length drift exceeds {MAX_LENGTH_DRIFT_FRACTION:.2f}")
        for artifact_field in ("stock_outputs_path", "prototype_outputs_path"):
            artifact_path = _resolve_relative(repo_root, row.get(artifact_field), f"row {index} {artifact_field}", errors)
            sha_field = artifact_field.replace("_path", "_sha256")
            expected_sha = str(row.get(sha_field, "")).strip()
            if not SHA256_PATTERN.match(expected_sha):
                errors.append(f"row {index} {sha_field} must be sha256:<64 lowercase hex chars>")
            elif artifact_path is not None and _file_sha256(artifact_path) != expected_sha:
                errors.append(f"row {index} {sha_field} must match {artifact_field}")

    return {"status": "FAIL" if errors else "PASS", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("quality_smoke_json", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = check_quality_smoke(args.quality_smoke_json, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
