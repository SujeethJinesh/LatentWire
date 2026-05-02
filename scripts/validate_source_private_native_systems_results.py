from __future__ import annotations

"""Validate native systems measurements against the LatentWire schema.

The native systems plan already defines required metrics and baseline rows.
This script turns that plan into an enforceable ingest gate: future NVIDIA /
vLLM / SGLang / C2C / KVComm / QJL / TurboQuant rows must type-check, include
all required fields, preserve source-exposure flags, and cover every required
baseline before `native_systems_complete` can become true.
"""

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PLAN_DIR = pathlib.Path("results/source_private_native_systems_benchmark_plan_20260501")
DEFAULT_SCHEMA = DEFAULT_PLAN_DIR / "native_systems_metric_schema.csv"
DEFAULT_BASELINES = DEFAULT_PLAN_DIR / "native_systems_baseline_rows.csv"
DEFAULT_OUTPUT = pathlib.Path("results/source_private_native_systems_result_ingest_gate_20260502")
REQUIRED_LATENTWIRE_MAX_FRAMED_BYTES = 64.0
BOOL_TRUE = {"1", "true", "yes", "y"}
BOOL_FALSE = {"0", "false", "no", "n"}


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: pathlib.Path | str) -> list[dict[str, str]]:
    with _resolve(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _load_measurements(inputs: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    files: list[str] = []
    for raw_pattern in inputs:
        matches = sorted(glob.glob(str(_resolve(raw_pattern))))
        if not matches and _resolve(raw_pattern).exists():
            matches = [str(_resolve(raw_pattern))]
        for match in matches:
            path = pathlib.Path(match)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                loaded = _read_csv(path)
            elif suffix in {".jsonl", ".json"}:
                if suffix == ".jsonl":
                    loaded = _read_jsonl(path)
                else:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, dict) and isinstance(value.get("rows"), list):
                        loaded = value["rows"]
                    elif isinstance(value, list):
                        loaded = value
                    else:
                        raise ValueError(f"{path} must be a JSON list or object with rows[]")
            else:
                raise ValueError(f"unsupported measurement file suffix for {path}")
            rows.extend(dict(row, _measurement_file=_display_path(path)) for row in loaded)
            files.append(_display_path(path))
    return rows, files


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    return None


def _parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _metric_required(schema_row: dict[str, str]) -> bool:
    return _parse_bool(schema_row.get("required", "")) is True


def _is_numeric_unit(unit: str) -> bool:
    return unit in {
        "count",
        "fraction",
        "ms",
        "ms/token",
        "requests/s",
        "tokens/s",
        "GB",
        "bytes/request",
        "tokens",
        "seconds",
    }


def _validate_metric_value(metric: str, unit: str, value: Any) -> str | None:
    if unit == "bool":
        return None if _parse_bool(value) is not None else f"{metric} must be boolean"
    if _is_numeric_unit(unit):
        parsed = _parse_float(value)
        if parsed is None:
            return f"{metric} must be finite numeric"
        if unit in {"count", "tokens"} and parsed < 0:
            return f"{metric} must be non-negative"
        if unit in {"ms", "ms/token", "requests/s", "tokens/s", "GB", "bytes/request", "seconds"} and parsed < 0:
            return f"{metric} must be non-negative"
        if metric == "accuracy" and not (0.0 <= parsed <= 1.0):
            return "accuracy must be in [0, 1]"
        if metric.startswith("paired_ci95") and not (-1.0 <= parsed <= 1.0):
            return f"{metric} must be in [-1, 1]"
        if metric.startswith("paired_delta") and not (-1.0 <= parsed <= 1.0):
            return f"{metric} must be in [-1, 1]"
        return None
    if str(value).strip() == "":
        return f"{metric} must be non-empty"
    return None


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("baseline_row_id") or row.get("method") or "").strip()


def _measurement_errors(
    *,
    row: dict[str, Any],
    schema: list[dict[str, str]],
    baseline_by_id: dict[str, dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    row_id = _row_id(row)
    if not row_id:
        errors.append("missing row_id/baseline_row_id")
        return errors
    baseline = baseline_by_id.get(row_id)
    if baseline is None:
        errors.append(f"row_id {row_id!r} is not in baseline schema")
    for metric in schema:
        name = str(metric["metric"])
        unit = str(metric["unit"])
        if not _metric_required(metric):
            continue
        if name not in row or str(row.get(name, "")).strip() == "":
            errors.append(f"missing required metric {name}")
            continue
        error = _validate_metric_value(name, unit, row[name])
        if error is not None:
            errors.append(error)
    if baseline is not None:
        for flag in ("source_text_exposed", "source_kv_exposed"):
            expected = _parse_bool(baseline.get(flag, ""))
            actual = _parse_bool(row.get(flag, ""))
            if expected is not None and actual is not None and expected != actual:
                errors.append(f"{flag}={actual} disagrees with baseline expectation {expected}")
        hidden = _parse_bool(row.get("source_hidden_or_score_vector_exposed", "false"))
        family = str(baseline.get("family", ""))
        if row_id.startswith("latentwire_packet") and hidden:
            errors.append("LatentWire packet rows must not expose raw hidden/score vectors")
        if row_id.startswith("latentwire_packet"):
            framed = _parse_float(row.get("framed_bytes_per_request"))
            if framed is not None and framed > REQUIRED_LATENTWIRE_MAX_FRAMED_BYTES:
                errors.append(
                    f"LatentWire packet framed bytes {framed} exceeds {REQUIRED_LATENTWIRE_MAX_FRAMED_BYTES}"
                )
        if family in {"cache_communication", "kv_communication", "quantized_projection", "quantized_kv"}:
            source_state = _parse_float(row.get("transferred_source_state_bytes"))
            if source_state is not None and source_state <= 0:
                errors.append("source-state baseline must report transferred_source_state_bytes > 0")
    return errors


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Native Systems Result Ingest Gate",
        "",
        f"- validator pass: `{payload['validator_pass']}`",
        f"- native systems complete: `{payload['native_systems_complete']}`",
        f"- paper native win allowed: `{payload['paper_native_win_allowed']}`",
        f"- measurement rows: `{payload['measurement_row_count']}`",
        f"- required baseline rows: `{payload['required_baseline_count']}`",
        f"- missing required rows: `{len(payload['missing_required_row_ids'])}`",
        f"- invalid measurement rows: `{len(payload['invalid_measurement_rows'])}`",
        "",
        "## Missing Required Rows",
        "",
    ]
    if payload["missing_required_row_ids"]:
        lines.extend(f"- `{row_id}`" for row_id in payload["missing_required_row_ids"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
            "## Lay Explanation",
            "",
            payload["lay_explanation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_row_status_csv(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fields = ["row_id", "required", "present_count", "valid_count", "status", "errors"]
    invalid_by_row = {row["row_id"]: row["errors"] for row in payload["invalid_measurement_rows"]}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["row_status"]:
            writer.writerow({**row, "errors": "; ".join(invalid_by_row.get(row["row_id"], []))})


def validate_native_systems_results(
    *,
    schema_path: pathlib.Path = DEFAULT_SCHEMA,
    baseline_rows_path: pathlib.Path = DEFAULT_BASELINES,
    measurement_inputs: tuple[str, ...] = (),
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = _read_csv(schema_path)
    baselines = _read_csv(baseline_rows_path)
    if not schema:
        raise ValueError("native systems metric schema is empty")
    if not baselines:
        raise ValueError("native systems baseline rows are empty")
    required_metrics = [row["metric"] for row in schema if _metric_required(row)]
    baseline_by_id = {str(row["row_id"]): row for row in baselines}
    required_row_ids = sorted(
        str(row["row_id"]) for row in baselines if _parse_bool(row.get("required_for_native_gate", "")) is True
    )
    measurements, measurement_files = _load_measurements(measurement_inputs)
    invalid_rows: list[dict[str, Any]] = []
    valid_measurements: list[dict[str, Any]] = []
    for index, row in enumerate(measurements):
        row_id = _row_id(row)
        errors = _measurement_errors(row=row, schema=schema, baseline_by_id=baseline_by_id)
        if errors:
            invalid_rows.append(
                {
                    "index": index,
                    "row_id": row_id or "<missing>",
                    "measurement_file": row.get("_measurement_file", ""),
                    "errors": errors,
                }
            )
        else:
            valid_measurements.append(row)
    valid_by_row: dict[str, int] = {}
    present_by_row: dict[str, int] = {}
    for row in measurements:
        row_id = _row_id(row)
        if row_id:
            present_by_row[row_id] = present_by_row.get(row_id, 0) + 1
    for row in valid_measurements:
        row_id = _row_id(row)
        valid_by_row[row_id] = valid_by_row.get(row_id, 0) + 1
    missing_required = [row_id for row_id in required_row_ids if valid_by_row.get(row_id, 0) == 0]
    row_status: list[dict[str, Any]] = []
    for row_id in required_row_ids:
        present_count = present_by_row.get(row_id, 0)
        valid_count = valid_by_row.get(row_id, 0)
        if valid_count > 0:
            status = "valid"
        elif present_count > 0:
            status = "invalid"
        else:
            status = "missing"
        row_status.append(
            {
                "row_id": row_id,
                "required": True,
                "present_count": present_count,
                "valid_count": valid_count,
                "status": status,
            }
        )
    native_complete = bool(not missing_required and not invalid_rows and len(valid_measurements) >= len(required_row_ids))
    validator_pass = bool(len(required_metrics) >= 40 and len(required_row_ids) >= 10 and all(row.get("row_id") for row in baselines))
    payload = {
        "gate": "source_private_native_systems_result_ingest_gate",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "validator_pass": validator_pass,
        "native_systems_complete": native_complete,
        "paper_native_win_allowed": native_complete,
        "measurement_row_count": len(measurements),
        "valid_measurement_row_count": len(valid_measurements),
        "required_metric_count": len(required_metrics),
        "required_metrics": required_metrics,
        "required_baseline_count": len(required_row_ids),
        "required_row_ids": required_row_ids,
        "missing_required_row_ids": missing_required,
        "invalid_measurement_rows": invalid_rows,
        "row_status": row_status,
        "input_files": {
            "schema": _display_path(schema_path),
            "schema_sha256": _sha256_file(schema_path),
            "baseline_rows": _display_path(baseline_rows_path),
            "baseline_rows_sha256": _sha256_file(baseline_rows_path),
            "measurement_files": measurement_files,
        },
        "decision": (
            "Native systems claims remain blocked until every required baseline row is ingested with all "
            "quality, latency, memory, traffic, payload-byte, and source-exposure fields. The current run "
            "validates the schema and correctly refuses native-systems-complete because required native "
            "measurement rows are missing."
            if not native_complete
            else "Native systems measurements satisfy the required schema and baseline coverage gate."
        ),
        "lay_explanation": (
            "This checker is a checklist with teeth. It will not let the paper say we have a real GPU systems "
            "win until every required method has the same accuracy, latency, memory, traffic, byte, and privacy "
            "fields filled in."
        ),
    }
    json_path = output_dir / "native_systems_result_ingest_gate.json"
    md_path = output_dir / "native_systems_result_ingest_gate.md"
    csv_path = output_dir / "native_systems_row_status.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_row_status_csv(csv_path, payload)
    manifest = {
        "gate": payload["gate"],
        "validator_pass": validator_pass,
        "native_systems_complete": native_complete,
        "files": [
            {"path": _display_path(json_path), "sha256": _sha256_file(json_path)},
            {"path": _display_path(md_path), "sha256": _sha256_file(md_path)},
            {"path": _display_path(csv_path), "sha256": _sha256_file(csv_path)},
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=pathlib.Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--baseline-rows", type=pathlib.Path, default=DEFAULT_BASELINES)
    parser.add_argument("--measurement", action="append", default=[])
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = validate_native_systems_results(
        schema_path=args.schema,
        baseline_rows_path=args.baseline_rows,
        measurement_inputs=tuple(args.measurement),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "validator_pass": payload["validator_pass"],
                "native_systems_complete": payload["native_systems_complete"],
                "measurement_row_count": payload["measurement_row_count"],
                "missing_required_rows": len(payload["missing_required_row_ids"]),
                "invalid_measurement_rows": len(payload["invalid_measurement_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
