#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_FIELDS: tuple[str, ...] = (
    "selector_method",
    "patch_corr",
    "quant_error_corr",
    "feature_persistence",
    "protected_ids",
    "bit_allocation",
    "help",
    "harm",
    "missed_help",
    "false_prune",
    "bytes",
    "compute",
    "stability",
)

_MISSING = object()


def _first(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        if isinstance(item, (int, float)):
            ids.append(int(item))
    return ids


def _bit_allocation(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    explicit = _first(row, ("bit_allocation", "mixed_bit_allocation"), _MISSING)
    if explicit is not _MISSING:
        return explicit if isinstance(explicit, dict) else {"value": explicit}

    low_bits = _first(row, ("low_bits",), config.get("low_bits"))
    high_bits = _first(row, ("high_bits",), config.get("high_bits"))
    protected = _first(row, ("protected_atoms", "protected_count", "protected"), None)
    if low_bits is None and high_bits is None and protected is None:
        return None
    return {
        "low_bits": low_bits,
        "high_bits": high_bits,
        "protected_count": int(protected) if isinstance(protected, (int, float)) else protected,
        "source": "count_only",
    }


def _help_value(row: dict[str, Any]) -> float | None:
    return _as_float(
        _first(
            row,
            (
                "help",
                "help_rate",
                "help_vs_prune_uniform_quant",
                "help_vs_no_pruning",
                "help_vs_full_precision",
                "help_vs_uniform",
                "route_help_vs_uniform",
                "answer_help_vs_uniform",
            ),
        )
    )


def _harm_value(row: dict[str, Any]) -> float | None:
    return _as_float(
        _first(
            row,
            (
                "harm",
                "harm_rate",
                "harm_vs_prune_uniform_quant",
                "harm_vs_no_pruning",
                "harm_vs_full_precision",
                "harm_vs_uniform",
                "route_harm_vs_uniform",
                "answer_harm_vs_uniform",
            ),
        )
    )


def normalize_row(row: dict[str, Any], *, input_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    normalized = {
        "selector_method": str(_first(row, ("selector_method", "method", "name"), "unknown")),
        "patch_corr": _as_float(
            _first(row, ("patch_corr", "patch_rank_correlation", "patch_effect_corr", "patch_correlation"))
        ),
        "quant_error_corr": _as_float(
            _first(row, ("quant_error_corr", "quant_error_rank_correlation", "quantization_error_correlation"))
        ),
        "feature_persistence": _as_float(
            _first(
                row,
                (
                    "feature_persistence",
                    "feature_overlap_persistence",
                    "shared_feature_recovery",
                    "atom_recovery",
                ),
            )
        ),
        "protected_ids": _as_id_list(_first(row, ("protected_ids", "selected_ids", "protected_atom_ids"))),
        "bit_allocation": _bit_allocation(row, config),
        "help": _help_value(row),
        "harm": _harm_value(row),
        "missed_help": _as_float(_first(row, ("missed_help", "missed_help_rate", "best_gate_missed_help"))),
        "false_prune": _as_float(_first(row, ("false_prune", "false_prune_rate"))),
        "bytes": _as_float(_first(row, ("bytes", "bytes_proxy", "bytes_estimate", "bytes_per_example"))),
        "compute": _as_float(_first(row, ("compute", "compute_proxy", "compute_fraction"))),
        "stability": _as_float(_first(row, ("stability", "selector_stability"))),
        "source": input_name,
    }
    return normalized


def _rows_from_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)], {}
    if not isinstance(payload, dict):
        return [], {}
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    rows = payload.get("rows", payload.get("method_rows", payload.get("telemetry_rows", [])))
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list):
        rows = []
    return [row for row in rows if isinstance(row, dict)], config


def load_and_normalize(paths: Sequence[Path]) -> list[dict[str, Any]]:
    if not paths:
        return [normalize_row(row, input_name="embedded_fixture", config=_fixture_payload()["config"]) for row in _fixture_payload()["rows"]]

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        source_rows, config = _rows_from_payload(payload)
        rows.extend(normalize_row(row, input_name=str(path), config=config) for row in source_rows)
    return rows


def _fixture_payload() -> dict[str, Any]:
    return {
        "config": {"low_bits": 2, "high_bits": 8},
        "rows": [
            {
                "method": "quant_error_protect",
                "patch_rank_correlation": 0.2712,
                "quant_error_rank_correlation": 0.8831,
                "feature_overlap_persistence": 0.214,
                "protected_ids": [1, 4, 7],
                "protected_atoms": 3,
                "help_vs_prune_uniform_quant": 0.1458,
                "harm_vs_prune_uniform_quant": 0.0,
                "missed_help_rate": 0.086,
                "false_prune_rate": 0.019,
                "bytes_proxy": 91.0,
                "compute_proxy": 681.0,
                "selector_stability": 0.72,
            },
            {
                "method": "universal_dictionary_persistence_protect",
                "patch_rank_correlation": 0.6078,
                "quant_error_rank_correlation": 0.441,
                "feature_overlap_persistence": 0.3303,
                "protected_ids": [0, 2, 5],
                "protected_atoms": 3,
                "help_vs_prune_uniform_quant": 0.2917,
                "harm_vs_prune_uniform_quant": 0.0,
                "missed_help_rate": 0.0,
                "false_prune_rate": 0.111,
                "bytes_proxy": 166.0,
                "compute_proxy": 446.4,
                "selector_stability": 1.0,
            },
        ],
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    missing_by_field = {
        field: sum(1 for row in rows if row.get(field) in (None, []) and field != "protected_ids")
        for field in SCHEMA_FIELDS
    }
    best_patch = max(rows, key=lambda row: row["patch_corr"] if row.get("patch_corr") is not None else float("-inf"), default=None)
    best_help = max(rows, key=lambda row: row["help"] if row.get("help") is not None else float("-inf"), default=None)
    return {
        "row_count": len(rows),
        "schema_fields": list(SCHEMA_FIELDS),
        "selector_methods": [row["selector_method"] for row in rows],
        "best_patch_corr_selector": None if best_patch is None else best_patch["selector_method"],
        "best_help_selector": None if best_help is None else best_help["selector_method"],
        "missing_by_field": missing_by_field,
    }


def build_payload(paths: Sequence[Path]) -> dict[str, Any]:
    rows = load_and_normalize(paths)
    return {
        "schema_version": 1,
        "generated_by": "scripts/analyze_frontier_selector_telemetry.py",
        "inputs": [str(path) for path in paths] or ["embedded_fixture"],
        "summary": summarize(rows),
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Frontier Selector Telemetry",
        "",
        "Unified fast telemetry for toy selector rows and future real route-pool rows.",
        "",
        f"- Rows: {payload['summary']['row_count']}",
        f"- Schema fields: {', '.join(payload['summary']['schema_fields'])}",
        f"- Best patch-corr selector: {payload['summary']['best_patch_corr_selector'] or '-'}",
        f"- Best help selector: {payload['summary']['best_help_selector'] or '-'}",
        "",
        "| Selector | Patch corr | Quant-error corr | Feature persistence | Protected ids | Bit allocation | Help | Harm | Missed help | False prune | Bytes | Compute | Stability |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {selector_method} | {patch_corr} | {quant_error_corr} | {feature_persistence} | {protected_ids} | {bit_allocation} | {help} | {harm} | {missed_help} | {false_prune} | {bytes} | {compute} | {stability} |".format(
                **{field: _fmt(row.get(field)) for field in SCHEMA_FIELDS}
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=[], help="Input JSON result file. Repeatable.")
    parser.add_argument("--output", required=True, type=Path, help="Output telemetry JSON path.")
    parser.add_argument("--output-md", type=Path, help="Output markdown path. Defaults to output with .md suffix.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = build_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = args.output_md or args.output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(payload, markdown)
    return payload


if __name__ == "__main__":
    main()
