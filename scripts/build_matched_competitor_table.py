#!/usr/bin/env python3
"""Build a lightweight matched competitor matrix from local JSONL/meta artifacts.

The table is intentionally tolerant of missing heavyweight competitor runs. A
missing row is reported as `missing` instead of being dropped, which keeps the
paper comparison checklist auditable as results are filled in.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RowSpec:
    row_id: str
    label: str
    family: str
    artifact: str
    method: str | None = None
    required: bool = True


@dataclass(frozen=True)
class RowStats:
    spec: RowSpec
    status: str
    n: int | None = None
    accuracy: float | None = None
    latency_sec: float | None = None
    token_proxy: float | None = None
    source: str = ""
    note: str = ""


DEFAULT_ROWS: tuple[RowSpec, ...] = (
    RowSpec(
        "target_alone",
        "Target alone",
        "LatentWire control",
        "results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl",
        "target_alone",
    ),
    RowSpec(
        "target_self_repair",
        "Target self-repair",
        "LatentWire control",
        "results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl",
        "target_self_repair",
    ),
    RowSpec(
        "selected_route_no_repair",
        "Selected route, no repair",
        "LatentWire",
        "results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl",
        "selected_route_no_repair",
    ),
    RowSpec(
        "selected_route_repair",
        "Selected route + repair",
        "LatentWire",
        "results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl",
        "process_repair_selected_route",
    ),
    RowSpec(
        "c2c",
        "C2C",
        "Direct competitor",
        "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
    ),
    RowSpec(
        "kvcomm",
        "KVComm",
        "Direct competitor",
        "results/kvcomm_gsm70_20260419/qwen_gsm70_kvcomm_ported.jsonl",
    ),
    RowSpec(
        "kvpress_none",
        "KVPress no press",
        "Compression control",
        "results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit20.jsonl",
    ),
    RowSpec(
        "kvpress_expected_attention",
        "KVPress expected-attn 0.5",
        "Compression control",
        "results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit20.jsonl",
    ),
    RowSpec(
        "latentmas_baseline_probe",
        "LatentMAS baseline harness probe",
        "Latent competitor probe",
        "results/latentmas_competitor_20260421/qwen25_05b_gsm1_baseline_probe.jsonl",
        required=False,
    ),
    RowSpec(
        "latentmas_text_mas_probe",
        "LatentMAS text-MAS harness probe",
        "Latent competitor probe",
        "results/latentmas_competitor_20260421/qwen25_05b_gsm1_text_mas_probe.jsonl",
        required=False,
    ),
    RowSpec(
        "latentmas_baseline",
        "LatentMAS baseline",
        "Latent competitor",
        "results/latentmas_competitor_20260421/gsm10_baseline.jsonl",
        required=False,
    ),
    RowSpec(
        "latentmas_text_mas",
        "LatentMAS text-MAS",
        "Latent competitor",
        "results/latentmas_competitor_20260421/gsm10_text_mas.jsonl",
        required=False,
    ),
    RowSpec(
        "latentmas_latent_mas",
        "LatentMAS latent-MAS",
        "Latent competitor",
        "results/latentmas_competitor_20260421/gsm10_latent_mas.jsonl",
        required=False,
    ),
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _meta_path(path: Path) -> Path:
    if path.name.endswith(".jsonl"):
        return path.with_suffix(".jsonl.meta.json")
    return path.with_suffix(path.suffix + ".meta.json")


def _first_number(mapping: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _count_from_records(records: Sequence[dict[str, Any]]) -> int:
    ids = {
        str(record.get("example_id", record.get("id", record.get("index", offset))))
        for offset, record in enumerate(records)
    }
    return len(ids)


def _records_for_method(records: Sequence[dict[str, Any]], method: str | None) -> list[dict[str, Any]]:
    if method is None:
        return list(records)
    return [record for record in records if record.get("method") == method]


def _token_proxy_from_records(records: Sequence[dict[str, Any]]) -> float | None:
    if not records:
        return None
    values: list[float] = []
    for record in records:
        trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
        direct = _first_number(
            record,
            (
                "generated_tokens",
                "generated_tokens_avg",
                "total_tokens",
                "target_prompt_token_count",
                "raw_target_token_count",
                "prompt_token_count",
            ),
        )
        if direct is not None:
            values.append(direct)
            continue
        traced = _first_number(trace, ("input_token_count", "output_token_count", "output_char_count"))
        if traced is not None:
            values.append(traced)
    if not values:
        return None
    return sum(values) / len(values)


def _latency_from_records(records: Sequence[dict[str, Any]]) -> float | None:
    values = [
        float(record["latency_sec"])
        for record in records
        if isinstance(record.get("latency_sec"), (int, float))
    ]
    if not values:
        return None
    return sum(values)


def _stats_from_method_summary(spec: RowSpec, meta: dict[str, Any]) -> RowStats | None:
    if not spec.method:
        return None
    method_summary = meta.get("method_summary")
    if not isinstance(method_summary, dict) or spec.method not in method_summary:
        return None
    summary = method_summary[spec.method]
    if not isinstance(summary, dict):
        return None
    metric_summary = meta.get("metric_summary") if isinstance(meta.get("metric_summary"), dict) else {}
    latency = _first_number(metric_summary, (f"{spec.method}_latency_sec", "latency_sec"))
    token_proxy = _first_number(
        metric_summary,
        (f"{spec.method}_generated_tokens_avg", f"{spec.method}_tokens", f"{spec.method}_tokens_per_sec"),
    )
    if token_proxy is None:
        token_proxy = _first_number(summary, ("generated_tokens_avg", "avg_tokens", "avg_bytes", "avg_bits"))
    return RowStats(
        spec=spec,
        status="present",
        n=int(summary.get("count", summary.get("num_examples", 0))) or None,
        accuracy=float(summary["accuracy"]) if isinstance(summary.get("accuracy"), (int, float)) else None,
        latency_sec=latency,
        token_proxy=token_proxy,
        source="meta:method_summary",
    )


def _stats_from_run_meta(spec: RowSpec, meta: dict[str, Any]) -> RowStats | None:
    accuracy = _first_number(meta, ("accuracy", "acc"))
    n = _first_number(meta, ("num_examples", "count", "n", "N", "limit"))
    correct = _first_number(meta, ("correct", "num_correct"))
    if accuracy is None and correct is not None and n:
        accuracy = correct / n
    if accuracy is None and n is None:
        return None
    token_proxy = _first_number(
        meta,
        ("generated_tokens_avg", "tokens_avg", "input_token_count", "output_char_count", "tokens_per_sec"),
    )
    return RowStats(
        spec=spec,
        status="present",
        n=int(n) if n is not None else None,
        accuracy=accuracy,
        latency_sec=_first_number(meta, ("latency_sec", "total_latency_sec")),
        token_proxy=token_proxy,
        source="meta:run",
    )


def _stats_from_records(spec: RowSpec, records: Sequence[dict[str, Any]]) -> RowStats | None:
    filtered = _records_for_method(records, spec.method)
    if not filtered:
        return None
    correct_values = [bool(record.get("correct")) for record in filtered if record.get("correct") is not None]
    accuracy = sum(correct_values) / len(correct_values) if correct_values else None
    return RowStats(
        spec=spec,
        status="present",
        n=_count_from_records(filtered),
        accuracy=accuracy,
        latency_sec=_latency_from_records(filtered),
        token_proxy=_token_proxy_from_records(filtered),
        source="jsonl",
    )


def _merge_stats(primary: RowStats, fallback: RowStats | None) -> RowStats:
    if fallback is None:
        return primary
    sources = primary.source
    used_fallback = (
        (primary.n is None and fallback.n is not None)
        or (primary.accuracy is None and fallback.accuracy is not None)
        or (primary.latency_sec is None and fallback.latency_sec is not None)
        or (primary.token_proxy is None and fallback.token_proxy is not None)
    )
    if used_fallback and fallback.source and fallback.source not in sources:
        sources = f"{sources}+{fallback.source}" if sources else fallback.source
    return RowStats(
        spec=primary.spec,
        status=primary.status,
        n=primary.n if primary.n is not None else fallback.n,
        accuracy=primary.accuracy if primary.accuracy is not None else fallback.accuracy,
        latency_sec=primary.latency_sec if primary.latency_sec is not None else fallback.latency_sec,
        token_proxy=primary.token_proxy if primary.token_proxy is not None else fallback.token_proxy,
        source=sources,
        note=primary.note,
    )


def summarize_row(spec: RowSpec, root: Path) -> RowStats:
    artifact = root / spec.artifact
    meta = _read_json(_meta_path(artifact))
    record_stats: RowStats | None = None
    records_loaded = False

    def records_fallback() -> RowStats | None:
        nonlocal record_stats, records_loaded
        if not records_loaded:
            record_stats = _stats_from_records(spec, _read_jsonl(artifact))
            records_loaded = True
        return record_stats

    if meta is not None:
        method_stats = _stats_from_method_summary(spec, meta)
        if method_stats is not None:
            return _merge_stats(method_stats, records_fallback())
        run_stats = _stats_from_run_meta(spec, meta)
        if run_stats is not None:
            return _merge_stats(run_stats, records_fallback())
    record_stats = records_fallback()
    if record_stats is not None:
        return record_stats
    note = "artifact missing" if not artifact.exists() else "no matching rows"
    if meta is None and artifact.exists():
        note = "meta missing; no aggregatable rows"
    return RowStats(spec=spec, status="missing", source="", note=note)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    return f"{value:.4f}"


def render_markdown(rows: Sequence[RowStats]) -> str:
    lines = [
        "# Matched Competitor Matrix",
        "",
        "Date: 2026-04-21",
        "",
        "This lightweight matrix reads existing JSONL and `.meta.json` artifacts only. Missing heavyweight rows stay explicit so the paper cannot accidentally compare against an incomplete competitor set.",
        "",
        "| Row | Family | Status | Accuracy | N | Latency sec | Token/byte proxy | Source | Artifact | Note |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.spec.label,
                    row.spec.family,
                    row.status,
                    _fmt_float(row.accuracy),
                    _fmt_int(row.n),
                    _fmt_metric(row.latency_sec),
                    _fmt_metric(row.token_proxy),
                    row.source or "-",
                    f"`{row.spec.artifact}`",
                    row.note or "-",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `Token/byte proxy` is intentionally a proxy because historical artifacts log different counters: generated tokens, trace token counts, or transported byte/bit averages.",
            "- `KVPress` rows are same-model compression controls, not semantic cross-model communication baselines.",
            "- `LatentMAS` harness probe rows use the cached Qwen2.5-0.5B model on `N=1`; they are plumbing checks, not fair competitor rows.",
            "- Full `LatentMAS` rows are expected to remain `missing` until bounded matched wrapper runs are executed; this is preferable to silently omitting them.",
            "- Promote a positive-method claim only when selected-route repair beats target self-repair on matched IDs with comparable repair, token, byte, and latency budgets.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_matrix(root: Path, rows: Sequence[RowSpec] = DEFAULT_ROWS) -> list[RowStats]:
    return [summarize_row(spec, root) for spec in rows]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/matched_competitor_matrix_20260421.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = build_matrix(args.repo_root)
    markdown = render_markdown(rows)
    output = args.repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
