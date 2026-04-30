from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_LEARNED_GATE = pathlib.Path(
    "results/source_private_learned_synonym_dictionary_packet_gate_20260429_seed_repeat/"
    "learned_synonym_dictionary_packet_gate.json"
)
DEFAULT_HELDOUT_GATE = pathlib.Path(
    "results/source_private_learned_synonym_dictionary_packet_gate_20260429_heldout_synonym/"
    "learned_synonym_dictionary_packet_gate.json"
)
DEFAULT_QJL_SUMMARY = pathlib.Path("results/source_private_tool_trace_qjl_residual_20260429_seed29_30/summary.json")
DEFAULT_KV_TABLE = pathlib.Path("results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json")

CSV_COLUMNS = (
    "row_group",
    "method",
    "surface",
    "artifact",
    "n",
    "budget_bytes",
    "payload_kind",
    "source_private",
    "decoder_side_info",
    "same_byte_matched",
    "accuracy",
    "target_accuracy",
    "best_destroying_control_accuracy",
    "delta_vs_target",
    "delta_vs_best_control",
    "controls_ok",
    "pass_rule_result",
    "p50_latency_ms",
    "p95_latency_ms",
    "kv_qjl_1bit_bytes_vs_packet",
    "kv_kivi_2bit_bytes_vs_packet",
    "paper_use",
    "caveat",
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _row(
    *,
    row_group: str,
    method: str,
    surface: str,
    artifact: pathlib.Path,
    n: int | None,
    budget_bytes: float | None,
    payload_kind: str,
    source_private: bool,
    decoder_side_info: bool,
    same_byte_matched: bool,
    accuracy: float | None,
    target_accuracy: float | None,
    best_destroying_control_accuracy: float | None,
    controls_ok: bool | None,
    pass_rule_result: str,
    p50_latency_ms: float | None,
    p95_latency_ms: float | None,
    paper_use: str,
    caveat: str,
    kv_qjl_1bit_bytes_vs_packet: float | None = None,
    kv_kivi_2bit_bytes_vs_packet: float | None = None,
) -> dict[str, Any]:
    delta_vs_target = None if accuracy is None or target_accuracy is None else accuracy - target_accuracy
    delta_vs_best = (
        None
        if accuracy is None or best_destroying_control_accuracy is None
        else accuracy - best_destroying_control_accuracy
    )
    try:
        artifact_name = str(artifact.relative_to(ROOT))
    except ValueError:
        artifact_name = str(artifact)
    return {
        "row_group": row_group,
        "method": method,
        "surface": surface,
        "artifact": artifact_name,
        "n": n,
        "budget_bytes": budget_bytes,
        "payload_kind": payload_kind,
        "source_private": source_private,
        "decoder_side_info": decoder_side_info,
        "same_byte_matched": same_byte_matched,
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "best_destroying_control_accuracy": best_destroying_control_accuracy,
        "delta_vs_target": delta_vs_target,
        "delta_vs_best_control": delta_vs_best,
        "controls_ok": controls_ok,
        "pass_rule_result": pass_rule_result,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "kv_qjl_1bit_bytes_vs_packet": kv_qjl_1bit_bytes_vs_packet,
        "kv_kivi_2bit_bytes_vs_packet": kv_kivi_2bit_bytes_vs_packet,
        "paper_use": paper_use,
        "caveat": caveat,
    }


def _learned_rows(gate_path: pathlib.Path, *, row_group: str, paper_use: str, caveat: str) -> list[dict[str, Any]]:
    gate = _read_json(gate_path)
    rows: list[dict[str, Any]] = []
    for summary in gate["rows"]:
        if summary["budget_bytes"] != 4:
            continue
        surface = f"{summary['direction']} {gate['candidate_atom_view']}"
        rows.append(
            _row(
                row_group=row_group,
                method="learned synonym dictionary packet",
                surface=surface,
                artifact=gate_path,
                n=summary["n"],
                budget_bytes=float(summary["budget_bytes"]),
                payload_kind="atom_packet",
                source_private=True,
                decoder_side_info=True,
                same_byte_matched=True,
                accuracy=summary["learned_synonym_dictionary_accuracy"],
                target_accuracy=summary["target_accuracy"],
                best_destroying_control_accuracy=summary["best_control_accuracy"],
                controls_ok=summary["controls_ok"],
                pass_rule_result="pass" if summary["pass_gate"] else "fail",
                p50_latency_ms=None,
                p95_latency_ms=None,
                paper_use=paper_use,
                caveat=caveat,
            )
        )
        direction_summary = _read_json(gate_path.parent / summary["direction"] / "summary.json")
        budget_summary = next(row for row in direction_summary["budget_summaries"] if row["budget_bytes"] == 4)
        for condition, label, use in (
            ("structured_text_matched", "same-byte structured text relay", "direct same-byte text control"),
            ("random_same_byte", "random same-byte packet", "source-destroying byte control"),
            ("answer_only_text", "answer-label text truncated to 4 bytes", "answer-only leakage control"),
        ):
            metrics = budget_summary["metrics"][condition]
            rows.append(
                _row(
                    row_group="same_surface_control",
                    method=label,
                    surface=surface,
                    artifact=gate_path.parent / summary["direction"] / "summary.json",
                    n=metrics["n"],
                    budget_bytes=metrics["mean_payload_bytes"],
                    payload_kind=condition,
                    source_private=condition == "structured_text_matched",
                    decoder_side_info=True,
                    same_byte_matched=True,
                    accuracy=metrics["accuracy"],
                    target_accuracy=summary["target_accuracy"],
                    best_destroying_control_accuracy=None,
                    controls_ok=metrics["accuracy"] <= summary["target_accuracy"] + 0.03,
                    pass_rule_result="control_ok" if metrics["accuracy"] <= summary["target_accuracy"] + 0.03 else "control_leak",
                    p50_latency_ms=metrics["p50_latency_ms"],
                    p95_latency_ms=metrics["p95_latency_ms"],
                    paper_use=use,
                    caveat="same benchmark surface; not a standalone communication method",
                )
            )
    return rows


def _compression_rows(summary_path: pathlib.Path) -> list[dict[str, Any]]:
    summary = _read_json(summary_path)
    budget = summary["budget_summaries"][0]
    metrics = budget["metrics"]
    target = metrics["target_only"]["accuracy"]
    row_specs = (
        (
            "compression_baseline",
            "scalar quantized source projection",
            "scalar_quantized_source",
            "scalar_uint8",
            budget["scalar_controls_ok"],
            "pass" if budget["scalar_source_packet_pass"] else "fail",
            "direct vector/source-coding comparator",
        ),
        (
            "compression_baseline",
            "QJL-style residual projection",
            "qjl_residual_source",
            "qjl_scalar_plus_sign",
            budget["qjl_controls_ok"],
            "pass" if budget["qjl_source_packet_pass"] else "fail",
            "direct quantization/sketch comparator",
        ),
        (
            "compression_control",
            "raw source sign sketch",
            "raw_source_sign_sketch",
            "raw_sign",
            None,
            "unpromoted",
            "raw sketch ablation",
        ),
    )
    rows: list[dict[str, Any]] = []
    for row_group, label, condition, payload_kind, controls_ok, pass_result, paper_use in row_specs:
        metric = metrics[condition]
        rows.append(
            _row(
                row_group=row_group,
                method=label,
                surface=f"{summary['train_family_set']}_to_{summary['eval_family_set']} {summary['candidate_view']}",
                artifact=summary_path,
                n=metric["n"],
                budget_bytes=metric["mean_payload_bytes"],
                payload_kind=payload_kind,
                source_private=True,
                decoder_side_info=True,
                same_byte_matched=False,
                accuracy=metric["accuracy"],
                target_accuracy=target,
                best_destroying_control_accuracy=budget["best_no_source_accuracy"],
                controls_ok=controls_ok,
                pass_rule_result=pass_result,
                p50_latency_ms=metric["p50_latency_ms"],
                p95_latency_ms=metric["p95_latency_ms"],
                paper_use=paper_use,
                caveat="not the synonym-stress candidate surface; use as source-coding comparator",
            )
        )
    return rows


def _kv_rows(kv_path: pathlib.Path) -> list[dict[str, Any]]:
    payload = _read_json(kv_path)
    rows: list[dict[str, Any]] = []
    for kv_row in payload["rows"]:
        condition = kv_row["condition"]
        if condition == "matched_packet":
            paper_use = "Mac endpoint packet systems row"
            pass_result = "pass"
        elif condition == "matched_byte_text_2":
            paper_use = "same-byte text endpoint control"
            pass_result = "fail"
        elif condition == "full_hidden_log":
            paper_use = "full-log rate baseline"
            pass_result = "higher_rate_comparator"
        else:
            paper_use = "structured text rate comparator"
            pass_result = "higher_rate_comparator"
        rows.append(
            _row(
                row_group="endpoint_systems",
                method=condition,
                surface=kv_row["surface"],
                artifact=kv_path,
                n=None,
                budget_bytes=kv_row["payload_bytes"],
                payload_kind=condition,
                source_private=condition != "matched_byte_text_2",
                decoder_side_info=True,
                same_byte_matched=condition in {"matched_packet", "matched_byte_text_2"},
                accuracy=kv_row["accuracy"],
                target_accuracy=None,
                best_destroying_control_accuracy=None,
                controls_ok=None,
                pass_rule_result=pass_result,
                p50_latency_ms=kv_row["p50_ttft_ms"],
                p95_latency_ms=kv_row["p95_ttft_ms"],
                kv_qjl_1bit_bytes_vs_packet=kv_row["qjl_1bit_bytes_vs_packet"],
                kv_kivi_2bit_bytes_vs_packet=kv_row["kivi_2bit_bytes_vs_packet"],
                paper_use=paper_use,
                caveat="KV/TurboQuant values are byte-floor accounting only, not kernel implementations",
            )
        )
    return rows


def build_systems_comparison_table(
    *,
    learned_gate: pathlib.Path,
    heldout_gate: pathlib.Path,
    qjl_summary: pathlib.Path,
    kv_table: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    learned_gate = _resolve(learned_gate)
    heldout_gate = _resolve(heldout_gate)
    qjl_summary = _resolve(qjl_summary)
    kv_table = _resolve(kv_table)

    rows = [
        *_learned_rows(
            learned_gate,
            row_group="learned_packet",
            paper_use="headline learned calibrated-dictionary packet",
            caveat="calibrated public dictionary; not held-out paraphrase transfer",
        ),
        *_learned_rows(
            heldout_gate,
            row_group="heldout_boundary",
            paper_use="negative boundary for held-out paraphrase generalization",
            caveat="calibration surface excludes transformed family-B phrases",
        ),
        *_compression_rows(qjl_summary),
        *_kv_rows(kv_table),
    ]
    headline_rows = [row for row in rows if row["row_group"] == "learned_packet" and row["pass_rule_result"] == "pass"]
    same_byte_text_rows = [
        row for row in rows if row["row_group"] == "same_surface_control" and row["method"] == "same-byte structured text relay"
    ]
    kv_nonpacket = [row for row in rows if row["row_group"] == "endpoint_systems" and row["method"] != "matched_packet"]
    payload = {
        "gate": "source_private_systems_comparison_table",
        "artifacts": {
            "learned_gate": str(learned_gate.relative_to(ROOT)),
            "heldout_gate": str(heldout_gate.relative_to(ROOT)),
            "qjl_summary": str(qjl_summary.relative_to(ROOT)),
            "kv_table": str(kv_table.relative_to(ROOT)),
        },
        "rows": rows,
        "headline": {
            "headline_learned_pass_rows": len(headline_rows),
            "headline_learned_min_delta_vs_target": min(row["delta_vs_target"] for row in headline_rows),
            "same_surface_text_max_accuracy": max(row["accuracy"] for row in same_byte_text_rows),
            "same_surface_text_max_delta_vs_target": max(row["delta_vs_target"] for row in same_byte_text_rows),
            "compression_scalar_accuracy": next(
                row["accuracy"] for row in rows if row["method"] == "scalar quantized source projection"
            ),
            "compression_qjl_accuracy": next(row["accuracy"] for row in rows if row["method"] == "QJL-style residual projection"),
            "min_endpoint_nonpacket_qjl_1bit_bytes_vs_packet": min(
                row["kv_qjl_1bit_bytes_vs_packet"]
                for row in kv_nonpacket
                if row["kv_qjl_1bit_bytes_vs_packet"] and row["kv_qjl_1bit_bytes_vs_packet"] > 0
            ),
            "claim_boundary": (
                "The learned packet has clean same-surface 4-byte rows and seed stability, but held-out paraphrase "
                "generalization fails; KV/TurboQuant rows are byte-floor accounting only."
            ),
        },
    }

    json_path = output_dir / "source_private_systems_comparison_table.json"
    csv_path = output_dir / "source_private_systems_comparison_table.csv"
    md_path = output_dir / "source_private_systems_comparison_table.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": payload["headline"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Systems Comparison Table",
        "",
        f"- headline learned pass rows: `{h['headline_learned_pass_rows']}`",
        f"- headline learned min delta vs target: `{h['headline_learned_min_delta_vs_target']:.3f}`",
        f"- same-surface text max accuracy: `{h['same_surface_text_max_accuracy']:.3f}`",
        f"- same-surface text max delta vs target: `{h['same_surface_text_max_delta_vs_target']:.3f}`",
        f"- scalar source-code comparator accuracy: `{h['compression_scalar_accuracy']:.3f}`",
        f"- QJL-style comparator accuracy: `{h['compression_qjl_accuracy']:.3f}`",
        f"- min endpoint non-packet QJL 1-bit byte ratio vs packet: "
        f"`{h['min_endpoint_nonpacket_qjl_1bit_bytes_vs_packet']:.1f}x`",
        "",
        "## Rows",
        "",
        "| Group | Method | Surface | Bytes | Accuracy | Target | Best control | Pass | Paper use |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['row_group']} | {row['method']} | {row['surface']} | "
            f"{_fmt(row['budget_bytes'])} | {_fmt(row['accuracy'])} | {_fmt(row['target_accuracy'])} | "
            f"{_fmt(row['best_destroying_control_accuracy'])} | {row['pass_rule_result']} | {row['paper_use']} |"
        )
    lines.extend(["", f"Claim boundary: {h['claim_boundary']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-gate", type=pathlib.Path, default=DEFAULT_LEARNED_GATE)
    parser.add_argument("--heldout-gate", type=pathlib.Path, default=DEFAULT_HELDOUT_GATE)
    parser.add_argument("--qjl-summary", type=pathlib.Path, default=DEFAULT_QJL_SUMMARY)
    parser.add_argument("--kv-table", type=pathlib.Path, default=DEFAULT_KV_TABLE)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_systems_comparison_table_20260429"),
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    payload = build_systems_comparison_table(
        learned_gate=args.learned_gate,
        heldout_gate=args.heldout_gate,
        qjl_summary=args.qjl_summary,
        kv_table=args.kv_table,
        output_dir=output_dir,
    )
    print(json.dumps({"output_dir": str(output_dir), "headline": payload["headline"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
