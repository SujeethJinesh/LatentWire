"""Build a concise bytes-vs-accuracy markdown table for the paper."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_meta(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _first_method_summary(meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    items = meta.get("method_summary", {})
    if not items:
        raise ValueError(f"No method_summary in {meta}")
    method = next(iter(items))
    return method, items[method]


def _accuracy_from_jsonl(path: pathlib.Path) -> float:
    total = 0
    correct = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            correct += int(bool(row.get("correct", False)))
    return correct / total if total else 0.0


def _row(split: str, method: str, accuracy: float, avg_bytes: float | None, notes: str) -> dict[str, Any]:
    return {
        "split": split,
        "method": method,
        "accuracy": accuracy,
        "avg_bytes": avg_bytes,
        "notes": notes,
    }


def _fmt_bytes(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    return f"{value:,.1f}"


def main() -> None:
    rows: list[dict[str, Any]] = []

    fixed70_meta = _load_meta(ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl.meta.json")
    _, fixed70 = _first_method_summary(fixed70_meta)
    rows.append(_row("gsm8k_eval_70", "target-alone", 0.04285714285714286, 0.0, "no source communication"))
    rows.append(_row("gsm8k_eval_70", "text-to-text", 0.1, None, "text communication baseline"))
    rows.append(_row("gsm8k_eval_70", "fixed prior", fixed70["accuracy"], fixed70.get("avg_bytes"), "best current internal same-pair branch"))

    shuffled70_meta = _load_meta(ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_shuffled_g010_pos05.jsonl.meta.json")
    _, shuffled70 = _first_method_summary(shuffled70_meta)
    rows.append(_row("gsm8k_eval_70", "shuffled fixed prior", shuffled70["accuracy"], shuffled70.get("avg_bytes"), "query-blind null"))

    sig70_meta = _load_meta(ROOT / "results/grouped_signature_transport_20260419/qwen_gsm70_grouped_signature_transport_w025_norank.jsonl.meta.json")
    _, sig70 = _first_method_summary(sig70_meta)
    rows.append(_row("gsm8k_eval_70", "grouped signature transport", sig70["accuracy"], sig70.get("avg_bytes"), "best current transport-only branch"))

    sub70_meta = _load_meta(ROOT / "results/grouped_subspace_transport_20260419/qwen_gsm70_grouped_subspace_transport_w010_norank.jsonl.meta.json")
    _, sub70 = _first_method_summary(sub70_meta)
    rows.append(_row("gsm8k_eval_70", "grouped subspace transport", sub70["accuracy"], sub70.get("avg_bytes"), "ties grouped signature transport"))

    subr4_meta = _load_meta(ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl.meta.json")
    _, subr4 = _first_method_summary(subr4_meta)
    rows.append(_row("gsm8k_eval_70", "grouped subspace transport + rank-4 residual", subr4["accuracy"], subr4.get("avg_bytes"), "best current transport-plus-correction branch"))

    covr4_meta = _load_meta(ROOT / "results/grouped_covariance_resid4_20260419/qwen_gsm70_grouped_covariance_transport_w010_r4.jsonl.meta.json")
    _, covr4 = _first_method_summary(covr4_meta)
    rows.append(_row("gsm8k_eval_70", "grouped covariance transport + rank-4 residual", covr4["accuracy"], covr4.get("avg_bytes"), "covariance-aware transport-plus-correction failure"))

    tmplr4_meta = _load_meta(ROOT / "results/grouped_template_resid4_20260419/qwen_gsm70_grouped_template_transport_w025_r4_cal64.jsonl.meta.json")
    _, tmplr4 = _first_method_summary(tmplr4_meta)
    rows.append(_row("gsm8k_eval_70", "grouped template transport + rank-4 residual", tmplr4["accuracy"], tmplr4.get("avg_bytes"), "attention-template transport-plus-correction probe (64-prompt calibration slice)"))

    tmpsubr4_meta = _load_meta(ROOT / "results/grouped_template_subspace_resid4_20260419/qwen_gsm70_grouped_template_subspace_transport_w010_r4_cal64.jsonl.meta.json")
    _, tmpsubr4 = _first_method_summary(tmpsubr4_meta)
    rows.append(_row("gsm8k_eval_70", "grouped template-subspace transport + rank-4 residual", tmpsubr4["accuracy"], tmpsubr4.get("avg_bytes"), "stacked grouped-penalty failure (64-prompt calibration slice)"))

    broadcast_meta = _load_meta(ROOT / "results/broadcast_template_transport_20260419/qwen_gsm70_broadcast_template_transport_w010_r4_cal64.jsonl.meta.json")
    _, broadcast = _first_method_summary(broadcast_meta)
    rows.append(_row("gsm8k_eval_70", "broadcast template transport + rank-4 residual", broadcast["accuracy"], broadcast.get("avg_bytes"), "rectangular 2->8 head transport probe (64-prompt calibration slice)"))

    broadcast_ot_meta = _load_meta(ROOT / "results/broadcast_template_ot_transport_20260419/qwen_gsm70_broadcast_template_ot_transport_w010_r4_cal64.jsonl.meta.json")
    _, broadcast_ot = _first_method_summary(broadcast_ot_meta)
    rows.append(_row("gsm8k_eval_70", "broadcast template OT transport + rank-4 residual", broadcast_ot["accuracy"], broadcast_ot.get("avg_bytes"), "rectangular Sinkhorn-style 2->8 head transport probe (64-prompt calibration slice)"))

    broadcast_peak_ot_meta = _load_meta(ROOT / "results/broadcast_peak_template_ot_transport_20260419/qwen_gsm70_broadcast_peak_template_ot_transport_w010_r4_cal64.jsonl.meta.json")
    _, broadcast_peak_ot = _first_method_summary(broadcast_peak_ot_meta)
    rows.append(_row("gsm8k_eval_70", "broadcast peak-template OT transport + rank-4 residual", broadcast_peak_ot["accuracy"], broadcast_peak_ot.get("avg_bytes"), "rectangular Sinkhorn-style 2->8 transport using peak-location templates (64-prompt calibration slice)"))

    can70_meta = _load_meta(ROOT / "results/grouped_canonical_transport_20260419/qwen_gsm70_grouped_canonical_transport_r8.jsonl.meta.json")
    _, can70 = _first_method_summary(can70_meta)
    rows.append(_row("gsm8k_eval_70", "grouped canonical transport", can70["accuracy"], can70.get("avg_bytes"), "low-rank canonical basis shortcut"))

    c2c70 = _accuracy_from_jsonl(ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl")
    rows.append(_row("gsm8k_eval_70", "C2C", c2c70, None, "strongest external baseline so far"))

    kvcomm70 = _accuracy_from_jsonl(ROOT / "results/kvcomm_gsm70_20260419/qwen_gsm70_kvcomm_ported.jsonl")
    rows.append(_row("gsm8k_eval_70", "KVComm-compatible replay", kvcomm70, None, "compatibility-lifted heterogeneous replay"))

    fixed100_meta = _load_meta(ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm100_attention_headbudget_prior_g010_pos05.jsonl.meta.json")
    _, fixed100 = _first_method_summary(fixed100_meta)
    rows.append(_row("gsm8k_100", "target-alone", 0.04, 0.0, "no source communication"))
    rows.append(_row("gsm8k_100", "text-to-text", 0.1, None, "text communication baseline"))
    rows.append(_row("gsm8k_100", "fixed prior", fixed100["accuracy"], fixed100.get("avg_bytes"), "best current internal branch on larger slice"))

    shuf100_meta = _load_meta(ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm100_attention_headbudget_prior_shuffled_g010_pos05.jsonl.meta.json")
    _, shuf100 = _first_method_summary(shuf100_meta)
    rows.append(_row("gsm8k_100", "shuffled fixed prior", shuf100["accuracy"], shuf100.get("avg_bytes"), "matched null"))

    c2c100 = _accuracy_from_jsonl(ROOT / "results/c2c_gsm100_20260418/qwen_gsm100_c2c.jsonl")
    rows.append(_row("gsm8k_100", "C2C", c2c100, None, "strongest external baseline so far"))

    rows.append(_row("svamp_eval_70", "target-alone", 0.07142857142857142, 0.0, "no source communication"))
    rows.append(_row("svamp_eval_70", "text-to-text", 0.4142857142857143, None, "text communication baseline"))
    rows.append(_row("svamp_eval_70", "grouped CCA fixed prior", 0.17142857142857143, None, "best current internal SVAMP branch"))
    rows.append(_row("svamp_eval_70", "grouped CCA shuffled null", 0.12857142857142856, None, "matched query-blind null"))

    c2c_svamp = _accuracy_from_jsonl(ROOT / "results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl")
    rows.append(_row("svamp_eval_70", "C2C", c2c_svamp, None, "strongest external baseline so far"))

    out = ROOT / "paper/bytes_accuracy_table_20260419.md"
    lines = [
        "# Bytes vs Accuracy Table (2026-04-19)",
        "",
        "| Split | Method | Accuracy | Avg bytes | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['method']} | {row['accuracy']:.4f} | {_fmt_bytes(row['avg_bytes'])} | {row['notes']} |"
        )
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
