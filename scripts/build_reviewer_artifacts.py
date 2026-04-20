"""Build current reviewer-facing paper artifacts from live result files."""

from __future__ import annotations

import json
import math
import pathlib
import random
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_prediction_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_meta_summary(meta_path: pathlib.Path) -> tuple[str, dict[str, Any]]:
    meta = _load_json(meta_path)
    items = meta.get("method_summary", {})
    if not items:
        raise ValueError(f"No method_summary in {meta_path}")
    method = next(iter(items))
    return method, items[method]


def _accuracy_from_jsonl(path: pathlib.Path) -> float:
    total = 0
    correct = 0
    for row in _load_prediction_records(path):
        total += 1
        correct += int(bool(row.get("correct", False)))
    return correct / total if total else 0.0


def _paired_stats(
    candidate_rows: dict[int, bool],
    baseline_rows: dict[int, bool],
    *,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    indices = sorted(set(candidate_rows) & set(baseline_rows))
    if not indices:
        raise ValueError("No paired indices")

    diffs = [
        (1.0 if candidate_rows[idx] else 0.0) - (1.0 if baseline_rows[idx] else 0.0)
        for idx in indices
    ]
    candidate_only = sum(candidate_rows[idx] and not baseline_rows[idx] for idx in indices)
    baseline_only = sum(baseline_rows[idx] and not candidate_rows[idx] for idx in indices)
    denom = candidate_only + baseline_only
    if denom:
        chi2 = (max(abs(candidate_only - baseline_only) - 1.0, 0.0) ** 2) / denom
        p_value = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        chi2 = 0.0
        p_value = 1.0

    rng = random.Random(0)
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lo_idx = int(0.025 * (len(boot) - 1))
    hi_idx = int(0.975 * (len(boot) - 1))
    return {
        "paired_n": float(len(indices)),
        "delta_accuracy": float(sum(diffs) / len(diffs)),
        "method_only": float(candidate_only),
        "baseline_only": float(baseline_only),
        "both_correct": float(sum(candidate_rows[idx] and baseline_rows[idx] for idx in indices)),
        "both_wrong": float(sum((not candidate_rows[idx]) and (not baseline_rows[idx]) for idx in indices)),
        "mcnemar_chi2": float(chi2),
        "mcnemar_p": float(p_value),
        "bootstrap_delta_low": float(boot[lo_idx]),
        "bootstrap_delta_high": float(boot[hi_idx]),
    }


def _compare_prediction_records(
    candidate_path: pathlib.Path,
    baseline_path: pathlib.Path,
    *,
    candidate_method: str,
    baseline_method: str,
    candidate_label: str,
    baseline_label: str,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    candidate_records = _load_prediction_records(candidate_path)
    baseline_records = _load_prediction_records(baseline_path)
    candidate_rows = {
        int(record["index"]): bool(record["correct"])
        for record in candidate_records
        if str(record["method"]) == candidate_method
    }
    baseline_rows = {
        int(record["index"]): bool(record["correct"])
        for record in baseline_records
        if str(record["method"]) == baseline_method
    }
    if not candidate_rows:
        raise ValueError(f"Missing candidate method {candidate_method} in {candidate_path}")
    if not baseline_rows:
        raise ValueError(f"Missing baseline method {baseline_method} in {baseline_path}")

    indices = sorted(set(candidate_rows) & set(baseline_rows))
    candidate_accuracy = sum(candidate_rows[idx] for idx in indices) / len(indices)
    baseline_accuracy = sum(baseline_rows[idx] for idx in indices) / len(indices)
    return {
        "method": candidate_label if candidate_method == baseline_method else f"{candidate_label} vs {baseline_label}",
        "candidate_method": candidate_method,
        "baseline_method": baseline_method,
        "candidate_label": candidate_label,
        "baseline_label": baseline_label,
        "candidate_accuracy": float(candidate_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        **_paired_stats(candidate_rows, baseline_rows, n_bootstrap=n_bootstrap),
    }


def _write_jsonl(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_paired_markdown(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    lines = [
        "| Method | Candidate Acc | Baseline Acc | Delta | Cand Only | Base Only | 95% Bootstrap Delta | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {cand:.4f} | {base:.4f} | {delta:+.4f} | {cand_only:.0f} | "
            "{base_only:.0f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} |".format(
                method=row["method"],
                cand=float(row["candidate_accuracy"]),
                base=float(row["baseline_accuracy"]),
                delta=float(row["delta_accuracy"]),
                cand_only=float(row["method_only"]),
                base_only=float(row["baseline_only"]),
                lo=float(row["bootstrap_delta_low"]),
                hi=float(row["bootstrap_delta_high"]),
                p=float(row["mcnemar_p"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _row(
    *,
    split: str,
    method: str,
    family: str,
    accuracy: float,
    avg_bytes: float | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "method": method,
        "family": family,
        "accuracy": accuracy,
        "avg_bytes": avg_bytes,
        "notes": notes,
    }


def _meta_row(
    *,
    split: str,
    method: str,
    family: str,
    meta_path: str,
    notes: str,
) -> dict[str, Any]:
    _, summary = _extract_meta_summary(ROOT / meta_path)
    return _row(
        split=split,
        method=method,
        family=family,
        accuracy=float(summary["accuracy"]),
        avg_bytes=summary.get("avg_bytes"),
        notes=notes,
    )


def _jsonl_row(
    *,
    split: str,
    method: str,
    family: str,
    jsonl_path: str,
    notes: str,
    avg_bytes: float | None = None,
) -> dict[str, Any]:
    return _row(
        split=split,
        method=method,
        family=family,
        accuracy=_accuracy_from_jsonl(ROOT / jsonl_path),
        avg_bytes=avg_bytes,
        notes=notes,
    )


def _fmt_bytes(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    return f"{value:,.1f}"


def _build_frontier_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.append(
        _row(
            split="gsm8k_eval_70",
            method="target-alone",
            family="control",
            accuracy=0.04285714285714286,
            avg_bytes=0.0,
            notes="no source communication",
        )
    )
    rows.append(
        _row(
            split="gsm8k_eval_70",
            method="text-to-text",
            family="control",
            accuracy=0.1,
            avg_bytes=None,
            notes="text communication baseline",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_70",
            method="shuffled fixed prior",
            family="selector-null",
            meta_path="results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_shuffled_g010_pos05.jsonl.meta.json",
            notes="query-blind prior null",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_70",
            method="fixed prior",
            family="selector",
            meta_path="results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl.meta.json",
            notes="best internal same-pair branch",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_70",
            method="grouped signature transport",
            family="transport",
            meta_path="results/grouped_signature_transport_20260419/qwen_gsm70_grouped_signature_transport_w025_norank.jsonl.meta.json",
            notes="best transport-only branch",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_70",
            method="grouped subspace + rank-4 residual",
            family="transport+correction",
            meta_path="results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl.meta.json",
            notes="best transport-plus-correction branch",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_70",
            method="bridge_ridge",
            family="bridge",
            meta_path="results/grouped_subspace_bridgeridge_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4_bridgeridge_cal64.jsonl.meta.json",
            notes="best bridge branch that survives held-out slices",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_70",
            method="QK-fidelity budget",
            family="query-conditioned selector",
            meta_path="results/qk_fidelity_budget_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4_qkbudget.jsonl.meta.json",
            notes="runtime query-conditioned selector on top of best transport+correction checkpoint",
        )
    )
    rows.append(
        _jsonl_row(
            split="gsm8k_eval_70",
            method="KVComm-compatible replay",
            family="external comparator",
            jsonl_path="results/kvcomm_gsm70_20260419/qwen_gsm70_kvcomm_ported.jsonl",
            notes="adjacent heterogeneous replay baseline",
        )
    )
    rows.append(
        _jsonl_row(
            split="gsm8k_eval_70",
            method="C2C",
            family="external comparator",
            jsonl_path="results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            notes="main external fair bar",
        )
    )

    rows.append(
        _row(
            split="gsm8k_eval_10_controlled",
            method="target-alone",
            family="control",
            accuracy=0.1,
            avg_bytes=0.0,
            notes="shared chat serialization + enable_thinking=false",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="bridge_ridge",
            family="bridge",
            meta_path="results/prompt_control_20260419/qwen_gsm10_grouped_subspace_bridgeridge_chat_thinking_false.jsonl.meta.json",
            notes="fair controlled bridge baseline",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="grouped rotational transport",
            family="geometry",
            meta_path="results/grouped_rotational_transport_20260420/qwen_gsm10_grouped_rotational_transport_w010_r4_cal64_chat.jsonl.meta.json",
            notes="first geometry-side branch to survive the controlled slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="grouped fitted rotation transport",
            family="geometry",
            meta_path="results/grouped_fitted_rotation_transport_20260420/qwen_gsm10_grouped_fitted_rotation_transport_w010_r4_cal64_chat.jsonl.meta.json",
            notes="calibration-fit gauge-fixing follow-up",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="grouped shared-basis transport",
            family="geometry",
            meta_path="results/grouped_shared_basis_transport_20260420/qwen_gsm10_grouped_shared_basis_transport_w010_r4_cal64_chat.jsonl.meta.json",
            notes="shared-basis coefficient-space transport",
        )
    )
    rows.append(
        _jsonl_row(
            split="gsm8k_eval_10_controlled",
            method="KVPress no-press",
            family="external comparator",
            jsonl_path="results/kvpress_expected_20260420/qwen_gsm10_no_press.jsonl",
            notes="exact external KVPress harness floor",
        )
    )
    rows.append(
        _jsonl_row(
            split="gsm8k_eval_10_controlled",
            method="KVPress ExpectedAttentionPress",
            family="external comparator",
            jsonl_path="results/kvpress_expected_20260420/qwen_gsm10_expected_attention.jsonl",
            notes="exact external Expected Attention comparator",
        )
    )

    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="readout adapter",
            family="stronger-teacher bridge",
            meta_path="results/bridge_ridge_qk_readout_adapter_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_readout_adapter_cal64_chat.jsonl.meta.json",
            notes="stronger prompt-local teacher survives smoke only",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="prediction-KL adapter",
            family="stronger-teacher bridge",
            meta_path="results/grouped_subspace_bridgepredkl_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_bridgepredkl_cal64_chat.jsonl.meta.json",
            notes="first prediction-level bridge teacher",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="prediction-KL bank",
            family="stronger-teacher bridge",
            meta_path="results/grouped_subspace_bridgepredklbank_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_bridgepredklbank_cal16_chat.jsonl.meta.json",
            notes="small modular bank follow-up to prediction-level teacher",
        )
    )

    return rows


def _build_paired_rows() -> list[dict[str, Any]]:
    specs = [
        {
            "candidate": "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "baseline": "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_shuffled_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "fixed_prior",
            "baseline_label": "shuffled_prior",
        },
        {
            "candidate": "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "baseline": "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_subspace_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": "results/grouped_subspace_bridgeridge_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4_bridgeridge_cal64.jsonl",
            "baseline": "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "bridge_ridge",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": "results/grouped_subspace_bridgeridge_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4_bridgeridge_cal64.jsonl",
            "baseline": "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "bridge_ridge",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "baseline": "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "grouped_subspace_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "baseline": "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "fixed_prior",
            "baseline_label": "c2c",
        },
        {
            "candidate": "results/prompt_control_20260419/qwen_gsm10_grouped_subspace_bridgeridge_chat_thinking_false.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "bridge_ridge_control",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/grouped_rotational_transport_20260420/qwen_gsm10_grouped_rotational_transport_w010_r4_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "grouped_rotational_transport",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/grouped_fitted_rotation_transport_20260420/qwen_gsm10_grouped_fitted_rotation_transport_w010_r4_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "grouped_fitted_rotation_transport",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/grouped_shared_basis_transport_20260420/qwen_gsm10_grouped_shared_basis_transport_w010_r4_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "grouped_shared_basis_transport",
            "baseline_label": "target_alone_control",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            _compare_prediction_records(
                ROOT / spec["candidate"],
                ROOT / spec["baseline"],
                candidate_method=spec["candidate_method"],
                baseline_method=spec["baseline_method"],
                candidate_label=spec["candidate_label"],
                baseline_label=spec["baseline_label"],
                n_bootstrap=1000,
            )
        )
    return rows


def _write_frontier_markdown(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    lines = [
        "# Current Bytes / Accuracy Frontier (2026-04-20)",
        "",
        "| Split | Method | Family | Accuracy | Avg bytes | Notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['method']} | {row['family']} | "
            f"{float(row['accuracy']):.4f} | {_fmt_bytes(row['avg_bytes'])} | {row['notes']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    frontier_rows = _build_frontier_rows()
    paired_rows = _build_paired_rows()

    frontier_json = ROOT / "paper/bytes_accuracy_frontier_20260420.json"
    frontier_md = ROOT / "paper/bytes_accuracy_table_20260420.md"
    paired_jsonl = ROOT / "paper/paired_flip_table_20260420.jsonl"
    paired_md = ROOT / "paper/paired_flip_table_20260420.md"

    frontier_json.write_text(json.dumps(frontier_rows, indent=2, sort_keys=True) + "\n")
    _write_frontier_markdown(frontier_rows, frontier_md)
    _write_jsonl(paired_rows, paired_jsonl)
    _write_paired_markdown(paired_rows, paired_md)

    print(frontier_json)
    print(frontier_md)
    print(paired_jsonl)
    print(paired_md)


if __name__ == "__main__":
    main()
