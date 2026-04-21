"""Build current reviewer-facing paper artifacts from live result files."""

from __future__ import annotations

import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict
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
            method="shared-plus-private asym adapter",
            family="modular bridge",
            meta_path="results/bridge_ridge_qk_asym_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_asym_adapter_cal64_chat.jsonl.meta.json",
            notes="AsymLoRA-style shared bottleneck plus private K/V residual heads",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="shared-plus-private dynmap adapter",
            family="modular bridge",
            meta_path="results/bridge_ridge_qk_asym_dynmap_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_asym_dynmap_adapter_cal64_chat.jsonl.meta.json",
            notes="shared-plus-private bridge with context-reweighted top-k teacher",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="xattn adapter",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_xattn_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_xattn_adapter_cal64_chat.jsonl.meta.json",
            notes="tiny query-conditioned cross-attention bridge over live K/V-side memory signals",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="xattn dynmap adapter",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_xattn_dynmap_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_xattn_dynmap_adapter_cal64_chat.jsonl.meta.json",
            notes="xattn bridge plus context-reweighted top-k output teacher",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="module adapter",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_module_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_module_adapter_cal64_chat.jsonl.meta.json",
            notes="slotted attention-side transfer module with nonlinear readout and prediction distillation",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="module replace",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_module_replace_cal64_chat.jsonl.meta.json",
            notes="slotted attention-side transfer module trained to predict full corrected K/V directly",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="span-aligned module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_spanalign_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_spanalign_module_replace_cal64_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from raw-prompt monotone span-aligned calibration pairs",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="byte-span module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_bytespan_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_bytespan_module_replace_cal16_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from dominant UTF-8 byte-overlap calibration pairs on a 16-prompt diagnostic slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="dynamic-aligned context-only module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.jsonl.meta.json",
            notes="matched dynalign null with the same candidate window but prediction-overlap scoring disabled on a 16-prompt diagnostic slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="dynamic-aligned module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_dynalign_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from context-plus-output-overlap token mixtures",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="dynamic-aligned interaction module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_dynalign_interact_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_interact_module_replace_cal16_chat.jsonl.meta.json",
            notes="dynalign module replace plus prompt-local interaction distillation on a 16-prompt diagnostic slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_eval_10_controlled",
            method="token-basis replace",
            family="token-native attention bridge",
            meta_path="results/bridge_ridge_qk_tokenbasis_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_tokenbasis_replace_cal64_chat.jsonl.meta.json",
            notes="slotted attention-side module constrained to a basis distilled from target next-token output rows",
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
            method="shared-plus-private asym adapter",
            family="modular bridge",
            meta_path="results/bridge_ridge_qk_asym_adapter_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_asym_adapter_cal64_chat.jsonl.meta.json",
            notes="AsymLoRA-style shared-plus-private bridge survives smoke and controlled slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="shared-plus-private dynmap adapter",
            family="modular bridge",
            meta_path="results/bridge_ridge_qk_asym_dynmap_adapter_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_asym_dynmap_adapter_cal64_chat.jsonl.meta.json",
            notes="shared-plus-private bridge with context-reweighted top-k teacher",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="xattn adapter",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_xattn_adapter_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_xattn_adapter_cal64_chat.jsonl.meta.json",
            notes="tiny query-conditioned cross-attention bridge over live K/V-side memory signals",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="xattn dynmap adapter",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_xattn_dynmap_adapter_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_xattn_dynmap_adapter_cal64_chat.jsonl.meta.json",
            notes="xattn bridge plus context-reweighted top-k output teacher",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="module adapter",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_module_adapter_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_module_adapter_cal64_chat.jsonl.meta.json",
            notes="slotted attention-side transfer module with nonlinear readout and prediction distillation",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="module replace",
            family="attention bridge",
            meta_path="results/bridge_ridge_qk_module_replace_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_module_replace_cal64_chat.jsonl.meta.json",
            notes="slotted attention-side transfer module trained to predict full corrected K/V directly",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="span-aligned module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_spanalign_module_replace_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_spanalign_module_replace_cal64_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from raw-prompt monotone span-aligned calibration pairs",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="byte-span module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_bytespan_module_replace_20260420_diag/qwen_gsm5_grouped_subspace_transport_w010_r4_bytespan_module_replace_cal16_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from dominant UTF-8 byte-overlap calibration pairs on a 16-prompt diagnostic slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="dynamic-aligned context-only module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen_gsm5_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.jsonl.meta.json",
            notes="matched dynalign null with the same candidate window but prediction-overlap scoring disabled on a 16-prompt diagnostic slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="contextual-aligned module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_ctxalign_module_replace_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_ctxalign_module_replace_cal64_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from context-weighted source-to-target token mixtures",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="dynamic-aligned module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_dynalign_module_replace_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.jsonl.meta.json",
            notes="direct-output slotted module fit from context-plus-output-overlap token mixtures",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="dynamic-aligned top-5 layer knockout",
            family="layer-localization ablation",
            meta_path="results/layer_knockout_20260420/qwen_gsm5_dynalign_top5_layerdrop.jsonl.meta.json",
            notes="dynalign module replace with translated signal removed from recurrent top layer-localization signature L27,L5,L23,L22,L8",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="dynamic-aligned offset-5 layer knockout",
            family="layer-localization ablation",
            meta_path="results/layer_knockout_20260420/qwen_gsm5_dynalign_offset5_layerdrop.jsonl.meta.json",
            notes="matched offset-layer knockout L26,L4,L21,L20,L7 for broad layer-budget sensitivity control",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="dynamic-aligned interaction module replace",
            family="token-remapped attention bridge",
            meta_path="results/bridge_ridge_qk_dynalign_interact_module_replace_20260420_diag/qwen_gsm5_grouped_subspace_transport_w010_r4_dynalign_interact_module_replace_cal16_chat.jsonl.meta.json",
            notes="dynalign module replace plus prompt-local interaction distillation on a 16-prompt diagnostic slice",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="token-basis replace",
            family="token-native attention bridge",
            meta_path="results/bridge_ridge_qk_tokenbasis_replace_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_tokenbasis_replace_cal64_chat.jsonl.meta.json",
            notes="direct-output slotted module constrained to a target next-token output basis",
        )
    )
    rows.append(
        _meta_row(
            split="gsm8k_5_controlled_smoke",
            method="shared-plus-private asym projector",
            family="projector bridge",
            meta_path="results/bridge_ridge_qk_asym_projector_20260420/qwen_gsm5_grouped_subspace_transport_w010_r4_asym_projector_cal64_chat.jsonl.meta.json",
            notes="shared-plus-private post-transport projector combining full-rank query projector with the paired K/V interface",
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
        {
            "candidate": "results/bridge_ridge_qk_asym_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_asym_adapter_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "shared_plus_private_asym_adapter",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_asym_dynmap_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_asym_dynmap_adapter_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "shared_plus_private_dynmap_adapter",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_xattn_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_xattn_adapter_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "xattn_adapter",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_xattn_dynmap_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_xattn_dynmap_adapter_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "xattn_dynmap_adapter",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_module_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_module_adapter_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "module_adapter",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_module_replace_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "module_replace",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_spanalign_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_spanalign_module_replace_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "spanalign_module_replace",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_bytespan_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_bytespan_module_replace_cal16_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "bytespan_module_replace",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "dynalign_ctxonly_module_replace",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_dynalign_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "dynalign_module_replace",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_dynalign_interact_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_interact_module_replace_cal16_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "dynalign_interact_module_replace",
            "baseline_label": "target_alone_control",
        },
        {
            "candidate": "results/bridge_ridge_qk_tokenbasis_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_tokenbasis_replace_cal64_chat.jsonl",
            "baseline": "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "target_alone",
            "candidate_label": "tokenbasis_replace",
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


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_layer_localization_rows() -> list[dict[str, Any]]:
    specs = [
        {
            "method": "shared_plus_private_asym_adapter",
            "family": "modular bridge",
            "jsonl": "results/bridge_ridge_qk_asym_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_asym_adapter_cal64_chat.jsonl",
            "notes": "shared-plus-private dense bridge on controlled gsm8k_eval_10",
        },
        {
            "method": "shared_plus_private_dynmap_adapter",
            "family": "modular bridge",
            "jsonl": "results/bridge_ridge_qk_asym_dynmap_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_asym_dynmap_adapter_cal64_chat.jsonl",
            "notes": "shared-plus-private bridge plus context-reweighted teacher on controlled gsm8k_eval_10",
        },
        {
            "method": "xattn_adapter",
            "family": "attention bridge",
            "jsonl": "results/bridge_ridge_qk_xattn_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_xattn_adapter_cal64_chat.jsonl",
            "notes": "tiny cross-attention bridge on controlled gsm8k_eval_10",
        },
        {
            "method": "xattn_dynmap_adapter",
            "family": "attention bridge",
            "jsonl": "results/bridge_ridge_qk_xattn_dynmap_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_xattn_dynmap_adapter_cal64_chat.jsonl",
            "notes": "cross-attention bridge plus context-reweighted teacher on controlled gsm8k_eval_10",
        },
        {
            "method": "module_adapter",
            "family": "attention bridge",
            "jsonl": "results/bridge_ridge_qk_module_adapter_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_module_adapter_cal64_chat.jsonl",
            "notes": "slotted attention-side module with nonlinear readout on controlled gsm8k_eval_10",
        },
        {
            "method": "module_replace",
            "family": "attention bridge",
            "jsonl": "results/bridge_ridge_qk_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_module_replace_cal64_chat.jsonl",
            "notes": "direct-output slotted attention-side module on controlled gsm8k_eval_10",
        },
        {
            "method": "spanalign_module_replace",
            "family": "token-remapped attention bridge",
            "jsonl": "results/bridge_ridge_qk_spanalign_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_spanalign_module_replace_cal64_chat.jsonl",
            "notes": "direct-output slotted module fit from raw-prompt monotone span-aligned calibration pairs on controlled gsm8k_eval_10",
        },
        {
            "method": "bytespan_module_replace",
            "family": "token-remapped attention bridge",
            "jsonl": "results/bridge_ridge_qk_bytespan_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_bytespan_module_replace_cal16_chat.jsonl",
            "notes": "direct-output slotted module fit from dominant UTF-8 byte-overlap calibration pairs on controlled gsm8k_eval_10 (16-prompt diagnostic calibration)",
        },
        {
            "method": "dynalign_ctxonly_module_replace",
            "family": "token-remapped attention bridge",
            "jsonl": "results/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.jsonl",
            "notes": "matched dynalign context-only null on controlled gsm8k_eval_10 (16-prompt diagnostic calibration)",
        },
        {
            "method": "dynalign_module_replace",
            "family": "token-remapped attention bridge",
            "jsonl": "results/bridge_ridge_qk_dynalign_module_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.jsonl",
            "notes": "direct-output slotted module fit from context-plus-output-overlap token mixtures on controlled gsm8k_eval_10",
        },
        {
            "method": "dynalign_interact_module_replace",
            "family": "token-remapped attention bridge",
            "jsonl": "results/bridge_ridge_qk_dynalign_interact_module_replace_20260420_diag/qwen_gsm10_grouped_subspace_transport_w010_r4_dynalign_interact_module_replace_cal16_chat.jsonl",
            "notes": "dynalign module replace plus prompt-local interaction distillation on controlled gsm8k_eval_10 (16-prompt diagnostic calibration)",
        },
        {
            "method": "tokenbasis_replace",
            "family": "token-native attention bridge",
            "jsonl": "results/bridge_ridge_qk_tokenbasis_replace_20260420/qwen_gsm10_grouped_subspace_transport_w010_r4_tokenbasis_replace_cal64_chat.jsonl",
            "notes": "direct-output slotted module constrained to a target next-token output basis on controlled gsm8k_eval_10",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        records = _load_prediction_records(ROOT / spec["jsonl"])
        by_layer: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "source_layers": [],
                "keep": [],
                "keep_fraction": [],
                "score_top": [],
                "score_gap": [],
                "score_entropy": [],
                "selected_mean_pos": [],
                "selected_span": [],
            }
        )
        example_count = 0
        correct_count = 0
        for record in records:
            if str(record.get("method")) != "rotalign_kv_gate_0.10":
                continue
            example_count += 1
            correct_count += int(bool(record.get("correct", False)))
            for trace in record.get("selector_trace", []):
                layer = int(trace["target_layer"])
                slot = by_layer[layer]
                slot["source_layers"].append(int(trace["source_layer"]))
                if "keep" in trace:
                    slot["keep"].append(float(trace["keep"]))
                if "keep_fraction" in trace:
                    slot["keep_fraction"].append(float(trace["keep_fraction"]))
                if "score_top" in trace:
                    slot["score_top"].append(float(trace["score_top"]))
                if "score_gap" in trace:
                    slot["score_gap"].append(float(trace["score_gap"]))
                if "score_entropy" in trace and trace["score_entropy"] is not None and not math.isnan(float(trace["score_entropy"])):
                    slot["score_entropy"].append(float(trace["score_entropy"]))
                if "selected_mean_pos" in trace:
                    slot["selected_mean_pos"].append(float(trace["selected_mean_pos"]))
                if "selected_min_pos" in trace and "selected_max_pos" in trace:
                    slot["selected_span"].append(float(trace["selected_max_pos"]) - float(trace["selected_min_pos"]))

        accuracy = float(correct_count / example_count) if example_count else 0.0
        for target_layer in sorted(by_layer):
            slot = by_layer[target_layer]
            source_counts = Counter(slot["source_layers"])
            source_layer_mode = int(source_counts.most_common(1)[0][0]) if source_counts else -1
            rows.append(
                {
                    "method": spec["method"],
                    "family": spec["family"],
                    "notes": spec["notes"],
                    "target_layer": int(target_layer),
                    "source_layer_mode": source_layer_mode,
                    "examples": int(example_count),
                    "accuracy": accuracy,
                    "mean_keep": _safe_mean(slot["keep"]),
                    "mean_keep_fraction": _safe_mean(slot["keep_fraction"]),
                    "mean_score_top": _safe_mean(slot["score_top"]),
                    "mean_score_gap": _safe_mean(slot["score_gap"]),
                    "mean_score_entropy": _safe_mean(slot["score_entropy"]),
                    "mean_selected_mean_pos": _safe_mean(slot["selected_mean_pos"]),
                    "mean_selected_span": _safe_mean(slot["selected_span"]),
                    "layer_score": _safe_mean(slot["score_top"]) * _safe_mean(slot["keep_fraction"]),
                }
            )
    return rows


def _write_layer_localization_markdown(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    lines = [
        "# Current Layer Localization (2026-04-20)",
        "",
        "Telemetry source: `selector_trace` from controlled `gsm8k_eval_10` runs under the fair shared-chat / `enable_thinking=false` Qwen control.",
        "",
        "| Method | Acc | Top target layers by layer_score | Mean keep frac (top layer) | Mean score top (top layer) | Mean score gap (top layer) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for method, method_rows in sorted(grouped.items()):
        ranked = sorted(method_rows, key=lambda row: (-float(row["layer_score"]), int(row["target_layer"])))
        top = ranked[:5]
        layer_summary = ", ".join(
            f"L{int(row['target_layer'])}<-S{int(row['source_layer_mode'])}"
            for row in top
        )
        lead = top[0] if top else None
        lines.append(
            "| {method} | {acc:.4f} | {layers} | {keep:.4f} | {score_top:.4f} | {score_gap:.4f} |".format(
                method=method,
                acc=float(method_rows[0]["accuracy"]) if method_rows else 0.0,
                layers=layer_summary or "-",
                keep=float(lead["mean_keep_fraction"]) if lead else 0.0,
                score_top=float(lead["mean_score_top"]) if lead else 0.0,
                score_gap=float(lead["mean_score_gap"]) if lead else 0.0,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    frontier_rows = _build_frontier_rows()
    paired_rows = _build_paired_rows()
    layer_rows = _build_layer_localization_rows()

    frontier_json = ROOT / "paper/bytes_accuracy_frontier_20260420.json"
    frontier_md = ROOT / "paper/bytes_accuracy_table_20260420.md"
    paired_jsonl = ROOT / "paper/paired_flip_table_20260420.jsonl"
    paired_md = ROOT / "paper/paired_flip_table_20260420.md"
    layer_jsonl = ROOT / "paper/layer_localization_20260420.jsonl"
    layer_md = ROOT / "paper/layer_localization_20260420.md"

    frontier_json.write_text(json.dumps(frontier_rows, indent=2, sort_keys=True) + "\n")
    _write_frontier_markdown(frontier_rows, frontier_md)
    _write_jsonl(paired_rows, paired_jsonl)
    _write_paired_markdown(paired_rows, paired_md)
    _write_jsonl(layer_rows, layer_jsonl)
    _write_layer_localization_markdown(layer_rows, layer_md)

    print(frontier_json)
    print(frontier_md)
    print(paired_jsonl)
    print(paired_md)
    print(layer_jsonl)
    print(layer_md)


if __name__ == "__main__":
    main()
