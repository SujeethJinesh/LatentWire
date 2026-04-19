"""Build the main paired-comparison artifact for the current paper draft."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.prediction_compare import (  # noqa: E402
    compare_prediction_records,
    load_prediction_records,
    write_jsonl,
    write_markdown,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    comparisons = [
        {
            "candidate": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_shuffled_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "fixed_prior",
            "baseline_label": "shuffled_prior",
        },
        {
            "candidate": ROOT / "results/grouped_signature_transport_20260419/qwen_gsm70_grouped_signature_transport_w025_norank.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_signature_transport",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/grouped_signature_transport_20260419/qwen_gsm70_grouped_signature_transport_w025_norank.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "grouped_signature_transport",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_subspace_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "grouped_subspace_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/grouped_template_resid4_20260419/qwen_gsm70_grouped_template_transport_w025_r4_cal64.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_template_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/grouped_template_resid4_20260419/qwen_gsm70_grouped_template_transport_w025_r4_cal64.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "grouped_template_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/grouped_template_resid4_20260419/qwen_gsm70_grouped_template_transport_w025_r4_cal64.jsonl",
            "baseline": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_template_resid4",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": ROOT / "results/grouped_template_subspace_resid4_20260419/qwen_gsm70_grouped_template_subspace_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_template_subspace_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/grouped_template_subspace_resid4_20260419/qwen_gsm70_grouped_template_subspace_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "grouped_template_subspace_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/grouped_template_subspace_resid4_20260419/qwen_gsm70_grouped_template_subspace_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_template_subspace_resid4",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": ROOT / "results/grouped_template_subspace_resid4_20260419/qwen_gsm70_grouped_template_subspace_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/grouped_template_resid4_20260419/qwen_gsm70_grouped_template_transport_w025_r4_cal64.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "grouped_template_subspace_resid4",
            "baseline_label": "grouped_template_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_template_transport_20260419/qwen_gsm70_broadcast_template_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_template_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/broadcast_template_transport_20260419/qwen_gsm70_broadcast_template_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "broadcast_template_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/broadcast_template_transport_20260419/qwen_gsm70_broadcast_template_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_template_resid4",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_template_ot_transport_20260419/qwen_gsm70_broadcast_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_template_ot_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/broadcast_template_ot_transport_20260419/qwen_gsm70_broadcast_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "broadcast_template_ot_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/broadcast_template_ot_transport_20260419/qwen_gsm70_broadcast_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_template_ot_resid4",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_template_ot_transport_20260419/qwen_gsm70_broadcast_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/broadcast_template_transport_20260419/qwen_gsm70_broadcast_template_transport_w010_r4_cal64.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_template_ot_resid4",
            "baseline_label": "broadcast_template_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_peak_template_ot_transport_20260419/qwen_gsm70_broadcast_peak_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_peak_template_ot_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/broadcast_peak_template_ot_transport_20260419/qwen_gsm70_broadcast_peak_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "broadcast_peak_template_ot_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/broadcast_peak_template_ot_transport_20260419/qwen_gsm70_broadcast_peak_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_peak_template_ot_resid4",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_peak_template_ot_transport_20260419/qwen_gsm70_broadcast_peak_template_ot_transport_w010_r4_cal64.jsonl",
            "baseline": ROOT / "results/broadcast_template_ot_transport_20260419/qwen_gsm70_broadcast_template_ot_transport_w010_r4_cal64.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_peak_template_ot_resid4",
            "baseline_label": "broadcast_template_ot_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_retrieval_spectrum_ot_transport_20260419/qwen_gsm70_broadcast_retrieval_spectrum_ot_transport_w010_r4_cal64_fair.jsonl",
            "baseline": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_retrieval_spectrum_ot_resid4",
            "baseline_label": "fixed_prior",
        },
        {
            "candidate": ROOT / "results/broadcast_retrieval_spectrum_ot_transport_20260419/qwen_gsm70_broadcast_retrieval_spectrum_ot_transport_w010_r4_cal64_fair.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "broadcast_retrieval_spectrum_ot_resid4",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/broadcast_retrieval_spectrum_ot_transport_20260419/qwen_gsm70_broadcast_retrieval_spectrum_ot_transport_w010_r4_cal64_fair.jsonl",
            "baseline": ROOT / "results/grouped_subspace_resid4_20260419/qwen_gsm70_grouped_subspace_transport_w010_r4.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_retrieval_spectrum_ot_resid4",
            "baseline_label": "grouped_subspace_resid4",
        },
        {
            "candidate": ROOT / "results/broadcast_retrieval_spectrum_ot_transport_20260419/qwen_gsm70_broadcast_retrieval_spectrum_ot_transport_w010_r4_cal64_fair.jsonl",
            "baseline": ROOT / "results/broadcast_peak_template_ot_transport_20260419/qwen_gsm70_broadcast_peak_template_ot_transport_w010_r4_cal64.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "rotalign_kv_gate_0.10",
            "candidate_label": "broadcast_retrieval_spectrum_ot_resid4",
            "baseline_label": "broadcast_peak_template_ot_resid4",
        },
        {
            "candidate": ROOT / "results/grouped_canonical_transport_20260419/qwen_gsm70_grouped_canonical_transport_r8.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "grouped_canonical_transport",
            "baseline_label": "c2c",
        },
        {
            "candidate": ROOT / "results/gsm8k_per_head_budget_20260418/predictions/qwen_gsm70_attention_headbudget_prior_g010_pos05.jsonl",
            "baseline": ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
            "candidate_method": "rotalign_kv_gate_0.10",
            "baseline_method": "c2c",
            "candidate_label": "fixed_prior",
            "baseline_label": "c2c",
        },
    ]

    rows = []
    for spec in comparisons:
        rows.append(
            compare_prediction_records(
                load_prediction_records(spec["candidate"]),
                load_prediction_records(spec["baseline"]),
                method=spec["candidate_method"],
                baseline_method=spec["baseline_method"],
                candidate_label=spec["candidate_label"],
                baseline_label=spec["baseline_label"],
                n_bootstrap=1000,
            )
        )

    out_jsonl = ROOT / "paper/paired_flip_table_20260419.jsonl"
    out_md = ROOT / "paper/paired_flip_table_20260419.md"
    write_jsonl(rows, out_jsonl)
    write_markdown(rows, out_md)
    print(out_md)


if __name__ == "__main__":
    main()
