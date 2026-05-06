from __future__ import annotations

from experimental.thoughtflow_fp8.phase2.rdu_no_retune_reproduction_check import build_report


POLICIES = (
    "full_cache",
    "rdu_topk",
    "rkv_like",
    "thin_kv_like",
    "longflow_like",
    "thoughtflow_saliency_recent",
    "tf_sparse_r0.55_p0.05_m0.12_a2",
)


def _result(offset: float = 0.0) -> dict[str, object]:
    per_trace = {
        "full_cache": [1.0, 1.1, 1.2],
        "rdu_topk": [1.5 + offset, 1.6 + offset, 1.8 + offset],
        "rkv_like": [1.8, 1.9, 2.1],
        "thin_kv_like": [1.7, 1.8, 2.0],
        "longflow_like": [2.1, 2.2, 2.4],
        "thoughtflow_saliency_recent": [1.65, 1.75, 1.95],
        "tf_sparse_r0.55_p0.05_m0.12_a2": [1.62, 1.72, 1.92],
    }
    rows = []
    for policy, values in per_trace.items():
        for trace_id, nll in enumerate(values):
            rows.append(
                {
                    "trace_id": trace_id,
                    "policy": policy,
                    "keep_rate": 1.0 if policy == "full_cache" else 0.2,
                    "retained_prefix_tokens": 10 if policy == "full_cache" else 2,
                    "continuation_tokens": 4,
                    "nll": nll,
                    "delta_nll_vs_full": nll - per_trace["full_cache"][trace_id],
                }
            )
    summary = {
        policy: {
            "n_traces": 3.0,
            "keep_rate": 1.0 if policy == "full_cache" else 0.2,
            "nll": sum(values) / len(values),
            "delta_nll_vs_full": sum(values) / len(values) - sum(per_trace["full_cache"]) / 3,
        }
        for policy, values in per_trace.items()
    }
    return {
        "model_name": "tiny",
        "keep_fraction": 0.2,
        "max_traces": 3,
        "max_length": 12,
        "continuation_tokens": 4,
        "n_scored_traces": 3,
        "summary": summary,
        "rows": rows,
        "paired_delta_nll_vs_rkv_like": {
            policy: {
                "n_pairs": 3.0,
                "mean_delta_nll_minus_rkv_like": summary[policy]["nll"] - summary["rkv_like"]["nll"],
                "ci95_low": summary[policy]["nll"] - summary["rkv_like"]["nll"] - 0.01,
                "ci95_high": summary[policy]["nll"] - summary["rkv_like"]["nll"] + 0.01,
            }
            for policy in POLICIES
            if policy != "rkv_like"
        },
        "paired_delta_nll_vs_thin_kv_like": {
            policy: {
                "n_pairs": 3.0,
                "mean_delta_nll_minus_thin_kv_like": summary[policy]["nll"] - summary["thin_kv_like"]["nll"],
                "ci95_low": summary[policy]["nll"] - summary["thin_kv_like"]["nll"] - 0.01,
                "ci95_high": summary[policy]["nll"] - summary["thin_kv_like"]["nll"] + 0.01,
            }
            for policy in POLICIES
            if policy != "thin_kv_like"
        },
        "status": "synthetic",
    }


def test_build_report_marks_measured_reproduction_and_separation() -> None:
    report = build_report(_result(), _result(offset=0.01))

    assert report["reproduction_pass"] is True
    assert report["cached_label"] == "cached_promoted_gate"
    assert report["measured_label"] == "measured_reproduction_rerun"
    assert report["measured_decision"]["best_compressed_policy"] == "rdu_topk"
    assert report["measured_decision"]["promotion_pass"] is True
    assert report["measured_family_separation"]["cross_family_margin_nll_vs_rdu"]["thin_kv_like"] > 0.03
    assert report["measured_family_separation"]["same_family_margin_nll_vs_rdu"][
        "tf_sparse_r0.55_p0.05_m0.12_a2"
    ] > 0.03
    assert report["measured_oracle_headroom"]["rdu_oracle_hit_rate"] == 1.0
    assert report["cached_vs_measured"]["policy_nll_delta_measured_minus_cached"]["rdu_topk"] > 0.0
