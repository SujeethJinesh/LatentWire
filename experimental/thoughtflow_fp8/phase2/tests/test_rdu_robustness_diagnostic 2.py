import pytest

from experimental.thoughtflow_fp8.phase2 import rdu_robustness_diagnostic as diag


def _row(trace_id: int, policy: str, nll: float) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "policy": policy,
        "keep_rate": 0.2 if policy != diag.FULL_POLICY else 1.0,
        "retained_prefix_tokens": 2,
        "continuation_tokens": 4,
        "nll": nll,
        "delta_nll_vs_full": nll - 1.0,
    }


def _fake_result() -> dict[str, object]:
    rows = []
    for trace_id in range(4):
        rows.extend(
            [
                _row(trace_id, diag.FULL_POLICY, 1.0),
                _row(trace_id, diag.RDU_POLICY, 1.50),
                _row(trace_id, diag.THIN_POLICY, 1.70),
                _row(trace_id, diag.RKV_POLICY, 1.90),
                _row(trace_id, "thoughtflow_saliency_recent", 1.80),
                _row(trace_id, "tf_sparse_r0.55_p0.05_m0.12_a2", 1.75),
            ]
        )
    return {
        "model_name": "toy",
        "keep_fraction": 0.2,
        "n_scored_traces": 4,
        "continuation_tokens": 4,
        "rows": rows,
    }


def test_deterministic_splits_are_fixed_and_cover_expected_partitions():
    splits = diag._deterministic_splits([0, 1, 2, 3, 4])

    assert [split["name"] for split in splits] == [
        "all_traces",
        "even_trace_ids",
        "odd_trace_ids",
        "first_half_trace_ids",
        "second_half_trace_ids",
    ]
    assert splits[1]["trace_ids"] == [0, 2, 4]
    assert splits[2]["trace_ids"] == [1, 3]
    assert splits[3]["trace_ids"] == [0, 1]
    assert splits[4]["trace_ids"] == [2, 3, 4]


def test_cached_rdu_diagnostic_reports_split_promotion_without_retuning():
    result = diag._run_from_result(_fake_result(), bootstrap_samples=25)

    assert result["status"].startswith("PROMOTED on cached full gate")
    assert result["split_mean_margin_passes"] == 4
    assert result["split_paired_mean_passes"] == 4
    assert result["split_promotion_passes"] == 4
    for split in result["split_results"]:
        decision = split["rdu_decision"]
        assert split["best_compressed_policy"] == diag.RDU_POLICY
        assert decision["promotion_pass"] is True
        assert decision["margin_vs_thin_kv_like"] == pytest.approx(0.2)
        assert decision["margin_vs_rkv_like"] == pytest.approx(0.4)
        assert decision["win_rate_vs_thin_kv_like"] == 1.0
