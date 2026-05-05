import torch

from experimental.thoughtflow_fp8.phase2 import kv_drop_quality_probe as probe


def test_prune_cache_keeps_sorted_prefix_indices():
    key = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    value = key + 100

    pruned = probe._prune_cache(((key, value),), {4, 1, 3})

    assert len(pruned) == 1
    assert torch.equal(pruned[0][0], key.index_select(2, torch.tensor([1, 3, 4])))
    assert torch.equal(pruned[0][1], value.index_select(2, torch.tensor([1, 3, 4])))


def test_status_marks_tie_window_as_mixed():
    summary = {
        "full_cache": {"nll": 1.0},
        "thoughtflow": {"nll": 1.10},
        "thoughtflow_recent": {"nll": 1.09},
        "rkv_like": {"nll": 1.11},
        "thin_kv_like": {"nll": 1.20},
    }

    assert probe._status(summary).startswith("MIXED")


def test_paired_deltas_report_trace_matched_difference():
    rows = [
        {"trace_id": 0, "policy": "rkv_like", "nll": 2.0},
        {"trace_id": 1, "policy": "rkv_like", "nll": 4.0},
        {"trace_id": 0, "policy": "thoughtflow_sweep_best", "nll": 1.5},
        {"trace_id": 1, "policy": "thoughtflow_sweep_best", "nll": 3.5},
    ]

    deltas = probe._paired_deltas(rows, baseline_policy="rkv_like", bootstrap_samples=20)

    assert deltas["thoughtflow_sweep_best"]["mean_delta_nll_minus_rkv_like"] == -0.5
