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


def test_paired_deltas_name_metric_for_requested_baseline():
    rows = [
        {"trace_id": 0, "policy": "thin_kv_like", "nll": 2.0},
        {"trace_id": 0, "policy": "thoughtflow_sparse", "nll": 1.75},
    ]

    deltas = probe._paired_deltas(rows, baseline_policy="thin_kv_like", bootstrap_samples=20)

    assert deltas["thoughtflow_sparse"]["mean_delta_nll_minus_thin_kv_like"] == -0.25


def test_sparse_sweep_configs_are_small_and_stable():
    configs = probe._sparse_sweep_configs()

    assert len(configs) == 24
    assert configs[0].name.startswith("tf_sparse_")


def test_sparse_sweep_policy_respects_budget_and_keeps_recent():
    config = probe.SparseSweepConfig(recent_fraction=0.55, phase_bonus=0.05, math_bonus=0.12, protect_anchors=2)
    policy = probe._make_sparse_sweep_policy(config)
    trace = [
        probe.Token("a", "anchor", 1.0),
        probe.Token("b", "anchor", 0.9),
        probe.Token("phase", "phase", 0.4),
        probe.Token("math", "math_state", 0.7),
        probe.Token("low", "reason", 0.1),
        probe.Token("recent1", "reason", 0.1),
        probe.Token("recent2", "reason", 0.1),
    ]

    kept = policy(trace, budget=5)

    assert len(kept) == 5
    assert {0, 1}.issubset(kept)
    assert {5, 6}.issubset(kept)
