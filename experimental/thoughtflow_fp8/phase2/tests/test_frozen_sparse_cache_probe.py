import torch
import pytest

from experimental.thoughtflow_fp8.phase2 import frozen_sparse_cache_probe as probe


def test_frozen_policy_set_contains_only_fixed_thoughtflow_candidates():
    policies = probe._frozen_policies()

    assert "thoughtflow_saliency_recent" in policies
    assert probe.FROZEN_SPARSE_POLICY_NAME in policies
    assert all("tf_sparse_" not in name or name == probe.FROZEN_SPARSE_POLICY_NAME for name in policies)


def test_status_alive_requires_mean_margin_and_paired_uncertainty():
    summary = {
        "full_cache": {"nll": 1.0},
        "thoughtflow_saliency_recent": {"nll": 1.10},
        probe.FROZEN_SPARSE_POLICY_NAME: {"nll": 1.20},
        "rkv_like": {"nll": 1.16},
        "thin_kv_like": {"nll": 1.14},
    }
    paired_vs_rkv = {
        "thoughtflow_saliency_recent": {"ci95_high": -0.01},
    }
    paired_vs_thin = {
        "thoughtflow_saliency_recent": {"ci95_high": -0.01},
    }

    assert probe._status(summary, paired_vs_rkv, paired_vs_thin).startswith("ALIVE")


def test_status_mixed_when_thinkv_uncertainty_crosses_zero():
    summary = {
        "full_cache": {"nll": 1.0},
        "thoughtflow_saliency_recent": {"nll": 1.10},
        probe.FROZEN_SPARSE_POLICY_NAME: {"nll": 1.20},
        "rkv_like": {"nll": 1.16},
        "thin_kv_like": {"nll": 1.14},
    }
    paired_vs_rkv = {
        "thoughtflow_saliency_recent": {"ci95_high": -0.01},
    }
    paired_vs_thin = {
        "thoughtflow_saliency_recent": {"ci95_high": 0.02},
    }

    assert "paired uncertainty remains" in probe._status(summary, paired_vs_rkv, paired_vs_thin)


def test_rdu_topk_uses_lag_bucket_mass_and_second_bucket_bonus():
    attention = torch.zeros(1, 1, 40, 40)
    attention[0, 0, 10, 2] = 8.0
    attention[0, 0, 18, 2] = 4.0
    attention[0, 0, 11, 3] = 9.0

    kept, scores = probe._rdu_topk_from_attentions((attention,), budget=1)
    score_by_index = {score["index"]: score for score in scores}

    assert kept == {2}
    assert score_by_index[2]["bucket_masses"]["b0_8_15"] == pytest.approx(8.0 / (8**0.5))
    assert score_by_index[2]["bucket_masses"]["b1_16_31"] == pytest.approx(1.0)
    assert score_by_index[2]["rdu"] > score_by_index[3]["rdu"]


def test_rdu_topk_breaks_score_ties_by_lower_position():
    attention = torch.zeros(1, 1, 20, 20)
    attention[0, 0, 8, 0] = 1.0
    attention[0, 0, 9, 1] = 1.0

    kept, _ = probe._rdu_topk_from_attentions((attention,), budget=1)

    assert kept == {0}


def test_rdu_telemetry_reports_labels_and_recurrence_buckets_separately():
    trace = [
        probe.Token("a", "anchor", 0.0),
        probe.Token("p", "phase", 0.0),
        probe.Token("m", "math_state", 0.0),
        probe.Token("x", "reason", 0.0),
    ]
    scores = [
        {"rdu": 1.0, "primary_bucket": "b0_8_15"},
        {"rdu": 0.5, "primary_bucket": "b1_16_31"},
        {"rdu": 0.0, "primary_bucket": "none"},
        {"rdu": 0.8, "primary_bucket": "b0_8_15"},
    ]

    telemetry = probe._rdu_retention_telemetry(trace, kept={0, 2}, scores=scores)

    assert telemetry["labels"]["anchor"]["retention_rate"] == 1.0
    assert telemetry["labels"]["phase"]["retention_rate"] == 0.0
    assert telemetry["labels"]["math_state"]["retention_rate"] == 1.0
    assert telemetry["recurrence_buckets"]["b0_8_15"]["total"] == 2.0
    assert telemetry["recurrence_buckets"]["b0_8_15"]["retained"] == 1.0
    assert telemetry["recurrence_buckets"]["none"]["retention_rate"] == 1.0


def test_status_evaluates_preregistered_rdu_rule_when_present():
    summary = {
        "full_cache": {"nll": 1.0},
        "thoughtflow_saliency_recent": {"nll": 1.20},
        probe.FROZEN_SPARSE_POLICY_NAME: {"nll": 1.30},
        probe.RDU_POLICY_NAME: {"nll": 1.10},
        "rkv_like": {"nll": 1.15},
        "thin_kv_like": {"nll": 1.14},
    }
    paired_vs_rkv = {probe.RDU_POLICY_NAME: {"ci95_high": -0.01}}
    paired_vs_thin = {probe.RDU_POLICY_NAME: {"ci95_high": -0.01}}

    assert probe._status(summary, paired_vs_rkv, paired_vs_thin).startswith("ALIVE")


def test_status_kills_rdu_when_preregistered_margin_fails():
    summary = {
        "full_cache": {"nll": 1.0},
        "thoughtflow_saliency_recent": {"nll": 1.20},
        probe.FROZEN_SPARSE_POLICY_NAME: {"nll": 1.30},
        probe.RDU_POLICY_NAME: {"nll": 1.12},
        "rkv_like": {"nll": 1.15},
        "thin_kv_like": {"nll": 1.14},
    }
    paired_vs_rkv = {probe.RDU_POLICY_NAME: {"ci95_high": -0.01}}
    paired_vs_thin = {probe.RDU_POLICY_NAME: {"ci95_high": -0.01}}

    assert probe._status(summary, paired_vs_rkv, paired_vs_thin).startswith("KILLED")
