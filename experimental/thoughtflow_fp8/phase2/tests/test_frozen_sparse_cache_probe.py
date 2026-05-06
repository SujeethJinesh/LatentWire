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
