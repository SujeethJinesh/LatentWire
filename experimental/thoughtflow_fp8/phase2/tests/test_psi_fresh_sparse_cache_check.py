import torch

from experimental.thoughtflow_fp8.phase2 import psi_fresh_sparse_cache_check as psi


def test_prefix_surprisal_uses_previous_token_logits():
    logits = torch.zeros(1, 4, 5)
    prefix_ids = [0, 1, 2, 3]
    logits[0, 0, 1] = 5.0
    logits[0, 1, 2] = -5.0
    logits[0, 2, 3] = 1.0

    scores = psi._prefix_surprisal_scores(logits, prefix_ids)

    assert scores[0]["surprisal"] == 0.0
    assert scores[2]["surprisal"] > scores[3]["surprisal"] > scores[1]["surprisal"]


def test_psi_topk_breaks_surprisal_ties_by_lower_position():
    logits = torch.zeros(1, 4, 5)
    prefix_ids = [0, 1, 2, 3]

    kept, _ = psi._psi_topk_from_logits(logits, prefix_ids, budget=1)

    assert kept == {1}


def test_promotion_requires_margin_uncertainty_and_best_compressed():
    summary = {
        "full_cache": {"nll": 1.0},
        "rkv_like": {"nll": 1.16},
        "thin_kv_like": {"nll": 1.15},
        "thoughtflow_saliency_recent": {"nll": 1.13},
        psi.PSI_POLICY_NAME: {"nll": 1.10},
    }
    paired_vs_rkv = {psi.PSI_POLICY_NAME: {"ci95_high": -0.01}}
    paired_vs_thin = {psi.PSI_POLICY_NAME: {"ci95_high": -0.01}}

    decision = psi._promotion_decision(summary, paired_vs_rkv, paired_vs_thin)

    assert decision["promotion_pass"] is True


def test_promotion_fails_when_psi_is_not_best_compressed():
    summary = {
        "full_cache": {"nll": 1.0},
        "rkv_like": {"nll": 1.16},
        "thin_kv_like": {"nll": 1.15},
        "thoughtflow_saliency_recent": {"nll": 1.09},
        psi.PSI_POLICY_NAME: {"nll": 1.10},
    }
    paired_vs_rkv = {psi.PSI_POLICY_NAME: {"ci95_high": -0.01}}
    paired_vs_thin = {psi.PSI_POLICY_NAME: {"ci95_high": -0.01}}

    decision = psi._promotion_decision(summary, paired_vs_rkv, paired_vs_thin)

    assert decision["promotion_pass"] is False

