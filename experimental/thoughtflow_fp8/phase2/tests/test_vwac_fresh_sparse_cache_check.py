import torch

from experimental.thoughtflow_fp8.phase2 import vwac_fresh_sparse_cache_check as vwac


def test_vwac_weights_future_attention_by_value_norm():
    attention = torch.zeros(1, 1, 4, 4)
    attention[0, 0, 2, 0] = 1.0
    attention[0, 0, 3, 1] = 1.0
    key = torch.zeros(1, 1, 4, 2)
    value = torch.zeros(1, 1, 4, 2)
    value[0, 0, 0] = torch.tensor([3.0, 4.0])
    value[0, 0, 1] = torch.tensor([1.0, 0.0])

    kept, scores = vwac._vwac_topk((attention,), ((key, value),), budget=1)
    by_index = {score["index"]: score for score in scores}

    assert kept == {0}
    assert by_index[0]["vwac"] > by_index[1]["vwac"]


def test_vwac_ties_break_by_lower_position():
    attention = torch.zeros(1, 1, 4, 4)
    key = torch.zeros(1, 1, 4, 2)
    value = torch.zeros(1, 1, 4, 2)

    kept, _ = vwac._vwac_topk((attention,), ((key, value),), budget=1)

    assert kept == {0}


def test_vwac_promotion_requires_best_compressed_row():
    summary = {
        "full_cache": {"nll": 1.0},
        "rkv_like": {"nll": 1.16},
        "thin_kv_like": {"nll": 1.15},
        "thoughtflow_saliency_recent": {"nll": 1.09},
        vwac.VWAC_POLICY_NAME: {"nll": 1.10},
    }
    paired_vs_rkv = {vwac.VWAC_POLICY_NAME: {"ci95_high": -0.01}}
    paired_vs_thin = {vwac.VWAC_POLICY_NAME: {"ci95_high": -0.01}}

    decision = vwac._promotion_decision(summary, paired_vs_rkv, paired_vs_thin)

    assert decision["promotion_pass"] is False
