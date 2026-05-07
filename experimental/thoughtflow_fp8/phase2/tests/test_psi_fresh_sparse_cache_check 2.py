import torch
import inspect

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


def test_run_accepts_explicit_model_revision_for_replay():
    signature = inspect.signature(psi.run)

    assert "model_revision" in signature.parameters
    assert signature.parameters["model_revision"].default == psi.DISTILGPT2_REVISION


def test_markdown_reports_model_and_tokenizer_revision(tmp_path):
    result = {
        "status": "KILLED",
        "model_name": "distilgpt2",
        "model_revision": "abc123",
        "tokenizer_revision": "abc123",
        "policy_name": psi.PSI_POLICY_NAME,
        "n_scored_traces": 0,
        "keep_fraction": 0.2,
        "max_length": 96,
        "continuation_tokens": 24,
        "input_paths": [],
        "summary": {"full_cache": {"nll": 1.0, "n_traces": 0, "keep_rate": 1.0, "delta_nll_vs_full": 0.0}},
        "decision": {
            "best_compressed_policy": "none",
            "margin_vs_rkv_like": 0.0,
            "paired_delta_vs_rkv_like": {},
            "margin_vs_thin_kv_like": 0.0,
            "paired_delta_vs_thin_kv_like": {},
            "promotion_pass": False,
        },
        "psi_topk_telemetry": {
            "aggregate": {
                "mean_surprisal": 0.0,
                "mean_kept_surprisal": 0.0,
                "max_surprisal": 0.0,
                "nonzero_tokens": 0.0,
            }
        },
    }

    output = tmp_path / "psi.md"
    psi.write_markdown(result, output)

    text = output.read_text(encoding="utf-8")
    assert "model revision: `abc123`" in text
    assert "tokenizer revision: `abc123`" in text
