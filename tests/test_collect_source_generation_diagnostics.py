import math

import torch

from scripts import collect_source_generation_diagnostics as diag


def test_step_stats_extracts_greedy_confidence_fields():
    logits = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32)

    stats = diag._step_stats(logits, token_id=1)

    assert stats.token_id == 1
    assert stats.chosen_logprob > -0.2
    assert stats.top1_logit == 3.0
    assert stats.top2_logit == 1.0
    assert stats.top1_top2_logit_margin == 2.0
    assert 0.0 < stats.top1_prob < 1.0
    assert stats.entropy > 0.0


def test_summarize_steps_handles_empty_and_nonempty_steps():
    empty = diag._summarize_steps([])
    assert empty["generated_token_ids"] == []
    assert empty["mean_chosen_logprob"] is None

    steps = [
        diag.StepStats(
            token_id=4,
            chosen_logprob=-0.1,
            top1_logit=2.0,
            top2_logit=1.0,
            top1_prob=0.8,
            top2_prob=0.1,
            top1_top2_logit_margin=1.0,
            entropy=0.5,
        ),
        diag.StepStats(
            token_id=5,
            chosen_logprob=-0.3,
            top1_logit=1.5,
            top2_logit=1.4,
            top1_prob=0.45,
            top2_prob=0.4,
            top1_top2_logit_margin=0.1,
            entropy=1.2,
        ),
    ]

    summary = diag._summarize_steps(steps)

    assert summary["generated_token_ids"] == [4, 5]
    assert summary["generated_tokens"] == 2
    assert math.isclose(summary["mean_chosen_logprob"], -0.2)
    assert summary["min_chosen_logprob"] == -0.3
    assert summary["final_chosen_logprob"] == -0.3
    assert math.isclose(summary["mean_entropy"], 0.85)
    assert math.isclose(summary["mean_top1_top2_logit_margin"], 0.55)
    assert summary["min_top1_prob"] == 0.45
