import math
from types import SimpleNamespace

import torch

from latent_bridge import evaluate
from scripts import collect_source_likelihood_sketch as collect


def test_candidate_text_falls_back_to_prediction_then_normalized():
    assert collect._candidate_text({"custom": " 42 "}, "custom") == "42"
    assert collect._candidate_text({"prediction": " 7 "}, "custom") == "7"
    assert collect._candidate_text({"normalized_prediction": "5"}, "prediction") == "5"
    assert collect._candidate_text({}, "prediction") == ""


def test_score_continuation_uses_next_token_logprobs(monkeypatch):
    calls = []

    def fake_step_with_past(model, current, past, device):
        del model, current, device
        calls.append(past)
        logits = torch.full((1, 4), -10.0)
        token_id = 1 if len(calls) == 1 else 2
        logits[0, token_id] = 0.0
        return logits, len(calls)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert text == "ab"
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[1, 2])

    monkeypatch.setattr(collect.evaluate, "_step_with_past", fake_step_with_past)

    score = collect._score_continuation(
        model=object(),
        tokenizer=FakeTokenizer(),
        prefix_state=evaluate.PrefixState(past_key_values=None, last_token=torch.tensor([[0]]), prefix_len=1),
        continuation="ab",
        device="cpu",
    )

    assert score["tokens"] == 2
    assert math.isclose(score["sum_logprob"], score["mean_logprob"] * 2)
    assert score["mean_logprob"] > -0.001
