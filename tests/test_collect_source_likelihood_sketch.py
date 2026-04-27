import json
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


def test_collect_supports_limit_and_resume(tmp_path, monkeypatch):
    target = tmp_path / "target.jsonl"
    source = tmp_path / "source.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    output = tmp_path / "sketch.jsonl"
    output_md = tmp_path / "sketch.md"
    rows = [
        {"example_id": "a", "method": "target_alone", "prediction": "1", "correct": False},
        {"example_id": "b", "method": "target_alone", "prediction": "2", "correct": True},
    ]
    target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source.write_text(
        "".join(
            json.dumps({**row, "method": "source_alone", "prediction": str(idx + 3)}) + "\n"
            for idx, row in enumerate(rows)
        ),
        encoding="utf-8",
    )
    eval_file.write_text(
        "".join(json.dumps({"prompt": f"p{row['example_id']}", "answer": row["prediction"]}) + "\n" for row in rows),
        encoding="utf-8",
    )

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __call__(self, text, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(input_ids=[1 if text.strip() in {"1", "3"} else 2])

    class FakeModel:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

    monkeypatch.setattr(collect, "_load_examples_by_id", lambda path: {
        "a": evaluate.GenerationExample(prompt="pa", answers=["1"]),
        "b": evaluate.GenerationExample(prompt="pb", answers=["2"]),
    })
    monkeypatch.setattr(collect, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()))
    monkeypatch.setattr(
        collect,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel()),
    )
    monkeypatch.setattr(
        collect.evaluate,
        "_prepare_prefix_state",
        lambda *args, **kwargs: evaluate.PrefixState(None, torch.tensor([[0]]), 1),
    )

    def fake_step_with_past(model, current, past, device):
        del model, current, past, device
        logits = torch.full((1, 4), -10.0)
        logits[0, 1] = 0.0
        return logits, None

    monkeypatch.setattr(collect.evaluate, "_step_with_past", fake_step_with_past)

    base_args = [
        "--source-model",
        "fake/source",
        "--eval-file",
        str(eval_file),
        "--candidate",
        f"target=path={target},method=target_alone",
        "--candidate",
        f"source=path={source},method=source_alone",
        "--reference-label",
        "target",
        "--device",
        "cpu",
        "--output-jsonl",
        str(output),
        "--output-md",
        str(output_md),
    ]
    args = collect.parse_args([*base_args, "--limit", "1"])
    args.command = "first"
    first = collect.collect(args)
    assert first["n"] == 1
    assert first["ordered_example_ids"] == ["a"]

    args = collect.parse_args([*base_args, "--limit", "2", "--resume"])
    args.command = "resume"
    second = collect.collect(args)
    assert second["n"] == 2
    assert second["skipped_existing"] == 1
    assert second["ordered_example_ids"] == ["a", "b"]
    assert "## Ordered Example IDs" in output_md.read_text(encoding="utf-8")
