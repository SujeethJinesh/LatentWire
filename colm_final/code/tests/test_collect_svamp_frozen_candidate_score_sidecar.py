import json
import math
from types import SimpleNamespace

import torch

from latent_bridge import evaluate
from scripts import collect_svamp_frozen_candidate_score_sidecar as collect


def _row(example_id: str, method: str, prediction: str, normalized: str, answer: str) -> dict:
    return {
        "answer": [answer],
        "correct": normalized == answer,
        "example_id": example_id,
        "method": method,
        "normalized_prediction": normalized,
        "prediction": prediction,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _target_set(tmp_path, *, target_prediction: str = "Target mentions 12 but answers 7") -> tuple:
    target_rows = [_row("a", "target_alone", target_prediction, "7", "12")]
    source_rows = [_row("a", "source_alone", "Source answer: 12", "12", "12")]
    _write_jsonl(tmp_path / "target.jsonl", target_rows)
    _write_jsonl(tmp_path / "source.jsonl", source_rows)
    target_set = {
        "artifacts": {
            "target": {"path": str(tmp_path / "target.jsonl"), "method": "target_alone"},
            "source": {"path": str(tmp_path / "source.jsonl"), "method": "source_alone"},
            "baselines": [],
            "controls": [],
        },
        "ids": {"clean_source_only": ["a"], "target_self_repair": []},
        "reference_ids": ["a"],
    }
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(json.dumps({"id": "a", "prompt": "Question?", "answer": "12"}) + "\n", encoding="utf-8")
    return target_set_path, eval_file


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
        prefix_state=evaluate.PrefixState(None, torch.tensor([[0]]), 1),
        continuation="ab",
        device="cpu",
    )

    assert score["tokens"] == 2
    assert math.isclose(score["sum_logprob"], score["mean_logprob"] * 2)
    assert score["mean_logprob"] > -0.001


def test_collect_emits_no_leak_target_side_sidecar(tmp_path, monkeypatch):
    target_set_path, eval_file = _target_set(tmp_path)
    output = tmp_path / "sidecar.jsonl"
    output_md = tmp_path / "sidecar.md"
    seen_prompts = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __call__(self, text, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(input_ids=[1 if text.strip().endswith("12") else 2])

    class FakeModel:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        collect,
        "_load_examples_by_id",
        lambda path: {"a": evaluate.GenerationExample(prompt="Question?", answers=["12"])},
    )
    monkeypatch.setattr(collect, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()))
    monkeypatch.setattr(collect, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel()))

    def fake_prepare(model, tokenizer, prompt, device, **kwargs):
        del model, tokenizer, device, kwargs
        seen_prompts.append(prompt)
        return evaluate.PrefixState(None, torch.tensor([[0]]), 1)

    monkeypatch.setattr(collect.evaluate, "_prepare_prefix_state", fake_prepare)

    def fake_step_with_past(model, current, past, device):
        del model, current, past, device
        logits = torch.full((1, 4), -10.0)
        logits[0, 1] = 0.0
        return logits, None

    monkeypatch.setattr(collect.evaluate, "_step_with_past", fake_step_with_past)

    payload = collect.main(
        [
            "--scorer-model",
            "fake/source",
            "--target-set-json",
            str(target_set_path),
            "--eval-file",
            str(eval_file),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--sidecar-bits",
            "16",
            "--output-jsonl",
            str(output),
            "--output-md",
            str(output_md),
            "--date",
            "2026-04-27",
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n"] == 1
    assert payload["candidate_pool"] == "target_side_only"
    assert row["sidecar_bits"] == 16
    assert {item["value"] for item in row["candidate_scores"]} == {"7", "12"}
    assert "source" not in {item["label"] for item in row["candidate_scores"]}
    assert row["candidate_scores"][0]["value"] == "12"
    assert not ({"answer", "correct", "candidate_correct"} & set(row))
    assert all(not ({"answer", "correct", "candidate_correct"} & set(item)) for item in row["candidate_scores"])
    assert seen_prompts and "Question?" in seen_prompts[0]
    assert "target_side_only" in output_md.read_text(encoding="utf-8")


def test_collect_does_not_add_source_only_value(tmp_path, monkeypatch):
    target_set_path, eval_file = _target_set(tmp_path, target_prediction="Target says 7")
    output = tmp_path / "sidecar.jsonl"

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __call__(self, text, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(input_ids=[1])

    class FakeModel:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        collect,
        "_load_examples_by_id",
        lambda path: {"a": evaluate.GenerationExample(prompt="Question?", answers=["12"])},
    )
    monkeypatch.setattr(collect, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()))
    monkeypatch.setattr(collect, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel()))
    monkeypatch.setattr(
        collect.evaluate,
        "_prepare_prefix_state",
        lambda *args, **kwargs: evaluate.PrefixState(None, torch.tensor([[0]]), 1),
    )
    monkeypatch.setattr(
        collect.evaluate,
        "_step_with_past",
        lambda *args, **kwargs: (torch.tensor([[0.0, 1.0, -1.0]]), None),
    )

    collect.main(
        [
            "--scorer-model",
            "fake/source",
            "--target-set-json",
            str(target_set_path),
            "--eval-file",
            str(eval_file),
            "--device",
            "cpu",
            "--output-jsonl",
            str(output),
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert {item["value"] for item in row["candidate_scores"]} == {"7"}
