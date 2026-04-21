from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


class FakeMethod:
    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.seen_batches: list[list[dict]] = []

    def run_batch(self, items: list[dict]) -> list[dict]:
        self.seen_batches.append(items)
        return self.records[: len(items)]

    def run_batch_vllm(self, items: list[dict]) -> list[dict]:
        return self.run_batch(items)


@pytest.fixture
def latentmas_eval_module():
    return pytest.importorskip(
        "scripts.run_latentmas_competitor_eval",
        reason="LatentMAS competitor wrapper is not implemented yet.",
    )


def test_latentmas_vendor_signatures_are_stable() -> None:
    root = Path("references/repos/LatentMAS")

    assert "max_new_tokens: int = 256" in (root / "methods/baseline.py").read_text(encoding="utf-8")
    assert "max_new_tokens_each: int = 256" in (root / "methods/text_mas.py").read_text(encoding="utf-8")
    assert "latent_steps: int = 10" in (root / "methods/latent_mas.py").read_text(encoding="utf-8")
    assert "def run_batch(self, items: List[Dict]) -> List[Dict]:" in (
        root / "methods/baseline.py"
    ).read_text(encoding="utf-8")
    assert "def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:" in (
        root / "methods/latent_mas.py"
    ).read_text(encoding="utf-8")


def test_load_latentwire_generation_items_converts_gsm_and_svamp_rows(
    tmp_path: Path, latentmas_eval_module
) -> None:
    latentmas_eval = latentmas_eval_module
    fixture = tmp_path / "eval.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Solve: 2 + 2",
                        "answer_text": "4",
                        "aliases": ["#### 4"],
                        "source_question": "2 + 2",
                    }
                ),
                json.dumps(
                    {
                        "question": "Tiffany has 12 bags and had 7 before. Difference?",
                        "answer": "5",
                        "aliases": ["5.0", "#### 5"],
                        "metadata": {"id": "svamp-1"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = latentmas_eval.load_latentwire_generation_items(fixture, limit=None)

    assert items == [
        {
            "id": "0",
            "question": "Solve: 2 + 2",
            "solution": "4",
            "gold": "4",
            "answers": ["4", "#### 4"],
        },
        {
            "id": "svamp-1",
            "question": "Tiffany has 12 bags and had 7 before. Difference?",
            "solution": "5",
            "gold": "5",
            "answers": ["5", "5.0", "#### 5"],
        },
    ]


def test_make_latentmas_method_passes_expected_constructor_arguments(
    monkeypatch, latentmas_eval_module
) -> None:
    latentmas_eval = latentmas_eval_module
    calls: list[tuple[str, dict]] = []

    class Baseline:
        def __init__(self, model, **kwargs) -> None:
            calls.append(("baseline", kwargs))

    class TextMAS:
        def __init__(self, model, **kwargs) -> None:
            calls.append(("text_mas", kwargs))

    class LatentMAS:
        def __init__(self, model, **kwargs) -> None:
            calls.append(("latent_mas", kwargs))

    monkeypatch.setattr(latentmas_eval, "BaselineMethod", Baseline)
    monkeypatch.setattr(latentmas_eval, "TextMASMethod", TextMAS)
    monkeypatch.setattr(latentmas_eval, "LatentMASMethod", LatentMAS)

    args = argparse.Namespace(
        method="latent_mas",
        max_new_tokens=64,
        temperature=0.01,
        top_p=1.0,
        generate_bs=2,
        use_vllm=False,
        latent_steps=12,
        task="gsm8k",
        prompt="sequential",
        device="cpu",
        device2="cpu",
        model_name="Qwen/Qwen3-4B",
    )

    latentmas_eval.make_latentmas_method(method_name="baseline", model=object(), args=args)
    latentmas_eval.make_latentmas_method(method_name="text_mas", model=object(), args=args)
    latentmas_eval.make_latentmas_method(method_name="latent_mas", model=object(), args=args)

    assert calls[0] == (
        "baseline",
        {
            "max_new_tokens": 64,
            "temperature": 0.01,
            "top_p": 1.0,
            "generate_bs": 2,
            "use_vllm": False,
            "args": args,
        },
    )
    assert calls[1] == (
        "text_mas",
        {
            "max_new_tokens_each": 64,
            "temperature": 0.01,
            "top_p": 1.0,
            "generate_bs": 2,
            "args": args,
        },
    )
    assert calls[2] == (
        "latent_mas",
        {
            "latent_steps": 12,
            "judger_max_new_tokens": 64,
            "temperature": 0.01,
            "top_p": 1.0,
            "generate_bs": 2,
            "args": args,
        },
    )


def test_runtime_args_map_svamp_to_native_math_prompt_without_losing_public_task(
    latentmas_eval_module,
) -> None:
    latentmas_eval = latentmas_eval_module
    args = argparse.Namespace(
        latentmas_root=Path("references/repos/LatentMAS"),
        model_name="mock-model",
        method="baseline",
        task="svamp",
        latentmas_task=None,
        prompt="sequential",
    )

    runtime_args = latentmas_eval._runtime_args_for_latentmas(args)

    assert args.task == "svamp"
    assert runtime_args.eval_task == "svamp"
    assert runtime_args.task == "gsm8k"


def test_run_eval_with_fake_method_writes_jsonl_and_meta(
    tmp_path: Path, latentmas_eval_module
) -> None:
    latentmas_eval = latentmas_eval_module
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"question": "What is 1 + 1?", "answer": "2", "metadata": {"id": "toy-1"}}) + "\n",
        encoding="utf-8",
    )
    prediction_output = tmp_path / "predictions.jsonl"
    fake = FakeMethod(
        [
            {
                "question": "What is 1 + 1?",
                "gold": "2",
                "solution": "2",
                "prediction": "2",
                "raw_prediction": "The answer is 2.",
                "correct": True,
                "agents": [
                    {
                        "name": "Judger",
                        "role": "judger",
                        "input": "What is 1 + 1?",
                        "input_ids": [1, 2, 3],
                        "input_tokens": ["What", " is", " 1"],
                        "output": "The answer is 2.",
                    }
                ],
            }
        ]
    )
    args = argparse.Namespace(
        latentmas_root=Path("references/repos/LatentMAS"),
        model_name="mock-model",
        method="baseline",
        task="svamp",
        prompt="sequential",
        eval_file=eval_file,
        limit=1,
        max_new_tokens=32,
        temperature=0.01,
        top_p=1.0,
        generate_bs=4,
        use_vllm=False,
        latent_steps=0,
        seed=7,
        device="cpu",
        device2="cpu",
        prediction_output=prediction_output,
    )

    summary = latentmas_eval.run_eval(
        args,
        model_factory=lambda parsed_args: object(),
        method_factory=lambda method_name, model, parsed_args: fake,
        clock=lambda: 10.0,
    )

    records = [json.loads(line) for line in prediction_output.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(prediction_output.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))

    assert fake.seen_batches[0][0]["question"] == "What is 1 + 1?"
    assert records[0]["id"] == "toy-1"
    assert records[0]["prediction"] == "2"
    assert records[0]["correct"] is True
    assert records[0]["trace"]["agent_count"] == 1
    assert records[0]["trace"]["input_token_count"] == 3
    assert meta == summary
    assert summary["accuracy"] == 1.0
    assert summary["method"] == "baseline"
    assert summary["task"] == "svamp"
    assert summary["num_examples"] == 1
