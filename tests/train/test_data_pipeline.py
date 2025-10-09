from typing import List, Dict

import sys
import types

import pytest

dummy_datasets = types.ModuleType("datasets")
dummy_datasets.load_dataset = lambda *args, **kwargs: None
dummy_datasets.Dataset = object
sys.modules.setdefault("datasets", dummy_datasets)

from latentwire.data_pipeline import prepare_training_data


@pytest.mark.parametrize("dataset", ["squad", "hotpot"])
def test_prepare_training_data(monkeypatch, dataset):
    fake_examples: List[Dict[str, str]] = [
        {"source": "Question: A?\nContext: Foo", "answer": "Bar"},
        {"source": "Question: B?\nContext: Baz", "answer": "Qux"},
    ]

    def fake_load_examples(**kwargs):
        return fake_examples

    monkeypatch.setattr("latentwire.data_pipeline.load_examples", fake_load_examples)
    texts, answers = prepare_training_data(dataset=dataset, samples=2, data_seed=0, hotpot_config="fullwiki")
    assert texts == [ex["source"] for ex in fake_examples]
    assert answers == [ex["answer"] for ex in fake_examples]
