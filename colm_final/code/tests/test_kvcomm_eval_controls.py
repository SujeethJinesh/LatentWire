from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from latent_bridge import kvcomm_eval
from latent_bridge.evaluate import _generation_example_id


def _example(prompt: str, source_question: str | None = None, example_id: str | None = None):
    return SimpleNamespace(
        prompt=prompt,
        source_question=source_question or "",
        example_id=example_id,
        answers=["1"],
    )


def test_controlled_source_prompt_matched_uses_source_question() -> None:
    examples = [_example("target prompt", "source prompt", "ex0")]

    prompt, source_id, source_index = kvcomm_eval._controlled_source_prompt(
        examples,
        0,
        source_reasoning_mode="plain",
        source_control="matched",
        shuffle_offset=1,
    )

    assert "source prompt" in prompt
    assert source_id == _generation_example_id(examples[0])
    assert source_index == 0


def test_controlled_source_prompt_zero_source_uses_matched_shape_prompt() -> None:
    examples = [_example("target prompt", "source prompt", "ex0")]

    prompt, source_id, source_index = kvcomm_eval._controlled_source_prompt(
        examples,
        0,
        source_reasoning_mode="plain",
        source_control="zero_source",
        shuffle_offset=1,
    )

    assert "source prompt" in prompt
    assert source_id == _generation_example_id(examples[0])
    assert source_index == 0


def test_controlled_source_prompt_target_only_uses_target_prompt() -> None:
    examples = [_example("target prompt", "source prompt", "ex0")]

    prompt, source_id, source_index = kvcomm_eval._controlled_source_prompt(
        examples,
        0,
        source_reasoning_mode="brief_analysis",
        source_control="target_only",
        shuffle_offset=1,
    )

    assert prompt == "target prompt"
    assert source_id == _generation_example_id(examples[0])
    assert source_index == 0


def test_controlled_source_prompt_shuffled_source_uses_different_example_when_possible() -> None:
    examples = [
        _example("target 0", "source 0", "ex0"),
        _example("target 1", "source 1", "ex1"),
        _example("target 2", "source 2", "ex2"),
    ]

    prompt, source_id, source_index = kvcomm_eval._controlled_source_prompt(
        examples,
        0,
        source_reasoning_mode="plain",
        source_control="shuffled_source",
        shuffle_offset=3,
    )

    assert source_index in {1, 2}
    assert source_id == _generation_example_id(examples[source_index])
    assert source_id != _generation_example_id(examples[0])
    assert f"source {source_index}" in prompt


def test_controlled_source_prompt_shuffled_source_requires_multiple_examples() -> None:
    with pytest.raises(ValueError, match="shuffled_source requires at least two examples"):
        kvcomm_eval._controlled_source_prompt(
            [_example("target", "source", "ex0")],
            0,
            source_reasoning_mode="plain",
            source_control="shuffled_source",
            shuffle_offset=1,
        )


def test_controlled_source_prompt_rejects_unknown_control() -> None:
    with pytest.raises(ValueError, match="Unsupported KVComm source control"):
        kvcomm_eval._controlled_source_prompt(
            [_example("target", "source", "ex0")],
            0,
            source_reasoning_mode="plain",
            source_control="bad",
            shuffle_offset=1,
        )


def test_resolve_selected_layers_keeps_at_least_one_layer() -> None:
    assert kvcomm_eval._resolve_selected_layers([3, 2, 1], 0.01) == [3]


def test_parse_source_control_modes_expands_all_and_dedupes() -> None:
    assert kvcomm_eval._parse_source_control_modes("matched, zero_source") == [
        "matched",
        "zero_source",
    ]
    assert kvcomm_eval._parse_source_control_modes("all,matched") == list(kvcomm_eval.SOURCE_CONTROLS)


def test_zero_past_key_values_preserves_shape_dtype_and_input() -> None:
    key = torch.ones(1, 2, dtype=torch.float32)
    value = torch.full((1, 2), 2.0, dtype=torch.float32)

    zeroed = kvcomm_eval._zero_past_key_values(((key, value),))

    assert torch.equal(key, torch.ones_like(key))
    assert torch.equal(value, torch.full_like(value, 2.0))
    assert zeroed[0][0].shape == key.shape
    assert zeroed[0][0].dtype == key.dtype
    assert torch.equal(zeroed[0][0], torch.zeros_like(key))
    assert torch.equal(zeroed[0][1], torch.zeros_like(value))


def test_answers_overlap_normalizes_string_answers() -> None:
    assert kvcomm_eval._answers_overlap([" 5.0 ", "#### 5"], ["5.0"])
    assert not kvcomm_eval._answers_overlap(["5"], ["6"])


def test_past_key_values_nbytes_counts_selected_layers() -> None:
    layer0 = (torch.zeros(1, 2, dtype=torch.float32), torch.zeros(1, 2, dtype=torch.float32))
    layer1 = (torch.zeros(1, 3, dtype=torch.float32), torch.zeros(1, 3, dtype=torch.float32))
    past_key_values = (layer0, layer1)

    assert kvcomm_eval._past_key_values_nbytes(past_key_values) == 40
    assert kvcomm_eval._past_key_values_nbytes(past_key_values, [1]) == 24
    assert kvcomm_eval._past_key_values_nbytes(past_key_values, [99]) == 0


def test_past_key_values_nbytes_supports_cache_objects() -> None:
    layer0 = (torch.zeros(1, 2, dtype=torch.float32), torch.zeros(1, 2, dtype=torch.float32))
    layer1 = (torch.zeros(1, 3, dtype=torch.float32), torch.zeros(1, 3, dtype=torch.float32))

    class Cache:
        def to_legacy_cache(self):
            return (layer0, layer1)

    assert kvcomm_eval._past_key_values_nbytes(Cache(), [0]) == 16
