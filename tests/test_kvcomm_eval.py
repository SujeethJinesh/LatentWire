from types import SimpleNamespace

from latent_bridge.evaluate import answer_labels_from_prompt, make_first_token_prefix_allowed_fn
from latent_bridge.kvcomm_eval import _build_prompts, _parse_grid, _resolve_selected_layers


class _TinyTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return {
            "A": [1],
            " B": [2],
            "C": [3],
        }.get(text, [10, 11])


def test_answer_labels_from_prompt_reads_choice_lines():
    prompt = "Question?\nChoices:\nA. alpha\nB. beta\nC. gamma\nAnswer:"
    assert answer_labels_from_prompt(prompt) == ["A", "B", "C"]


def test_first_token_prefix_allows_only_answer_letters_then_eos():
    allowed = make_first_token_prefix_allowed_fn(
        tokenizer=_TinyTokenizer(),
        labels=["A", "B", "C"],
        prompt_length=3,
    )
    assert allowed(0, SimpleNamespace(shape=[1, 3])) == [1, 2, 3]
    assert allowed(0, SimpleNamespace(shape=[1, 4])) == [99]


def test_build_prompts_prefers_source_question_for_source_side():
    ex = SimpleNamespace(
        prompt="Solve this.\nAnswer:",
        source_question="What is 2 + 2?",
    )
    source_prompt, target_prompt = _build_prompts(ex, "brief_analysis")
    assert "What is 2 + 2?" in source_prompt
    assert target_prompt == "Solve this.\nAnswer:"


def test_resolve_selected_layers_keeps_at_least_one_layer():
    ranking = [5, 3, 1, 0]
    assert _resolve_selected_layers(ranking, 0.1) == [5]
    assert _resolve_selected_layers(ranking, 0.5) == [5, 3]


def test_parse_grid_reads_comma_separated_floats():
    assert _parse_grid("0.25, 0.5,1.0") == [0.25, 0.5, 1.0]
