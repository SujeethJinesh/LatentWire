from types import SimpleNamespace

from latent_bridge.kvcomm_eval import _build_prompts, _parse_grid, _resolve_selected_layers


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
