from scripts import analyze_source_private_arc_llama8b_failure_probe as probe


def test_candidate_rules_mark_source_index_rules_as_diagnostic() -> None:
    rows = [
        {
            "feature": {
                "source_selected_index": 0,
                "packet_margin": 0.1,
                "packet_code_l2": 0.9,
                "packet_scale": 0.2,
            }
        },
        {
            "feature": {
                "source_selected_index": 1,
                "packet_margin": 0.2,
                "packet_code_l2": 1.1,
                "packet_scale": 0.3,
            }
        },
    ]

    rules = {probe._rule_name(rule): rule for rule in probe._candidate_rules(rows)}

    assert rules["source_matches_llama_prediction"]["deployable_without_source_index"] is False
    assert rules["source_index_eq:0"]["deployable_without_source_index"] is False
    assert rules["packet_margin_ge:0.1"]["deployable_without_source_index"] is True
    assert rules["always_qwen"]["deployable_without_source_index"] is True


def test_evaluate_router_uses_expected_branch() -> None:
    rows = [
        {
            "feature": {"source_selected_index": 0, "packet_margin": 0.2},
            "llama_prediction_index": 0,
            "qwen_prediction_index": 1,
            "llama_correct": True,
            "qwen_correct": False,
        },
        {
            "feature": {"source_selected_index": 1, "packet_margin": 0.05},
            "llama_prediction_index": 0,
            "qwen_prediction_index": 1,
            "llama_correct": False,
            "qwen_correct": True,
        },
    ]

    diagnostic = probe._evaluate_router(
        {"kind": "source_matches_llama_prediction", "deployable_without_source_index": False},
        rows,
    )
    deployable = probe._evaluate_router(
        {"kind": "packet_margin_ge", "threshold": 0.1, "deployable_without_source_index": True},
        rows,
    )

    assert diagnostic["accuracy"] == 1.0
    assert diagnostic["deployable_without_source_index"] is False
    assert deployable["accuracy"] == 1.0
    assert deployable["deployable_without_source_index"] is True


def test_summarize_split_reports_packet_loss_and_oracle_headroom() -> None:
    rows = [
        {
            "row_id": "Mercury_1",
            "prefix": "Mercury",
            "source_correct": True,
            "llama_correct": False,
            "qwen_correct": True,
            "cached_tiny_correct": False,
            "same_byte_text_correct": True,
            "target_correct": False,
        },
        {
            "row_id": "MCAS_1",
            "prefix": "MCAS",
            "source_correct": False,
            "llama_correct": True,
            "qwen_correct": False,
            "cached_tiny_correct": False,
            "same_byte_text_correct": False,
            "target_correct": False,
        },
    ]

    summary = probe._summarize_split(rows)

    assert summary["source_accuracy"] == 0.5
    assert summary["llama_packet_accuracy"] == 0.5
    assert summary["source_to_llama_packet_loss"] == 0.0
    assert summary["llama_qwen_oracle_accuracy"] == 1.0
    assert summary["same_byte_text_minus_llama"] == 0.0
