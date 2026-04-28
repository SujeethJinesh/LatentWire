from __future__ import annotations

from scripts import run_source_private_hidden_repair_packet_smoke as repair_gate


def test_hidden_repair_benchmark_executes_real_buggy_cases() -> None:
    examples = repair_gate.make_benchmark(examples=16, candidates=4, seed=28)

    assert len(examples) == 16
    assert any("IndexError" in example.actual_repr for example in examples)
    assert all("REPAIR_DIAG=" in example.private_test_log for example in examples)
    assert all("REPAIR_DIAG=" not in example.target_prompt for example in examples)


def test_holdout_family_set_is_disjoint_and_executable() -> None:
    core = repair_gate.make_benchmark(examples=8, candidates=4, seed=28, family_set="core")
    holdout = repair_gate.make_benchmark(examples=8, candidates=4, seed=28, family_set="holdout")

    assert {example.family_name for example in core}.isdisjoint({example.family_name for example in holdout})
    assert len({example.family_name for example in holdout}) == 8
    assert any("ValueError" in example.actual_repr for example in holdout)
    assert all("REPAIR_DIAG=" in example.private_test_log for example in holdout)


def test_matched_repair_packet_beats_controls() -> None:
    examples = repair_gate.make_benchmark(examples=32, candidates=4, seed=7)
    _, summary = repair_gate.run_budget(examples=examples, seed=7, budget_bytes=2)

    assert summary["pass_gate"] is True
    assert summary["metrics"]["matched_repair_packet"]["accuracy"] == 1.0
    assert summary["best_source_destroying_control_accuracy"] <= summary["best_no_source_accuracy"] + 0.02
    assert summary["metrics"]["structured_text_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]


def test_full_hidden_log_oracle_works_but_truncated_text_does_not() -> None:
    examples = repair_gate.make_benchmark(examples=24, candidates=4, seed=11)
    _, summary = repair_gate.run_budget(examples=examples, seed=11, budget_bytes=8)

    assert summary["metrics"]["full_hidden_log"]["accuracy"] == 1.0
    assert summary["metrics"]["full_diag_text"]["accuracy"] == 1.0
    assert summary["metrics"]["structured_text_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]


def test_reviewer_risk_rows_separate_packet_from_matched_byte_text() -> None:
    examples = repair_gate.make_benchmark(examples=24, candidates=4, seed=11)
    _, summary = repair_gate.run_budget(examples=examples, seed=11, budget_bytes=2)

    assert summary["pass_gate"] is True
    assert summary["candidate_pool_recall"] == 1.0
    assert summary["matched_selector_accuracy"] == 1.0
    assert summary["metrics"]["structured_json_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert summary["metrics"]["structured_free_text_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert summary["metrics"]["helper_only_no_log"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert summary["metrics"]["diag_masked_full_log"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert summary["metrics"]["expected_actual_masked_full_log"]["accuracy"] == 1.0
    assert summary["metrics"]["test_name_masked_full_log"]["accuracy"] == 1.0


def test_structured_relays_are_oracles_only_when_budget_reveals_diag() -> None:
    examples = repair_gate.make_benchmark(examples=24, candidates=4, seed=11)
    _, small = repair_gate.run_budget(examples=examples, seed=11, budget_bytes=2)
    _, large = repair_gate.run_budget(examples=examples, seed=11, budget_bytes=32)

    assert small["metrics"]["structured_json_matched"]["accuracy"] == small["metrics"]["target_only"]["accuracy"]
    assert small["metrics"]["structured_free_text_matched"]["accuracy"] == small["metrics"]["target_only"]["accuracy"]
    assert large["metrics"]["structured_json_matched"]["accuracy"] == 1.0
    assert large["metrics"]["structured_free_text_matched"]["accuracy"] == 1.0
    assert large["pass_gate"] is False


def test_shuffled_source_uses_nonself_source_id() -> None:
    examples = repair_gate.make_benchmark(examples=16, candidates=4, seed=13)
    rows, _ = repair_gate.run_budget(examples=examples, seed=13, budget_bytes=2)

    for row in rows:
        assert row["conditions"]["shuffled_source"]["source_example_id"] != row["example_id"]
