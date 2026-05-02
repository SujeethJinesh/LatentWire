from __future__ import annotations

from scripts import run_source_private_testlog_packet_strict_small as testlog_gate


def test_testlog_budget_sweep_has_exact_id_parity() -> None:
    examples = testlog_gate.make_benchmark(examples=48, candidates=4, seed=5)
    rows, summary = testlog_gate.run_budget(examples=examples, seed=5, budget_bytes=2)

    assert len(rows) == 48
    assert summary["exact_id_parity"] is True
    assert summary["exact_id_count"] == 48
    assert summary["candidate_pool_recall"] == 1.0


def test_matched_testlog_packet_beats_source_destroying_controls() -> None:
    examples = testlog_gate.make_benchmark(examples=64, candidates=4, seed=7)
    _, summary = testlog_gate.run_budget(examples=examples, seed=7, budget_bytes=2)

    assert summary["pass_gate"] is True
    assert summary["metrics"]["matched_testlog_packet"]["accuracy"] == 1.0
    assert 0.0 < summary["best_no_source_accuracy"] < 1.0
    assert summary["best_no_source_accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert summary["best_source_destroying_control_accuracy"] <= summary["best_no_source_accuracy"] + 0.02
    assert summary["metrics"]["structured_text_matched"]["accuracy"] <= summary["best_no_source_accuracy"] + 0.02


def test_full_log_oracle_works_but_matched_byte_log_text_does_not() -> None:
    examples = testlog_gate.make_benchmark(examples=32, candidates=4, seed=13)
    _, summary = testlog_gate.run_budget(examples=examples, seed=13, budget_bytes=32)

    assert summary["metrics"]["full_structured_log"]["accuracy"] == 1.0
    assert summary["metrics"]["full_signature_text"]["accuracy"] == 1.0
    assert summary["metrics"]["structured_text_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert (
        summary["metrics"]["full_structured_log"]["mean_payload_bytes"]
        > summary["metrics"]["structured_text_matched"]["mean_payload_bytes"]
    )


def test_sweep_summary_promotes_lowest_passing_budget() -> None:
    examples = testlog_gate.make_benchmark(examples=24, candidates=4, seed=17)
    summaries = [
        testlog_gate.run_budget(examples=examples, seed=17, budget_bytes=budget)[1]
        for budget in [2, 4, 8]
    ]
    sweep = testlog_gate.summarize_sweep(summaries)

    assert sweep["strict_small_pass"] is True
    assert sweep["passing_budgets"] == [2, 4, 8]
    assert sweep["best_budget_bytes"] == 2


def test_leakage_audit_separates_public_prompt_from_private_log() -> None:
    examples = testlog_gate.make_benchmark(examples=12, candidates=4, seed=19)
    rows, _ = testlog_gate.run_budget(examples=examples, seed=19, budget_bytes=2)
    audit = testlog_gate._leakage_audit(examples, {2: rows})

    assert audit["public_target_answer_label_candidate_pool_hits"] == len(examples)
    assert audit["public_target_private_log_hits"] == 0
    assert audit["target_prompt_trace_sig_hits"] == 0
    assert audit["over_budget_counts"]["2"]["matched_testlog_packet"] == 0
    assert audit["packet_copy_counts"]["2"]["matched_testlog_packet"]["payload_contains_answer_label"] == 0


def test_shuffled_source_uses_nonself_packet_source() -> None:
    examples = testlog_gate.make_benchmark(examples=16, candidates=4, seed=23)
    rows, _ = testlog_gate.run_budget(examples=examples, seed=23, budget_bytes=2)

    for row in rows:
        shuffled = row["conditions"]["shuffled_source"]
        assert shuffled["source_example_id"] != row["example_id"]
