from __future__ import annotations

from scripts import run_source_private_evidence_packet_strict_small as strict_gate


def test_strict_small_budget_sweep_has_exact_id_parity() -> None:
    examples = strict_gate.make_benchmark(examples=48, candidates=4, seed=5)
    rows, summary = strict_gate.run_budget(examples=examples, seed=5, budget_bytes=4)

    assert len(rows) == 48
    assert summary["exact_id_parity"] is True
    assert summary["exact_id_count"] == 48
    assert summary["candidate_pool_recall"] == 1.0


def test_matched_syndrome_beats_all_source_destroying_controls() -> None:
    examples = strict_gate.make_benchmark(examples=64, candidates=4, seed=7)
    _, summary = strict_gate.run_budget(examples=examples, seed=7, budget_bytes=2)

    assert summary["pass_gate"] is True
    assert summary["metrics"]["matched_syndrome"]["accuracy"] == 1.0
    assert 0.0 < summary["best_no_source_accuracy"] < 1.0
    assert summary["best_no_source_accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert summary["best_source_destroying_control_accuracy"] <= summary["best_no_source_accuracy"] + 0.02
    assert summary["metrics"]["structured_text_matched"]["accuracy"] <= summary["best_no_source_accuracy"] + 0.02


def test_full_text_oracles_work_but_matched_byte_text_does_not() -> None:
    examples = strict_gate.make_benchmark(examples=32, candidates=4, seed=13)
    _, summary = strict_gate.run_budget(examples=examples, seed=13, budget_bytes=32)

    assert summary["metrics"]["full_structured_text"]["accuracy"] == 1.0
    assert summary["metrics"]["full_evidence_oracle"]["accuracy"] == 1.0
    assert summary["metrics"]["structured_text_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]
    assert (
        summary["metrics"]["full_structured_text"]["mean_payload_bytes"]
        > summary["metrics"]["structured_text_matched"]["mean_payload_bytes"]
    )


def test_sweep_summary_promotes_lowest_passing_budget_when_all_pass() -> None:
    examples = strict_gate.make_benchmark(examples=24, candidates=4, seed=17)
    summaries = [
        strict_gate.run_budget(examples=examples, seed=17, budget_bytes=budget)[1]
        for budget in [2, 4, 8]
    ]
    sweep = strict_gate.summarize_sweep(summaries)

    assert sweep["strict_small_pass"] is True
    assert sweep["passing_budgets"] == [2, 4, 8]
    assert sweep["best_budget_bytes"] == 2
