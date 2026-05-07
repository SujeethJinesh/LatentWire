from __future__ import annotations

from experimental.thoughtflow_fp8.phase2.rdu_independent_trace_reproduction_check import build_report
from experimental.thoughtflow_fp8.phase2.tests.test_rdu_no_retune_reproduction_check import _result


def test_build_report_marks_independent_trace_reproduction() -> None:
    cached = _result()
    measured = _result(offset=-0.02)
    measured["max_traces"] = 96
    measured["input_paths"] = ["results/independent/source_alone.jsonl"]

    report = build_report(
        cached,
        measured,
        measured_label="measured_independent",
        trace_input_paths=("results/independent/source_alone.jsonl",),
    )

    assert report["reproduction_pass"] is True
    assert report["diagnostic_type"] == "measured_no_retuning_independent_trace_slice_against_cached_frozen_gate"
    assert report["measured_label"] == "measured_independent"
    assert report["trace_input_paths"] == ["results/independent/source_alone.jsonl"]
    assert report["measured_decision"]["best_compressed_policy"] == "rdu_topk"
    assert report["strict_family_pass"]["same_family_positive"] is True
    assert report["strict_family_pass"]["cross_family_positive"] is True
    assert report["measured_oracle_headroom"]["rdu_gap_to_per_trace_oracle"] == 0.0
    assert report["measured_failure_decomposition"]["group_summaries"]["all"]["n"] > 0


def test_build_report_fails_when_same_family_separation_fails() -> None:
    cached = _result()
    measured = _result(offset=-0.02)
    measured["summary"]["tf_sparse_r0.55_p0.05_m0.12_a2"]["nll"] = (
        measured["summary"]["rdu_topk"]["nll"] - 0.001
    )

    report = build_report(
        cached,
        measured,
        measured_label="measured_independent",
        trace_input_paths=("results/independent/source_alone.jsonl",),
    )

    assert report["reproduction_pass"] is False
    assert report["strict_family_pass"]["same_family_positive"] is False
