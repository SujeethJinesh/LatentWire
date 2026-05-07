from __future__ import annotations

from experimental.thoughtflow_fp8.phase2.rdu_alt_surface_reproduction_check import build_report
from experimental.thoughtflow_fp8.phase2.tests.test_rdu_no_retune_reproduction_check import _result


def test_build_report_marks_alternate_surface_reproduction() -> None:
    cached = _result()
    measured = _result(offset=-0.02)
    measured["max_length"] = 16
    measured["continuation_tokens"] = 6

    report = build_report(cached, measured, measured_label="measured_alt_surface")

    assert report["reproduction_pass"] is True
    assert report["diagnostic_type"] == "measured_no_retuning_alternate_surface_against_cached_frozen_gate"
    assert report["cached_label"] == "cached_promoted_gate"
    assert report["measured_label"] == "measured_alt_surface"
    assert report["surface_changes"]["max_length"] == {"cached": 12, "measured": 16}
    assert report["surface_changes"]["continuation_tokens"] == {"cached": 4, "measured": 6}
    assert report["measured_decision"]["best_compressed_policy"] == "rdu_topk"
    assert report["strict_family_pass"]["same_family_positive"] is True
    assert report["strict_family_pass"]["cross_family_positive"] is True
    assert report["measured_family_separation"]["cross_family_margin_nll_vs_rdu"]["thin_kv_like"] > 0.03
    assert report["measured_oracle_headroom"]["rdu_gap_to_per_trace_oracle"] == 0.0


def test_build_report_fails_when_same_family_separation_fails() -> None:
    cached = _result()
    measured = _result(offset=-0.02)
    measured["max_length"] = 16
    measured["continuation_tokens"] = 6
    measured["summary"]["tf_sparse_r0.55_p0.05_m0.12_a2"]["nll"] = (
        measured["summary"]["rdu_topk"]["nll"] - 0.001
    )

    report = build_report(cached, measured, measured_label="measured_alt_surface")

    assert report["reproduction_pass"] is False
    assert report["strict_family_pass"]["same_family_positive"] is False
