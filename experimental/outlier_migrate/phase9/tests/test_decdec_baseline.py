from experimental.outlier_migrate.phase9 import check_om_phase9_decdec_baseline as checker


def test_decdec_baseline_checker_reports_artifact_complete_as_pass() -> None:
    rows = [
        {
            "no_recoverable_static_gap": False,
            "recoveries": {
                "decdec_reactive_top1_proxy": 0.4,
                "static_top10": 0.2,
                "random_reactive_top1": 0.1,
            },
        },
        {
            "no_recoverable_static_gap": False,
            "recoveries": {
                "decdec_reactive_top1_proxy": 0.2,
                "static_top10": 0.1,
                "random_reactive_top1": 0.0,
            },
        },
    ]

    summary = checker.summarize_recovery(rows, "decdec_reactive_top1_proxy")

    assert summary["median_recovery"] == 0.30000000000000004
    assert summary["included_trace_count"] == 2


def test_decdec_baseline_summary_excludes_no_gap_rows() -> None:
    rows = [
        {
            "no_recoverable_static_gap": False,
            "recoveries": {
                "decdec_reactive_top1_proxy": 0.6,
                "static_top10": 0.2,
                "random_reactive_top1": 0.1,
            },
        },
        {
            "no_recoverable_static_gap": True,
            "recoveries": {
                "decdec_reactive_top1_proxy": None,
                "static_top10": None,
                "random_reactive_top1": None,
            },
        },
    ]

    summary = checker.summarize_recovery(rows, "decdec_reactive_top1_proxy")

    assert summary["median_recovery"] == 0.6
    assert summary["included_trace_count"] == 1
    assert summary["no_recoverable_static_gap_count"] == 1
