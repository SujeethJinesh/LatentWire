from outlier_migrate.metrics import bootstrap_median_ci, perplexity, recovery, summarize_recovery


def test_recovery_returns_zero_without_static_gap() -> None:
    assert recovery(1.0, 1.0, 0.9) == 0.0


def test_recovery_computes_fraction_recovered() -> None:
    assert recovery(1.0, 3.0, 2.0) == 0.5


def test_bootstrap_median_ci_is_deterministic() -> None:
    left = bootstrap_median_ci([0.0, 1.0, 2.0], samples=20, seed=7)
    right = bootstrap_median_ci([0.0, 1.0, 2.0], samples=20, seed=7)
    assert left == right


def test_summarize_recovery_reports_count_and_median() -> None:
    summary = summarize_recovery([0.0, 0.5, 1.0], seed=3)
    assert summary.trace_count == 3
    assert summary.median_recovery == 0.5
    assert perplexity(0.0) == 1.0
