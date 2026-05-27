from experimental.outlier_migrate.phase9 import check_om_phase9_mpred as checker


def summary(median: float, low: float, high: float) -> dict:
    return {"median_recovery": median, "bootstrap_ci95": {"ci95_low": low, "ci95_high": high}}


def base_summaries() -> dict:
    return {
        "m11b_top5": summary(0.40, 0.20, 0.60),
        "m11b_top10": summary(0.35, 0.10, 0.55),
        "static_top10": summary(0.10, -0.10, 0.30),
        "random_walk_top5": summary(0.05, -0.10, 0.20),
        "mpred_random_alpha_top5": summary(0.10, -0.05, 0.20),
        "mpred_top5_alpha_0_5": summary(0.20, 0.00, 0.40),
        "mpred_top5_alpha_0_8": summary(0.30, 0.10, 0.50),
        "mpred_top5_alpha_0_95": summary(0.70, 0.62, 0.80),
        "mpred_top5_alpha_0_99": summary(0.25, 0.05, 0.45),
        "mpred_top10_alpha_0_95": summary(0.45, 0.20, 0.65),
    }


def test_mpred_passes_with_non_overlapping_gain() -> None:
    decision, _reasons, details = checker.decision_from_summaries(base_summaries(), "nemotron")

    assert decision == checker.PASS
    assert details["best_mpred_regime"] == "mpred_top5_alpha_0_95"


def test_mpred_granite_tightening_passes_without_median_gain() -> None:
    summaries = base_summaries()
    summaries["m11b_top5"] = summary(0.45, -1.30, 1.00)
    summaries["mpred_top5_alpha_0_95"] = summary(0.30, 0.10, 0.50)

    decision, _reasons, _details = checker.decision_from_summaries(summaries, "granite")

    assert decision == checker.PASS_TIGHTENS_GRANITE


def test_mpred_kills_when_random_alpha_matches_best_arm() -> None:
    summaries = base_summaries()
    for regime in checker.MPRED_REGIMES:
        summaries[regime] = summary(0.25, 0.00, 0.45)
    summaries["mpred_random_alpha_top5"] = summary(0.30, 0.00, 0.50)

    decision, _reasons, _details = checker.decision_from_summaries(summaries, "nemotron")

    assert decision == checker.KILL


def test_mpred_decision_accepts_partial_high_value_subset() -> None:
    summaries = {
        "m11b_top5": summary(0.40, 0.20, 0.60),
        "m11b_top10": summary(0.35, 0.10, 0.55),
        "static_top10": summary(0.10, -0.10, 0.30),
        "mpred_top10_alpha_0_95": summary(0.70, 0.62, 0.80),
    }

    decision, _reasons, details = checker.decision_from_summaries(summaries, "nemotron")

    assert decision == checker.PASS
    assert details["best_mpred_regime"] == "mpred_top10_alpha_0_95"
