from experimental.sinkaware.phase2.sink_predictability_probe import _run_case


def test_static_prior_fails_random_queries() -> None:
    row = _run_case("random", seed=0, n=256, dim=32)

    assert row["static_r2"] < 0.05


def test_low_rank_query_model_recovers_low_rank_case() -> None:
    row = _run_case("low_rank", seed=0, n=256, dim=32)

    assert row["rank4_query_r2"] > 0.95
