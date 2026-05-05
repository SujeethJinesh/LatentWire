from experimental.sinkaware.phase2.qk_sink_cost_model import _run


def test_qk_cost_model_has_low_rank_rows() -> None:
    result = _run(model_name="sshleifer/tiny-gpt2", sink_tokens=4)

    assert {row["rank"] for row in result["rows"]} == {1, 2, 4, 8}
    assert all(row["cost_ratio_vs_exact"] > 0 for row in result["rows"])
