from __future__ import annotations

import json

from scripts.build_source_private_conditional_pq_iclr_colm_v2_status import build_status


def test_build_status_promotes_scoped_colm_and_blocks_iclr(tmp_path) -> None:
    conditional_summary = tmp_path / "conditional.json"
    schema_grid = tmp_path / "schema.json"
    systems_waterfall = tmp_path / "systems.json"
    conditional_summary.write_text(
        json.dumps(
            {
                "summary": {
                    "decisive_disjoint_n500_rows": 4,
                    "decisive_disjoint_n500_pass_rows": 4,
                    "less_diagnostic_decisive_rows": 2,
                    "less_diagnostic_decisive_pass_rows": 2,
                    "budget2_decisive_rows": 1,
                    "budget2_decisive_pass_rows": 1,
                    "min_decisive_source_accuracy": 0.996,
                    "max_decisive_best_control_accuracy": 0.302,
                    "min_decisive_ci95_low_vs_best_control": 0.658,
                    "cross_family_pass_rows": 0,
                    "cross_family_rows": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    schema_grid.write_text(
        json.dumps(
            {
                "summary": {
                    "pass_rows": 0,
                    "rows": 28,
                    "max_source_minus_best_control": 0.007812,
                    "max_ci95_low_vs_best_control": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    systems_waterfall.write_text(
        json.dumps(
            {
                "pass_gate": True,
                "checks": {
                    "private_state_exposure_separated": True,
                    "receiver_exact": True,
                },
                "rows": [
                    {
                        "row_type": "method",
                        "record_bytes": 5,
                        "payload_bytes": 2,
                        "source_kv_exposed": False,
                    },
                    {
                        "row_type": "method",
                        "record_bytes": 7,
                        "payload_bytes": 4,
                        "source_kv_exposed": False,
                    },
                    {
                        "row_type": "baseline",
                        "record_bytes": 21504,
                        "payload_bytes": None,
                        "source_kv_exposed": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    status = build_status(
        conditional_summary_path=conditional_summary,
        schema_grid_path=schema_grid,
        systems_waterfall_path=systems_waterfall,
    )

    assert status["readiness"]["colm_v2"] == "scoped_positive_method_ready_for_writeup"
    assert status["readiness"]["iclr"] == "blocked_by_cross_family_or_broader_benchmark_positive_gate"
    assert status["readiness"]["cross_family_blocked"] is True
    assert status["evidence"]["systems"]["method_record_bytes_range"] == [5, 7]
    assert status["evidence"]["systems"]["min_kv_floor_record_bytes"] == 21504
    assert status["next_exact_gate"]["name"] == "public_conditioned_conditional_pq_resurrection_gate"
