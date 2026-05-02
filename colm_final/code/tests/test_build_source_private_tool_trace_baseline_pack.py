from __future__ import annotations

from scripts.build_source_private_tool_trace_baseline_pack import _best_budget_summary, aggregate_baseline_pack


def test_best_budget_summary_selects_declared_budget() -> None:
    sweep = {
        "best_budget_bytes": 4,
        "budget_summaries": [
            {"budget_bytes": 2, "value": "small"},
            {"budget_bytes": 4, "value": "best"},
        ],
    }

    assert _best_budget_summary(sweep)["value"] == "best"


def test_aggregate_baseline_pack_keeps_claim_boundary() -> None:
    condition_metrics = {
        name: {
            "accuracy": 0.25,
            "correct": 1,
            "mean_payload_bytes": 0.0,
            "max_payload_bytes": 0,
            "mean_payload_tokens": 0.0,
            "p50_latency_ms": 0.0,
        }
        for name in [
            "target_only",
            "target_wrapper",
            "zero_source",
            "shuffled_source",
            "random_same_byte",
            "answer_only",
            "answer_masked",
            "target_derived_sidecar",
            "structured_text_matched",
        ]
    }
    condition_metrics["matched_repair_packet"] = condition_metrics["full_hidden_log"] = condition_metrics["full_diag_text"] = {
        "accuracy": 1.0,
        "correct": 4,
        "mean_payload_bytes": 2.0,
        "max_payload_bytes": 2,
        "mean_payload_tokens": 1.0,
        "p50_latency_ms": 0.0,
    }
    sweep = {"best_budget_bytes": 2, "budget_summaries": [{"budget_bytes": 2, "n": 4, "metrics": condition_metrics}]}
    seed_repeat = {
        "pass_gate": True,
        "min_primary_delta_target_low": 0.5,
        "min_primary_delta_control_low": 0.49,
        "max_destruction_matched_accuracy": 0.25,
        "n_surfaces": 1,
        "n_primary_rows": 1,
        "n_destruction_rows": 1,
        "by_model": {},
        "rows": [
            {
                "surface": "s",
                "family_set": "core",
                "seed": 1,
                "run_id": "r",
                "model": "m",
                "prompt_mode": "trace_no_hint",
                "pass_gate": True,
                "matched_accuracy": 1.0,
                "target_only_accuracy": 0.25,
                "best_control_accuracy": 0.25,
                "packet_valid_rate": 1.0,
                "mean_packet_bytes": 2.0,
                "delta_target_low": 0.5,
                "delta_target_high": 0.75,
                "delta_control_low": 0.5,
                "delta_control_high": 0.75,
            }
        ],
    }

    pack = aggregate_baseline_pack(deterministic=[("surface", sweep)], seed_repeat=seed_repeat)

    assert pack["pass_gate"] is True
    assert "not raw-log" in pack["claim_boundary"]
    assert pack["systems"]["deterministic_text_relay_mean_accuracy"] == 0.25
    assert any(row["condition"] == "target_wrapper" for row in pack["deterministic_rows"])
    assert any("JSON" in row["gap"] for row in pack["remaining_gaps"])
