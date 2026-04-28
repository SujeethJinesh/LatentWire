from __future__ import annotations

import json

from scripts import build_source_private_tool_trace_figures as figures


def _summary() -> dict:
    return {
        "budget_summaries": [
            {
                "budget_bytes": 2,
                "metrics": {
                    "matched_repair_packet": {"accuracy": 1.0},
                    "structured_json_matched": {"accuracy": 0.25},
                    "structured_free_text_matched": {"accuracy": 0.25},
                    "structured_text_matched": {"accuracy": 0.25},
                    "full_diag_text": {"accuracy": 1.0, "mean_payload_bytes": 14.0},
                    "full_hidden_log": {"accuracy": 1.0, "mean_payload_bytes": 370.0},
                },
            },
            {
                "budget_bytes": 32,
                "metrics": {
                    "matched_repair_packet": {"accuracy": 1.0},
                    "structured_json_matched": {"accuracy": 1.0},
                    "structured_free_text_matched": {"accuracy": 1.0},
                    "structured_text_matched": {"accuracy": 0.25},
                    "full_diag_text": {"accuracy": 1.0, "mean_payload_bytes": 14.0},
                    "full_hidden_log": {"accuracy": 1.0, "mean_payload_bytes": 370.0},
                },
            },
        ]
    }


def test_extract_rate_rows_includes_packet_and_text_relays() -> None:
    rows = figures._extract_rate_rows(_summary(), _summary())
    labels = {row["interface"] for row in rows}

    assert {"packet", "json", "free_text", "hidden_log_prefix", "full_diag", "full_log"} <= labels
    assert any(row["interface"] == "json" and row["bytes"] == 32 and row["accuracy"] == 1.0 for row in rows)
    assert any(row["interface"] == "packet" and row["bytes"] == 2 and row["accuracy"] == 1.0 for row in rows)


def test_writes_svg_and_csv_assets(tmp_path) -> None:
    core = tmp_path / "core.json"
    holdout = tmp_path / "holdout.json"
    core.write_text(json.dumps(_summary()), encoding="utf-8")
    holdout.write_text(json.dumps(_summary()), encoding="utf-8")
    out = tmp_path / "figures"

    figures.main(
        [
            "--output-dir",
            str(out),
            "--core-summary",
            str(core),
            "--holdout-summary",
            str(holdout),
        ]
    )

    assert (out / "source_private_setup.svg").read_text(encoding="utf-8").startswith("<svg")
    assert "Accuracy versus communicated bytes" in (out / "rate_curve.svg").read_text(encoding="utf-8")
    assert "surface,interface,bytes,accuracy" in (out / "rate_curve.csv").read_text(encoding="utf-8")
